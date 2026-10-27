// test/storeContract.ts
import { describe, expect, it } from "vitest";
import type { Event, Lesson, LessonUse, MemoryStore, Outcome, Run, Scope } from "../src/index.js";

export function describeStoreContract(
  name: string,
  createStore: () => Promise<MemoryStore> | MemoryStore,
): void {
  describe(`MemoryStore Contract: ${name}`, () => {
    const scopeA: Scope = { namespace: "app-test", userId: "user-1", agentId: "agent-1" };

    it("handles runs CRUD and nextSeq", async () => {
      const store = await createStore();
      await store.init();

      const runId = "run_test_1";
      const run: Omit<Run, "events"> = {
        runId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        task: "solve mystery",
        startedAt: "2026-10-27T10:00:00.000+00:00",
        endedAt: null,
        outcome: null,
        meta: { model: "fake-gpt" },
        extra: { attempt: 1 },
      };

      await store.insertRun(run);

      // Verify nextSeq starts at 1
      const initialSeq = await store.nextSeq(scopeA, runId);
      expect(initialSeq).toBe(1);

      // Verify getRun
      let fetched = await store.getRun(scopeA, runId);
      expect(fetched).not.toBeNull();
      expect(fetched?.runId).toBe(runId);
      expect(fetched?.task).toBe("solve mystery");
      expect(fetched?.meta).toEqual({ model: "fake-gpt" });
      expect(fetched?.extra).toEqual({ attempt: 1 });
      expect(fetched?.events).toEqual([]);

      // Insert events
      const ev1: Event = {
        eventId: "ev_1",
        runId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        seq: 1,
        ts: "2026-10-27T10:01:00.000+00:00",
        type: "step",
        input: "input payload",
        tokens: 10,
        costUsd: 0.01,
        latencyMs: 150,
        extra: { stepNum: 1 },
      };

      const ev2: Event = {
        eventId: "ev_2",
        runId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        seq: 2,
        ts: "2026-10-27T10:02:00.000+00:00",
        type: "tool_call",
        tool: "search",
        input: "{\"q\":\"query\"}",
        output: "{\"res\":\"found\"}",
        tokens: 25,
        costUsd: 0.02,
        latencyMs: 300,
        extra: { toolCallId: "call_1" },
      };

      await store.insertEvent(ev1);
      const afterEv1Seq = await store.nextSeq(scopeA, runId);
      expect(afterEv1Seq).toBe(2);

      await store.insertEvent(ev2);
      const afterEv2Seq = await store.nextSeq(scopeA, runId);
      expect(afterEv2Seq).toBe(3);

      // Verify events in order
      fetched = await store.getRun(scopeA, runId);
      expect(fetched?.events).toHaveLength(2);
      expect(fetched?.events[0]?.seq).toBe(1);
      expect(fetched?.events[0]?.type).toBe("step");
      expect(fetched?.events[0]?.extra).toEqual({ stepNum: 1 });
      expect(fetched?.events[1]?.seq).toBe(2);
      expect(fetched?.events[1]?.tool).toBe("search");

      // Update run
      const outcome: Outcome = { status: "success" };
      await store.updateRun(scopeA, runId, {
        endedAt: "2026-10-27T10:05:00.000+00:00",
        outcome,
        meta: { model: "fake-gpt", cost: 0.03, totalSteps: 2 },
      });

      fetched = await store.getRun(scopeA, runId);
      expect(fetched?.endedAt).toBe("2026-10-27T10:05:00.000+00:00");
      expect(fetched?.outcome).toEqual(outcome);
      expect(fetched?.meta).toEqual({ model: "fake-gpt", cost: 0.03, totalSteps: 2 });

      await store.close();
    });

    it("handles lessons CRUD, query filters, search ordering, and delete", async () => {
      const store = await createStore();
      await store.init();

      const lesson1: Lesson = {
        lessonId: "les_1",
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        text: "Avoid looping without changing search parameters",
        trigger: "search fails twice",
        kind: "avoid",
        provenance: {
          createdBy: "distiller",
          sourceRunIds: ["run_1"],
          sourceEventIds: ["ev_1"],
        },
        confidence: 0.8,
        uses: 5,
        helped: 4,
        hurt: 1,
        createdAt: "2026-10-27T10:00:00.000+00:00",
        updatedAt: "2026-10-27T10:00:00.000+00:00",
        status: "active",
        replacedBy: null,
        embedding: new Float32Array([1, 0, 0]),
        extra: { tag: "search" },
      };

      const lesson2: Lesson = {
        lessonId: "les_2",
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        text: "Prefer checking filter parameters before retrying",
        trigger: "api returns empty list",
        kind: "prefer",
        provenance: {
          createdBy: "manual",
          sourceRunIds: [],
          sourceEventIds: [],
        },
        confidence: 0.6,
        uses: 2,
        helped: 1,
        hurt: 1,
        createdAt: "2026-10-27T10:05:00.000+00:00",
        updatedAt: "2026-10-27T10:05:00.000+00:00",
        status: "candidate",
        replacedBy: null,
        embedding: new Float32Array([0.707106, 0.707106, 0]),
        extra: { tag: "api" },
      };

      const lesson3: Lesson = {
        lessonId: "les_3",
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        text: "Fix broken handoff by verifying schema",
        trigger: "handoff fails",
        kind: "fix",
        provenance: {
          createdBy: "distiller",
          sourceRunIds: ["run_2"],
          sourceEventIds: ["ev_5"],
        },
        confidence: 0.9,
        uses: 0,
        helped: 0,
        hurt: 0,
        createdAt: "2026-10-27T10:10:00.000+00:00",
        updatedAt: "2026-10-27T10:10:00.000+00:00",
        status: "active",
        replacedBy: "les_1",
        embedding: new Float32Array([0, 1, 0]),
        extra: { tag: "handoff" },
      };

      await store.insertLesson(lesson1);
      await store.insertLesson(lesson2);
      await store.insertLesson(lesson3);

      // Verify getLesson
      const fetched1 = await store.getLesson(scopeA, "les_1");
      expect(fetched1).not.toBeNull();
      expect(fetched1?.text).toBe(lesson1.text);
      expect(fetched1?.embedding).toBeInstanceOf(Float32Array);
      expect(Array.from(fetched1?.embedding ?? [])).toEqual([1, 0, 0]);
      expect(fetched1?.provenance.createdBy).toBe("distiller");

      // Verify updateLesson
      const updated1 = await store.updateLesson(scopeA, "les_1", {
        text: "Updated instruction",
        confidence: 0.95,
      });
      expect(updated1.text).toBe("Updated instruction");
      expect(updated1.confidence).toBe(0.95);

      // List filtering by status
      const activeList = await store.listLessons(scopeA, { status: ["active"] });
      expect(activeList).toHaveLength(2);
      expect(activeList.map((l) => l.lessonId).sort()).toEqual(["les_1", "les_3"]);

      const candidateList = await store.listLessons(scopeA, { status: ["candidate"] });
      expect(candidateList).toHaveLength(1);
      expect(candidateList[0]?.lessonId).toBe("les_2");

      // List filtering by kind
      const fixList = await store.listLessons(scopeA, { kind: ["fix"] });
      expect(fixList).toHaveLength(1);
      expect(fixList[0]?.lessonId).toBe("les_3");

      // List filtering by minConfidence
      const highConf = await store.listLessons(scopeA, { minConfidence: 0.9 });
      expect(highConf).toHaveLength(2); // les_1 (0.95) and les_3 (0.9)

      // List filtering by textContains
      const filteredText = await store.listLessons(scopeA, { textContains: "handoff" });
      expect(filteredText).toHaveLength(1);
      expect(filteredText[0]?.lessonId).toBe("les_3");

      // Search lessons ordered by cosine similarity
      // Query vector [1, 0, 0] should match les_1 best (sim ~1.0), then les_2 (sim ~0.707), then les_3 (sim ~0)
      const hits = await store.searchLessons(scopeA, new Float32Array([1, 0, 0]), 10);
      expect(hits).toHaveLength(3);
      expect(hits[0]?.lesson.lessonId).toBe("les_1");
      expect(hits[0]?.similarity).toBeCloseTo(1.0, 4);
      expect(hits[1]?.lesson.lessonId).toBe("les_2");
      expect(hits[1]?.similarity).toBeCloseTo(0.7071, 3);
      expect(hits[2]?.lesson.lessonId).toBe("les_3");
      expect(hits[2]?.similarity).toBeCloseTo(0.0, 4);

      // Search with limit k
      const top2 = await store.searchLessons(scopeA, new Float32Array([1, 0, 0]), 2);
      expect(top2).toHaveLength(2);
      expect(top2[0]?.lesson.lessonId).toBe("les_1");
      expect(top2[1]?.lesson.lessonId).toBe("les_2");

      // Search with query filter (status: ["active"])
      const activeHits = await store.searchLessons(scopeA, new Float32Array([1, 0, 0]), 10, {
        status: ["active"],
      });
      expect(activeHits).toHaveLength(2);
      expect(activeHits.map((h) => h.lesson.lessonId)).toEqual(["les_1", "les_3"]);

      // Real delete
      // First verify les_3 has replacedBy = "les_1"
      let les3 = await store.getLesson(scopeA, "les_3");
      expect(les3?.replacedBy).toBe("les_1");

      // Delete les_1
      const deleted = await store.deleteLesson(scopeA, "les_1");
      expect(deleted).toBe(true);

      // les_1 is gone
      const deletedCheck = await store.getLesson(scopeA, "les_1");
      expect(deletedCheck).toBeNull();

      // les_3's replacedBy should now be NULL
      les3 = await store.getLesson(scopeA, "les_3");
      expect(les3?.replacedBy).toBeNull();

      // Deleting again returns false
      const deletedAgain = await store.deleteLesson(scopeA, "les_1");
      expect(deletedAgain).toBe(false);

      await store.close();
    });

    it("handles lesson uses upsert and cascade deletion", async () => {
      const store = await createStore();
      await store.init();

      const runId = "run_u1";
      await store.insertRun({
        runId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        task: "task",
        startedAt: "2026-10-27T10:00:00.000+00:00",
        endedAt: null,
        outcome: null,
        meta: {},
        extra: {},
      });

      const lessonId = "les_u1";
      await store.insertLesson({
        lessonId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        text: "use test",
        trigger: "trigger",
        kind: "avoid",
        provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
        confidence: 0.5,
        uses: 0,
        helped: 0,
        hurt: 0,
        createdAt: "2026-10-27T10:00:00.000+00:00",
        updatedAt: "2026-10-27T10:00:00.000+00:00",
        status: "active",
        replacedBy: null,
        embedding: null,
        extra: {},
      });

      const use: LessonUse = {
        runId,
        lessonId,
        namespace: scopeA.namespace,
        userId: scopeA.userId,
        agentId: scopeA.agentId,
        helped: null,
      };

      const res1 = await store.upsertLessonUse(use);
      expect(res1.inserted).toBe(true);

      const res2 = await store.upsertLessonUse({ ...use, helped: true });
      expect(res2.inserted).toBe(false);

      // Deleting lesson also removes its uses
      await store.deleteLesson(scopeA, lessonId);

      await store.close();
    });

    it("handles transaction commit and rollback", async () => {
      const store = await createStore();
      await store.init();

      const runIdSuccess = "run_tx_success";
      await store.transaction(async () => {
        await store.insertRun({
          runId: runIdSuccess,
          namespace: scopeA.namespace,
          userId: scopeA.userId,
          agentId: scopeA.agentId,
          task: "tx success",
          startedAt: "2026-10-27T10:00:00.000+00:00",
          endedAt: null,
          outcome: null,
          meta: {},
          extra: {},
        });
      });

      const fetchedSuccess = await store.getRun(scopeA, runIdSuccess);
      expect(fetchedSuccess).not.toBeNull();

      const runIdFail = "run_tx_fail";
      await expect(
        store.transaction(async () => {
          await store.insertRun({
            runId: runIdFail,
            namespace: scopeA.namespace,
            userId: scopeA.userId,
            agentId: scopeA.agentId,
            task: "tx fail",
            startedAt: "2026-10-27T10:00:00.000+00:00",
            endedAt: null,
            outcome: null,
            meta: {},
            extra: {},
          });
          throw new Error("Intentional rollback error");
        }),
      ).rejects.toThrow("Intentional rollback error");

      const fetchedFail = await store.getRun(scopeA, runIdFail);
      expect(fetchedFail).toBeNull();

      await store.close();
    });
  });
}
