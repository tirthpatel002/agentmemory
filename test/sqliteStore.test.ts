// test/sqliteStore.test.ts
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  AgentMemory,
  cosine,
  fromBlob,
  InvalidEventError,
  InvalidTransitionError,
  MemoryError,
  NotFoundError,
  RunClosedError,
  SCHEMA_SQL,
  SqliteStore,
  toBlob,
} from "../src/index.js";
import type { Lesson, Scope } from "../src/index.js";
import { describeStoreContract } from "./storeContract.js";

// Run the full store contract suite against SqliteStore in-memory
describeStoreContract("SqliteStore (:memory:)", () => new SqliteStore({ path: ":memory:" }));

describe("SqliteStore & AgentMemory M1 Unit Tests", () => {
  const scope1: Scope = { namespace: "org-alpha", userId: "usr-1", agentId: "agt-1" };
  const scope2: Scope = { namespace: "org-beta", userId: "usr-1", agentId: "agt-1" };

  it("exports all error types properly inheriting from MemoryError", () => {
    expect(new NotFoundError("not found")).toBeInstanceOf(MemoryError);
    expect(new RunClosedError("closed")).toBeInstanceOf(MemoryError);
    expect(new InvalidEventError("invalid")).toBeInstanceOf(MemoryError);
    expect(new InvalidTransitionError("invalid transition")).toBeInstanceOf(MemoryError);
  });

  it("schema.sql matches embedded SCHEMA_SQL and is idempotent with schema_version = 2", async () => {
    const sqlFilePath = path.resolve(__dirname, "../src/store/schema.sql");
    const sqlFileContent = fs.readFileSync(sqlFilePath, "utf-8").trim();
    expect(SCHEMA_SQL.trim()).toBe(sqlFileContent);

    const store = new SqliteStore({ path: ":memory:" });
    await store.init();
    // Second init call must be idempotent
    await store.init();

    // Query schema_version directly
    const rows = (store as unknown as { db: { prepare: (sql: string) => { all: () => { version: number }[] } } }).db
      .prepare("SELECT version FROM schema_version")
      .all();

    expect(rows).toEqual([{ version: 2 }]);
    await store.close();
  });

  it("enforces scope isolation across getRun, listLessons, and forget", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const mem = new AgentMemory({ store });
    await mem.init();

    // 1. Run isolation: startRun in scope1
    const runId = await mem.startRun(scope1, "scope isolation task");

    // getRun in scope1 succeeds
    const runInScope1 = await mem.getRun(scope1, runId);
    expect(runInScope1.runId).toBe(runId);

    // getRun in scope2 with same runId throws NotFoundError
    await expect(mem.getRun(scope2, runId)).rejects.toThrow(NotFoundError);

    // 2. Lesson isolation: insert lesson directly into store under scope1
    const lessonId = "les_iso_test_1";
    const lesson: Lesson = {
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Only visible in scope1",
      trigger: "isolation test",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.75,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000+00:00",
      updatedAt: "2026-10-27T12:00:00.000+00:00",
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: {},
    };
    await store.insertLesson(lesson);

    // listLessons in scope1 sees it
    const listScope1 = await mem.listLessons(scope1);
    expect(listScope1).toHaveLength(1);
    expect(listScope1[0]?.lessonId).toBe(lessonId);

    // listLessons in scope2 returns empty list
    const listScope2 = await mem.listLessons(scope2);
    expect(listScope2).toHaveLength(0);

    // forget in scope2 throws NotFoundError
    await expect(mem.forget(scope2, lessonId)).rejects.toThrow(NotFoundError);

    // lesson is still active in scope1
    expect(await mem.listLessons(scope1)).toHaveLength(1);

    // forget in scope1 succeeds
    await mem.forget(scope1, lessonId);
    expect(await mem.listLessons(scope1)).toHaveLength(0);

    await mem.close();
  });

  it("supports open schema: custom event type, custom lesson kind, and extra JSON round-trips", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    await mem.init();

    const customEventType = "user_navigation_telemetry";
    const customLessonKind = "architecture_invariant";
    const runExtra = { traceId: "tr-987", tags: ["experiment", "v2"] };
    const eventExtra = { uiComponent: "navbar", mouseCoords: { x: 100, y: 250 } };
    const lessonExtra = { reviewerAudit: { ok: true, tier: 3 } };

    const runId = await mem.startRun(scope1, "open schema run", { model: "custom-agent" }, runExtra);

    const eventId = await mem.record(scope1, runId, {
      type: customEventType,
      tool: "custom_sensor",
      input: { sensorId: "s1" },
      output: { reading: 42 },
      extra: eventExtra,
    });

    const run = await mem.getRun(scope1, runId);
    expect(run.extra).toEqual(runExtra);
    expect(run.events).toHaveLength(1);
    expect(run.events[0]?.type).toBe(customEventType);
    expect(run.events[0]?.eventId).toBe(eventId);
    expect(run.events[0]?.extra).toEqual(eventExtra);

    // Insert lesson with custom kind and extra
    const lessonId = "les_custom_kind";
    await mem.store.insertLesson({
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Custom kind instruction text",
      trigger: "when custom trigger occurs",
      kind: customLessonKind,
      provenance: { createdBy: "manual", sourceRunIds: [runId], sourceEventIds: [eventId] },
      confidence: 0.9,
      uses: 1,
      helped: 1,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000+00:00",
      updatedAt: "2026-10-27T12:00:00.000+00:00",
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: lessonExtra,
    });

    const lessons = await mem.listLessons(scope1, { kind: [customLessonKind] });
    expect(lessons).toHaveLength(1);
    expect(lessons[0]?.kind).toBe(customLessonKind);
    expect(lessons[0]?.extra).toEqual(lessonExtra);

    await mem.close();
  });

  it("sequences events 1..n and cuts input/output to 8 KB", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    const runId = await mem.startRun(scope1, "cut test task");

    // Event 1 with normal text
    await mem.record(scope1, runId, {
      type: "step",
      input: "normal input",
      output: "normal output",
    });

    // Event 2 with >8 KB string payload
    const bigInput = "X".repeat(12000);
    const bigOutput = "Y".repeat(10000);
    await mem.record(scope1, runId, {
      type: "step",
      input: bigInput,
      output: bigOutput,
    });

    // Event 3 with >8 KB JSON object payload
    const bigObj = { data: "Z".repeat(10000) };
    await mem.record(scope1, runId, {
      type: "step",
      input: bigObj,
    });

    const run = await mem.getRun(scope1, runId);
    expect(run.events).toHaveLength(3);

    // Check seq numbers
    expect(run.events[0]?.seq).toBe(1);
    expect(run.events[1]?.seq).toBe(2);
    expect(run.events[2]?.seq).toBe(3);

    // Check 8 KB cuts
    const ev2Input = run.events[1]?.input as string;
    const ev2Output = run.events[1]?.output as string;
    expect(ev2Input.length).toBe(8192);
    expect(ev2Input).toBe(bigInput.slice(0, 8192));
    expect(ev2Output.length).toBe(8192);
    expect(ev2Output).toBe(bigOutput.slice(0, 8192));

    const ev3Input = run.events[2]?.input as string;
    expect(ev3Input.length).toBe(8192);
    expect(ev3Input).toBe(JSON.stringify(bigObj).slice(0, 8192));

    await mem.close();
  });

  it("computes endRun totals (cost and totalSteps)", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    const runId = await mem.startRun(scope1, "totals run", { model: "test-model", cost: 0.01 });

    await mem.record(scope1, runId, { type: "step", costUsd: 0.02 });
    await mem.record(scope1, runId, { type: "tool_call", costUsd: 0.05 });
    await mem.record(scope1, runId, { type: "tool_result", costUsd: 0.01 });

    const lessonsReturned = await mem.endRun(scope1, runId, { status: "success" });
    expect(lessonsReturned).toEqual([]);

    const run = await mem.getRun(scope1, runId);
    expect(run.endedAt).not.toBeNull();
    expect(run.meta.totalSteps).toBe(3);
    expect(run.meta.cost).toBeCloseTo(0.09, 5); // 0.01 initial + 0.02 + 0.05 + 0.01

    await mem.close();
  });

  it("throws RunClosedError on ended runs", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    const runId = await mem.startRun(scope1, "closed run test");

    await mem.record(scope1, runId, { type: "step" });
    await mem.endRun(scope1, runId, { status: "success" });

    // Recording on ended run throws RunClosedError
    await expect(mem.record(scope1, runId, { type: "step" })).rejects.toThrow(RunClosedError);

    // Ending already ended run throws RunClosedError
    await expect(mem.endRun(scope1, runId, { status: "success" })).rejects.toThrow(RunClosedError);

    await mem.close();
  });

  it("throws InvalidEventError for fail/partial outcome without failureType", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    const runId = await mem.startRun(scope1, "failureType check");

    // Fail status without failureType throws
    await expect(
      mem.endRun(scope1, runId, { status: "fail", reason: "Something crashed" }),
    ).rejects.toThrow(InvalidEventError);

    // Partial status without failureType throws
    await expect(
      mem.endRun(scope1, runId, { status: "partial", reason: "Halfway done" }),
    ).rejects.toThrow(InvalidEventError);

    // With failureType succeeds
    await expect(
      mem.endRun(scope1, runId, {
        status: "fail",
        failureType: "loop",
        reason: "Detected infinite loop",
      }),
    ).resolves.toEqual([]);

    await mem.close();
  });

  it("supports editLesson method on AgentMemory facade", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    await mem.init();
    const lessonId = "les_edit_1";
    await mem.store.insertLesson({
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Initial lesson text",
      trigger: "initial trigger",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
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

    const updated = await mem.editLesson(scope1, lessonId, "Edited lesson instruction");
    expect(updated.text).toBe("Edited lesson instruction");
    expect(updated.updatedAt).not.toBe("2026-10-27T10:00:00.000+00:00");

    // Editing in wrong scope throws NotFoundError
    await expect(mem.editLesson(scope2, lessonId, "Wrong scope edit")).rejects.toThrow(NotFoundError);

    await mem.close();
  });

  it("verifies exact Float32Array blob round-trip and cosine calculations", () => {
    const original = new Float32Array([
      0.0,
      1.0,
      -1.0,
      3.1415927,
      -0.0000456,
      123456.78,
      Number.MIN_VALUE,
    ]);

    const blob = toBlob(original);
    const recovered = fromBlob(blob);

    expect(recovered.length).toBe(original.length);
    for (let i = 0; i < original.length; i++) {
      expect(recovered[i]).toBe(original[i]);
    }

    // Cosine verification
    const v1 = new Float32Array([1, 0, 0]);
    const v2 = new Float32Array([1, 0, 0]);
    const v3 = new Float32Array([0, 1, 0]);
    expect(cosine(v1, v2)).toBeCloseTo(1.0, 5);
    expect(cosine(v1, v3)).toBeCloseTo(0.0, 5);
    expect(cosine(new Float32Array([]), new Float32Array([]))).toBe(0);
  });

  it("grep check: every SQL statement touching runs/events/lessons/lesson_uses includes scope filters", () => {
    const sqliteSourcePath = path.resolve(__dirname, "../src/store/sqlite.ts");
    const sourceCode = fs.readFileSync(sqliteSourcePath, "utf-8");

    // Match all string literals (backtick template literals and quoted strings)
    const regex = /`([\s\S]*?)`|"([^"\n]+)"/g;
    let m: RegExpExecArray | null;
    let checkedCount = 0;

    const tables = ["runs", "events", "lessons", "lesson_uses"];

    while ((m = regex.exec(sourceCode)) !== null) {
      const content = (m[1] ?? m[2] ?? "").trim();
      const touchesTable = tables.some((tbl) => new RegExp(`\\b${tbl}\\b`, "i").test(content));

      if (touchesTable && /(SELECT|INSERT|UPDATE|DELETE)/i.test(content)) {
        checkedCount++;
        expect(content).toContain("namespace");
        expect(content).toContain("user_id");
        expect(content).toContain("agent_id");
      }
    }

    // Ensure we checked all queries in sqlite.ts
    expect(checkedCount).toBeGreaterThanOrEqual(15);
  });

  it("forget leaves no trace of the lesson text in a file DB after vacuum()", async () => {
    const tmpDir = os.tmpdir();
    const dbPath = path.join(tmpDir, `agentmemory-vacuum-test-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
    const walPath = `${dbPath}-wal`;

    const store = new SqliteStore({ path: dbPath });
    await store.init();

    const canary = "SUPER_SECRET_LESSON_INSTRUCTION_987654321_CANARY";
    const lessonId = "les_vacuum_1";

    await store.insertLesson({
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: canary,
      trigger: "vacuum trigger",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.9,
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

    const canaryBuf = Buffer.from(canary);

    // Verify data exists on disk across main db file or wal file
    let fileBuffer = fs.readFileSync(dbPath);
    let walBuffer = fs.existsSync(walPath) ? fs.readFileSync(walPath) : Buffer.alloc(0);
    expect(fileBuffer.includes(canaryBuf) || walBuffer.includes(canaryBuf)).toBe(true);

    // Delete the lesson
    const deleted = await store.deleteLesson(scope1, lessonId);
    expect(deleted).toBe(true);

    // Run vacuum (which checkpoints and reclaims all pages)
    store.vacuum();
    await store.close();

    // Check that NO file on disk (neither main DB file nor WAL file) contains the canary string
    fileBuffer = fs.readFileSync(dbPath);
    walBuffer = fs.existsSync(walPath) ? fs.readFileSync(walPath) : Buffer.alloc(0);
    expect(fileBuffer.includes(canaryBuf)).toBe(false);
    expect(walBuffer.includes(canaryBuf)).toBe(false);

    // Clean up temporary files
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);
    if (fs.existsSync(`${dbPath}-wal`)) fs.unlinkSync(`${dbPath}-wal`);
    if (fs.existsSync(`${dbPath}-shm`)) fs.unlinkSync(`${dbPath}-shm`);
  });
});
