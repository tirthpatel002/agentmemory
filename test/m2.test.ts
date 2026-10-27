// test/m2.test.ts
import { describe, expect, it, vi } from "vitest";
import {
  AgentMemory,
  asPrompt,
  DEFAULT_MEMORY_CONFIG,
  mergeConfig,
  NotFoundError,
  SqliteStore,
  WeightedRetriever,
  wordJaccardSimilarity,
} from "../src/index.js";
import type { Embedder, Lesson, Retriever, Scope } from "../src/index.js";

describe("Checkpoint M2: Config, Retriever, recall, asPrompt, markUsed", () => {
  const scope1: Scope = { namespace: "test-space", userId: "usr-m2", agentId: "agt-m2" };
  const scope2: Scope = { namespace: "other-space", userId: "usr-other", agentId: "agt-other" };

  it("Partial config merges with defaults", () => {
    // 1. undefined returns exact defaults
    const cfgDefault = mergeConfig();
    expect(cfgDefault).toEqual(DEFAULT_MEMORY_CONFIG);

    // 2. Partial weights override
    const cfgWeights = mergeConfig({ weights: { similarity: 0.85 } });
    expect(cfgWeights.weights.similarity).toBe(0.85);
    expect(cfgWeights.weights.helpfulness).toBe(0.3);
    expect(cfgWeights.weights.recency).toBe(0.1);
    expect(cfgWeights.minSimilarity).toBe(0.3);
    expect(cfgWeights.recencyHalfLifeDays).toBe(14);
    expect(cfgWeights.retire.minUses).toBe(3);

    // 3. Partial retire override
    const cfgRetire = mergeConfig({ retire: { minUses: 5 } });
    expect(cfgRetire.retire.minUses).toBe(5);
    expect(cfgRetire.retire.rule).toBe("hurtGtHelped");

    // 4. Overriding top-level knobs
    const cfgKnobs = mergeConfig({ minSimilarity: 0.45, maxLessonsPerRun: 5 });
    expect(cfgKnobs.minSimilarity).toBe(0.45);
    expect(cfgKnobs.maxLessonsPerRun).toBe(5);
    expect(cfgKnobs.weights).toEqual(DEFAULT_MEMORY_CONFIG.weights);

    // 5. Passed through AgentMemory constructor
    const mem = new AgentMemory({
      config: {
        minSimilarity: 0.5,
        weights: { similarity: 0.7 },
      },
    });
    expect(mem.config.minSimilarity).toBe(0.5);
    expect(mem.config.weights.similarity).toBe(0.7);
    expect(mem.config.weights.helpfulness).toBe(0.3);
    expect(mem.config.weights.recency).toBe(0.1);
  });

  it("ranking order with fixed fake embeddings (similarity vs helpfulness vs recency, with fake clock)", async () => {
    // Fixed reference time: 2026-10-27T12:00:00.000Z
    const refTime = new Date("2026-10-27T12:00:00.000Z");
    let fakeCurrentTime = new Date(refTime.getTime());
    const retriever = new WeightedRetriever({ now: () => fakeCurrentTime });

    const store = new SqliteStore({ path: ":memory:" });
    const fakeEmbedder: Embedder = async (texts: string[]) => {
      // Return fixed query vector [1, 0, 0]
      return texts.map(() => new Float32Array([1, 0, 0]));
    };

    const mem = new AgentMemory({ store, embedder: fakeEmbedder, retriever });
    await mem.init();

    // Lesson 1: Updated right now (age 0d -> recency = 1.0), unused (helped 0, hurt 0 -> helpfulness = 0.5)
    // Embedding [1, 0, 0] -> cosine similarity with [1, 0, 0] = 1.0
    // Expected score: 0.6 * 1.0 + 0.3 * 0.5 + 0.1 * 1.0 = 0.60 + 0.15 + 0.10 = 0.85
    const l1: Lesson = {
      lessonId: "les_rank_1",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Lesson 1 text",
      trigger: "trigger 1",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: refTime.toISOString(),
      updatedAt: refTime.toISOString(),
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    };

    // Lesson 2: Updated 14 days ago (recency = e^(-14/14) = e^(-1) ≈ 0.367879)
    // High helpfulness: helped 9, hurt 0 -> helpfulness = (9+1)/(9+0+2) = 10/11 ≈ 0.909091
    // Embedding [0.8, 0.6, 0] -> cosine similarity with [1, 0, 0] = 0.8
    // Expected score: 0.6 * 0.8 + 0.3 * (10/11) + 0.1 * e^(-1)
    // = 0.48 + 0.272727 + 0.036788 = 0.789515
    const l2Time = new Date(refTime.getTime() - 14 * 24 * 60 * 60 * 1000);
    const l2: Lesson = {
      lessonId: "les_rank_2",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Lesson 2 text",
      trigger: "trigger 2",
      kind: "fix",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 9,
      helped: 9,
      hurt: 0,
      createdAt: l2Time.toISOString(),
      updatedAt: l2Time.toISOString(),
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([0.8, 0.6, 0]),
      extra: {},
    };

    // Lesson 3: Updated 28 days ago (recency = e^(-28/14) = e^(-2) ≈ 0.135335)
    // Low helpfulness: helped 1, hurt 5 -> helpfulness = (1+1)/(1+5+2) = 2/8 = 0.25
    // Embedding [0.95, 0.31225, 0] -> cosine similarity with [1, 0, 0] = 0.95
    // Expected score: 0.6 * 0.95 + 0.3 * 0.25 + 0.1 * e^(-2)
    // = 0.57 + 0.075 + 0.013534 = 0.658534
    const l3Time = new Date(refTime.getTime() - 28 * 24 * 60 * 60 * 1000);
    const l3: Lesson = {
      lessonId: "les_rank_3",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Lesson 3 text",
      trigger: "trigger 3",
      kind: "prefer",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 6,
      helped: 1,
      hurt: 5,
      createdAt: l3Time.toISOString(),
      updatedAt: l3Time.toISOString(),
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([0.95, 0.3122498998, 0]),
      extra: {},
    };

    await store.insertLesson(l1);
    await store.insertLesson(l2);
    await store.insertLesson(l3);

    const recalled = await mem.recall(scope1, "test task", 10);
    expect(recalled).toHaveLength(3);

    // Ranking order: L1 > L2 > L3
    expect(recalled[0]?.lessonId).toBe("les_rank_1");
    expect(recalled[0]?.score).toBeCloseTo(0.85, 3);

    expect(recalled[1]?.lessonId).toBe("les_rank_2");
    expect(recalled[1]?.score).toBeCloseTo(0.7895, 3);

    expect(recalled[2]?.lessonId).toBe("les_rank_3");
    expect(recalled[2]?.score).toBeCloseTo(0.6585, 3);

    // Advance clock by 28 days: l1 is now 28 days old
    fakeCurrentTime = new Date(refTime.getTime() + 28 * 24 * 60 * 60 * 1000);
    const recalledAfterAdvance = await mem.recall(scope1, "test task", 10);
    expect(recalledAfterAdvance).toHaveLength(3);
    // L1's recency decayed from 1.0 to e^(-2) ≈ 0.1353, decreasing its score by ~0.086
    const newL1 = recalledAfterAdvance.find((l) => l.lessonId === "les_rank_1");
    expect(newL1?.score).toBeCloseTo(0.6 * 1.0 + 0.3 * 0.5 + 0.1 * Math.exp(-2), 3);

    await mem.close();
  });

  it("minSimilarity cut excludes lessons below threshold", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const fakeEmbedder: Embedder = async (texts) => texts.map(() => new Float32Array([1, 0, 0]));
    const mem = new AgentMemory({ store, embedder: fakeEmbedder });
    await mem.init();

    // L_high: sim = 1.0 (> 0.3)
    await store.insertLesson({
      lessonId: "les_high_sim",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "High similarity lesson",
      trigger: "high",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    });

    // L_low: sim = 0.2 (< 0.3 minSimilarity)
    await store.insertLesson({
      lessonId: "les_low_sim",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Low similarity lesson",
      trigger: "low",
      kind: "fix",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 10,
      helped: 10,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([0.2, 0.9797958971, 0]),
      extra: {},
    });

    const recalled = await mem.recall(scope1, "any query", 5);
    expect(recalled).toHaveLength(1);
    expect(recalled[0]?.lessonId).toBe("les_high_sim");

    await mem.close();
  });

  it("config override changes ranking (e.g. weights)", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const fakeEmbedder: Embedder = async (texts) => texts.map(() => new Float32Array([1, 0, 0]));

    // L1: high similarity (1.0), lower helpfulness (0.5), recency 1.0
    // L2: moderate similarity (0.6), high helpfulness (0.9), recency 1.0
    const nowIso = "2026-10-27T12:00:00.000Z";
    await store.init();
    await store.insertLesson({
      lessonId: "les_sim_heavy",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Similarity heavy",
      trigger: "t1",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: nowIso,
      updatedAt: nowIso,
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    });

    await store.insertLesson({
      lessonId: "les_help_heavy",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Helpfulness heavy",
      trigger: "t2",
      kind: "fix",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 10,
      helped: 10,
      hurt: 0, // helpfulness = (10+1)/(10+0+2) = 11/12 ≈ 0.9167
      createdAt: nowIso,
      updatedAt: nowIso,
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([0.6, 0.8, 0]), // sim = 0.6
      extra: {},
    });

    // Case A: Default weights (0.6 sim, 0.3 help, 0.1 rec)
    // les_sim_heavy: 0.6*1.0 + 0.3*0.5 + 0.1*1.0 = 0.85
    // les_help_heavy: 0.6*0.6 + 0.3*0.9167 + 0.1*1.0 = 0.36 + 0.275 + 0.1 = 0.735
    // => les_sim_heavy wins
    const memDefault = new AgentMemory({ store, embedder: fakeEmbedder });
    const recalledA = await memDefault.recall(scope1, "task", 5);
    expect(recalledA[0]?.lessonId).toBe("les_sim_heavy");
    expect(recalledA[1]?.lessonId).toBe("les_help_heavy");

    // Case B: Custom weights emphasizing helpfulness (0.1 sim, 0.8 help, 0.1 rec)
    // les_sim_heavy: 0.1*1.0 + 0.8*0.5 + 0.1*1.0 = 0.60
    // les_help_heavy: 0.1*0.6 + 0.8*0.9167 + 0.1*1.0 = 0.06 + 0.733 + 0.1 = 0.893
    // => les_help_heavy wins
    const memCustom = new AgentMemory({
      store,
      embedder: fakeEmbedder,
      config: { weights: { similarity: 0.1, helpfulness: 0.8, recency: 0.1 } },
    });
    const recalledB = await memCustom.recall(scope1, "task", 5);
    expect(recalledB[0]?.lessonId).toBe("les_help_heavy");
    expect(recalledB[1]?.lessonId).toBe("les_sim_heavy");

    await store.close();
  });

  it("markUsed counters and retirement after 3 uses with hurt > helped", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const mem = new AgentMemory({ store });
    await mem.init();

    const lessonId = "les_retire_test";
    await store.insertLesson({
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Retirement test lesson",
      trigger: "retire trigger",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.7,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: "2026-10-27T10:00:00.000Z",
      updatedAt: "2026-10-27T10:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: {},
    });

    const run1 = await mem.startRun(scope1, "run 1");
    const run2 = await mem.startRun(scope1, "run 2");
    const run3 = await mem.startRun(scope1, "run 3");

    // Use 1: hurt (false)
    await mem.markUsed(scope1, run1, [lessonId], false);
    let les = await store.getLesson(scope1, lessonId);
    expect(les?.uses).toBe(1);
    expect(les?.helped).toBe(0);
    expect(les?.hurt).toBe(1);
    expect(les?.status).toBe("active");

    // Use 2: hurt ("hurt" string outcome)
    await mem.markUsed(scope1, run2, lessonId, "hurt");
    les = await store.getLesson(scope1, lessonId);
    expect(les?.uses).toBe(2);
    expect(les?.helped).toBe(0);
    expect(les?.hurt).toBe(2);
    expect(les?.status).toBe("active");

    // Use 3: hurt (false) -> uses >= 3 (retire.minUses = 3) and hurt (3) > helped (0)
    await mem.markUsed(scope1, run3, [lessonId], false);
    les = await store.getLesson(scope1, lessonId);
    expect(les?.uses).toBe(3);
    expect(les?.helped).toBe(0);
    expect(les?.hurt).toBe(3);
    // Retirement triggered: status becomes "replaced", replacedBy is null
    expect(les?.status).toBe("replaced");
    expect(les?.replacedBy).toBeNull();

    await mem.close();
  });

  it("retired lessons not recalled", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const fakeEmbedder: Embedder = async (texts) => texts.map(() => new Float32Array([1, 0, 0]));
    const mem = new AgentMemory({ store, embedder: fakeEmbedder });
    await mem.init();

    // Insert active lesson and replaced lesson
    await store.insertLesson({
      lessonId: "les_active_alive",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Still active",
      trigger: "alive",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 1,
      helped: 1,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    });

    await store.insertLesson({
      lessonId: "les_retired_dead",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Retired replaced lesson",
      trigger: "dead",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 3,
      helped: 0,
      hurt: 3,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "replaced",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    });

    const recalled = await mem.recall(scope1, "any query", 10);
    expect(recalled).toHaveLength(1);
    expect(recalled[0]?.lessonId).toBe("les_active_alive");

    await mem.close();
  });

  it("scope isolation for recall/markUsed", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const fakeEmbedder: Embedder = async (texts) => texts.map(() => new Float32Array([1, 0, 0]));
    const mem = new AgentMemory({ store, embedder: fakeEmbedder });
    await mem.init();

    const lessonId = "les_scope_1";
    await store.insertLesson({
      lessonId,
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Scope 1 secret lesson",
      trigger: "scope1",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: new Float32Array([1, 0, 0]),
      extra: {},
    });

    const runIdScope2 = await mem.startRun(scope2, "scope2 task");

    // recall in scope2 must never return scope1 lessons
    const recalledScope2 = await mem.recall(scope2, "query", 5);
    expect(recalledScope2).toHaveLength(0);

    // recall in scope1 sees it
    const recalledScope1 = await mem.recall(scope1, "query", 5);
    expect(recalledScope1).toHaveLength(1);
    expect(recalledScope1[0]?.lessonId).toBe(lessonId);

    // markUsed in scope2 referencing scope1's lesson throws NotFoundError
    await expect(mem.markUsed(scope2, runIdScope2, [lessonId], true)).rejects.toThrow(NotFoundError);

    // scope1's lesson counters remain untouched
    const les = await store.getLesson(scope1, lessonId);
    expect(les?.uses).toBe(0);
    expect(les?.helped).toBe(0);

    await mem.close();
  });

  it("asPrompt output snapshot", () => {
    // 1. Empty lessons returns empty string
    expect(asPrompt([])).toBe("");

    // 2. Exact match with SPEC.md lines 341-346
    const lessons: Lesson[] = [
      {
        lessonId: "les_1",
        namespace: "test",
        userId: "u",
        agentId: "a",
        text: "When the search tool returns 0 results twice: stop retrying the same query; widen it once, then report.",
        trigger: "search tool returns 0",
        kind: "avoid",
        provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
        confidence: 0.9,
        uses: 5,
        helped: 5,
        hurt: 0,
        createdAt: "2026-10-27T10:00:00.000Z",
        updatedAt: "2026-10-27T10:00:00.000Z",
        status: "active",
        replacedBy: null,
        embedding: null,
        extra: {},
      },
      {
        lessonId: "les_2",
        namespace: "test",
        userId: "u",
        agentId: "a",
        text: "When a write fails with a conflict: re-read the record before writing again.",
        trigger: "write conflict",
        kind: "fix",
        provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
        confidence: 0.85,
        uses: 3,
        helped: 3,
        hurt: 0,
        createdAt: "2026-10-27T10:00:00.000Z",
        updatedAt: "2026-10-27T10:00:00.000Z",
        status: "active",
        replacedBy: null,
        embedding: null,
        extra: {},
      },
      {
        lessonId: "les_3",
        namespace: "test",
        userId: "u",
        agentId: "a",
        text: "When an API returns an empty list: check the filter parameters before retrying.",
        trigger: "api empty list",
        kind: "prefer",
        provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
        confidence: 0.7,
        uses: 0,
        helped: 0,
        hurt: 0,
        createdAt: "2026-10-27T10:00:00.000Z",
        updatedAt: "2026-10-27T10:00:00.000Z",
        status: "candidate", // Candidate rendered as trial
        replacedBy: null,
        embedding: null,
        extra: {},
      },
    ];

    const rendered = asPrompt(lessons);
    const expected = [
      "Lessons from earlier runs (follow unless the task says otherwise):",
      "1. [avoid] When the search tool returns 0 results twice: stop retrying the same query; widen it once, then report.",
      "2. [fix] When a write fails with a conflict: re-read the record before writing again.",
      "3. [prefer, trial] When an API returns an empty list: check the filter parameters before retrying.",
    ].join("\n");

    expect(rendered).toBe(expected);

    // Facade method behaves identically
    const mem = new AgentMemory();
    expect(mem.asPrompt(lessons)).toBe(expected);
    expect(mem.asPrompt([])).toBe("");
  });

  it("injected custom Retriever is used (spy)", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const dummyLesson: Lesson = {
      lessonId: "les_spy_1",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Spy lesson",
      trigger: "spy trigger",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.9,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: "2026-10-27T12:00:00.000Z",
      updatedAt: "2026-10-27T12:00:00.000Z",
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: {},
      score: 0.99,
    };

    const spyRetriever: Retriever = {
      rank: vi.fn().mockResolvedValue([dummyLesson]),
    };

    const mem = new AgentMemory({ store, retriever: spyRetriever });
    await mem.init();

    const result = await mem.recall(scope1, "investigate anomalies", 4);
    expect(result).toEqual([dummyLesson]);
    expect(spyRetriever.rank).toHaveBeenCalledTimes(1);
    expect(spyRetriever.rank).toHaveBeenCalledWith(
      scope1,
      "investigate anomalies",
      null, // No embedder injected
      4,
      store,
      mem.config,
    );

    await mem.close();
  });

  it("falls back to Jaccard word-overlap similarity when no embedder is provided", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    // No embedder provided
    const mem = new AgentMemory({ store });
    await mem.init();

    // Verify wordJaccardSimilarity unit behavior
    expect(wordJaccardSimilarity("search query failed", "search query failed")).toBe(1.0);
    expect(wordJaccardSimilarity("foo bar", "baz qux")).toBe(0.0);

    const nowIso = "2026-10-27T12:00:00.000Z";

    // Lesson matching task keywords
    await store.insertLesson({
      lessonId: "les_jaccard_match",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "When the search tool returns 0 results: widen the query before retrying.",
      trigger: "search tool query returns 0 results",
      kind: "avoid",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: nowIso,
      updatedAt: nowIso,
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: {},
    });

    // Lesson with completely disjoint words
    await store.insertLesson({
      lessonId: "les_jaccard_unrelated",
      namespace: scope1.namespace,
      userId: scope1.userId,
      agentId: scope1.agentId,
      text: "Ensure database transactions always execute rollback on exceptions.",
      trigger: "database transactions rollback",
      kind: "fix",
      provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: 0.8,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: nowIso,
      updatedAt: nowIso,
      status: "active",
      replacedBy: null,
      embedding: null,
      extra: {},
    });

    const recalled = await mem.recall(scope1, "search tool returns 0 results", 5);
    expect(recalled).toHaveLength(1);
    expect(recalled[0]?.lessonId).toBe("les_jaccard_match");
    expect(recalled[0]?.score).toBeGreaterThan(0);

    await mem.close();
  });

  describe.runIf(process.env["LIVE"])("Live API checks (skipped in normal CI / without keys)", () => {
    it("runs only when LIVE env is set", () => {
      expect(process.env["LIVE"]).toBeTruthy();
    });
  });
});
