import { describe, expect, it } from "vitest";
import { AgentMemory, InvalidTransitionError, NotFoundError, SqliteStore } from "../src/index.js";
import type { Retriever, Scope } from "../src/index.js";

const scope: Scope = { namespace: "sup", userId: "u1", agentId: "a1" };
const mk = (retriever?: Retriever) =>
  new AgentMemory({ store: new SqliteStore({ path: ":memory:" }), ...(retriever ? { retriever } : {}), config: { minLexicalSimilarity: 0 } });

describe("supersession", () => {
  it("remember with supersedes marks the old version replaced and recall returns only the newest", async () => {
    const m = mk();
    const v1 = await m.remember(scope, { text: "Pylon API rate limit is 100 requests per minute", trigger: "pylon rate limit" });
    const v2 = await m.remember(scope, { text: "Pylon API rate limit is 500 requests per minute", trigger: "pylon rate limit", supersedes: v1.lessonId });
    const old = (await m.listLessons(scope, { status: ["replaced"] }))[0]!;
    expect(old.lessonId).toBe(v1.lessonId);
    expect(old.replacedBy).toBe(v2.lessonId);
    const got = await m.recall(scope, "what is the pylon rate limit", 5);
    expect(got.map((l) => l.lessonId)).toEqual([v2.lessonId]);
    expect(v2.extra.supersedes).toEqual([v1.lessonId]);
  });

  it("history returns the chain oldest-first and current follows replacedBy", async () => {
    const m = mk();
    const a = await m.remember(scope, { text: "v1 limit 100" });
    const b = await m.remember(scope, { text: "v2 limit 250", supersedes: a.lessonId });
    const c = await m.remember(scope, { text: "v3 limit 500", supersedes: b.lessonId });
    expect((await m.history(scope, a.lessonId)).map((l) => l.text)).toEqual(["v1 limit 100", "v2 limit 250", "v3 limit 500"]);
    expect((await m.current(scope, a.lessonId)).lessonId).toBe(c.lessonId);
  });

  it("supersede links two existing lessons", async () => {
    const m = mk();
    const a = await m.remember(scope, { text: "office is in Pune" });
    const b = await m.remember(scope, { text: "office moved to Bangalore" });
    const r = await m.supersede(scope, a.lessonId, b.lessonId);
    expect(r.status).toBe("replaced");
    expect((await m.listLessons(scope)).map((l) => l.lessonId)).toEqual([b.lessonId]);
  });

  it("rejects bad transitions atomically", async () => {
    const m = mk();
    const a = await m.remember(scope, { text: "x1" });
    await expect(m.supersede(scope, a.lessonId, a.lessonId)).rejects.toBeInstanceOf(InvalidTransitionError);
    await expect(m.remember(scope, { text: "x2", supersedes: "nope" })).rejects.toBeInstanceOf(NotFoundError);
    expect((await m.listLessons(scope)).length).toBe(1); // failed remember inserted nothing
    const b = await m.remember(scope, { text: "x3", supersedes: a.lessonId });
    await expect(m.remember(scope, { text: "x4", supersedes: a.lessonId })).rejects.toBeInstanceOf(InvalidTransitionError);
    await expect(m.supersede(scope, b.lessonId, a.lessonId)).rejects.toBeInstanceOf(InvalidTransitionError);
  });

  it("recall drops non-active lessons even from a custom retriever", async () => {
    const leaky: Retriever = { rank: async (s, _t, _e, _k, store) => store.listLessons(s, { status: ["active", "replaced"] }) };
    const m = mk(leaky);
    const a = await m.remember(scope, { text: "old" });
    const b = await m.remember(scope, { text: "new", supersedes: a.lessonId });
    expect((await m.recall(scope, "anything")).map((l) => l.lessonId)).toEqual([b.lessonId]);
  });

  it("supersession is scoped: another scope cannot replace my lesson", async () => {
    const m = mk();
    const a = await m.remember(scope, { text: "mine" });
    await expect(m.remember({ ...scope, userId: "u2" }, { text: "theirs", supersedes: a.lessonId })).rejects.toBeInstanceOf(NotFoundError);
  });
});
