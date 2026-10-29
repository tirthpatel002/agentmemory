import fs from "node:fs";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  AgentMemory,
  bm25Similarities,
  DistillError,
  LlmDistiller,
  mergeConfig,
  SqliteStore,
  type EndRunError,
  type Lesson,
  type LLM,
  type Scope,
} from "../src/index.js";

// Lessons Muse Glimmer distilled in the 2026-10-29 live test (Jaccard scored them 0.14-0.18 and recall was empty).
const LIVE = {
  emptyCsv: {
    text: "Do not email the sales CSV if the query returns zero rows; abort and investigate instead of sending an empty attachment.",
    trigger: "SQL query returns 0 rows but email is still sent with empty CSV",
  },
  partition: {
    text: "Verify the correct sales table/partition exists before querying; sales data may be partitioned by year.",
    trigger: "Finance reports CSV is empty and sales table is partitioned, query sales_2026 instead",
  },
  pdf: {
    text: "When pdf.render times out with a suggested alternative template, do not retry the same template; switch to the suggested template immediately.",
    trigger: "pdf.render timeout error mentions an alternative template such as 'weekly-lite'",
  },
};
const CSV_TASK = "Export the sales report as CSV and email finance";
const PDF_TASK = "Render the weekly marketing PDF summary";
const UNRELATED_TASK = "Book a flight to Berlin for next Tuesday";

const scope: Scope = { namespace: "live-fixes", userId: "_system", agentId: "agt" };

function lesson(id: string, l: { text: string; trigger: string }): Lesson {
  const now = new Date().toISOString();
  return {
    lessonId: id, ...scope, text: l.text, trigger: l.trigger, kind: "avoid",
    provenance: { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
    confidence: 0.7, uses: 0, helped: 0, hurt: 0, createdAt: now, updatedAt: now,
    status: "active", replacedBy: null, embedding: null, extra: {},
  };
}

describe("lexical recall without an embedder (BM25)", () => {
  const docs = [LIVE.emptyCsv, LIVE.partition, LIVE.pdf].map((l) => `${l.trigger} ${l.text}`);
  const min = mergeConfig().minLexicalSimilarity;

  it("scores the live-test examples above the threshold and unrelated lessons below it", () => {
    const csv = bm25Similarities(CSV_TASK, docs);
    const pdf = bm25Similarities(PDF_TASK, docs);
    const none = bm25Similarities(UNRELATED_TASK, docs);
    expect(csv[0]).toBeGreaterThanOrEqual(min);
    expect(csv[1]).toBeGreaterThanOrEqual(min);
    expect(csv[2]).toBeLessThan(min);
    expect(pdf[2]).toBeGreaterThanOrEqual(min);
    expect(pdf[0]).toBeLessThan(min);
    expect(pdf[1]).toBeLessThan(min);
    for (const s of none) expect(s).toBeLessThan(min);
    for (const s of [...csv, ...pdf, ...none]) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("recall() returns related lessons and skips unrelated ones", async () => {
    const store = new SqliteStore({ path: ":memory:" });
    const mem = new AgentMemory({ store });
    await mem.init();
    await store.insertLesson(lesson("les_csv", LIVE.emptyCsv));
    await store.insertLesson(lesson("les_part", LIVE.partition));
    await store.insertLesson(lesson("les_pdf", LIVE.pdf));
    const csv = (await mem.recall(scope, CSV_TASK, 5)).map((l) => l.lessonId);
    expect(csv.sort()).toEqual(["les_csv", "les_part"]);
    const pdf = (await mem.recall(scope, PDF_TASK, 5)).map((l) => l.lessonId);
    expect(pdf).toEqual(["les_pdf"]);
    expect(await mem.recall(scope, UNRELATED_TASK, 5)).toEqual([]);
    await mem.close();
  });
});

async function failedRun(mem: AgentMemory) {
  const runId = await mem.startRun(scope, CSV_TASK);
  const evId = await mem.record(scope, runId, { type: "error", tool: "email.send", error: "CSV is empty" });
  return { runId, evId };
}

describe("distiller prompt and error reporting", () => {
  it("spells out the exact lesson schema with an example", async () => {
    const seen: string[] = [];
    const llm: LLM = async (messages) => {
      seen.push(messages.map((m) => m.content).join("\n"));
      return '{"lessons": []}';
    };
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }), llm });
    await mem.init();
    const { runId } = await failedRun(mem);
    await mem.endRun(scope, runId, { status: "fail", failureType: "empty_success" });
    const prompt = seen[0]!;
    for (const field of ["\"text\"", "\"trigger\"", "\"kind\"", "\"source_event_ids\"", "\"confidence\"", "\"updates_lesson_id\"", "Example"]) {
      expect(prompt).toContain(field);
    }
    expect(prompt).toMatch(/"avoid" \| "prefer" \| "fix"/);
    await mem.close();
  });

  it("throws DistillError when every lesson uses the wrong field names (live-test reply shape)", async () => {
    const d = new LlmDistiller({ llm: async () => '{"lessons":[{"lesson":"switch template","event_ids":["ev_x"]}]}' });
    const run = { runId: "r", ...scope, task: "t", startedAt: "", endedAt: "", meta: {}, extra: {},
      outcome: { status: "fail" as const, failureType: "loop" as const },
      events: [{ eventId: "ev_x", runId: "r", ...scope, seq: 1, ts: "", type: "error" }] };
    await expect(d.distill(run, [], mergeConfig())).rejects.toBeInstanceOf(DistillError);
  });

  it("endRun reports model errors through errors and onError, still records a note and does not throw", async () => {
    const onError = vi.fn<(e: EndRunError) => void>();
    const llm: LLM = async () => '{"lessons": [ {"text": "cut off';
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }), llm, onError });
    await mem.init();
    const { runId } = await failedRun(mem);
    const res = await mem.endRun(scope, runId, { status: "fail", failureType: "empty_success" });
    expect(res).toEqual([]);
    expect(res.errors).toHaveLength(1);
    expect(res.errors[0]?.name).toBe("DistillError");
    expect(res.errors[0]?.stage).toBe("distill");
    expect(onError).toHaveBeenCalledTimes(1);
    const run = await mem.getRun(scope, runId);
    expect(run.endedAt).not.toBeNull();
    const note = run.events.find((e) => e.type === "note");
    expect(note?.error).toContain("not valid JSON");
    await mem.close();
  });

  it("successful endRun has an empty errors list", async () => {
    const mem = new AgentMemory({ store: new SqliteStore({ path: ":memory:" }) });
    await mem.init();
    const { runId } = await failedRun(mem);
    const res = await mem.endRun(scope, runId, { status: "fail", failureType: "loop" });
    expect(res.errors).toEqual([]);
    await mem.close();
  });
});

describe("package.json scripts", () => {
  it("every script that runs a source file points at a file that exists", () => {
    const root = path.resolve(__dirname, "..");
    const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf-8")) as { scripts: Record<string, string> };
    for (const [name, cmd] of Object.entries(pkg.scripts)) {
      for (const file of cmd.match(/\S+\.(?:ts|js|mjs|cjs)\b/g) ?? []) {
        expect(fs.existsSync(path.join(root, file)), `${name}: ${file}`).toBe(true);
      }
    }
  });
});
