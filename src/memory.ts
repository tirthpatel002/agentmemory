// src/memory.ts
import { ulid } from "ulid";
import type { DeepPartial, MemoryConfig } from "./config.js";
import { mergeConfig } from "./config.js";
import { InvalidEventError, NotFoundError, RunClosedError } from "./errors.js";
import type { Retriever } from "./retriever.js";
import { WeightedRetriever } from "./retriever.js";
import { SqliteStore } from "./store/sqlite.js";
import type { LessonQuery, MemoryStore } from "./store/store.js";
import type {
  Embedder,
  Event,
  EventInput,
  Extra,
  Lesson,
  LLM,
  Outcome,
  Run,
  RunMeta,
  Scope,
} from "./types.js";

function nowIso(): string {
  const d = new Date();
  const pad = (n: number) => String(Math.floor(Math.abs(n))).padStart(2, "0");
  const year = d.getFullYear();
  const month = pad(d.getMonth() + 1);
  const day = pad(d.getDate());
  const hours = pad(d.getHours());
  const minutes = pad(d.getMinutes());
  const seconds = pad(d.getSeconds());
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  const offsetMinutes = -d.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const offsetHours = pad(Math.floor(Math.abs(offsetMinutes) / 60));
  const offsetMins = pad(Math.abs(offsetMinutes) % 60);
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${ms}${sign}${offsetHours}:${offsetMins}`;
}

function validateScope(scope: Scope): void {
  if (
    !scope ||
    typeof scope.namespace !== "string" ||
    scope.namespace.trim() === "" ||
    typeof scope.userId !== "string" ||
    scope.userId.trim() === "" ||
    typeof scope.agentId !== "string" ||
    scope.agentId.trim() === ""
  ) {
    throw new InvalidEventError("Scope must provide non-empty namespace, userId, and agentId");
  }
}

function cutPayload(val: unknown, maxBytes = 8192): unknown {
  if (val === undefined) return undefined;
  if (val === null) return null;
  const str = typeof val === "string" ? val : JSON.stringify(val);
  if (str === undefined) return undefined;
  if (str.length > maxBytes) {
    return str.slice(0, maxBytes);
  }
  return str;
}

export function asPrompt(lessons: Lesson[]): string {
  if (!lessons || lessons.length === 0) {
    return "";
  }

  const header = "Lessons from earlier runs (follow unless the task says otherwise):";
  const items = lessons.map((lesson, idx) => {
    const num = idx + 1;
    const isTrial =
      lesson.status === "candidate" ||
      Boolean((lesson as unknown as { trial?: boolean }).trial) ||
      Boolean(lesson.extra?.trial);
    const tag = isTrial ? `[${lesson.kind}, trial]` : `[${lesson.kind}]`;
    return `${num}. ${tag} ${lesson.text}`;
  });

  return [header, ...items].join("\n");
}

export interface AgentMemoryOptions {
  store?: MemoryStore | undefined;
  llm?: LLM | undefined;
  embedder?: Embedder | undefined;
  retriever?: Retriever | undefined;
  config?: DeepPartial<MemoryConfig> | Partial<MemoryConfig> | undefined;
}

export class AgentMemory {
  public readonly store: MemoryStore;
  public readonly config: MemoryConfig;
  public readonly llm?: LLM | undefined;
  public readonly embedder?: Embedder | undefined;
  public readonly retriever: Retriever;
  private initialized = false;

  constructor(opts?: AgentMemoryOptions) {
    this.store = opts?.store ?? new SqliteStore({ path: "memory.db" });
    this.llm = opts?.llm;
    this.embedder = opts?.embedder;
    this.config = mergeConfig(opts?.config);
    this.retriever = opts?.retriever ?? new WeightedRetriever();
  }

  async init(): Promise<void> {
    if (!this.initialized) {
      await this.store.init();
      this.initialized = true;
    }
  }

  private async ensureInit(): Promise<void> {
    if (!this.initialized) {
      await this.init();
    }
  }

  async startRun(scope: Scope, task: string, meta?: RunMeta, extra?: Extra): Promise<string> {
    validateScope(scope);
    await this.ensureInit();

    const runId = `run_${ulid()}`;
    const startedAt = nowIso();

    const run: Omit<Run, "events"> = {
      runId,
      namespace: scope.namespace,
      userId: scope.userId,
      agentId: scope.agentId,
      task,
      startedAt,
      endedAt: null,
      outcome: null,
      meta: meta ?? {},
      extra: extra ?? {},
    };

    await this.store.insertRun(run);
    return runId;
  }

  async record(scope: Scope, runId: string, event: EventInput): Promise<string> {
    validateScope(scope);
    await this.ensureInit();

    const run = await this.store.getRun(scope, runId);
    if (!run) {
      throw new NotFoundError(`Run ${runId} not found in scope`);
    }

    if (run.endedAt !== null) {
      throw new RunClosedError(`Run ${runId} has ended`);
    }

    const seq = await this.store.nextSeq(scope, runId);
    const eventId = `ev_${ulid()}`;
    const ts = nowIso();

    const ev: Event = {
      eventId,
      runId,
      namespace: scope.namespace,
      userId: scope.userId,
      agentId: scope.agentId,
      seq,
      ts,
      type: event.type,
      tokens: event.tokens ?? 0,
      costUsd: event.costUsd ?? 0,
      latencyMs: event.latencyMs ?? 0,
      extra: event.extra ?? {},
    };

    if (event.tool !== undefined) ev.tool = event.tool;
    if (event.error !== undefined) ev.error = event.error;
    if (event.input !== undefined) ev.input = cutPayload(event.input);
    if (event.output !== undefined) ev.output = cutPayload(event.output);

    await this.store.insertEvent(ev);
    return eventId;
  }

  async endRun(scope: Scope, runId: string, outcome: Outcome): Promise<Lesson[]> {
    validateScope(scope);
    await this.ensureInit();

    if (outcome.status !== "success" && !outcome.failureType) {
      throw new InvalidEventError("Failure outcome requires a failureType");
    }

    const run = await this.store.getRun(scope, runId);
    if (!run) {
      throw new NotFoundError(`Run ${runId} not found in scope`);
    }

    if (run.endedAt !== null) {
      throw new RunClosedError(`Run ${runId} has already ended`);
    }

    const endedAt = nowIso();
    const eventCostTotal = run.events.reduce((sum, e) => sum + (e.costUsd ?? 0), 0);
    const cost = (run.meta?.cost ?? 0) + eventCostTotal;
    const totalSteps = run.events.length;

    const updatedMeta: RunMeta = {
      ...run.meta,
      cost,
      totalSteps,
    };

    await this.store.updateRun(scope, runId, {
      endedAt,
      outcome,
      meta: updatedMeta,
    });

    // M1 / M2: Learning (Distiller/Merger) returns empty list; built in later checkpoints
    return [];
  }

  async recall(scope: Scope, task: string, k = 5): Promise<Lesson[]> {
    validateScope(scope);
    await this.ensureInit();

    let taskEmbedding: Float32Array | null = null;
    if (this.embedder) {
      const embeddings = await this.embedder([task]);
      const first = embeddings[0];
      if (first) {
        taskEmbedding = first;
      }
    }

    return this.retriever.rank(scope, task, taskEmbedding, k, this.store, this.config);
  }

  asPrompt(lessons: Lesson[]): string {
    return asPrompt(lessons);
  }

  async markUsed(
    scope: Scope,
    runId: string,
    lessonIds: string[] | string,
    helped?: boolean | null | "helped" | "hurt" | "neutral",
  ): Promise<void> {
    validateScope(scope);
    await this.ensureInit();

    const run = await this.store.getRun(scope, runId);
    if (!run) {
      throw new NotFoundError(`Run ${runId} not found in scope`);
    }

    let helpedBool: boolean | null = null;
    if (typeof helped === "boolean") {
      helpedBool = helped;
    } else if (helped === "helped") {
      helpedBool = true;
    } else if (helped === "hurt") {
      helpedBool = false;
    } else {
      helpedBool = null;
    }

    const ids = Array.isArray(lessonIds) ? lessonIds : [lessonIds];

    await this.store.transaction(async () => {
      for (const lessonId of ids) {
        const lesson = await this.store.getLesson(scope, lessonId);
        if (!lesson) {
          throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
        }

        const { inserted, previousHelped } = await this.store.upsertLessonUse({
          runId,
          lessonId,
          namespace: scope.namespace,
          userId: scope.userId,
          agentId: scope.agentId,
          helped: helpedBool,
        });

        let uses = lesson.uses;
        let helpedCount = lesson.helped;
        let hurtCount = lesson.hurt;

        if (inserted) {
          uses += 1;
          if (helpedBool === true) helpedCount += 1;
          if (helpedBool === false) hurtCount += 1;
        } else if (previousHelped !== undefined && previousHelped !== helpedBool) {
          if (previousHelped === true) helpedCount -= 1;
          if (previousHelped === false) hurtCount -= 1;
          if (helpedBool === true) helpedCount += 1;
          if (helpedBool === false) hurtCount += 1;
        }

        const patch: Partial<Lesson> = {
          uses,
          helped: Math.max(0, helpedCount),
          hurt: Math.max(0, hurtCount),
        };

        // Retirement rule: uses >= retire.minUses and hurt > helped -> status replaced (replaced_by NULL)
        if (
          lesson.status === "active" &&
          this.config.retire.rule === "hurtGtHelped" &&
          uses >= this.config.retire.minUses &&
          hurtCount > helpedCount
        ) {
          patch.status = "replaced";
          patch.replacedBy = lesson.replacedBy ?? null;
        }

        await this.store.updateLesson(scope, lessonId, patch);
      }
    });
  }

  async listLessons(scope: Scope, q?: LessonQuery): Promise<Lesson[]> {
    validateScope(scope);
    await this.ensureInit();

    const query: LessonQuery = {
      status: q?.status ?? ["active"],
      ...(q?.kind !== undefined ? { kind: q.kind } : {}),
      ...(q?.minConfidence !== undefined ? { minConfidence: q.minConfidence } : {}),
      ...(q?.textContains !== undefined ? { textContains: q.textContains } : {}),
      ...(q?.limit !== undefined ? { limit: q.limit } : {}),
    };

    return this.store.listLessons(scope, query);
  }

  async editLesson(scope: Scope, lessonId: string, text: string): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();

    const existing = await this.store.getLesson(scope, lessonId);
    if (!existing) {
      throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    }

    const updatedAt = nowIso();
    return this.store.updateLesson(scope, lessonId, { text, updatedAt });
  }

  async forget(scope: Scope, lessonId: string): Promise<void> {
    validateScope(scope);
    await this.ensureInit();

    const existing = await this.store.getLesson(scope, lessonId);
    if (!existing) {
      throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    }

    await this.store.deleteLesson(scope, lessonId);
  }

  async getRun(scope: Scope, runId: string): Promise<Run> {
    validateScope(scope);
    await this.ensureInit();

    const run = await this.store.getRun(scope, runId);
    if (!run) {
      throw new NotFoundError(`Run ${runId} not found in scope`);
    }

    return run;
  }

  async close(): Promise<void> {
    await this.store.close();
  }
}
