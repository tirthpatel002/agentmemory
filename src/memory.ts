// src/memory.ts
import { ulid } from "ulid";
import type { MemoryConfig } from "./config.js";
import { DEFAULT_MEMORY_CONFIG } from "./config.js";
import { InvalidEventError, NotFoundError, RunClosedError } from "./errors.js";
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

export interface AgentMemoryOptions {
  store?: MemoryStore | undefined;
  llm?: LLM | undefined;
  embedder?: Embedder | undefined;
  config?: Partial<MemoryConfig> | undefined;
}

export class AgentMemory {
  public readonly store: MemoryStore;
  public readonly config: MemoryConfig;
  public readonly llm?: LLM | undefined;
  public readonly embedder?: Embedder | undefined;
  private initialized = false;

  constructor(opts?: AgentMemoryOptions) {
    this.store = opts?.store ?? new SqliteStore({ path: "memory.db" });
    this.llm = opts?.llm;
    this.embedder = opts?.embedder;
    this.config = {
      ...DEFAULT_MEMORY_CONFIG,
      ...opts?.config,
      weights: {
        ...DEFAULT_MEMORY_CONFIG.weights,
        ...opts?.config?.weights,
      },
      retire: {
        ...DEFAULT_MEMORY_CONFIG.retire,
        ...opts?.config?.retire,
      },
      quarantine: {
        ...DEFAULT_MEMORY_CONFIG.quarantine,
        ...opts?.config?.quarantine,
      },
    };
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

    // M1: Learning not implemented yet, returns empty list
    return [];
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
