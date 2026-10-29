// src/memory.ts
import { ulid } from "ulid";
import type { DeepPartial, MemoryConfig } from "./config.js";
import { mergeConfig } from "./config.js";
import type { Distiller } from "./distiller.js";
import { LlmDistiller } from "./distiller.js";
import { cosine } from "./embed.js";
import {
  InvalidEventError,
  InvalidTransitionError,
  NotFoundError,
  RunClosedError,
} from "./errors.js";
import type { Merger } from "./merger.js";
import { EmbeddingMerger } from "./merger.js";
import type { Retriever } from "./retriever.js";
import { WeightedRetriever } from "./retriever.js";
import { bm25Similarities } from "./lexical.js";
import { SqliteStore } from "./store/sqlite.js";
import type { LessonQuery, MemoryStore } from "./store/store.js";
import type {
  DraftLesson,
  Embedder,
  Event,
  EventInput,
  Extra,
  Lesson,
  LessonStatus,
  LLM,
  Outcome,
  Provenance,
  Run,
  RunMeta,
  Scope,
} from "./types.js";
import type { LessonValidator } from "./validator.js";
import { BasicValidator } from "./validator.js";

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

export interface RememberInput {
  text: string;
  trigger?: string | undefined;
  kind?: string | undefined;
  /** Lesson id(s) this memory replaces; they become status `replaced`. */
  supersedes?: string | string[] | null | undefined;
  confidence?: number | undefined;
  createdAt?: string | undefined;
  lessonId?: string | undefined;
  embedding?: Float32Array | null | undefined;
  provenance?: Provenance | undefined;
  extra?: Extra | undefined;
}

export interface AgentMemoryOptions {
  store?: MemoryStore | undefined;
  llm?: LLM | undefined;
  embedder?: Embedder | undefined;
  distiller?: Distiller | undefined;
  merger?: Merger | undefined;
  retriever?: Retriever | undefined;
  validator?: LessonValidator | undefined;
  config?: DeepPartial<MemoryConfig> | Partial<MemoryConfig> | undefined;
  /**
   * Called when a model-backed step fails inside endRun (the run is still closed and
   * endRun does not throw). Default: console.warn.
   */
  onError?: ((err: EndRunError) => void) | undefined;
}

export interface EndRunError {
  stage: "distill";
  runId: string;
  name: string;
  message: string;
  error: unknown;
}

/** Lessons written by endRun; `errors` (non-enumerable) lists model failures that were absorbed. */
export type EndRunResult = Lesson[] & { readonly errors: EndRunError[] };

function endRunResult(lessons: Lesson[], errors: EndRunError[] = []): EndRunResult {
  Object.defineProperty(lessons, "errors", { value: errors, enumerable: false });
  return lessons as EndRunResult;
}

export class AgentMemory {
  public readonly store: MemoryStore;
  public readonly config: MemoryConfig;
  public readonly llm?: LLM | undefined;
  public readonly embedder?: Embedder | undefined;
  public readonly distiller: Distiller;
  public readonly merger: Merger;
  public readonly retriever: Retriever;
  public readonly validator: LessonValidator;
  private initialized = false;
  private readonly onError: (err: EndRunError) => void;

  constructor(opts?: AgentMemoryOptions) {
    this.store = opts?.store ?? new SqliteStore({ path: "memory.db" });
    this.llm = opts?.llm;
    this.embedder = opts?.embedder;
    this.config = mergeConfig(opts?.config);
    this.retriever = opts?.retriever ?? new WeightedRetriever();
    this.validator =
      opts?.validator ??
      new BasicValidator({
        mergeThreshold: this.config.mergeThreshold,
        jaccardThreshold: this.config.jaccardThreshold,
        embedder: this.embedder,
      });
    this.merger = opts?.merger ?? new EmbeddingMerger({ embedder: this.embedder });
    this.distiller = opts?.distiller ?? new LlmDistiller({ llm: this.llm });
    this.onError =
      opts?.onError ??
      ((e) => console.warn(`agentmemory: endRun ${e.stage} failed for ${e.runId}: ${e.name}: ${e.message}`));
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

  async endRun(scope: Scope, runId: string, outcome: Outcome): Promise<EndRunResult> {
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

    const fullRun: Run = {
      ...run,
      endedAt,
      outcome,
      meta: updatedMeta,
    };

    let drafts: DraftLesson[] = [];
    try {
      const activeLessons = await this.store.listLessons(scope, { status: ["active"] });
      drafts = await this.distiller.distill(fullRun, activeLessons, this.config);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const errName = err instanceof Error ? err.name : "Error";
      const seq = await this.store.nextSeq(scope, runId);
      const ev: Event = {
        eventId: `ev_${ulid()}`,
        runId,
        namespace: scope.namespace,
        userId: scope.userId,
        agentId: scope.agentId,
        seq,
        ts: nowIso(),
        type: "note",
        error: errMsg,
        tokens: 0,
        costUsd: 0,
        latencyMs: 0,
        extra: { stage: "distill", errorName: errName },
      };
      await this.store.insertEvent(ev);
      const info: EndRunError = { stage: "distill", runId, name: errName, message: errMsg, error: err };
      try {
        this.onError(info);
      } catch {
        // a throwing hook must not break the caller's agent
      }
      return endRunResult([], [info]);
    }

    if (!drafts || drafts.length === 0) {
      return endRunResult([]);
    }

    const toProcess = drafts.slice(0, this.config.maxLessonsPerRun);
    const results: Lesson[] = [];

    for (const draft of toProcess) {
      const currentActive = await this.store.listLessons(scope, { status: ["active"] });
      const valResult = await this.validator.validate(draft, { scope, active: currentActive });

      let draftEmbedding: Float32Array | null = null;
      if (this.embedder) {
        const embs = await this.embedder([`${draft.trigger} ${draft.text}`]);
        draftEmbedding = embs[0] ?? null;
      }

      if (!valResult.passed) {
        // Failed validation: stored as rejected
        const cappedConfidence = Math.min(draft.confidence, this.config.maxInitialConfidence);
        const rejectedLesson: Lesson = {
          lessonId: `les_${ulid()}`,
          namespace: scope.namespace,
          userId: scope.userId,
          agentId: scope.agentId,
          text: draft.text,
          trigger: draft.trigger,
          kind: draft.kind,
          provenance: {
            createdBy: "distiller",
            sourceRunIds: [runId],
            sourceEventIds: draft.sourceEventIds,
            ...(run.meta?.model ? { distillerModel: run.meta.model } : {}),
            validation: {
              validator: this.validator.name,
              passed: false,
              reasons: valResult.reasons,
              at: nowIso(),
            },
          },
          confidence: cappedConfidence,
          uses: 0,
          helped: 0,
          hurt: 0,
          createdAt: nowIso(),
          updatedAt: nowIso(),
          status: "rejected",
          replacedBy: null,
          embedding: draftEmbedding,
          extra: draft.extra ?? {},
        };

        await this.store.insertLesson(rejectedLesson);
        results.push(rejectedLesson);
        continue;
      }

      // Passed validation: run merger
      const decision = await this.merger.decide(
        draft,
        draftEmbedding,
        scope,
        this.store,
        this.config,
      );

      if (decision.action === "strengthen") {
        const existing = await this.store.getLesson(scope, decision.lessonId);
        if (existing) {
          const newSourceRunIds = Array.from(new Set([...existing.provenance.sourceRunIds, runId]));
          const newSourceEventIds = Array.from(
            new Set([...existing.provenance.sourceEventIds, ...draft.sourceEventIds]),
          );
          const newConfidence = Math.min(1, existing.confidence + 0.1);
          const newText = decision.text ?? existing.text;
          let newEmbedding = existing.embedding;
          if (decision.text && this.embedder) {
            const embs = await this.embedder([`${existing.trigger} ${newText}`]);
            newEmbedding = embs[0] ?? existing.embedding;
          }

          const updated = await this.store.updateLesson(scope, existing.lessonId, {
            confidence: newConfidence,
            text: newText,
            embedding: newEmbedding,
            provenance: {
              ...existing.provenance,
              sourceRunIds: newSourceRunIds,
              sourceEventIds: newSourceEventIds,
            },
            updatedAt: nowIso(),
          });
          results.push(updated);
        }
      } else if (decision.action === "replace") {
        const initialStatus: LessonStatus = this.config.quarantine.enabled ? "candidate" : "active";
        const newLessonId = `les_${ulid()}`;
        const cappedConfidence = Math.min(draft.confidence, this.config.maxInitialConfidence);

        const newLesson: Lesson = {
          lessonId: newLessonId,
          namespace: scope.namespace,
          userId: scope.userId,
          agentId: scope.agentId,
          text: draft.text,
          trigger: draft.trigger,
          kind: draft.kind,
          provenance: {
            createdBy: "distiller",
            sourceRunIds: [runId],
            sourceEventIds: draft.sourceEventIds,
            ...(run.meta?.model ? { distillerModel: run.meta.model } : {}),
            validation: {
              validator: this.validator.name,
              passed: true,
              reasons: valResult.reasons,
              at: nowIso(),
            },
          },
          confidence: cappedConfidence,
          uses: 0,
          helped: 0,
          hurt: 0,
          createdAt: nowIso(),
          updatedAt: nowIso(),
          status: initialStatus,
          replacedBy: null,
          embedding: draftEmbedding,
          extra: {
            ...(draft.extra ?? {}),
            replacesLessonId: decision.lessonId,
          },
        };

        await this.store.insertLesson(newLesson);

        if (!this.config.quarantine.enabled) {
          await this.store.updateLesson(scope, decision.lessonId, {
            status: "replaced",
            replacedBy: newLessonId,
            updatedAt: nowIso(),
          });
        }

        results.push(newLesson);
      } else {
        // action === "insert"
        const initialStatus: LessonStatus = this.config.quarantine.enabled ? "candidate" : "active";
        const newLessonId = `les_${ulid()}`;
        const cappedConfidence = Math.min(draft.confidence, this.config.maxInitialConfidence);

        const newLesson: Lesson = {
          lessonId: newLessonId,
          namespace: scope.namespace,
          userId: scope.userId,
          agentId: scope.agentId,
          text: draft.text,
          trigger: draft.trigger,
          kind: draft.kind,
          provenance: {
            createdBy: "distiller",
            sourceRunIds: [runId],
            sourceEventIds: draft.sourceEventIds,
            ...(run.meta?.model ? { distillerModel: run.meta.model } : {}),
            validation: {
              validator: this.validator.name,
              passed: true,
              reasons: valResult.reasons,
              at: nowIso(),
            },
          },
          confidence: cappedConfidence,
          uses: 0,
          helped: 0,
          hurt: 0,
          createdAt: nowIso(),
          updatedAt: nowIso(),
          status: initialStatus,
          replacedBy: null,
          embedding: draftEmbedding,
          extra: draft.extra ?? {},
        };

        await this.store.insertLesson(newLesson);
        results.push(newLesson);
      }
    }

    return endRunResult(results);
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

    const activeLessons = await this.retriever.rank(
      scope,
      task,
      taskEmbedding,
      k,
      this.store,
      this.config,
    );

    const activeOnly = activeLessons.filter((l) => l.status === "active");
    const trialSlots = this.config.quarantine.trialSlots;
    if (trialSlots <= 0) {
      return activeOnly;
    }

    const candidateLessons = await this.store.listLessons(scope, { status: ["candidate"] });
    const validatedCandidates = candidateLessons.filter(
      (c) => c.provenance.validation === undefined || c.provenance.validation.passed !== false,
    );

    if (validatedCandidates.length === 0) {
      return activeOnly;
    }

    interface ScoredCandidate {
      lesson: Lesson;
      sim: number;
    }
    const scored: ScoredCandidate[] = [];

    const lexical = bm25Similarities(
      task,
      validatedCandidates.map((c) => `${c.trigger} ${c.text}`),
    );
    for (const [i, c] of validatedCandidates.entries()) {
      let sim = 0;
      let threshold = this.config.minSimilarity;
      if (taskEmbedding !== null && c.embedding !== null) {
        sim = Math.max(0, Math.min(1, cosine(taskEmbedding, c.embedding)));
      } else {
        sim = lexical[i] ?? 0;
        threshold = this.config.minLexicalSimilarity;
      }
      if (sim >= threshold) {
        scored.push({ lesson: c, sim });
      }
    }

    scored.sort((a, b) => b.sim - a.sim);
    const trialLessons = scored.slice(0, trialSlots).map((s) => ({
      ...s.lesson,
      score: s.sim,
      extra: {
        ...s.lesson.extra,
        trial: true,
      },
    }));

    return [...activeOnly, ...trialLessons];
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

        // Promotion rule: candidate -> active after promoteAfterHelpedUses helped uses unless requireApproval
        if (
          lesson.status === "candidate" &&
          !this.config.quarantine.requireApproval &&
          helpedCount >= this.config.quarantine.promoteAfterHelpedUses
        ) {
          patch.status = "active";
          if (typeof lesson.extra?.replacesLessonId === "string") {
            await this.store.updateLesson(scope, lesson.extra.replacesLessonId, {
              status: "replaced",
              replacedBy: lesson.lessonId,
              updatedAt: nowIso(),
            });
          }
        }

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

    let embedding = existing.embedding;
    if (this.embedder) {
      const embeddings = await this.embedder([`${existing.trigger} ${text}`]);
      const first = embeddings[0];
      if (first) {
        embedding = first;
      }
    }

    const draft: DraftLesson = {
      text,
      trigger: existing.trigger,
      kind: existing.kind,
      sourceEventIds: existing.provenance.sourceEventIds,
      confidence: existing.confidence,
      updatesLessonId: null,
    };

    const active = (await this.store.listLessons(scope, { status: ["active"] })).filter(
      (l) => l.lessonId !== lessonId,
    );
    const valResult = await this.validator.validate(draft, { scope, active });

    const updatedAt = nowIso();
    let status = existing.status;
    const provenance: Provenance = {
      ...existing.provenance,
      validation: {
        validator: this.validator.name,
        passed: valResult.passed,
        reasons: valResult.reasons,
        at: updatedAt,
      },
    };

    if (!valResult.passed) {
      status = "candidate";
    }

    return this.store.updateLesson(scope, lessonId, {
      text,
      embedding,
      status,
      provenance,
      updatedAt,
    });
  }

  async approveLesson(scope: Scope, lessonId: string, by: string, note?: string): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();

    const lesson = await this.store.getLesson(scope, lessonId);
    if (!lesson) {
      throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    }

    if (lesson.status !== "candidate" && lesson.status !== "rejected") {
      throw new InvalidTransitionError(
        `Cannot approve lesson ${lessonId} with status '${lesson.status}'`,
      );
    }

    const updatedAt = nowIso();
    const provenance: Provenance = {
      ...lesson.provenance,
      reviewedBy: by,
      reviewedAt: updatedAt,
      ...(note !== undefined ? { reviewNote: note } : {}),
    };

    if (typeof lesson.extra?.replacesLessonId === "string") {
      await this.store.updateLesson(scope, lesson.extra.replacesLessonId, {
        status: "replaced",
        replacedBy: lesson.lessonId,
        updatedAt,
      });
    }

    return this.store.updateLesson(scope, lessonId, {
      status: "active",
      provenance,
      updatedAt,
    });
  }

  async rejectLesson(scope: Scope, lessonId: string, by: string, note?: string): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();

    const lesson = await this.store.getLesson(scope, lessonId);
    if (!lesson) {
      throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    }

    if (lesson.status !== "candidate" && lesson.status !== "active") {
      throw new InvalidTransitionError(
        `Cannot reject lesson ${lessonId} with status '${lesson.status}'`,
      );
    }

    const updatedAt = nowIso();
    const provenance: Provenance = {
      ...lesson.provenance,
      reviewedBy: by,
      reviewedAt: updatedAt,
      ...(note !== undefined ? { reviewNote: note } : {}),
    };

    return this.store.updateLesson(scope, lessonId, {
      status: "rejected",
      provenance,
      updatedAt,
    });
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

  /**
   * Store a memory directly (no run/distiller). If `supersedes` names older lessons, they are
   * marked `replaced` (replacedBy = new id) in the same transaction, so recall only sees the newest.
   */
  async remember(scope: Scope, input: RememberInput): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();
    if (typeof input?.text !== "string" || input.text.trim() === "") {
      throw new InvalidEventError("remember() needs non-empty text");
    }
    const olds = input.supersedes === undefined || input.supersedes === null
      ? [] : Array.isArray(input.supersedes) ? input.supersedes : [input.supersedes];
    const trigger = input.trigger ?? "";
    let embedding: Float32Array | null = input.embedding ?? null;
    if (embedding === null && this.embedder) {
      embedding = (await this.embedder([`${trigger} ${input.text}`.trim()]))[0] ?? null;
    }
    const at = input.createdAt ?? nowIso();
    const lesson: Lesson = {
      ...scope,
      lessonId: input.lessonId ?? ulid(),
      text: input.text,
      trigger,
      kind: input.kind ?? "prefer",
      provenance: input.provenance ?? { createdBy: "manual", sourceRunIds: [], sourceEventIds: [] },
      confidence: input.confidence ?? 0.7,
      uses: 0,
      helped: 0,
      hurt: 0,
      createdAt: at,
      updatedAt: at,
      status: "active",
      replacedBy: null,
      embedding,
      extra: { ...(input.extra ?? {}), ...(olds.length ? { supersedes: olds } : {}) },
    };
    await this.store.transaction(async () => {
      for (const id of olds) await this.checkSupersedable(scope, id, lesson.lessonId);
      await this.store.insertLesson(lesson);
      for (const id of olds) {
        await this.store.updateLesson(scope, id, { status: "replaced", replacedBy: lesson.lessonId, updatedAt: nowIso() });
      }
    });
    return lesson;
  }

  /** Mark an existing lesson as replaced by another existing lesson (newest wins in recall). */
  async supersede(scope: Scope, oldLessonId: string, newLessonId: string): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();
    return this.store.transaction(async () => {
      const newer = await this.store.getLesson(scope, newLessonId);
      if (!newer) throw new NotFoundError(`Lesson ${newLessonId} not found in scope`);
      if (newer.status !== "active" && newer.status !== "candidate") {
        throw new InvalidTransitionError(`Lesson ${newLessonId} is '${newer.status}' and cannot supersede`);
      }
      await this.checkSupersedable(scope, oldLessonId, newLessonId);
      // both are heads of their chains (active/candidate), so no cycle is possible
      return this.store.updateLesson(scope, oldLessonId, { status: "replaced", replacedBy: newLessonId, updatedAt: nowIso() });
    });
  }

  /** Follow replacedBy links to the newest version of a lesson. */
  async current(scope: Scope, lessonId: string): Promise<Lesson> {
    validateScope(scope);
    await this.ensureInit();
    let l = await this.store.getLesson(scope, lessonId);
    if (!l) throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    const seen = new Set<string>();
    while (l.replacedBy && !seen.has(l.lessonId)) {
      seen.add(l.lessonId);
      const next: Lesson | null = await this.store.getLesson(scope, l.replacedBy);
      if (!next) break;
      l = next;
    }
    return l;
  }

  /** Version chain containing lessonId, oldest first, newest last. */
  async history(scope: Scope, lessonId: string): Promise<Lesson[]> {
    validateScope(scope);
    await this.ensureInit();
    const head = await this.current(scope, lessonId);
    const replaced = await this.store.listLessons(scope, { status: ["replaced"] });
    const byNext = new Map<string, Lesson>();
    for (const l of replaced) if (l.replacedBy) byNext.set(l.replacedBy, l);
    const out: Lesson[] = [head];
    const seen = new Set([head.lessonId]);
    let cur = byNext.get(head.lessonId);
    while (cur && !seen.has(cur.lessonId)) {
      out.unshift(cur);
      seen.add(cur.lessonId);
      cur = byNext.get(cur.lessonId);
    }
    return out;
  }

  private async checkSupersedable(scope: Scope, oldId: string, newId: string): Promise<void> {
    if (oldId === newId) throw new InvalidTransitionError("A lesson cannot supersede itself");
    const old = await this.store.getLesson(scope, oldId);
    if (!old) throw new NotFoundError(`Lesson ${oldId} not found in scope`);
    if (old.status !== "active" && old.status !== "candidate") {
      throw new InvalidTransitionError(`Lesson ${oldId} is '${old.status}' and cannot be superseded`);
    }
  }

  async close(): Promise<void> {
    await this.store.close();
  }
}
