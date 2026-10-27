// src/store/sqlite.ts
import Database from "better-sqlite3";
import type { Database as DatabaseType } from "better-sqlite3";
import { cosine, fromBlob, toBlob } from "../embed.js";
import { NotFoundError } from "../errors.js";
import type { Event, FailureType, Lesson, LessonStatus, LessonUse, Outcome, Run, RunStatus, Scope } from "../types.js";
import { SCHEMA_SQL, SCHEMA_VERSION } from "./schema.js";
import type { LessonQuery, MemoryStore, VectorHit } from "./store.js";

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

interface SqliteRunRow {
  run_id: string;
  namespace: string;
  user_id: string;
  agent_id: string;
  task: string;
  started_at: string;
  ended_at: string | null;
  status: RunStatus | null;
  reason: string | null;
  failure_type: FailureType | null;
  meta: string;
  extra: string;
}

interface SqliteEventRow {
  event_id: string;
  run_id: string;
  namespace: string;
  user_id: string;
  agent_id: string;
  seq: number;
  ts: string;
  type: string;
  tool: string | null;
  input: string | null;
  output: string | null;
  error: string | null;
  tokens: number;
  cost_usd: number;
  latency_ms: number;
  extra: string;
}

interface SqliteLessonRow {
  lesson_id: string;
  namespace: string;
  user_id: string;
  agent_id: string;
  text: string;
  trigger: string;
  kind: string;
  provenance: string;
  confidence: number;
  uses: number;
  helped: number;
  hurt: number;
  created_at: string;
  updated_at: string;
  status: LessonStatus;
  replaced_by: string | null;
  embedding: Buffer | null;
  extra: string;
}

export class SqliteStore implements MemoryStore {
  private readonly db: DatabaseType;

  constructor(opts?: { path?: string | undefined }) {
    const dbPath = opts?.path ?? "memory.db";
    this.db = new Database(dbPath);
  }

  async init(): Promise<void> {
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.db.exec(SCHEMA_SQL);

    const versionRow = this.db
      .prepare("SELECT version FROM schema_version LIMIT 1")
      .get() as { version: number } | undefined;

    if (!versionRow) {
      this.db
        .prepare("INSERT INTO schema_version (version) VALUES (?)")
        .run(SCHEMA_VERSION);
    }
  }

  async close(): Promise<void> {
    this.db.close();
  }

  vacuum(): void {
    this.db.pragma("wal_checkpoint(TRUNCATE)");
    this.db.exec("VACUUM");
    this.db.pragma("wal_checkpoint(TRUNCATE)");
  }

  async insertRun(run: Omit<Run, "events">): Promise<void> {
    const status = run.outcome?.status ?? null;
    const reason = run.outcome?.reason ?? null;
    const failureType = run.outcome?.failureType ?? null;

    this.db
      .prepare(
        `INSERT INTO runs (
          run_id, namespace, user_id, agent_id, task, started_at, ended_at, status, reason, failure_type, meta, extra
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        run.runId,
        run.namespace,
        run.userId,
        run.agentId,
        run.task,
        run.startedAt,
        run.endedAt ?? null,
        status,
        reason,
        failureType,
        JSON.stringify(run.meta ?? {}),
        JSON.stringify(run.extra ?? {}),
      );
  }

  async updateRun(
    scope: Scope,
    runId: string,
    patch: Partial<Pick<Run, "endedAt" | "outcome" | "meta" | "extra">>,
  ): Promise<void> {
    const existing = await this.getRun(scope, runId);
    if (!existing) {
      return;
    }

    const endedAt = patch.endedAt !== undefined ? patch.endedAt : existing.endedAt;
    const outcome = patch.outcome !== undefined ? patch.outcome : existing.outcome;
    const meta = patch.meta !== undefined ? patch.meta : existing.meta;
    const extra = patch.extra !== undefined ? patch.extra : existing.extra;

    const status = outcome?.status ?? null;
    const reason = outcome?.reason ?? null;
    const failureType = outcome?.failureType ?? null;

    this.db
      .prepare(
        `UPDATE runs SET
          ended_at = ?,
          status = ?,
          reason = ?,
          failure_type = ?,
          meta = ?,
          extra = ?
        WHERE run_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .run(
        endedAt,
        status,
        reason,
        failureType,
        JSON.stringify(meta ?? {}),
        JSON.stringify(extra ?? {}),
        runId,
        scope.namespace,
        scope.userId,
        scope.agentId,
      );
  }

  async getRun(scope: Scope, runId: string): Promise<Run | null> {
    const runRow = this.db
      .prepare(
        `SELECT run_id, namespace, user_id, agent_id, task, started_at, ended_at, status, reason, failure_type, meta, extra
        FROM runs
        WHERE run_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .get(runId, scope.namespace, scope.userId, scope.agentId) as SqliteRunRow | undefined;

    if (!runRow) {
      return null;
    }

    const eventRows = this.db
      .prepare(
        `SELECT event_id, run_id, namespace, user_id, agent_id, seq, ts, type, tool, input, output, error, tokens, cost_usd, latency_ms, extra
        FROM events
        WHERE run_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?
        ORDER BY seq ASC`,
      )
      .all(runId, scope.namespace, scope.userId, scope.agentId) as SqliteEventRow[];

    const events: Event[] = eventRows.map((e) => {
      const ev: Event = {
        eventId: e.event_id,
        runId: e.run_id,
        namespace: e.namespace,
        userId: e.user_id,
        agentId: e.agent_id,
        seq: e.seq,
        ts: e.ts,
        type: e.type,
        tokens: e.tokens,
        costUsd: e.cost_usd,
        latencyMs: e.latency_ms,
        extra: JSON.parse(e.extra) as Record<string, unknown>,
      };
      if (e.tool !== null) ev.tool = e.tool;
      if (e.input !== null) ev.input = e.input;
      if (e.output !== null) ev.output = e.output;
      if (e.error !== null) ev.error = e.error;
      return ev;
    });

    let outcome: Outcome | null = null;
    if (runRow.status) {
      outcome = {
        status: runRow.status,
      };
      if (runRow.reason !== null) outcome.reason = runRow.reason;
      if (runRow.failure_type !== null) outcome.failureType = runRow.failure_type;
    }

    return {
      runId: runRow.run_id,
      namespace: runRow.namespace,
      userId: runRow.user_id,
      agentId: runRow.agent_id,
      task: runRow.task,
      startedAt: runRow.started_at,
      endedAt: runRow.ended_at,
      outcome,
      meta: JSON.parse(runRow.meta) as Record<string, unknown>,
      extra: JSON.parse(runRow.extra) as Record<string, unknown>,
      events,
    };
  }

  async insertEvent(event: Event): Promise<void> {
    const inputStr =
      event.input === undefined || event.input === null
        ? null
        : typeof event.input === "string"
          ? event.input
          : JSON.stringify(event.input);

    const outputStr =
      event.output === undefined || event.output === null
        ? null
        : typeof event.output === "string"
          ? event.output
          : JSON.stringify(event.output);

    this.db
      .prepare(
        `INSERT INTO events (
          event_id, run_id, namespace, user_id, agent_id, seq, ts, type, tool, input, output, error, tokens, cost_usd, latency_ms, extra
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        event.eventId,
        event.runId,
        event.namespace,
        event.userId,
        event.agentId,
        event.seq,
        event.ts,
        event.type,
        event.tool ?? null,
        inputStr,
        outputStr,
        event.error ?? null,
        event.tokens ?? 0,
        event.costUsd ?? 0,
        event.latencyMs ?? 0,
        JSON.stringify(event.extra ?? {}),
      );
  }

  async nextSeq(scope: Scope, runId: string): Promise<number> {
    const row = this.db
      .prepare(
        `SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
        FROM events
        WHERE run_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .get(runId, scope.namespace, scope.userId, scope.agentId) as { next_seq: number } | undefined;

    return row ? row.next_seq : 1;
  }

  async insertLesson(lesson: Lesson): Promise<void> {
    const embeddingBlob = lesson.embedding ? toBlob(lesson.embedding) : null;

    this.db
      .prepare(
        `INSERT INTO lessons (
          lesson_id, namespace, user_id, agent_id, text, trigger, kind, provenance, confidence, uses, helped, hurt, created_at, updated_at, status, replaced_by, embedding, embedding_model, extra
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        lesson.lessonId,
        lesson.namespace,
        lesson.userId,
        lesson.agentId,
        lesson.text,
        lesson.trigger,
        lesson.kind,
        JSON.stringify(lesson.provenance ?? {}),
        lesson.confidence,
        lesson.uses ?? 0,
        lesson.helped ?? 0,
        lesson.hurt ?? 0,
        lesson.createdAt,
        lesson.updatedAt,
        lesson.status,
        lesson.replacedBy ?? null,
        embeddingBlob,
        null,
        JSON.stringify(lesson.extra ?? {}),
      );
  }

  async updateLesson(scope: Scope, lessonId: string, patch: Partial<Lesson>): Promise<Lesson> {
    const existing = await this.getLesson(scope, lessonId);
    if (!existing) {
      throw new NotFoundError(`Lesson ${lessonId} not found in scope`);
    }

    const updated: Lesson = {
      ...existing,
      ...patch,
      updatedAt: patch.updatedAt ?? existing.updatedAt,
    };

    const embeddingBlob = updated.embedding ? toBlob(updated.embedding) : null;

    this.db
      .prepare(
        `UPDATE lessons SET
          text = ?,
          trigger = ?,
          kind = ?,
          provenance = ?,
          confidence = ?,
          uses = ?,
          helped = ?,
          hurt = ?,
          updated_at = ?,
          status = ?,
          replaced_by = ?,
          embedding = ?,
          extra = ?
        WHERE lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .run(
        updated.text,
        updated.trigger,
        updated.kind,
        JSON.stringify(updated.provenance),
        updated.confidence,
        updated.uses,
        updated.helped,
        updated.hurt,
        updated.updatedAt,
        updated.status,
        updated.replacedBy,
        embeddingBlob,
        JSON.stringify(updated.extra),
        lessonId,
        scope.namespace,
        scope.userId,
        scope.agentId,
      );

    return updated;
  }

  async getLesson(scope: Scope, lessonId: string): Promise<Lesson | null> {
    const row = this.db
      .prepare(
        `SELECT lesson_id, namespace, user_id, agent_id, text, trigger, kind, provenance, confidence, uses, helped, hurt, created_at, updated_at, status, replaced_by, embedding, extra
        FROM lessons
        WHERE lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .get(lessonId, scope.namespace, scope.userId, scope.agentId) as SqliteLessonRow | undefined;

    if (!row) {
      return null;
    }

    return this.mapLessonRow(row);
  }

  async listLessons(scope: Scope, q?: LessonQuery): Promise<Lesson[]> {
    let sql =
      `SELECT lesson_id, namespace, user_id, agent_id, text, trigger, kind, provenance, confidence, uses, helped, hurt, created_at, updated_at, status, replaced_by, embedding, extra
      FROM lessons
      WHERE namespace = ? AND user_id = ? AND agent_id = ?`;

    const params: unknown[] = [scope.namespace, scope.userId, scope.agentId];

    if (q?.status && q.status.length > 0) {
      const placeholders = q.status.map(() => "?").join(", ");
      sql += ` AND status IN (${placeholders})`;
      params.push(...q.status);
    }

    if (q?.kind && q.kind.length > 0) {
      const placeholders = q.kind.map(() => "?").join(", ");
      sql += ` AND kind IN (${placeholders})`;
      params.push(...q.kind);
    }

    if (q?.minConfidence !== undefined) {
      sql += " AND confidence >= ?";
      params.push(q.minConfidence);
    }

    if (q?.textContains) {
      sql += " AND instr(text, ?) > 0";
      params.push(q.textContains);
    }

    sql += " ORDER BY created_at ASC";

    if (q?.limit !== undefined) {
      sql += " LIMIT ?";
      params.push(q.limit);
    }

    const rows = this.db.prepare(sql).all(...params) as SqliteLessonRow[];
    return rows.map((r) => this.mapLessonRow(r));
  }

  async deleteLesson(scope: Scope, lessonId: string): Promise<boolean> {
    const existing = this.db
      .prepare(
        `SELECT lesson_id
        FROM lessons
        WHERE lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .get(lessonId, scope.namespace, scope.userId, scope.agentId) as { lesson_id: string } | undefined;

    if (!existing) {
      return false;
    }

    this.db
      .prepare(
        `UPDATE lessons SET replaced_by = NULL
        WHERE replaced_by = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .run(lessonId, scope.namespace, scope.userId, scope.agentId);

    this.db
      .prepare(
        `DELETE FROM lesson_uses
        WHERE lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .run(lessonId, scope.namespace, scope.userId, scope.agentId);

    this.db
      .prepare(
        `DELETE FROM lessons
        WHERE lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .run(lessonId, scope.namespace, scope.userId, scope.agentId);

    return true;
  }

  async searchLessons(scope: Scope, query: Float32Array, k: number, q?: LessonQuery): Promise<VectorHit[]> {
    const lessons = await this.listLessons(scope, q);
    const hits: VectorHit[] = [];

    for (const lesson of lessons) {
      if (!lesson.embedding) {
        continue;
      }
      const rawSim = cosine(query, lesson.embedding);
      const similarity = Math.max(0, Math.min(1, rawSim));
      hits.push({ lesson, similarity });
    }

    hits.sort((a, b) => b.similarity - a.similarity);
    return hits.slice(0, k);
  }

  async upsertLessonUse(
    use: LessonUse,
  ): Promise<{ inserted: boolean; previousHelped?: boolean | null | undefined }> {
    const existing = this.db
      .prepare(
        `SELECT run_id, lesson_id, helped
        FROM lesson_uses
        WHERE run_id = ? AND lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
      )
      .get(use.runId, use.lessonId, use.namespace, use.userId, use.agentId) as
      | { run_id: string; lesson_id: string; helped: number | null }
      | undefined;

    const markedAt = nowIso();
    const helpedInt = use.helped === null ? null : use.helped ? 1 : 0;

    if (existing) {
      const previousHelped = existing.helped === null ? null : existing.helped === 1;
      this.db
        .prepare(
          `UPDATE lesson_uses SET helped = ?, marked_at = ?
          WHERE run_id = ? AND lesson_id = ? AND namespace = ? AND user_id = ? AND agent_id = ?`,
        )
        .run(helpedInt, markedAt, use.runId, use.lessonId, use.namespace, use.userId, use.agentId);
      return { inserted: false, previousHelped };
    }

    this.db
      .prepare(
        `INSERT INTO lesson_uses (
          run_id, lesson_id, namespace, user_id, agent_id, helped, marked_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(use.runId, use.lessonId, use.namespace, use.userId, use.agentId, helpedInt, markedAt);
    return { inserted: true };
  }

  async transaction<T>(fn: () => Promise<T>): Promise<T> {
    this.db.prepare("BEGIN").run();
    try {
      const result = await fn();
      this.db.prepare("COMMIT").run();
      return result;
    } catch (err) {
      try {
        this.db.prepare("ROLLBACK").run();
      } catch {
        // Rollback ignored if already rolled back
      }
      throw err;
    }
  }

  private mapLessonRow(row: SqliteLessonRow): Lesson {
    return {
      lessonId: row.lesson_id,
      namespace: row.namespace,
      userId: row.user_id,
      agentId: row.agent_id,
      text: row.text,
      trigger: row.trigger,
      kind: row.kind,
      provenance: JSON.parse(row.provenance),
      confidence: row.confidence,
      uses: row.uses,
      helped: row.helped,
      hurt: row.hurt,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      status: row.status,
      replacedBy: row.replaced_by,
      embedding: row.embedding ? fromBlob(row.embedding) : null,
      extra: JSON.parse(row.extra),
    };
  }
}
