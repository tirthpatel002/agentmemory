// src/store/store.ts
import type { Event, Lesson, LessonStatus, LessonUse, Run, Scope } from "../types.js";

export interface LessonQuery {
  status?: LessonStatus[] | undefined;
  kind?: string[] | undefined;
  minConfidence?: number | undefined;
  textContains?: string | undefined;
  limit?: number | undefined;
}

export interface VectorHit {
  lesson: Lesson;
  similarity: number; // cosine, 0-1
}

export interface MemoryStore {
  init(): Promise<void>;
  close(): Promise<void>;

  insertRun(run: Omit<Run, "events">): Promise<void>;
  updateRun(
    scope: Scope,
    runId: string,
    patch: Partial<Pick<Run, "endedAt" | "outcome" | "meta" | "extra">>,
  ): Promise<void>;
  getRun(scope: Scope, runId: string): Promise<Run | null>;
  insertEvent(event: Event): Promise<void>;
  nextSeq(scope: Scope, runId: string): Promise<number>;

  insertLesson(lesson: Lesson): Promise<void>;
  updateLesson(scope: Scope, lessonId: string, patch: Partial<Lesson>): Promise<Lesson>;
  getLesson(scope: Scope, lessonId: string): Promise<Lesson | null>;
  listLessons(scope: Scope, q?: LessonQuery): Promise<Lesson[]>;
  deleteLesson(scope: Scope, lessonId: string): Promise<boolean>;
  searchLessons(scope: Scope, query: Float32Array, k: number, q?: LessonQuery): Promise<VectorHit[]>;

  upsertLessonUse(use: LessonUse): Promise<{ inserted: boolean }>;
  transaction<T>(fn: () => Promise<T>): Promise<T>;
}
