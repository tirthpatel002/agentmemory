// src/types.ts
// Known values are constants; any other string is allowed (open schema).
export const EVENT_TYPES = ["step", "tool_call", "tool_result", "error", "note"] as const;
export const LESSON_KINDS = ["avoid", "prefer", "fix"] as const;

export type EventType = (typeof EVENT_TYPES)[number] | (string & {});
export type LessonKind = (typeof LESSON_KINDS)[number] | (string & {});

export type RunStatus = "success" | "fail" | "partial";
export type FailureType = "loop" | "empty_success" | "broken_handoff" | "cost_blowup" | "false_done";
export type LessonStatus = "candidate" | "active" | "replaced" | "deleted" | "rejected";
export type Extra = Record<string, unknown>; // free JSON, stored as-is

export interface Scope {
  namespace: string; // e.g. "app-a", "app-b"
  userId: string; // end user or "_system"
  agentId: string;
}

export interface EventInput {
  type: EventType;
  tool?: string | undefined;
  input?: unknown; // JSON text, cut to 8 KB
  output?: unknown; // JSON text, cut to 8 KB
  error?: string | undefined;
  tokens?: number | undefined;
  costUsd?: number | undefined;
  latencyMs?: number | undefined;
  extra?: Extra | undefined;
}

export interface Event extends EventInput, Scope {
  eventId: string;
  runId: string;
  seq: number;
  ts: string;
}

export interface Outcome {
  status: RunStatus;
  reason?: string | undefined;
  failureType?: FailureType | undefined; // failureType required unless success
}

export interface Provenance {
  createdBy: "distiller" | "manual" | (string & {});
  sourceRunIds: string[];
  sourceEventIds: string[];
  distillerModel?: string | undefined;
  validation?: { validator: string; passed: boolean; reasons: string[]; at: string } | undefined;
  reviewedBy?: string | undefined;
  reviewedAt?: string | undefined;
  reviewNote?: string | undefined;
}

export interface Lesson extends Scope {
  lessonId: string;
  text: string; // one instruction, max 300 chars
  trigger: string; // when it applies, max 200 chars
  kind: LessonKind;
  provenance: Provenance;
  confidence: number; // 0-1
  uses: number;
  helped: number;
  hurt: number;
  createdAt: string; // ISO 8601 with offset
  updatedAt: string; // ISO 8601 with offset
  status: LessonStatus;
  replacedBy: string | null;
  embedding: Float32Array | null;
  extra: Extra;
  score?: number | undefined; // filled by recall()
}

export interface RunMeta {
  model?: string | undefined;
  cost?: number | undefined;
  totalSteps?: number | undefined;
}

export interface Run extends Scope {
  runId: string;
  task: string;
  startedAt: string;
  endedAt: string | null;
  outcome: Outcome | null;
  meta: RunMeta;
  extra: Extra;
  events: Event[];
}

export interface LessonUse extends Scope {
  runId: string;
  lessonId: string;
  helped: boolean | null; // null = not judged
}

// Injected functions; defaults wrap @hackkit/connector chat() and embed()
export type LLM = (messages: { role: string; content: string }[], opts: { json: boolean }) => Promise<string>;
export type Embedder = (texts: string[]) => Promise<Float32Array[]>;
