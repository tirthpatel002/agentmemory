-- store/schema.sql, run by SqliteStore.init() (better-sqlite3). schema_version = 2
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  namespace     TEXT NOT NULL,
  user_id       TEXT NOT NULL,
  agent_id      TEXT NOT NULL,
  task          TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  ended_at      TEXT,
  status        TEXT CHECK (status IN ('success','fail','partial')),
  reason        TEXT,
  failure_type  TEXT CHECK (failure_type IN ('loop','empty_success','broken_handoff','cost_blowup','false_done')),
  meta          TEXT NOT NULL DEFAULT '{}',     -- JSON: model, cost, totalSteps
  extra         TEXT NOT NULL DEFAULT '{}'      -- JSON, caller-defined
);

CREATE TABLE IF NOT EXISTS events (
  event_id    TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  namespace   TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  agent_id    TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  ts          TEXT NOT NULL,
  type        TEXT NOT NULL,                    -- open: built-ins or any custom string
  tool        TEXT,
  input       TEXT,
  output      TEXT,
  error       TEXT,
  tokens      INTEGER NOT NULL DEFAULT 0,
  cost_usd    REAL NOT NULL DEFAULT 0,
  latency_ms  INTEGER NOT NULL DEFAULT 0,
  extra       TEXT NOT NULL DEFAULT '{}',
  UNIQUE (run_id, seq)
);

CREATE TABLE IF NOT EXISTS lessons (
  lesson_id        TEXT PRIMARY KEY,
  namespace        TEXT NOT NULL,
  user_id          TEXT NOT NULL,
  agent_id         TEXT NOT NULL,
  text             TEXT NOT NULL,
  trigger          TEXT NOT NULL,
  kind             TEXT NOT NULL,                 -- open: avoid|prefer|fix or custom
  provenance       TEXT NOT NULL DEFAULT '{}',    -- JSON Provenance (createdBy, sources, validation, review)
  confidence       REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  uses             INTEGER NOT NULL DEFAULT 0,
  helped           INTEGER NOT NULL DEFAULT 0,
  hurt             INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'candidate'
                   CHECK (status IN ('candidate','active','replaced','deleted','rejected')),
  replaced_by      TEXT REFERENCES lessons(lesson_id) ON DELETE SET NULL,
  embedding        BLOB,                          -- Float32Array bytes (see toBlob)
  embedding_model  TEXT,
  extra            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS lesson_uses (
  run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  lesson_id  TEXT NOT NULL REFERENCES lessons(lesson_id) ON DELETE CASCADE,
  namespace  TEXT NOT NULL,
  user_id    TEXT NOT NULL,
  agent_id   TEXT NOT NULL,
  helped     INTEGER,                             -- 1, 0, or NULL (not judged)
  marked_at  TEXT NOT NULL,
  PRIMARY KEY (run_id, lesson_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_scope     ON runs(namespace, user_id, agent_id, started_at);
CREATE INDEX IF NOT EXISTS idx_events_run     ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_scope   ON events(namespace, user_id, agent_id, type);
CREATE INDEX IF NOT EXISTS idx_lessons_scope  ON lessons(namespace, user_id, agent_id, status);
CREATE INDEX IF NOT EXISTS idx_uses_lesson    ON lesson_uses(lesson_id);
CREATE INDEX IF NOT EXISTS idx_uses_scope     ON lesson_uses(namespace, user_id, agent_id);
