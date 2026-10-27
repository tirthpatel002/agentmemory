# agentmemory

Project-agnostic memory for agents. Records each agent run step by step, turns finished runs into at most 3 short lessons, and hands the best lessons back at the start of the next similar task. Lessons can be listed, edited and truly deleted.

Storage: one SQLite file (better-sqlite3), embeddings stored as Float32Array blobs. Everything is scoped by namespace, user and agent. New lessons sit in quarantine until validated. Storage goes through a `MemoryStore` interface, and the learning parts (`Distiller`, `Merger`, `Retriever`, `LessonValidator`) are small classes passed into the constructor so they can be swapped. `LLM` and `Embedder` are injected functions.

## Setup

```bash
pnpm install
pnpm typecheck && pnpm test
```

Unit tests run offline with fake LLM / embedder functions. `pnpm eval` (live) needs keys in a git-ignored `.env` (see `.env.example`).

Full spec and build plan: [docs/SPEC.md](docs/SPEC.md).

Licence: MIT.
