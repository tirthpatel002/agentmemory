# agentmemory

Records each agent run step by step, turns finished runs into at most 3 short lessons, and hands the best lessons back at the start of the next similar task. Lessons can be listed, edited and truly deleted.

Storage: one SQLite file (better-sqlite3), embeddings stored as Float32Array blobs. The learning parts (`Distiller`, `Merger`, `Retriever`) are small classes behind fixed interfaces so they can be swapped.

## Setup

```bash
pnpm install
pnpm typecheck && pnpm test
```

Unit tests run offline with fake LLM / embedder functions. `pnpm eval` (live) needs keys in a git-ignored `.env` (see `.env.example`).

Licence: MIT.
