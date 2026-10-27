// agentmemory: AgentMemory facade, Distiller, Merger, Retriever. Implemented in M1-M3.
import Database from "better-sqlite3";

export const PACKAGE_NAME = "agentmemory";

/** Scaffold check that the native better-sqlite3 binding loads. */
export function sqliteVersion(): string {
  const db = new Database(":memory:");
  const row = db.prepare("select sqlite_version() as v").get() as { v: string };
  db.close();
  return row.v;
}
