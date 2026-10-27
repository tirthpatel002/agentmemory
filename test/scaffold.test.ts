import { describe, expect, it } from "vitest";
import { PACKAGE_NAME, sqliteVersion } from "../src/index.js";

describe("scaffold", () => {
  it("exports its package name", () => {
    expect(PACKAGE_NAME).toBe("agentmemory");
  });
  it("loads better-sqlite3 natively", () => {
    expect(sqliteVersion()).toMatch(/^3\./);
  });
});
