import { describe, expect, it } from "vitest";
import { createOpenAICompatEmbedder, EmbedderError } from "../src/index.js";

function fakeFetch(calls: { url: string; body: any; auth: string }[], status = 200) {
  return (async (url: string, init: any) => {
    const body = JSON.parse(init.body);
    calls.push({ url, body, auth: init.headers.authorization });
    if (status !== 200) return new Response("nope", { status });
    const data = body.input.map((t: string, i: number) => ({ index: i, embedding: [t.length, i, 1] })).reverse();
    return new Response(JSON.stringify({ data, usage: { prompt_tokens: 7, total_tokens: 7 } }), { status: 200 });
  }) as unknown as typeof fetch;
}

describe("createOpenAICompatEmbedder", () => {
  it("batches, orders by index, reports usage", async () => {
    const calls: any[] = [];
    let tokens = 0;
    const embed = createOpenAICompatEmbedder({
      baseURL: "https://example.test/v1/", model: "m", apiKey: "k", batchSize: 2,
      fetch: fakeFetch(calls), onUsage: (u) => (tokens += u.promptTokens),
    });
    const out = await embed(["a", "bb", "ccc"]);
    expect(calls.length).toBe(2);
    expect(calls[0].url).toBe("https://example.test/v1/embeddings");
    expect(calls[0].auth).toBe("Bearer k");
    expect(calls[0].body).toEqual({ model: "m", input: ["a", "bb"] });
    expect(out.map((v) => Array.from(v))).toEqual([[1, 0, 1], [2, 1, 1], [3, 0, 1]]);
    expect(out[0]).toBeInstanceOf(Float32Array);
    expect(tokens).toBe(14);
  });
  it("throws EmbedderError on HTTP failure", async () => {
    const embed = createOpenAICompatEmbedder({ baseURL: "https://x/v1", model: "m", apiKey: "k", fetch: fakeFetch([], 401) });
    await expect(embed(["a"])).rejects.toBeInstanceOf(EmbedderError);
  });
});
