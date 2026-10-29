// src/embedders/openaiCompat.ts
// Embedder for any OpenAI-compatible POST {baseURL}/embeddings endpoint.
import type { Embedder } from "../types.js";

export interface OpenAICompatEmbedderOptions {
  /** e.g. "https://api.openai.com/v1"; "/embeddings" is appended. */
  baseURL: string;
  model: string;
  apiKey: string;
  /** Max inputs per request (default 64). */
  batchSize?: number;
  /** Optional output dimensions, sent as `dimensions` when set. */
  dimensions?: number;
  /** Request timeout in ms (default 60000). */
  timeoutMs?: number;
  /** Called after each request with the reported token usage, if any. */
  onUsage?: (usage: { promptTokens: number; totalTokens: number }) => void;
  /** Injected fetch, for tests. */
  fetch?: typeof fetch;
}

export class EmbedderError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "EmbedderError";
  }
}

interface EmbeddingResponse {
  data?: { index?: number; embedding?: number[] }[];
  usage?: { prompt_tokens?: number; total_tokens?: number };
}

export function createOpenAICompatEmbedder(opts: OpenAICompatEmbedderOptions): Embedder {
  const url = opts.baseURL.replace(/\/+$/, "") + "/embeddings";
  const batchSize = Math.max(1, opts.batchSize ?? 64);
  const doFetch = opts.fetch ?? fetch;
  const timeoutMs = opts.timeoutMs ?? 60_000;

  return async (texts: string[]): Promise<Float32Array[]> => {
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const body: Record<string, unknown> = { model: opts.model, input: batch };
      if (opts.dimensions !== undefined) body.dimensions = opts.dimensions;
      const res = await doFetch(url, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${opts.apiKey}` },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!res.ok) {
        const detail = (await res.text().catch(() => "")).slice(0, 300);
        throw new EmbedderError(`embeddings request failed: HTTP ${res.status} ${detail}`, res.status);
      }
      const json = (await res.json()) as EmbeddingResponse;
      const data = json.data ?? [];
      if (data.length !== batch.length) {
        throw new EmbedderError(`expected ${batch.length} embeddings, got ${data.length}`);
      }
      const sorted = [...data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      for (const d of sorted) {
        if (!Array.isArray(d.embedding)) throw new EmbedderError("response item missing embedding");
        out.push(Float32Array.from(d.embedding));
      }
      if (opts.onUsage && json.usage) {
        opts.onUsage({
          promptTokens: json.usage.prompt_tokens ?? 0,
          totalTokens: json.usage.total_tokens ?? json.usage.prompt_tokens ?? 0,
        });
      }
    }
    return out;
  };
}
