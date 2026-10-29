// src/retriever.ts
import type { MemoryConfig } from "./config.js";
import { bm25Similarities } from "./lexical.js";
import type { MemoryStore } from "./store/store.js";
import type { Lesson, Scope } from "./types.js";

export interface Retriever {
  rank(
    scope: Scope,
    task: string,
    taskEmbedding: Float32Array | null,
    k: number,
    store: MemoryStore,
    cfg: MemoryConfig,
  ): Promise<Lesson[]>; // each with score
}

export interface WeightedRetrieverOptions {
  now?: (() => Date) | undefined;
}

/**
 * Simple text-overlap similarity fallback (Jaccard similarity on lowercased word tokens).
 *
 * Algorithm:
 * 1. Tokenizes query task and lesson target (trigger + text) by extracting lowercased alphanumeric words (\b\w+\b).
 * 2. Builds two distinct sets of words: A (task) and B (lesson trigger + text).
 * 3. Computes the intersection |A ∩ B| and union |A ∪ B|.
 * 4. Returns |A ∩ B| / |A ∪ B| (or 0 if both sets are empty), producing a similarity score in [0, 1].
 */
export function wordJaccardSimilarity(task: string, lessonText: string): number {
  const tokenize = (s: string): Set<string> => {
    const tokens = s.toLowerCase().match(/\b\w+\b/g);
    return new Set(tokens ?? []);
  };

  const taskWords = tokenize(task);
  const lessonWords = tokenize(lessonText);

  if (taskWords.size === 0 || lessonWords.size === 0) {
    return 0;
  }

  let intersectionSize = 0;
  for (const word of taskWords) {
    if (lessonWords.has(word)) {
      intersectionSize++;
    }
  }

  const unionSize = taskWords.size + lessonWords.size - intersectionSize;
  if (unionSize === 0) {
    return 0;
  }

  return intersectionSize / unionSize;
}

export class WeightedRetriever implements Retriever {
  private readonly now: () => Date;

  constructor(opts?: WeightedRetrieverOptions) {
    this.now = opts?.now ?? (() => new Date());
  }

  async rank(
    scope: Scope,
    task: string,
    taskEmbedding: Float32Array | null,
    k: number,
    store: MemoryStore,
    cfg: MemoryConfig,
  ): Promise<Lesson[]> {
    if (k <= 0) {
      return [];
    }

    const { similarity: wSim, helpfulness: wHelp, recency: wRec } = cfg.weights;
    const nowMs = this.now().getTime();
    const halfLife = cfg.recencyHalfLifeDays > 0 ? cfg.recencyHalfLifeDays : 14;

    interface ScoredHit {
      lesson: Lesson;
      score: number;
      similarity: number;
    }

    const candidates: ScoredHit[] = [];

    if (taskEmbedding !== null) {
      // Vector search in store (status active only)
      const hits = await store.searchLessons(scope, taskEmbedding, 10000, { status: ["active"] });
      for (const hit of hits) {
        if (hit.lesson.status !== "active") {
          continue;
        }

        const similarity = Math.max(0, Math.min(1, hit.similarity));
        if (similarity < cfg.minSimilarity) {
          continue;
        }

        const helped = hit.lesson.helped;
        const hurt = hit.lesson.hurt;
        const helpfulness = (helped + 1) / (helped + hurt + 2);

        const updatedMs = Date.parse(hit.lesson.updatedAt);
        const ageMs = Math.max(0, nowMs - (Number.isNaN(updatedMs) ? nowMs : updatedMs));
        const ageDays = ageMs / (1000 * 60 * 60 * 24);
        const recency = Math.exp(-ageDays / halfLife);

        const score = wSim * similarity + wHelp * helpfulness + wRec * recency;
        candidates.push({ lesson: hit.lesson, score, similarity });
      }
    } else {
      // Fallback without embedder: normalised BM25 over trigger + text of the active lessons in scope
      const lessons = (await store.listLessons(scope, { status: ["active"] })).filter(
        (l) => l.status === "active",
      );
      const sims = bm25Similarities(
        task,
        lessons.map((l) => `${l.trigger} ${l.text}`),
      );
      for (const [i, lesson] of lessons.entries()) {
        const similarity = sims[i] ?? 0;

        if (similarity < cfg.minLexicalSimilarity) {
          continue;
        }

        const helped = lesson.helped;
        const hurt = lesson.hurt;
        const helpfulness = (helped + 1) / (helped + hurt + 2);

        const updatedMs = Date.parse(lesson.updatedAt);
        const ageMs = Math.max(0, nowMs - (Number.isNaN(updatedMs) ? nowMs : updatedMs));
        const ageDays = ageMs / (1000 * 60 * 60 * 24);
        const recency = Math.exp(-ageDays / halfLife);

        const score = wSim * similarity + wHelp * helpfulness + wRec * recency;
        candidates.push({ lesson, score, similarity });
      }
    }

    candidates.sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      if (b.similarity !== a.similarity) {
        return b.similarity - a.similarity;
      }
      return b.lesson.lessonId.localeCompare(a.lesson.lessonId);
    });

    return candidates.slice(0, k).map((c) => ({
      ...c.lesson,
      score: c.score,
    }));
  }
}
