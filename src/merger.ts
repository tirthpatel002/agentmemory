// src/merger.ts
import type { MemoryConfig } from "./config.js";
import { wordJaccardSimilarity } from "./retriever.js";
import type { MemoryStore } from "./store/store.js";
import type { DraftLesson, Embedder, Lesson, Scope } from "./types.js";

export type MergeDecision =
  | { action: "insert" }
  | { action: "strengthen"; lessonId: string; text?: string | undefined }
  | { action: "replace"; lessonId: string };

export interface Merger {
  decide(
    draft: DraftLesson,
    draftEmbedding: Float32Array | null,
    scope: Scope,
    store: MemoryStore,
    cfg: MemoryConfig,
  ): Promise<MergeDecision>;
}

export interface EmbeddingMergerOptions {
  embedder?: Embedder | undefined;
}

export class EmbeddingMerger implements Merger {
  private readonly embedder?: Embedder | undefined;

  constructor(opts?: EmbeddingMergerOptions) {
    this.embedder = opts?.embedder;
  }

  async decide(
    draft: DraftLesson,
    draftEmbedding: Float32Array | null,
    scope: Scope,
    store: MemoryStore,
    cfg: MemoryConfig,
  ): Promise<MergeDecision> {
    let embedding = draftEmbedding;
    if (embedding === null && this.embedder) {
      const embs = await this.embedder([`${draft.trigger} ${draft.text}`]);
      embedding = embs[0] ?? null;
    }

    if (embedding !== null) {
      const hits = await store.searchLessons(scope, embedding, 1000, {
        status: ["active", "candidate", "rejected"],
      });
      const best = hits[0];
      if (best && best.similarity >= cfg.mergeThreshold) {
        return this.evaluateMatch(draft, best.lesson);
      }
    } else {
      const allLessons = await store.listLessons(scope, {
        status: ["active", "candidate", "rejected"],
      });
      const combinedDraft = `${draft.trigger} ${draft.text}`;
      let bestLesson: Lesson | null = null;
      let bestSim = 0;

      for (const lesson of allLessons) {
        const combinedLesson = `${lesson.trigger} ${lesson.text}`;
        const sim = wordJaccardSimilarity(combinedDraft, combinedLesson);
        if (sim > bestSim) {
          bestSim = sim;
          bestLesson = lesson;
        }
      }

      if (bestLesson && bestSim >= cfg.jaccardThreshold) {
        return this.evaluateMatch(draft, bestLesson);
      }
    }

    return { action: "insert" };
  }

  private evaluateMatch(draft: DraftLesson, match: Lesson): MergeDecision {
    if (draft.kind === match.kind) {
      const shouldUpdateText =
        draft.updatesLessonId === match.lessonId &&
        draft.text.length > match.text.length;

      return {
        action: "strengthen",
        lessonId: match.lessonId,
        ...(shouldUpdateText ? { text: draft.text } : {}),
      };
    }

    const isOpposite =
      (draft.kind === "avoid" && match.kind === "prefer") ||
      (draft.kind === "prefer" && match.kind === "avoid");

    if (isOpposite && draft.updatesLessonId === match.lessonId && match.status === "active") {
      return {
        action: "replace",
        lessonId: match.lessonId,
      };
    }

    return { action: "insert" };
  }
}
