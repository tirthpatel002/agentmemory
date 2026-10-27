// src/validator.ts
import { cosine } from "./embed.js";
import { wordJaccardSimilarity } from "./retriever.js";
import type { DraftLesson, Embedder, Lesson, Scope } from "./types.js";

export interface LessonValidator {
  name: string;
  validate(
    draft: DraftLesson,
    ctx: { scope: Scope; active: Lesson[] },
  ): Promise<{ passed: boolean; reasons: string[] }>;
}

export interface BasicValidatorOptions {
  mergeThreshold?: number | undefined;
  jaccardThreshold?: number | undefined;
  embedder?: Embedder | undefined;
}

const INJECTION_PHRASES = [
  "ignore previous instructions",
  "you are now",
  "system prompt",
  "disregard the above",
];

export class BasicValidator implements LessonValidator {
  public readonly name = "BasicValidator";
  private readonly mergeThreshold: number;
  private readonly jaccardThreshold: number;
  private readonly embedder?: Embedder | undefined;

  constructor(opts?: BasicValidatorOptions) {
    this.mergeThreshold = opts?.mergeThreshold ?? 0.88;
    this.jaccardThreshold = opts?.jaccardThreshold ?? 0.6;
    this.embedder = opts?.embedder;
  }

  async validate(
    draft: DraftLesson,
    ctx: { scope: Scope; active: Lesson[] },
  ): Promise<{ passed: boolean; reasons: string[] }> {
    const reasons: string[] = [];

    // 1. Instruction-injection phrases
    const lowerText = draft.text.toLowerCase();
    const lowerTrigger = draft.trigger.toLowerCase();

    for (const phrase of INJECTION_PHRASES) {
      if (lowerText.includes(phrase) || lowerTrigger.includes(phrase)) {
        reasons.push(`Prompt injection detected: contains "${phrase}"`);
      }
    }

    // 2. Contradiction: active lesson in scope with similarity >= threshold, opposite advice, no updatesLessonId
    const draftCombined = `${draft.trigger} ${draft.text}`;
    let draftEmbedding: Float32Array | null = null;
    if (this.embedder) {
      const embs = await this.embedder([draftCombined]);
      draftEmbedding = embs[0] ?? null;
    }

    for (const activeLesson of ctx.active) {
      const isOpposite =
        (draft.kind === "avoid" && activeLesson.kind === "prefer") ||
        (draft.kind === "prefer" && activeLesson.kind === "avoid");

      if (!isOpposite) {
        continue;
      }

      if (draft.updatesLessonId && draft.updatesLessonId === activeLesson.lessonId) {
        continue;
      }

      let similarity = 0;
      let threshold = this.jaccardThreshold;

      if (draftEmbedding !== null && activeLesson.embedding !== null) {
        similarity = Math.max(0, Math.min(1, cosine(draftEmbedding, activeLesson.embedding)));
        threshold = this.mergeThreshold;
      } else {
        const activeCombined = `${activeLesson.trigger} ${activeLesson.text}`;
        similarity = wordJaccardSimilarity(draftCombined, activeCombined);
        threshold = this.jaccardThreshold;
      }

      if (similarity >= threshold) {
        reasons.push(
          `Contradicts active lesson ${activeLesson.lessonId} ("${activeLesson.text}") with opposite advice without updatesLessonId (similarity: ${similarity.toFixed(2)} >= ${threshold})`,
        );
      }
    }

    return {
      passed: reasons.length === 0,
      reasons,
    };
  }
}
