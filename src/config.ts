// src/config.ts

export interface MemoryConfig {
  maxLessonsPerRun: number; // 3
  weights: { similarity: number; helpfulness: number; recency: number }; // 0.6 / 0.3 / 0.1
  recencyHalfLifeDays: number; // 14
  minSimilarity: number; // 0.3 : below this a lesson is never recalled
  mergeThreshold: number; // 0.88 cosine
  jaccardThreshold: number; // 0.6 when no embedder
  maxInitialConfidence: number; // 0.7
  retire: { minUses: number; rule: "hurtGtHelped" }; // 3, hurt > helped -> replaced
  quarantine: {
    enabled: boolean; // true : new lessons start as "candidate"
    promoteAfterHelpedUses: number; // 2 : candidate -> active after this many helped uses
    trialSlots: number; // 1 : candidates (validator passed) recall may add as trials
    requireApproval: boolean; // false : true means only approveLesson() activates
  };
}

export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  maxLessonsPerRun: 3,
  weights: { similarity: 0.6, helpfulness: 0.3, recency: 0.1 },
  recencyHalfLifeDays: 14,
  minSimilarity: 0.3,
  mergeThreshold: 0.88,
  jaccardThreshold: 0.6,
  maxInitialConfidence: 0.7,
  retire: { minUses: 3, rule: "hurtGtHelped" },
  quarantine: {
    enabled: true,
    promoteAfterHelpedUses: 2,
    trialSlots: 1,
    requireApproval: false,
  },
};

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export function mergeConfig(partial?: DeepPartial<MemoryConfig> | Partial<MemoryConfig>): MemoryConfig {
  if (!partial) {
    return {
      ...DEFAULT_MEMORY_CONFIG,
      weights: { ...DEFAULT_MEMORY_CONFIG.weights },
      retire: { ...DEFAULT_MEMORY_CONFIG.retire },
      quarantine: { ...DEFAULT_MEMORY_CONFIG.quarantine },
    };
  }

  return {
    maxLessonsPerRun: partial.maxLessonsPerRun ?? DEFAULT_MEMORY_CONFIG.maxLessonsPerRun,
    weights: {
      similarity: partial.weights?.similarity ?? DEFAULT_MEMORY_CONFIG.weights.similarity,
      helpfulness: partial.weights?.helpfulness ?? DEFAULT_MEMORY_CONFIG.weights.helpfulness,
      recency: partial.weights?.recency ?? DEFAULT_MEMORY_CONFIG.weights.recency,
    },
    recencyHalfLifeDays: partial.recencyHalfLifeDays ?? DEFAULT_MEMORY_CONFIG.recencyHalfLifeDays,
    minSimilarity: partial.minSimilarity ?? DEFAULT_MEMORY_CONFIG.minSimilarity,
    mergeThreshold: partial.mergeThreshold ?? DEFAULT_MEMORY_CONFIG.mergeThreshold,
    jaccardThreshold: partial.jaccardThreshold ?? DEFAULT_MEMORY_CONFIG.jaccardThreshold,
    maxInitialConfidence: partial.maxInitialConfidence ?? DEFAULT_MEMORY_CONFIG.maxInitialConfidence,
    retire: {
      minUses: partial.retire?.minUses ?? DEFAULT_MEMORY_CONFIG.retire.minUses,
      rule: partial.retire?.rule ?? DEFAULT_MEMORY_CONFIG.retire.rule,
    },
    quarantine: {
      enabled: partial.quarantine?.enabled ?? DEFAULT_MEMORY_CONFIG.quarantine.enabled,
      promoteAfterHelpedUses:
        partial.quarantine?.promoteAfterHelpedUses ?? DEFAULT_MEMORY_CONFIG.quarantine.promoteAfterHelpedUses,
      trialSlots: partial.quarantine?.trialSlots ?? DEFAULT_MEMORY_CONFIG.quarantine.trialSlots,
      requireApproval: partial.quarantine?.requireApproval ?? DEFAULT_MEMORY_CONFIG.quarantine.requireApproval,
    },
  };
}
