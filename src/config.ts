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
