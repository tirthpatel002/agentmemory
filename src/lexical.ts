// src/lexical.ts
// Lexical similarity used when no Embedder is configured: BM25 term weighting over the
// lesson texts in scope, normalised to [0, 1] so it can share the ranking formula.

const STOPWORDS = new Set(
  (
    "a an the and or but as of to for in on at by with from into onto is are was were be been being " +
    "it its this that these those if then than so do does did done can could should would will shall " +
    "i me my we our you your they them their he she his her there here what which who whom when where " +
    "how why all any some each every no nor only own same too very just also via per about over under"
  ).split(" "),
);

/** Light suffix stemmer so "partitioned"/"partition" and "reports"/"report" match. */
export function stem(word: string): string {
  let w = word;
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 5 && w.endsWith("ing")) w = w.slice(0, -3);
  else if (w.length > 4 && w.endsWith("ed")) w = w.slice(0, -2);
  else if (w.length > 4 && w.endsWith("es")) w = w.slice(0, -2);
  else if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
  return w;
}

/** Lowercase, split on anything non-alphanumeric (so "pdf.render" -> pdf, render), drop stopwords, stem. */
export function lexicalTokens(text: string): string[] {
  const raw = text.toLowerCase().split(/[^a-z0-9]+/);
  const out: string[] = [];
  for (const t of raw) {
    if (t.length < 2 || STOPWORDS.has(t)) continue;
    out.push(stem(t));
  }
  return out;
}

export interface Bm25Options {
  k1?: number | undefined; // 1.2
  b?: number | undefined; // 0.75
}

/**
 * Normalised BM25 similarity of `query` against each document in `docs` (the corpus).
 *
 * score(d) = sum over query terms t in d of idf(t) * tf'(t, d)  /  sum over query terms t of idf(t)
 *   tf'(t, d) = tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))   (BM25 saturation, 1 at tf=1, |d|=avgdl)
 *   idf(t)    = ln(1 + (N - df + 0.5) / (df + 0.5)), with df floored at 1
 * The df floor stops query words that no lesson contains from swamping tiny corpora, so the
 * score reads as "IDF-weighted share of the query covered by this lesson". Clamped to [0, 1].
 */
export function bm25Similarities(query: string, docs: string[], opts?: Bm25Options): number[] {
  const k1 = opts?.k1 ?? 1.2;
  const b = opts?.b ?? 0.75;
  const qTerms = [...new Set(lexicalTokens(query))];
  if (qTerms.length === 0 || docs.length === 0) return docs.map(() => 0);

  const docTf = docs.map((d) => {
    const m = new Map<string, number>();
    const toks = lexicalTokens(d);
    for (const t of toks) m.set(t, (m.get(t) ?? 0) + 1);
    return { m, len: toks.length };
  });
  const n = docs.length;
  const avgdl = docTf.reduce((s, d) => s + d.len, 0) / n || 1;
  const idf = new Map<string, number>();
  for (const t of qTerms) {
    const df = Math.max(1, docTf.filter((d) => d.m.has(t)).length);
    idf.set(t, Math.log(1 + (n - df + 0.5) / (df + 0.5)));
  }
  const maxScore = qTerms.reduce((s, t) => s + idf.get(t)!, 0);

  return docTf.map((d) => {
    if (d.len === 0 || maxScore === 0) return 0;
    let score = 0;
    for (const t of qTerms) {
      const tf = d.m.get(t);
      if (!tf) continue;
      const sat = (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * d.len) / avgdl));
      score += idf.get(t)! * sat;
    }
    return Math.max(0, Math.min(1, score / maxScore));
  });
}
