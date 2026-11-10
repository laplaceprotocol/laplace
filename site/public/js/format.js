// How the page writes numbers, times and bins. Pure functions, so the wording rules are tested.

const MINUS = "−";
const HOUR_MS = 3_600_000;

export const issueTime = (issue) => new Date(issue * HOUR_MS);

/** "2026-10-28 07:00 UTC" (seconds with `seconds: true`). */
export function utc(date, { seconds = false } = {}) {
  const iso = date.toISOString();
  return `${iso.slice(0, 10)} ${iso.slice(11, seconds ? 19 : 16)} UTC`;
}

/** "07:00 UTC" (or "07:07:29 UTC" with `seconds: true`). */
export const clock = (date, { seconds = false } = {}) => `${date.toISOString().slice(11, seconds ? 19 : 16)} UTC`;

export const thousands = (n) => n.toLocaleString("en-US");

/** A gain in nats per question with its sign, e.g. "+0.0181" or "−0.0619"; a dash when there is none. */
export function gain(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return (value > 0 ? "+" : MINUS) + text;
}

/** A plain signed number in the page's minus sign, e.g. a log score. */
export function signed(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return (value < 0 ? MINUS : "") + Math.abs(value).toFixed(digits);
}

/** Basis points as a percentage, e.g. 4126 -> "41.3%". */
export const percent = (bps, digits = 1) => `${(bps / 100).toFixed(digits)}%`;

/** A short hash for reading, e.g. "0xff70…122a"; the full value stays in the link or the title. */
export const short = (hex, keep = 4) => `${hex.slice(0, 2 + keep)}…${hex.slice(-keep)}`;

/** The edges of a horizon's ten bins in percent of log return, from the feed's four positive edges. */
export function binEdges(positive) {
  return [...positive.map((e) => -e).reverse(), 0, ...positive];
}

/** Labels of the ten bins: "below −3%", "−3 to −1.5%", …, "+3% or more". */
export function binLabels(positive) {
  const edges = binEdges(positive);
  const pct = (e) => (e > 0 ? `+${e}` : e < 0 ? `${MINUS}${-e}` : "0");
  const labels = [`below ${pct(edges[0])}%`];
  for (let k = 1; k < edges.length; k++) labels.push(`${pct(edges[k - 1])} to ${pct(edges[k])}%`);
  labels.push(`${pct(edges[edges.length - 1])}% or more`);
  return labels;
}

/** The price range of each bin from a starting price: returns are log returns, so a price is start · e^(r/100). */
export function binPrices(start, positive) {
  const at = binEdges(positive).map((e) => start * Math.exp(e / 100));
  return Array.from({ length: at.length + 1 }, (_, k) => [k === 0 ? null : at[k - 1], k === at.length ? null : at[k]]);
}

export function price(value) {
  return value.toLocaleString("en-US", { maximumSignificantDigits: 6 });
}

/** The basis points a forecast puts on a move beyond ±edges[k] (k = 0 for the innermost edge). */
export function beyond(bps, k) {
  const tail = 4 - k; // bins outside the edge on each side
  return bps.slice(0, tail).reduce((a, b) => a + b, 0) + bps.slice(bps.length - tail).reduce((a, b) => a + b, 0);
}

/** Plural-aware counts: "1 issue", "80 issues". */
export const count = (n, word) => `${thousands(n)} ${word}${n === 1 ? "" : "s"}`;
