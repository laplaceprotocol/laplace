// What the page says about the record that the record's files do not: the rules' terms, the ladder and display
// names. Every fact here is copied from the document it links to; the numbers of the live record are never here.

export const REPO = "https://github.com/laplaceprotocol/laplace";
export const doc = (path) => `${REPO}/blob/main/${path}`;

export const NOTES = {
  poc1: "docs/research/2026-10-23-poc1-majors.md",
  poc2: "docs/research/2026-10-24-poc2-evolution.md",
  events: "docs/research/2026-10-25-scheduled-events.md",
  learned: "docs/research/2026-10-27-learned-layer.md",
  gbm: "docs/research/2026-10-27-ev-gbm.md",
};

/** The pre-registered rules (docs/judging), by the path the scoreboard names them with. */
export const RULES = {
  "docs/judging/r1.md": {
    name: "Rule r1",
    title: "the Demon against GARCH and HAR",
    registered: "2026-10-25",
    question:
      "Does the Demon of release r1 forecast the 1- and 4-hour questions better than the best classical volatility " +
      "models?",
    claims: [{ label: "the classical baselines", opponents: ["garch", "har"] }],
    day30: "2026-11-24 00:00 UTC",
    looks: [
      { day: 90, after: "2027-01-23", z: 4.36 },
      { day: 180, after: "2027-04-23", z: 2.99 },
      { day: 365, after: "2027-10-25", z: 1.97 },
    ],
  },
  "docs/judging/r2.md": {
    name: "Rule r2",
    title: "EV-GBM against opponents fitted on the same data",
    registered: "2026-10-28",
    question:
      "Does EV-GBM, a learned correction on top of a calendar-aware classical model, forecast the 1- and 4-hour " +
      "questions better than (A) GARCH and HAR and (B) EV-HAR, all fitted on the same data?",
    claims: [
      { label: "claim A, the classical bar", opponents: ["garch-r15", "har-r15"] },
      { label: "claim B, the learning itself", opponents: ["ev-har-r15"] },
    ],
    day30: "2026-11-28 00:00 UTC",
    looks: [
      { day: 90, after: "2027-01-27", z: 4.36 },
      { day: 180, after: "2027-04-27", z: 2.99 },
      { day: 365, after: "2027-10-29", z: 1.97 },
    ],
  },
};

/** The two forecasters the rules judge, shown in the odds: by identity, which never changes. */
export const JUDGED = [
  { id: "0x97bc9657cb5b4af73b46e0031ac385cab039e72fb9fb8edf591bd457f149914e", rule: "docs/judging/r1.md" },
  { id: "0xb968a022ac9be499bdd06a67a764ee53bdb38c606c7f8e1fc2748f1972845454", rule: "docs/judging/r2.md" },
];

export const NAMES = {
  naive: "naive",
  ewma: "EWMA",
  garch: "GARCH",
  har: "HAR",
  monkey: "the monkey",
  "rv-har": "RV-HAR",
  "iv-har": "IV-HAR",
  "ev-har": "EV-HAR",
  "ev-net": "EV-NET",
  "garch-r15": "GARCH-r15",
  "har-r15": "HAR-r15",
  "ev-har-r15": "EV-HAR-r15",
  "ev-gbm-r15": "EV-GBM",
  demon: "Demon",
};

export const label = (name) => NAMES[name] ?? name;

/**
 * The ladder, from the simplest forecaster to the most elaborate. Each rung lists its instances in the record and,
 * above the reference, one line of development evidence: backtests inspected while building, so a reason to be here,
 * not a result.
 */
export const LADDER = [
  {
    rung: "naive",
    adds: "Last week's volatility and nothing else: the seed every evolved program started from, and the reference for " +
      "every gain below.",
    instances: [{ name: "naive", fit: "release r1" }],
    evidence: null,
  },
  {
    rung: "GARCH(1,1)-t",
    adds: "Volatility that clusters, with fat tails.",
    instances: [
      { name: "garch", fit: "fitted to 2026-06-01 (r1)", role: "opponent in rule r1" },
      { name: "garch-r15", fit: "refitted to 2026-09-27 (r1.5)", role: "opponent in rule r2, claim A" },
    ],
    evidence: { text: "+0.0297 over naive, all horizons, on the PoC 2 holdout.", note: "poc2" },
  },
  {
    rung: "HAR-RV",
    adds: "Volatility over the last hour, day, week and month, combined by regression.",
    instances: [
      { name: "har", fit: "fitted to 2026-06-01 (r1)", role: "opponent in rule r1" },
      { name: "har-r15", fit: "refitted to 2026-09-27 (r1.5)", role: "opponent in rule r2, claim A" },
    ],
    evidence: {
      text: "+0.0268 over naive, all horizons, level with GARCH (−0.0030, −0.0134 to +0.0066), on the PoC 2 holdout.",
      note: "poc2",
    },
  },
  {
    rung: "EV-HAR",
    adds: "Implied volatility and the calendar: weekends, the US session, FOMC, CPI and jobs releases.",
    instances: [
      { name: "ev-har", fit: "fitted to 2026-06-01 (r1.3)" },
      { name: "ev-har-r15", fit: "refitted to 2026-09-27 (r1.5)", role: "opponent in rule r2, claim B" },
    ],
    evidence: {
      text: "+0.0167 over GARCH at 1 and 4 hours (+0.0060 to +0.0269), ahead in 5 of 7 months, rolling test.",
      note: "learned",
    },
  },
  {
    rung: "EV-GBM",
    adds: "Boosted trees that correct EV-HAR's probabilities.",
    instances: [{ name: "ev-gbm-r15", fit: "fitted to 2026-09-27, stopped to 2026-10-27 (r1.5)", role: "judged by rule r2" }],
    evidence: {
      text: "+0.0150 over EV-HAR at 1 and 4 hours (+0.0098 to +0.0206), ahead in 7 of 7 months, rolling test.",
      note: "gbm",
    },
  },
  {
    rung: "the Demon",
    adds: "A pool of programs that language models evolved, with fixed weights.",
    instances: [
      { name: "demon", fit: "programs evolved on data to 2026-06-01, weights fitted to 2026-08-16 (r1)", role: "judged by rule r1" },
    ],
    evidence: {
      text: "+0.0210 over GARCH at 1 and 4 hours (+0.0087 to +0.0335) on the PoC 2 holdout, where EV-HAR alone got +0.0185.",
      note: "events",
    },
  },
];
