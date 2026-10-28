// Scores so far: each rule's running statistic, the ladder and every forecaster, from scoreboard.json as published.
// Nothing here is a verdict: the rules give one only at their scheduled looks.

import { LADDER, NOTES, RULES, doc, label } from "./content.js";
import { fill, h, link } from "./dom.js";
import { count, gain, short, signed, thousands } from "./format.js";

const LOOK_DAYS = 90; // the first scheduled look of every rule so far
const BLOCKS = [72, 168]; // the rules' two bootstrap block lengths, in issues; a z needs two blocks
const DASH = "—";

export function renderRules(root, { board, manifest }) {
  const hashes = new Map((manifest.judging ?? []).map((j) => [j.path, j.keccak256]));
  fill(
    root,
    (board.rules ?? []).map((entry) => rule(entry, hashes.get(entry.rule), manifest.issue)),
  );
}

function rule(entry, hash, issue) {
  const terms = RULES[entry.rule];
  const name = terms?.name ?? entry.rule;
  const elapsed = Math.max(0, issue - entry.start_issue + 1);
  const claims = terms?.claims ?? [{ label: null, opponents: Object.keys(entry.vs) }];
  const rows = claims.flatMap((claim) => [
    claims.length > 1 ? h("tr", { class: "group" }, h("th", { colspan: 5, scope: "rowgroup" }, claim.label)) : null,
    claim.opponents.filter((o) => entry.vs[o]).map((o) => opponentRow(o, entry.vs[o])),
  ]);
  return h(
    "article",
    { class: "rule" },
    h(
      "header",
      {},
      h("h3", {}, name, h("span", { class: "muted" }, ` · ${terms?.title ?? ""}`)),
      h("span", { class: "badge" }, entry.status),
    ),
    terms ? h("p", {}, terms.question) : null,
    h(
      "table",
      { class: "stats" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", { scope: "col" }, `${label(entry.judged.name)} against`),
          h("th", { scope: "col", class: "num" }, "gain, 1 h + 4 h"),
          h("th", { scope: "col", class: "num" }, "issues"),
          BLOCKS.map((b) => h("th", { scope: "col", class: "num" }, `z (${b})`)),
        ),
      ),
      h("tbody", {}, rows),
    ),
    h("progress", {
      max: 24 * LOOK_DAYS,
      value: Math.min(elapsed, 24 * LOOK_DAYS),
      "aria-label": `${(elapsed / 24).toFixed(1)} of the ${LOOK_DAYS} days before the first look`,
    }),
    h(
      "p",
      { class: "small" },
      `Day ${(elapsed / 24).toFixed(1)} of the ${LOOK_DAYS} before the first look, counted from issue ` +
        `${entry.start_issue}. Gains are in nats per question; z is the gain over its bootstrap standard error with ` +
        `blocks of ${BLOCKS.join(" or ")} issues, shown once ${BLOCKS.map((b) => 2 * b).join(" or ")} issues count. `,
      terms ? lookText(terms) : null,
      " ",
      hash ? `Its keccak256, ${short(hash)}, is in this hour's manifest. ` : null,
      link(doc(entry.rule), `Read ${name.toLowerCase()}`),
      ".",
    ),
  );
}

function opponentRow(name, s) {
  return h(
    "tr",
    {},
    h("th", { scope: "row" }, label(name)),
    h("td", { class: "num" }, gain(s.mean)),
    h("td", { class: "num" }, thousands(s.issues)),
    BLOCKS.map((b) => {
      const z = s.z?.[String(b)];
      return h("td", { class: "num" }, z === null || z === undefined ? DASH : signed(z, 2));
    }),
  );
}

function lookText(terms) {
  const looks = terms.looks.map((l) => `after ${l.after} (day ${l.day}, z ≥ ${l.z})`);
  return (
    `Registered ${terms.registered}. A direction report after ${terms.day30}, without a verdict; ` +
    `looks ${looks.slice(0, -1).join(", ")} and ${looks.at(-1)}. A claim needs the threshold with both block lengths.`
  );
}

export function renderLadder(root, { board, manifest }) {
  const ids = new Map(manifest.forecasters.map((f) => [f.name, f.id]));
  const rungs = LADDER.map((rung) => {
    const instances = rung.instances.filter((i) => ids.has(i.name));
    return h(
      "tbody",
      {},
      h(
        "tr",
        { class: "rung" },
        h(
          "th",
          { colspan: 2, scope: "rowgroup" },
          rung.rung,
          h("small", {}, rung.adds),
          rung.evidence
            ? h(
                "small",
                { class: "evidence" },
                h("em", {}, "Development evidence: "),
                rung.evidence.text,
                " ",
                link(doc(NOTES[rung.evidence.note]), "Note"),
              )
            : null,
        ),
      ),
      instances.map((instance) =>
        h(
          "tr",
          {},
          h(
            "td",
            {},
            h("span", { class: "name" }, label(instance.name)),
            h("small", {}, [instance.fit, instance.role].filter(Boolean).join(" · ")),
          ),
          h("td", { class: "num" }, liveGain(instance.name, board.forecasters?.[ids.get(instance.name)])),
        ),
      ),
    );
  });
  fill(
    root,
    h(
      "table",
      { class: "ladder" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", { scope: "col" }, "rung, and its forecasters in the record"),
          h("th", { scope: "col", class: "num" }, "live: gain over naive, 1 h + 4 h, running"),
        ),
      ),
      rungs,
    ),
  );
}

function liveGain(name, entry) {
  if (name === "naive") return "reference";
  const p = entry?.vs?.naive?.primary;
  if (!p || p.mean === null) return "not settled yet";
  return [gain(p.mean), h("small", {}, count(p.issues, "issue"))];
}

export function renderTable(root, { board, manifest }) {
  const rows = manifest.forecasters.map((f) => {
    const entry = board.forecasters?.[f.id];
    const vs = (base) => (f.name === base ? DASH : primary(entry?.vs?.[base]?.primary));
    return h(
      "tr",
      {},
      h(
        "th",
        { scope: "row", title: f.origin ?? "" },
        h("span", { class: "name" }, label(f.name)),
        h("code", {}, short(f.id)),
      ),
      h("td", {}, f.kind === "pool" ? "pool" : f.public ? "program, public" : "program, private until revealed"),
      h("td", { class: "num" }, entry ? thousands(entry.questions) : DASH),
      h("td", { class: "num" }, entry ? signed(entry.mean_log_score) : DASH),
      ["naive", "garch", "har"].map((b) => h("td", { class: "num" }, vs(b))),
    );
  });
  fill(
    root,
    h(
      "div",
      { class: "scroll" },
      h(
        "table",
        { class: "everyone" },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { scope: "col" }, "forecaster"),
            h("th", { scope: "col" }, "kind"),
            h("th", { scope: "col", class: "num" }, "questions scored"),
            h("th", { scope: "col", class: "num" }, "mean log score"),
            h("th", { scope: "col", class: "num" }, "vs naive"),
            h("th", { scope: "col", class: "num" }, "vs GARCH (r1)"),
            h("th", { scope: "col", class: "num" }, "vs HAR (r1)"),
          ),
        ),
        h("tbody", {}, rows),
      ),
    ),
  );
}

function primary(p) {
  if (!p || p.mean === null) return DASH;
  return [gain(p.mean), h("small", {}, count(p.issues, "issue"))];
}
