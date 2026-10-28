// The latest commitment, as latest.json and the issue's manifest describe it. The in-browser check re-reads it from
// the chain; this section only reports it.

import { COMMIT_WINDOW_S, EXPLORER, LEDGER } from "./chain.js";
import { REPO, label } from "./content.js";
import { fill, h, link } from "./dom.js";
import { clock, count, issueTime, short, thousands, utc } from "./format.js";

const LATE_MS = 20 * 60_000; // Laplace Watch also alerts at :20 if the hour's issue is not public

export function renderStatus(root, { latest, manifest, board, record, now = Date.now() }) {
  const opens = issueTime(latest.issue);
  const committed = new Date(latest.timestamp * 1000);
  const delay = latest.timestamp - opens.getTime() / 1000;
  const next = latest.issue + 1;
  const failures = Object.entries(manifest.failed ?? {});
  const late = now > issueTime(next).getTime() + LATE_MS;
  const file = (name) => link(new URL(`issues/${latest.issue}/${name}`, record).href, name);
  fill(
    root,
    late
      ? h(
          "p",
          { class: "notice" },
          `The ${clock(issueTime(next))} issue (${next}) is not public yet, although its window closed at ` +
            `${clock(new Date(issueTime(next).getTime() + COMMIT_WINDOW_S * 1000))}. The ledger's lastIssue() says ` +
            "whether it was committed.",
        )
      : null,
    h(
      "dl",
      { class: "facts" },
      h("dt", {}, "Issue"),
      h("dd", {}, `${latest.issue}, the forecasts made at ${utc(opens)}`),
      h("dt", {}, "Committed"),
      h(
        "dd",
        {},
        `${clock(committed, { seconds: true })}, ${minutes(delay)} into its 15-minute window, in block `,
        link(`${EXPLORER}/block/${latest.block}`, thousands(latest.block)),
        ", ",
        link(`${EXPLORER}/tx/${latest.tx}`, `transaction ${short(latest.tx)}`),
      ),
      h("dt", {}, "Forecasts"),
      h(
        "dd",
        {},
        `${thousands(manifest.leaves)} from ${count(manifest.forecasters.length, "forecaster")} of release ${manifest.release}, `,
        "root ",
        h("code", { title: latest.root }, short(latest.root)),
      ),
      h("dt", {}, "Failures"),
      h("dd", {}, failures.length ? failures.map(([name, why]) => `${label(name)} (${why})`).join("; ") : "none this hour"),
      h("dt", {}, "Manifest"),
      h(
        "dd",
        {},
        h("code", { title: latest.manifest }, short(latest.manifest)),
        `, engine ${manifest.engine.version} at `,
        link(`${REPO}/commit/${manifest.engine.commit}`, manifest.engine.commit.slice(0, 7)),
      ),
      h("dt", {}, "Files"),
      h("dd", { class: "files" }, file("manifest.json"), file("forecasts.json"), file("commitment.json")),
      h("dt", {}, "Ledger"),
      h(
        "dd",
        {},
        "LaplaceLedger v2, ",
        link(`${EXPLORER}/address/${LEDGER}`, h("code", {}, short(LEDGER, 6))),
        " on Robinhood Chain (chain id 4663), source verified on ",
        link(`https://repo.sourcify.dev/4663/${LEDGER}`, "Sourcify"),
      ),
      board
        ? [
            h("dt", {}, "Settled"),
            h(
              "dd",
              {},
              `${count(board.settled_issues, "issue")}, ${board.first_issue} to ${board.last_issue}; the record began at ` +
                `${utc(issueTime(board.first_issue))}`,
            ),
          ]
        : null,
    ),
    h(
      "p",
      { class: "small" },
      `The next issue, ${next}, is made at ${clock(issueTime(next))} and committed by ` +
        `${clock(new Date(issueTime(next).getTime() + COMMIT_WINDOW_S * 1000))}.`,
    ),
  );
}

function minutes(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m} min ${s} s`;
}
