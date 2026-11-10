// The live page. It reads only the public record and Robinhood Chain: latest.json names the newest issue, whose
// manifest and forecasts files give the odds, and scoreboard.json gives the scores. Each section renders on its own,
// so one file that cannot be read never blanks the others.

import { renderCheck } from "./check.js";
import { fill, h, link } from "./dom.js";
import { renderOdds } from "./odds.js";
import { renderLadder, renderRules, renderTable } from "./scores.js";
import { renderStatus } from "./status.js";

const RECORD = new URL(document.querySelector('meta[name="laplace:record"]').content, document.baseURI);
const REFRESH_MS = 5 * 60_000;
const decoder = new TextDecoder();

async function bytes(path, init) {
  const response = await fetch(new URL(path, RECORD), init);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

const json = async (path, init) => JSON.parse(decoder.decode(await bytes(path, init)));

async function issueFiles(issue) {
  const [manifestBytes, forecastsBytes, commitment] = await Promise.all([
    bytes(`issues/${issue}/manifest.json`),
    bytes(`issues/${issue}/forecasts.json`),
    json(`issues/${issue}/commitment.json`),
  ]);
  return { manifestBytes, forecastsBytes, commitment };
}

const section = (id) => document.getElementById(id);

function problem(el, what, error) {
  fill(
    el,
    h(
      "p",
      { class: "notice" },
      `Could not read ${what} from this browser (${error.message}). The files are public at `,
      link(RECORD.href, RECORD.href),
      ".",
    ),
  );
}

// What is on screen: the issue, its manifest and the scoreboard. A poll that finds no new issue only refreshes the
// status line (a late issue shows up there); a missing scoreboard is read again.
let shown = null;

async function load() {
  let latest;
  try {
    latest = await json("latest.json", { cache: "no-cache" });
  } catch (error) {
    if (shown) return; // keep what is on screen; the next poll tries again
    for (const id of ["status", "odds", "rules", "ladder", "everyone"]) problem(section(id), "the record", error);
    return;
  }
  if (shown?.latest.issue === latest.issue && shown.board) {
    renderStatus(section("status"), { ...shown, latest, record: RECORD });
    return;
  }

  const [files, board] = await Promise.allSettled([issueFiles(latest.issue), json("scoreboard.json")]);
  const scoreboard = board.status === "fulfilled" ? board.value : null;
  const check = {
    latest: latest.issue,
    load: (issue) =>
      issue === latest.issue && files.status === "fulfilled" ? Promise.resolve(files.value) : issueFiles(issue),
    bytes,
  };
  if (files.status === "rejected") {
    for (const id of ["status", "odds", "rules", "ladder", "everyone"]) {
      problem(section(id), `issue ${latest.issue}`, files.reason);
    }
    renderCheck(section("check"), check);
    return; // not marked as shown, so the next poll tries this issue again
  }
  const manifest = JSON.parse(decoder.decode(files.value.manifestBytes));
  const forecasts = JSON.parse(decoder.decode(files.value.forecastsBytes));
  shown = { latest, manifest, board: scoreboard };

  for (const el of document.querySelectorAll("[data-issue]")) el.textContent = String(latest.issue);
  renderStatus(section("status"), { latest, manifest, board: scoreboard, record: RECORD });
  renderOdds(section("odds"), { manifest, forecasts });
  if (scoreboard) {
    renderRules(section("rules"), { board: scoreboard, manifest });
    renderLadder(section("ladder"), { board: scoreboard, manifest });
    renderTable(section("everyone"), { board: scoreboard, manifest });
  } else {
    for (const id of ["rules", "ladder", "everyone"]) problem(section(id), "the scoreboard", board.reason);
  }
  renderCheck(section("check"), check);
}

load();
setInterval(() => {
  if (document.visibilityState === "visible") load();
}, REFRESH_MS);
