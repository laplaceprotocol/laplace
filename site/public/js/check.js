// Check an issue in the browser: rebuild it from its published files, compare it with what Robinhood Chain says, then
// prove one forecast against the committed root, here and through the ledger's own verifyForecast. Nothing runs until
// the visitor asks, so a page view costs the public RPC endpoints nothing.

import { keccakHex } from "./abi.js";
import { COMMIT_WINDOW_S, EXPLORER, committedEvent, rootOf, verifyForecast } from "./chain.js";
import { label } from "./content.js";
import { fill, h, link } from "./dom.js";
import { clock, issueTime, short, thousands, utc } from "./format.js";
import { leafHash, proofOf, programId, verifyIssue, verifyProof } from "./record.js";

/** `load(issue)` resolves to { manifestBytes, forecastsBytes, commitment }; `bytes(path)` reads any public file. */
export function renderCheck(root, { latest, load, bytes }) {
  const input = h("input", { type: "number", name: "issue", value: latest, min: 1, step: 1, inputmode: "numeric" });
  const steps = h("ol", { class: "steps", "aria-live": "polite" });
  const forecast = h("div", { class: "forecast-check", hidden: true });
  const programs = h("div", { class: "programs-check", hidden: true });
  const form = h("form", { class: "check-form" }, h("label", {}, "Issue ", input), h("button", { type: "submit" }, "Check it here"));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    run(Number(input.value));
  });
  fill(root, form, steps, forecast, programs);

  async function run(issue) {
    fill(steps);
    forecast.hidden = true;
    programs.hidden = true;

    let files;
    let s = add(steps, `Downloading issue ${issue} from the public record…`);
    try {
      files = await load(issue);
      done(
        s,
        `Downloaded manifest.json (${thousands(files.manifestBytes.length)} bytes) and forecasts.json ` +
          `(${thousands(files.forecastsBytes.length)} bytes).`,
      );
    } catch (error) {
      return failed(s, `Could not download issue ${issue}: ${error.message}`);
    }

    let issueCheck;
    s = add(steps, "Rebuilding the Merkle tree…");
    try {
      issueCheck = verifyIssue(files.manifestBytes, files.forecastsBytes);
      done(
        s,
        `${thousands(issueCheck.leaves)} forecasts rebuild the root ${short(issueCheck.root)}, as the manifest ` +
          `declares, and ${issueCheck.pools} pools are exactly their members' forecasts mixed with the published weights.`,
        `keccak256(manifest.json) = ${issueCheck.manifestHash}`,
      );
    } catch (error) {
      return failed(s, `The files do not rebuild: ${error.message}`);
    }

    s = add(steps, "Asking the ledger for the root…");
    try {
      const { root: onChain, endpoint } = await rootOf(issue);
      if (onChain === issueCheck.root) done(s, `LaplaceLedger.rootOf(${issue}) is the same root.`, answeredBy(endpoint));
      else failed(s, `LaplaceLedger.rootOf(${issue}) is ${onChain}, not the rebuilt root.`, answeredBy(endpoint));
    } catch (error) {
      failed(s, `Could not read the ledger: ${error.message}`);
    }

    s = add(steps, "Reading the commitment event…");
    try {
      const { event, endpoint } = await committedEvent(issue, files.commitment.block);
      if (!event) {
        failed(s, `No Committed event for issue ${issue} in block ${thousands(files.commitment.block)}, the block its ` + "commitment.json names.", answeredBy(endpoint));
      } else if (
        event.root !== issueCheck.root ||
        event.manifest !== issueCheck.manifestHash ||
        event.leaves !== issueCheck.leaves
      ) {
        failed(s, "The Committed event carries another root, manifest hash or leaf count.", answeredBy(endpoint));
      } else {
        done(
          s,
          [
            `The Committed event carries the same root, manifest hash and leaf count: block ${thousands(event.block)}, `,
            link(`${EXPLORER}/tx/${event.tx}`, `transaction ${short(event.tx)}`),
            ".",
          ],
          answeredBy(endpoint),
        );
        const opens = issueTime(issue).getTime() / 1000;
        const inside = event.timestamp >= opens && event.timestamp < opens + COMMIT_WINDOW_S;
        (inside ? done : failed)(
          add(steps, ""),
          `Committed at ${utc(new Date(event.timestamp * 1000), { seconds: true })}, ${inside ? "inside" : "outside"} ` +
            `the issue's window (${clock(issueTime(issue))} to ${clock(new Date((opens + COMMIT_WINDOW_S) * 1000))}), ` +
            `before the first of its questions settled at ${clock(issueTime(issue + 1))}.`,
        );
      }
    } catch (error) {
      failed(s, `Could not read the event: ${error.message}`);
    }

    renderForecast(forecast, issueCheck);
    renderPrograms(programs, issueCheck.manifest, bytes);
  }
}

function renderForecast(root, { manifest, forecasts, levels }) {
  const names = [...new Set(forecasts.forecasts.map((e) => e.forecaster))];
  const pick = (name, options, value) =>
    h(
      "select",
      { name, "aria-label": name },
      options.map(([v, text]) => h("option", { value: v, selected: String(v) === String(value) }, text)),
    );
  const who = pick(
    "forecaster",
    names.map((n) => [n, label(n)]),
    names.includes("demon") ? "demon" : names[0],
  );
  const asset = pick(
    "asset",
    manifest.feed.assets.map((a) => [a, a]),
    manifest.feed.assets[0],
  );
  const horizon = pick(
    "horizon",
    manifest.feed.horizons.map((x) => [x, `${x} h`]),
    manifest.feed.horizons[0],
  );
  const body = h("div", {});
  const steps = h("ol", { class: "steps", "aria-live": "polite" });
  const ask = h("button", { type: "button" }, "Check this forecast");
  const toggle = h("button", { type: "button", class: "secondary" }, "Change one basis point");
  let tampered = false;

  function current() {
    const entry = forecasts.forecasts.find(
      (e) => e.forecaster === who.value && e.asset === asset.value && e.horizon === Number(horizon.value),
    );
    const id = manifest.forecasters.find((f) => f.name === who.value).id;
    const index = manifest.feed.assets.indexOf(asset.value);
    const bps = tampered ? nudge(entry.bps) : entry.bps;
    const committed = leafHash(manifest.issue, id, index, entry.horizon, entry.bps);
    return {
      id,
      index,
      horizon: entry.horizon,
      committed: entry.bps,
      bps,
      leaf: leafHash(manifest.issue, id, index, entry.horizon, bps),
      proof: proofOf(levels, committed),
    };
  }

  function show() {
    const c = current();
    toggle.textContent = tampered ? "Use the committed forecast" : "Change one basis point";
    fill(steps);
    fill(
      body,
      h(
        "p",
        { class: "bps", "aria-label": "basis points of the ten bins" },
        c.bps.map((b, k) => h("span", { class: b === c.committed[k] ? null : "changed" }, String(b))),
      ),
      h(
        "dl",
        { class: "facts" },
        h("dt", {}, "leaf"),
        h("dd", {}, h("code", {}, c.leaf)),
        h("dt", {}, "proof"),
        h(
          "dd",
          {},
          h(
            "details",
            {},
            h("summary", {}, `${c.proof.length} sibling hashes`),
            h(
              "ol",
              { class: "proof" },
              c.proof.map((p) => h("li", {}, h("code", {}, p))),
            ),
          ),
        ),
      ),
    );
  }

  ask.addEventListener("click", async () => {
    const c = current();
    fill(steps);
    const here = verifyProof(c.proof, manifest.root, c.leaf);
    (here ? done : failed)(
      add(steps, ""),
      here ? "Checked here: the proof leads to the committed root." : "Checked here: the proof does not lead to the root.",
    );
    const s = add(steps, "Asking LaplaceLedger.verifyForecast…");
    try {
      const { ok, endpoint } = await verifyForecast({
        issue: manifest.issue,
        forecaster: c.id,
        asset: c.index,
        horizon: c.horizon,
        bps: c.bps,
        proof: c.proof,
      });
      (ok ? done : failed)(s, `LaplaceLedger.verifyForecast answers ${ok}.`, answeredBy(endpoint));
    } catch (error) {
      failed(s, `Could not ask the ledger: ${error.message}`);
    }
  });
  toggle.addEventListener("click", () => {
    tampered = !tampered;
    show();
  });
  for (const select of [who, asset, horizon]) select.addEventListener("change", show);

  fill(
    root,
    h("h4", {}, "One forecast, against the ledger"),
    h(
      "p",
      { class: "small" },
      "Pick a forecast. Its leaf and proof are computed here, and the ledger checks them against the root it stores. " +
        "Change one basis point and both checks fail.",
    ),
    h("div", { class: "pick" }, who, asset, horizon),
    body,
    h("div", { class: "actions" }, ask, toggle),
    steps,
  );
  root.hidden = false;
  show();
}

/** The same forecast with one basis point moved from its largest bin to a neighbour: still a valid forecast. */
export function nudge(bps) {
  const out = [...bps];
  const top = out.indexOf(Math.max(...out));
  out[top] -= 1;
  out[top + 1 < out.length ? top + 1 : top - 1] += 1;
  return out;
}

function renderPrograms(root, manifest, bytes) {
  const programs = manifest.forecasters.filter((f) => f.kind === "program" && f.public);
  const steps = h("ol", { class: "steps", "aria-live": "polite" });
  const go = h("button", { type: "button", class: "secondary" }, `Check the ${programs.length} public programs`);
  go.addEventListener("click", async () => {
    go.disabled = true;
    fill(steps);
    for (const f of programs) {
      const s = add(steps, `${label(f.name)}…`);
      try {
        const base = `releases/${manifest.release}/`;
        const source = await bytes(base + f.path);
        const params = f.params ? await bytes(base + f.params.path) : undefined;
        if (params && keccakHex(params) !== f.params.keccak256) throw new Error("its parameter file has another hash");
        const id = programId(source, params);
        const what = params ? `${f.path} and its parameter file (${thousands(params.length)} bytes)` : f.path;
        if (id === f.id) done(s, `${label(f.name)}: ${what} hash to its identity, ${short(id)}.`);
        else failed(s, `${label(f.name)}: ${what} hash to ${short(id)}, not to its identity ${short(f.id)}.`);
      } catch (error) {
        failed(s, `${label(f.name)}: ${error.message}`);
      }
    }
    go.disabled = false;
  });
  fill(
    root,
    h("h4", {}, "The public programs"),
    h(
      "p",
      { class: "small" },
      "A program's identity is the hash of its source, and of its parameter file if it has one, so every leaf commits " +
        "to the exact code that made it. This downloads the release's public programs: about 1 MB, most of it " +
        "EV-GBM's trees.",
    ),
    h("div", { class: "actions" }, go),
    steps,
  );
  root.hidden = false;
}

function add(list, text) {
  const item = h("li", { class: "pending" }, h("span", { class: "text" }, text));
  list.append(item);
  return item;
}

function done(item, main, detail) {
  settle(item, "ok", main, detail);
}

function failed(item, main, detail) {
  settle(item, "fail", main, detail);
}

function settle(item, state, main, detail) {
  item.className = state;
  fill(
    item,
    h("span", { class: "mark", role: "img", "aria-label": state === "ok" ? "passed" : "failed" }),
    h("span", { class: "text" }, main, detail ? h("small", {}, detail) : null),
  );
}

const answeredBy = (endpoint) => `answered by ${new URL(endpoint).host}`;
