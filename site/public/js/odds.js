// This hour's odds: the ten-bin forecasts of the forecasters the rules judge, for every asset and horizon, drawn from
// the issue's forecasts file exactly as committed.

import { JUDGED, RULES, label } from "./content.js";
import { fill, h } from "./dom.js";
import { beyond, binLabels, binPrices, clock, issueTime, percent, price } from "./format.js";
import { N_BINS } from "./record.js";

const CLOSE = 3; // position of the close in a manifest's last candle (open, high, low, close, volume)
const W = 320;
const PLOT = { left: 4, right: 316, top: 6, bottom: 108 };

export function renderOdds(root, { manifest, forecasts }) {
  const bps = new Map(forecasts.forecasts.map((e) => [`${e.forecaster} ${e.asset} ${e.horizon}`, e.bps]));
  const failed = new Set(Object.keys(manifest.failed ?? {}));
  const judged = JUDGED.map((j) => ({ ...j, forecaster: manifest.forecasters.find((f) => f.id === j.id) }))
    .filter((j) => j.forecaster)
    .map((j, i) => ({ ...j, name: j.forecaster.name, series: i }));
  const horizons = manifest.feed.horizons;
  const grid = h("div", { class: "odds-grid" });
  const tabs = horizons.map((horizon) =>
    h(
      "button",
      { type: "button", role: "tab", class: "tab", "aria-selected": "false", onclick: () => show(horizon) },
      horizon === 1 ? "1 hour" : `${horizon} hours`,
    ),
  );

  function show(horizon) {
    tabs.forEach((t, i) => t.setAttribute("aria-selected", String(horizons[i] === horizon)));
    const scale = Math.max(
      ...manifest.feed.assets.flatMap((a) => judged.flatMap((j) => bps.get(`${j.name} ${a} ${horizon}`) ?? [0])),
    );
    fill(
      grid,
      manifest.feed.assets.map((asset) => card({ manifest, bps, judged, asset, horizon, scale })),
    );
  }

  fill(
    root,
    h(
      "div",
      { class: "legend" },
      judged.map((j) =>
        h(
          "span",
          { class: `key s${j.series}` },
          h("i", { "aria-hidden": "true" }),
          `${label(j.name)} `,
          h("small", {}, `judged by ${RULES[j.rule].name.toLowerCase()}`),
          failed.has(j.name) ? h("small", { class: "warn" }, " · no forecast this hour (listed as failed)") : null,
        ),
      ),
    ),
    h("div", { class: "tabs", role: "tablist", "aria-label": "Horizon" }, tabs),
    grid,
  );
  show(horizons[0]);
}

function card({ manifest, bps, judged, asset, horizon, scale }) {
  const edges = manifest.feed.edges_pct[String(horizon)];
  const start = manifest.inputs.last_candle[asset][CLOSE];
  const issue = manifest.issue;
  const ends = issueTime(issue + horizon);
  const series = judged.map((j) => ({ ...j, bps: bps.get(`${j.name} ${asset} ${horizon}`) })).filter((s) => s.bps);
  const labels = binLabels(edges);
  const prices = binPrices(start, edges);
  const big = edges[1];
  return h(
    "article",
    { class: "odds-card" },
    h(
      "header",
      {},
      h("h3", {}, asset),
      h("span", { class: "muted" }, `${price(start)} at ${clock(issueTime(issue))} → close at ${when(ends, issue)}`),
    ),
    chart(series, labels, edges, scale, `${asset}, ${horizon} hour${horizon === 1 ? "" : "s"}`),
    h(
      "p",
      { class: "tail" },
      `Beyond ±${big}%: `,
      series.map((s, i) => [
        i ? " · " : "",
        h("span", { class: `s${s.series}` }, `${label(s.name)} ${percent(beyond(s.bps, 1))}`),
      ]),
    ),
    h(
      "details",
      {},
      h("summary", {}, "Exact numbers"),
      h(
        "div",
        { class: "scroll" },
        h(
          "table",
          { class: "bins" },
          h("thead", {}, h("tr", {}, h("th", {}, "log return"), h("th", {}, "close"), series.map((s) => h("th", {}, label(s.name))))),
          h(
            "tbody",
            {},
            labels.map((text, k) =>
              h(
                "tr",
                {},
                h("td", {}, text),
                h("td", { class: "num" }, range(prices[k])),
                series.map((s) => h("td", { class: "num" }, `${s.bps[k]} bp`)),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

function chart(series, labels, edges, scale, name) {
  const slot = (PLOT.right - PLOT.left) / N_BINS;
  const gap = 6; // between bins; a bin's bars touch
  const bar = (slot - gap) / Math.max(1, series.length);
  const y = (v) => PLOT.bottom - ((PLOT.bottom - PLOT.top) * v) / scale;
  const ticks = [...edges.map((e) => -e).reverse(), 0, ...edges];
  return h(
    "svg",
    { viewBox: `0 0 ${W} 124`, class: "chart", role: "img", "aria-label": `${name}: ten-bin forecasts` },
    labels.map((text, k) =>
      h(
        "g",
        {},
        series.map((s, i) => {
          const x = PLOT.left + k * slot + gap / 2 + i * bar;
          const top = y(s.bps[k]);
          return h(
            "rect",
            { x, y: top, width: bar, height: PLOT.bottom - top, class: `bar s${s.series}` },
            h("title", {}, `${label(s.name)}: ${percent(s.bps[k])} on ${text}`),
          );
        }),
      ),
    ),
    h("line", { x1: PLOT.left, x2: PLOT.right, y1: PLOT.bottom, y2: PLOT.bottom, class: "axis" }),
    ticks.map((e, k) =>
      h(
        "text",
        { x: PLOT.left + (k + 1) * slot, y: 120, class: "tick" },
        e === 0 ? "0" : `${e > 0 ? "+" : "−"}${Math.abs(e)}`,
      ),
    ),
  );
}

function range([lo, hi]) {
  if (lo === null) return `below ${price(hi)}`;
  if (hi === null) return `${price(lo)} or more`;
  return `${price(lo)} to ${price(hi)}`;
}

function when(date, issue) {
  const sameDay = date.toISOString().slice(0, 10) === issueTime(issue).toISOString().slice(0, 10);
  return sameDay ? clock(date) : `${clock(date)} on ${date.toISOString().slice(5, 10)}`;
}
