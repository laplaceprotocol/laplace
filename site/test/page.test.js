import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { RPCS } from "../public/js/chain.js";
import { parseHeaders, pointAtProxy, recordUrl } from "../preview.mjs";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const html = read("../public/index.html");
const headers = parseHeaders(read("../public/_headers"));
const record = /<meta name="laplace:record" content="([^"]+)"/.exec(html)[1];

test("the policy lets the page reach the record and the RPCs, and nothing else", () => {
  const csp = Object.fromEntries(
    headers["Content-Security-Policy"].split(";").map((d) => {
      const [name, ...values] = d.trim().split(/\s+/);
      return [name, values];
    }),
  );
  assert.deepEqual(csp["default-src"], ["'none'"]);
  assert.deepEqual(csp["script-src"], ["'self'"]);
  assert.deepEqual(csp["connect-src"].sort(), ["'self'", new URL(record).origin, ...RPCS].sort());
});

test("the page loads no script, style or font from anywhere else", () => {
  for (const [, url] of html.matchAll(/<(?:script|link)\b[^>]*\b(?:src|href)="([^"]+)"/g)) {
    assert.doesNotMatch(url, /^(https?:)?\/\//, url);
  }
});

test("the corrections on the page are the corrections log, in order", () => {
  const fromDoc = [...read("../../docs/corrections.md").matchAll(/^## (\d{4}-\d{2}-\d{2}) · (.+)$/gm)].map(
    ([, date, title]) => `${date} ${title}`,
  );
  const fromPage = [...html.matchAll(/<h3><time datetime="(\d{4}-\d{2}-\d{2})">\1<\/time> ([^<]+)<\/h3>/g)].map(
    ([, date, title]) => `${date} ${title}`,
  );
  assert.ok(fromDoc.length > 0);
  assert.deepEqual(fromPage, fromDoc);
});

test("the preview points the page at its proxy and proxies only the record's files", () => {
  assert.match(pointAtProxy(html, "/v1/"), /<meta name="laplace:record" content="\/v1\/" \/>/);
  assert.throws(() => pointAtProxy("<html></html>", "/v1/"));
  for (const file of [
    "latest.json",
    "scoreboard.json",
    "issues/498103/manifest.json",
    "issues/498095/settlement-24h.json",
    "releases/r1.5/programs/ev-gbm-r15.params.json",
  ]) {
    assert.equal(recordUrl(`/v1/${file}`), `${record}${file}`);
  }
  for (const path of ["/v1///example.com/x", "/v1/../secret", "/v1/releases/../programs/x", "/v1/issues/1/a/b.json"]) {
    assert.equal(recordUrl(path), null, path);
  }
});
