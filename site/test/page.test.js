import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { RPCS } from "../public/js/chain.js";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const html = read("../public/index.html");
const record = /<meta name="laplace:record" content="([^"]+)"/.exec(html)[1];
// The headers public/_headers sets (one rule, for every path): indented "Name: value" lines.
const headers = Object.fromEntries(
  [...read("../public/_headers").matchAll(/^\s+([A-Za-z-]+):\s*(.+)$/gm)].map(([, name, value]) => [name, value.trim()]),
);

test("the policy lets the page reach the record and the RPCs, and nothing else", () => {
  const csp = Object.fromEntries(
    headers["Content-Security-Policy"].split(";").map((d) => {
      const [name, ...values] = d.trim().split(/\s+/);
      return [name, values];
    }),
  );
  assert.deepEqual(csp["default-src"], ["'none'"]);
  assert.deepEqual(csp["script-src"], ["'self'"]);
  assert.deepEqual(csp["connect-src"].sort(), [new URL(record).origin, ...RPCS].sort());
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
