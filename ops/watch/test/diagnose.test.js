import assert from "node:assert/strict";
import { test } from "node:test";

import { diagnose, heartbeat, published } from "../src/index.js";

const ISSUE = 498_020; // 2026-10-24 20:00 UTC
const at = (minutes) => (ISSUE * 60 + minutes) * 60_000;

test("a published current hour is fine", () => {
  assert.equal(published(at(20), ISSUE), true);
  assert.equal(published(at(20), ISSUE - 1), false);
  assert.equal(published(at(20), null), false);
  assert.equal(diagnose({ now: at(20), latest: ISSUE, lastIssue: ISSUE }), null);
});

test("tells a missing commitment from a missing publication", () => {
  assert.match(diagnose({ now: at(20), latest: ISSUE - 1, lastIssue: ISSUE - 1 }), /issue 498020 \(2026-10-24 20:00 UTC\) is not committed/);
  assert.match(diagnose({ now: at(20), latest: ISSUE - 1, lastIssue: ISSUE }), /committed on-chain but not published/);
  assert.match(diagnose({ now: at(20), latest: null, lastIssue: null }), /the ledger could not be read/);
});

test("the heartbeat says how far behind the record is", () => {
  assert.match(heartbeat({ now: at(5), latest: ISSUE }), /0 hour\(s\) behind/);
  assert.match(heartbeat({ now: at(5), latest: null }), /unreadable/);
});
