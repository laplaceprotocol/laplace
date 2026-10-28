import assert from "node:assert/strict";
import { test } from "node:test";

import { nudge } from "../public/js/check.js";
import { beyond, binLabels, binPrices, clock, gain, issueTime, short, signed, utc } from "../public/js/format.js";

test("gains carry their sign, and nothing is shown as zero gain that is not", () => {
  assert.equal(gain(0.01810174448804298), "+0.0181");
  assert.equal(gain(-0.06190849176429205), "−0.0619");
  assert.equal(gain(0.00001), "0.0000");
  assert.equal(gain(null), "—");
  assert.equal(signed(-1.5223), "−1.5223");
});

test("issues are hours since the epoch, in UTC", () => {
  assert.equal(utc(issueTime(498020)), "2026-10-24 20:00 UTC");
  assert.equal(clock(new Date(1793171249 * 1000), { seconds: true }), "07:07:29 UTC");
  assert.equal(short("0xff708f93b0d7f69b177682099026b2efaa3b6f44af68979ac4a097db77d4122a"), "0xff70…122a");
});

test("bins read as the feed defines them", () => {
  const edges = [0.3, 0.7, 1.5, 3.0];
  assert.deepEqual(binLabels(edges), [
    "below −3%",
    "−3 to −1.5%",
    "−1.5 to −0.7%",
    "−0.7 to −0.3%",
    "−0.3 to 0%",
    "0 to +0.3%",
    "+0.3 to +0.7%",
    "+0.7 to +1.5%",
    "+1.5 to +3%",
    "+3% or more",
  ]);
  const prices = binPrices(100, edges);
  assert.equal(prices.length, 10);
  assert.deepEqual(prices[0], [null, 100 * Math.exp(-0.03)]);
  assert.deepEqual(prices[5], [100, 100 * Math.exp(0.003)]);
  assert.deepEqual(prices[9], [100 * Math.exp(0.03), null]);
});

test("the odds of a move beyond an edge add the bins outside it", () => {
  const bps = [1, 3, 86, 784, 4126, 4126, 784, 86, 3, 1];
  assert.equal(beyond(bps, 0), 1748);
  assert.equal(beyond(bps, 1), 180);
  assert.equal(beyond(bps, 3), 2);
});

test("the tamper button keeps a valid forecast", () => {
  const bps = [1, 3, 86, 784, 4126, 4126, 784, 86, 3, 1];
  const changed = nudge(bps);
  assert.equal(changed.reduce((a, b) => a + b, 0), 10_000);
  assert.deepEqual(changed, [1, 3, 86, 784, 4125, 4127, 784, 86, 3, 1]);
  assert.deepEqual(nudge([1, 1, 1, 1, 1, 1, 1, 1, 1, 9991]).slice(-2), [2, 9990]);
});
