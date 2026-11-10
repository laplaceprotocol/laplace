import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { toHex } from "../public/js/abi.js";
import { LEDGER, committedEvent, encodeVerifyForecast, rootOf, rpc } from "../public/js/chain.js";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const vectors = read("../../contracts/test/vectors/record.json");

/** A fetch that answers JSON-RPC calls from a table of method -> result, or fails for some endpoints. */
function fakeFetch(results, { down = [] } = {}) {
  const calls = [];
  const impl = async (url, init) => {
    const body = JSON.parse(init.body);
    calls.push({ url, ...body });
    if (down.includes(url)) throw new TypeError("network error");
    return { json: async () => ({ jsonrpc: "2.0", id: body.id, result: results[body.method] }) };
  };
  return { impl, calls };
}

test("verifyForecast calldata matches eth_abi", () => {
  const leaf = vectors.leaves[0];
  const data = encodeVerifyForecast({ issue: vectors.issue, ...leaf });
  assert.equal(toHex(data), read("fixtures/verify-forecast.json").calldata);
});

test("a call fails over to the next endpoint and says which one answered", async () => {
  const root = "0xff708f93b0d7f69b177682099026b2efaa3b6f44af68979ac4a097db77d4122a";
  const { impl, calls } = fakeFetch({ eth_call: root }, { down: ["https://a.example"] });
  const answer = await rootOf(498103, { endpoints: ["https://a.example", "https://b.example"], fetchImpl: impl });
  assert.deepEqual(answer, { root, endpoint: "https://b.example" });
  assert.equal(calls[1].params[0].to, LEDGER);
  assert.equal(calls[1].params[0].data, "0x9d7384d9" + "0".repeat(59) + "799b7");
});

test("no endpoint answering is an error, not a false", async () => {
  const { impl } = fakeFetch({}, { down: ["https://a.example"] });
  await assert.rejects(rpc("eth_call", [], { endpoints: ["https://a.example"], fetchImpl: impl }), /no endpoint answered/);
});

test("the Committed event is read from the log and the block", async () => {
  // The chain's answer for issue 498103.
  const log = {
    address: "0x15aae63a0ef391559153bcf9c56e5e2be916b2be",
    topics: [
      "0x2e76edc61f3807746b081aaee8287c0dbc32099d44cfbfa79e71f881fbba2b66",
      "0x00000000000000000000000000000000000000000000000000000000000799b7",
    ],
    data:
      "0xff708f93b0d7f69b177682099026b2efaa3b6f44af68979ac4a097db77d4122a" +
      "50b0d868ed22469019523b61fd4508608d0563ec6a9ad88cac0ab601d8ae4e71" +
      "0000000000000000000000000000000000000000000000000000000000000261",
    blockNumber: "0x5f4320e",
    transactionHash: "0x3070c5de89254aa54db944e87f9a1132f13ee5058b0b81efa4799dd34ede756c",
    removed: false,
  };
  const { impl, calls } = fakeFetch({ eth_getLogs: [log], eth_getBlockByNumber: { timestamp: "0x6ae19f31" } });
  const { event } = await committedEvent(498103, 99889678, { endpoints: ["https://a.example"], fetchImpl: impl });
  assert.deepEqual(event, {
    root: "0xff708f93b0d7f69b177682099026b2efaa3b6f44af68979ac4a097db77d4122a",
    manifest: "0x50b0d868ed22469019523b61fd4508608d0563ec6a9ad88cac0ab601d8ae4e71",
    leaves: 609,
    tx: "0x3070c5de89254aa54db944e87f9a1132f13ee5058b0b81efa4799dd34ede756c",
    block: 99889678,
    timestamp: 1793171249,
  });
  assert.deepEqual(calls[0].params[0].topics, log.topics);
  assert.equal(calls[0].params[0].fromBlock, "0x5f4320e");

  const empty = fakeFetch({ eth_getLogs: [] });
  const none = await committedEvent(498103, 99889678, { endpoints: ["https://a.example"], fetchImpl: empty.impl });
  assert.equal(none.event, null);
});
