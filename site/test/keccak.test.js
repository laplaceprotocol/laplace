import assert from "node:assert/strict";
import { test } from "node:test";

import { eventTopic, fromHex, keccakHex, selector, toHex, word } from "../public/js/abi.js";

const bytes = (s) => new TextEncoder().encode(s);

// Expected values from eth_hash (the engine's keccak), around the 136-byte rate and across several blocks.
test("keccak256 matches the engine's", () => {
  const twice = new Uint8Array(512).map((_, i) => i % 256);
  const cases = [
    [bytes(""), "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"],
    [bytes("abc"), "0x4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"],
    [bytes("a".repeat(135)), "0x34367dc248bbd832f4e3e69dfaac2f92638bd0bbd18f2912ba4ef454919cf446"],
    [bytes("a".repeat(136)), "0xa6c4d403279fe3e0af03729caada8374b5ca54d8065329a3ebcaeb4b60aa386e"],
    [bytes("a".repeat(137)), "0xd869f639c7046b4929fc92a4d988a8b22c55fbadb802c0c66ebcd484f1915f39"],
    [twice, "0xf55ba327291604f0e5be6651752398b7be2331aad65f5763ce067df95cc13be1"],
  ];
  for (const [input, expected] of cases) assert.equal(keccakHex(input), expected);
});

test("selectors and topics of the ledger's interface", () => {
  assert.equal(toHex(selector("rootOf(uint64)")), "0x9d7384d9");
  assert.equal(toHex(selector("lastIssue()")), "0x35028d1e"); // as ops/watch hard-codes it
  assert.equal(
    eventTopic("Committed(uint64,bytes32,bytes32,uint32)"),
    "0x2e76edc61f3807746b081aaee8287c0dbc32099d44cfbfa79e71f881fbba2b66",
  );
});

test("hex and words round-trip", () => {
  assert.equal(toHex(fromHex("0x00ff10")), "0x00ff10");
  assert.throws(() => fromHex("0xabc"));
  assert.throws(() => fromHex("12"));
  assert.equal(toHex(word(498103)), "0x" + "0".repeat(59) + "799b7");
  assert.equal(toHex(word(0)), "0x" + "0".repeat(64));
  assert.throws(() => word(-1));
});
