import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  buildTree,
  canonical,
  leafHash,
  mixBasisPoints,
  programId,
  proofOf,
  rootOf,
  verifyIssue,
  verifyProof,
} from "../public/js/record.js";

const read = (path) => readFileSync(new URL(path, import.meta.url));
// The vectors the engine and the contract are tested against (python -m laplace_engine.vectors).
const vectors = JSON.parse(read("../../contracts/test/vectors/record.json"));
// A real issue as published: https://data.laplaceprotocol.com/v1/issues/498103/
const issue = (name) => read(`fixtures/issue-498103/${name}`);

test("leaves, root and proofs match the shared vectors", () => {
  const hashes = vectors.leaves.map((l) => leafHash(vectors.issue, l.forecaster, l.asset, l.horizon, l.bps));
  assert.deepEqual(
    hashes,
    vectors.leaves.map((l) => l.hash),
  );
  const levels = buildTree(hashes);
  assert.equal(rootOf(levels), vectors.root);
  for (const l of vectors.leaves) {
    assert.deepEqual(proofOf(levels, l.hash), l.proof);
    assert.equal(verifyProof(l.proof, vectors.root, l.hash), true);
  }
});

test("a changed basis point no longer proves", () => {
  const l = vectors.leaves[0];
  const bps = [...l.bps];
  bps[4] -= 1;
  bps[5] += 1;
  const changed = leafHash(vectors.issue, l.forecaster, l.asset, l.horizon, bps);
  assert.notEqual(changed, l.hash);
  assert.equal(verifyProof(l.proof, vectors.root, changed), false);
  assert.throws(() => leafHash(vectors.issue, l.forecaster, l.asset, l.horizon, [...l.bps.slice(1), 0]));
});

test("trees refuse duplicates and carry an odd node up", () => {
  const [a, b, c] = vectors.leaves.slice(0, 3).map((l) => l.hash);
  assert.throws(() => buildTree([a, a]));
  assert.throws(() => buildTree([]));
  const levels = buildTree([c, b, a]);
  assert.equal(levels[1].length, 2);
  assert.equal(levels[1][1], [a, b, c].sort()[2]);
});

test("a real issue rebuilds its committed root and its pools", () => {
  const commitment = JSON.parse(issue("commitment.json"));
  const result = verifyIssue(issue("manifest.json"), issue("forecasts.json"));
  assert.equal(result.manifestHash, commitment.manifest);
  assert.equal(result.root, commitment.root);
  assert.equal(result.leaves, commitment.leaves);
  assert.equal(result.pools, 8);
});

test("a tampered forecasts file is refused", () => {
  const forecasts = JSON.parse(issue("forecasts.json"));
  const entry = forecasts.forecasts.find((e) => e.forecaster === "naive" && e.asset === "BTC" && e.horizon === 1);
  entry.bps[4] -= 1;
  entry.bps[5] += 1;
  const tampered = new TextEncoder().encode(JSON.stringify(forecasts));
  assert.throws(() => verifyIssue(issue("manifest.json"), tampered), /rebuilt root/);
});

// Expected values from the engine's quantize.mix_basis_points: ties in the remainders go to the lower bin.
test("pools mix rounded forecasts in integers, as the engine does", () => {
  const a = [1, 9, 90, 900, 4000, 4000, 900, 90, 9, 1];
  const b = [1, 1, 1, 1, 4996, 4996, 1, 1, 1, 1];
  const c = [9991, 1, 1, 1, 1, 1, 1, 1, 1, 1];
  assert.deepEqual(mixBasisPoints([a], [1_000_000]), a);
  assert.deepEqual(mixBasisPoints([a, b], [500_000, 500_000]), [1, 5, 46, 451, 4498, 4498, 450, 45, 5, 1]);
  assert.deepEqual(mixBasisPoints([a, c], [333_333, 666_667]), [6661, 4, 31, 300, 1334, 1334, 300, 31, 4, 1]);
  assert.throws(() => mixBasisPoints([a, b], [500_000, 400_000]));
});

test("canonical JSON sorts keys and leaves no whitespace", () => {
  assert.equal(canonical({ b: [1, { d: "x", c: 2 }], a: "é" }), '{"a":"\\u00e9","b":[1,{"c":2,"d":"x"}]}');
  assert.throws(() => canonical({ a: 0.5 }));
});

test("a program's identity is its source's hash, with its parameter file if it has one", () => {
  const release = JSON.parse(read("../../engine/releases/r1.5/release.json"));
  const program = (name) => release.forecasters.find((f) => f.name === name);
  const file = (path) => read(`../../engine/releases/r1.5/${path}`);
  const naive = program("naive");
  assert.equal(programId(file(naive.path)), naive.id);
  const gbm = program("ev-gbm-r15");
  assert.equal(programId(file(gbm.path), file(gbm.params.path)), gbm.id);
  assert.notEqual(programId(file(gbm.path)), gbm.id);
});
