// The live record's format, exactly as the engine (engine/src/laplace_engine/record.py, quantize.py, verify.py) and
// LaplaceLedger define it, so that a browser can rebuild an issue's Merkle root from its published files.
//
// A leaf is keccak256(keccak256(abi.encode(uint64 issue, bytes32 forecaster, uint8 asset, uint16 horizon,
// uint16[10] bps))). Inner nodes hash the sorted pair of their children (OpenZeppelin's MerkleProof convention), and
// an odd node at the end of a level is carried up unchanged. Hashes are "0x…" lowercase hex strings throughout: for
// strings of equal length their order is the order of the bytes.

import { bytes32, concat, fromHex, keccakHex, toHex, word } from "./abi.js";
import { keccak256 } from "./keccak.js";

export const N_BINS = 10;
export const TOTAL = 10_000; // basis points in a forecast
export const WEIGHT_SCALE = 1_000_000; // pool weights are integers in parts per million

export function leafHash(issue, forecaster, asset, horizon, bps) {
  if (bps.length !== N_BINS || bps.reduce((a, b) => a + b, 0) !== TOTAL || Math.min(...bps) < 1) {
    throw new Error("bps must be ten positive integers summing to 10,000");
  }
  const encoded = concat([word(issue), bytes32(forecaster), word(asset), word(horizon), ...bps.map(word)]);
  return keccakHex(keccak256(encoded));
}

export function hashPair(a, b) {
  return keccakHex(a < b ? concat([fromHex(a), fromHex(b)]) : concat([fromHex(b), fromHex(a)]));
}

/** The tree's levels, from the sorted leaf hashes up to the root. */
export function buildTree(leafHashes) {
  let level = [...leafHashes].sort();
  if (level.length === 0) throw new Error("a tree needs at least one leaf");
  if (new Set(level).size !== level.length) throw new Error("duplicate leaves");
  const levels = [level];
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i + 1 < level.length; i += 2) next.push(hashPair(level[i], level[i + 1]));
    if (level.length % 2 === 1) next.push(level[level.length - 1]);
    levels.push((level = next));
  }
  return levels;
}

export function rootOf(levels) {
  return levels[levels.length - 1][0];
}

/** The sibling hashes from a leaf up to the root, as LaplaceLedger.verifyForecast takes them. */
export function proofOf(levels, leaf) {
  let index = levels[0].indexOf(leaf);
  if (index < 0) throw new Error("not a leaf of this tree");
  const siblings = [];
  for (const level of levels.slice(0, -1)) {
    if ((index ^ 1) < level.length) siblings.push(level[index ^ 1]);
    index = Math.floor(index / 2);
  }
  return siblings;
}

export function verifyProof(proof, root, leaf) {
  return proof.reduce(hashPair, leaf) === root;
}

/** Canonical JSON, as the engine hashes it: sorted keys, no whitespace, ASCII only. */
export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return `{${keys.map((k) => `${ascii(JSON.stringify(k))}:${canonical(value[k])}`).join(",")}}`;
  }
  if (typeof value === "number" && !Number.isInteger(value)) throw new Error("canonical JSON here holds integers only");
  return ascii(JSON.stringify(value));
}

function ascii(json) {
  return json.replace(/[\u007f-￿]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

/** A pool of rounded forecasts with integer weights, rounded the engine's way (quantize.mix_basis_points). */
export function mixBasisPoints(forecasts, weights) {
  if (
    forecasts.length !== weights.length ||
    weights.reduce((a, b) => a + b, 0) !== WEIGHT_SCALE ||
    Math.min(...weights) < 0 ||
    forecasts.some((f) => f.length !== N_BINS || f.reduce((a, b) => a + b, 0) !== TOTAL)
  ) {
    throw new Error("a pool needs basis-point forecasts and non-negative weights summing to WEIGHT_SCALE");
  }
  const numerators = Array.from({ length: N_BINS }, (_, k) => forecasts.reduce((n, f, i) => n + weights[i] * f[k], 0));
  const floors = numerators.map((n) => Math.floor(n / WEIGHT_SCALE));
  const remainders = numerators.map((n) => n % WEIGHT_SCALE);
  // Largest remainders first, ties by bin; then every bin gets at least one basis point, taken from the largest.
  const order = [...floors.keys()].sort((a, b) => remainders[b] - remainders[a] || a - b);
  const missing = TOTAL - floors.reduce((a, b) => a + b, 0);
  for (const k of order.slice(0, missing)) floors[k] += 1;
  while (Math.min(...floors) < 1) {
    floors[floors.indexOf(Math.min(...floors))] += 1;
    floors[floors.indexOf(Math.max(...floors))] -= 1;
  }
  return floors;
}

/** A pool specification's members and weights at one horizon. */
export function membersAt(spec, horizon) {
  if (spec.kind === "linear-pool") return spec.members;
  if (spec.kind === "linear-pool-by-horizon") return spec.horizons[String(horizon)];
  throw new Error(`unknown pool kind ${spec.kind}`);
}

/** A program's identity: keccak256 of its source, or with a parameter file keccak256(abi.encode(both hashes)). */
export function programId(source, params) {
  const own = keccak256(source);
  return params === undefined ? toHex(own) : keccakHex(concat([own, keccak256(params)]));
}

/**
 * Rebuild an issue from its published manifest and forecasts files (their exact bytes), as the reference verifier
 * does: every leaf, the Merkle root, and every pool's forecasts from its members'. Throws on the first mismatch.
 */
export function verifyIssue(manifestBytes, forecastsBytes) {
  const decoder = new TextDecoder();
  const m = JSON.parse(decoder.decode(manifestBytes));
  const f = JSON.parse(decoder.decode(forecastsBytes));
  const manifestHash = keccakHex(manifestBytes);
  check(f.manifest === manifestHash, "the forecasts file names another manifest");
  check(f.issue === m.issue && f.root === m.root, "issue or root differs between the files");

  const ids = new Map(m.forecasters.map((x) => [x.name, x.id]));
  const assets = m.feed.assets;
  const leaves = f.forecasts.map((e) => {
    const asset = assets.indexOf(e.asset);
    check(ids.has(e.forecaster) && asset >= 0, `malformed forecast: ${e.forecaster} ${e.asset}`);
    return leafHash(m.issue, ids.get(e.forecaster), asset, e.horizon, e.bps);
  });
  const levels = buildTree(leaves);
  check(rootOf(levels) === m.root, `rebuilt root ${rootOf(levels)} differs from the declared ${m.root}`);
  check(leaves.length === m.leaves, "leaf count differs from the manifest");

  const byQuestion = new Map(f.forecasts.map((e) => [`${e.forecaster} ${e.asset} ${e.horizon}`, e.bps]));
  const names = new Map(m.forecasters.map((x) => [x.id, x.name]));
  let pools = 0;
  for (const x of m.forecasters) {
    if (x.kind !== "pool" || !f.forecasts.some((e) => e.forecaster === x.name)) continue;
    check(keccakHex(new TextEncoder().encode(canonical(x.spec))) === x.id, `pool ${x.name}: id is not its spec's hash`);
    for (const asset of assets) {
      for (const horizon of m.feed.horizons) {
        const members = membersAt(x.spec, horizon);
        const inputs = members.map((mm) => byQuestion.get(`${names.get(mm.forecaster)} ${asset} ${horizon}`));
        check(inputs.every(Boolean), `pool ${x.name}: a member has no forecast for ${asset} ${horizon}h`);
        const expected = mixBasisPoints(inputs, members.map((mm) => mm.weight));
        const published = byQuestion.get(`${x.name} ${asset} ${horizon}`);
        check(published?.join() === expected.join(), `pool ${x.name} ${asset} ${horizon}h is not its members' mix`);
      }
    }
    pools += 1;
  }
  return { manifest: m, forecasts: f, manifestHash, root: m.root, leaves: leaves.length, levels, pools };
}

function check(ok, message) {
  if (!ok) throw new Error(message);
}
