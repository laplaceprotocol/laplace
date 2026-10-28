// The few pieces of Solidity's ABI encoding the page needs: hex strings, 32-byte words and function selectors.

import { keccak256 } from "./keccak.js";

const encoder = new TextEncoder();

/** "0x…" lowercase hex of a Uint8Array. */
export function toHex(bytes) {
  let out = "0x";
  for (const b of bytes) out += b.toString(16).padStart(2, "0");
  return out;
}

/** The bytes of a "0x…" hex string. */
export function fromHex(hex) {
  if (!/^0x([0-9a-fA-F]{2})*$/.test(hex)) throw new Error(`not hex: ${hex}`);
  const out = new Uint8Array((hex.length - 2) / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(2 + 2 * i, 4 + 2 * i), 16);
  return out;
}

export function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** An unsigned integer (a safe integer or a BigInt) as one big-endian 32-byte word. */
export function word(value) {
  let v = BigInt(value);
  if (v < 0n || v >= 1n << 256n) throw new Error(`not a uint256: ${value}`);
  const out = new Uint8Array(32);
  for (let i = 31; v > 0n; i--, v >>= 8n) out[i] = Number(v & 0xffn);
  return out;
}

/** A bytes32 value ("0x" and 64 hex digits) as a word. */
export function bytes32(hex) {
  const out = fromHex(hex);
  if (out.length !== 32) throw new Error(`not a bytes32: ${hex}`);
  return out;
}

export function keccakHex(bytes) {
  return toHex(keccak256(bytes));
}

/** The first four bytes of keccak256 of a function signature, e.g. "rootOf(uint64)". */
export function selector(signature) {
  return keccak256(encoder.encode(signature)).subarray(0, 4);
}

/** keccak256 of an event signature: the event's first topic. */
export function eventTopic(signature) {
  return keccakHex(encoder.encode(signature));
}
