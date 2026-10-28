// Reading LaplaceLedger on Robinhood Chain straight from public RPC endpoints, so the page never vouches for the
// chain itself: roots, commitment events and verifyForecast answers come from the chain to the browser.

import { bytes32, concat, eventTopic, fromHex, selector, toHex, word } from "./abi.js";

export const CHAIN_ID = 4663;
export const LEDGER = "0x15AaE63A0EF391559153bCF9c56E5E2BE916b2BE";
export const RPCS = ["https://rpc.mainnet.chain.robinhood.com", "https://robinhood.drpc.org"];
export const EXPLORER = "https://robinhoodchain.blockscout.com";
export const COMMIT_WINDOW_S = 15 * 60;

const ROOT_OF = selector("rootOf(uint64)");
const VERIFY_FORECAST = selector("verifyForecast(uint64,bytes32,uint8,uint16,uint16[10],bytes32[])");
const COMMITTED = eventTopic("Committed(uint64,bytes32,bytes32,uint32)");

/** One JSON-RPC call, tried on each endpoint in turn; the answer says which endpoint gave it. */
export async function rpc(method, params, { endpoints = RPCS, fetchImpl = fetch } = {}) {
  let last;
  for (const url of endpoints) {
    try {
      const response = await fetchImpl(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
      });
      const body = await response.json();
      if (body.error) throw new Error(body.error.message ?? "RPC error");
      if (body.result === undefined) throw new Error("no result");
      return { result: body.result, endpoint: url };
    } catch (error) {
      last = error;
    }
  }
  throw new Error(`no endpoint answered ${method}: ${last?.message ?? "unknown error"}`);
}

async function call(data, options) {
  const { result, endpoint } = await rpc("eth_call", [{ to: LEDGER, data: toHex(data) }, "latest"], options);
  return { result, endpoint };
}

/** The root the ledger stores for an issue (zero if none). */
export async function rootOf(issue, options) {
  const { result, endpoint } = await call(concat([ROOT_OF, word(issue)]), options);
  return { root: decodeWord(result, 0), endpoint };
}

/** The ledger's own answer to whether a forecast is in the record committed for its issue. */
export async function verifyForecast({ issue, forecaster, asset, horizon, bps, proof }, options) {
  const { result, endpoint } = await call(encodeVerifyForecast({ issue, forecaster, asset, horizon, bps, proof }), options);
  return { ok: BigInt(decodeWord(result, 0)) === 1n, endpoint };
}

export function encodeVerifyForecast({ issue, forecaster, asset, horizon, bps, proof }) {
  // Head: four static words, the static uint16[10] inline, then the offset of the dynamic proof (15 words in).
  const head = [word(issue), bytes32(forecaster), word(asset), word(horizon), ...bps.map(word), word(15 * 32)];
  return concat([VERIFY_FORECAST, ...head, word(proof.length), ...proof.map(bytes32)]);
}

/**
 * The issue's Committed event in a given block, and that block's time. The block is only a place to look: the
 * event, its issue and its contract come from the chain, and the ledger accepts one commitment per issue, ever.
 */
export async function committedEvent(issue, block, options) {
  const at = `0x${block.toString(16)}`;
  const filter = { address: LEDGER, fromBlock: at, toBlock: at, topics: [COMMITTED, toHex(word(issue))] };
  const { result: logs, endpoint } = await rpc("eth_getLogs", [filter], options);
  const log = logs.find((l) => l.address.toLowerCase() === LEDGER.toLowerCase() && !l.removed);
  if (!log) return { event: null, endpoint };
  const { result: header } = await rpc("eth_getBlockByNumber", [at, false], options);
  return {
    event: {
      root: decodeWord(log.data, 0),
      manifest: decodeWord(log.data, 1),
      leaves: Number(BigInt(decodeWord(log.data, 2))),
      tx: log.transactionHash,
      block: Number(BigInt(log.blockNumber)),
      timestamp: Number(BigInt(header.timestamp)),
    },
    endpoint,
  };
}

function decodeWord(hex, index) {
  const bytes = fromHex(hex);
  if (bytes.length < 32 * (index + 1)) throw new Error("short return data");
  return toHex(bytes.subarray(32 * index, 32 * (index + 1)));
}
