// Laplace Watch: an off-box dead-man's switch for the hourly record.
//
// Every hour at :20 UTC it checks that the current hour's issue is public; if not, it asks the ledger whether the
// issue was at least committed, and messages the operator on Telegram. It runs on Cloudflare, so it still alerts when
// the machine that runs the engine is down. Once a week it reports that it is armed, so its own silence is noticed.

const LATEST = "https://data.laplaceprotocol.com/v1/latest.json";
const RPCS = ["https://rpc.mainnet.chain.robinhood.com", "https://robinhood.drpc.org"];
const LEDGER = "0x15AaE63A0EF391559153bCF9c56E5E2BE916b2BE";
const LAST_ISSUE = "0x35028d1e"; // lastIssue()
const HEARTBEAT_CRON = "0 8 * * 1";
const HOUR_MS = 3_600_000;

function hourLabel(issue) {
  return new Date(issue * HOUR_MS).toISOString().slice(0, 16).replace("T", " ");
}

/** Whether the issue of the hour that contains `now` is public. */
export function published(now, latest) {
  return latest !== null && latest >= Math.floor(now / HOUR_MS);
}

/** What is wrong with the record at `now`, or null if the current hour's issue is public. */
export function diagnose({ now, latest, lastIssue }) {
  if (published(now, latest)) return null;
  const current = Math.floor(now / HOUR_MS);
  let state;
  if (lastIssue === null) state = "not public, and the ledger could not be read";
  else if (lastIssue >= current) state = "committed on-chain but not published";
  else state = "not committed";
  return (
    `🚨 Laplace Watch · issue ${current} (${hourLabel(current)} UTC) is ${state}.\n` +
    `Latest public issue: ${latest ?? "unreadable"}; the ledger's last issue: ${lastIssue ?? "unreadable"}.`
  );
}

export function heartbeat({ now, latest }) {
  const current = Math.floor(now / HOUR_MS);
  const behind = latest === null ? "unreadable" : `${current - latest} hour(s) behind the clock`;
  return `🛡️ Laplace Watch is armed. Latest public issue: ${latest ?? "unreadable"} (${behind}).`;
}

async function latestIssue() {
  try {
    const response = await fetch(`${LATEST}?t=${Date.now()}`, { cf: { cacheTtl: 0 } });
    if (!response.ok) return null;
    const body = await response.json();
    return Number.isInteger(body.issue) ? body.issue : null;
  } catch {
    return null;
  }
}

async function ledgerLastIssue() {
  const call = { jsonrpc: "2.0", id: 1, method: "eth_call", params: [{ to: LEDGER, data: LAST_ISSUE }, "latest"] };
  for (const url of RPCS) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(call),
      });
      const body = await response.json();
      if (typeof body.result === "string") return Number(BigInt(body.result));
    } catch {
      // try the next endpoint
    }
  }
  return null;
}

async function telegram(env, text) {
  await fetch(`https://api.telegram.org/bot${env.TG_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ chat_id: env.TG_CHAT, text, disable_web_page_preview: true }),
  });
}

export default {
  async scheduled(controller, env) {
    const now = controller.scheduledTime;
    const latest = await latestIssue();
    if (controller.cron === HEARTBEAT_CRON) {
      await telegram(env, heartbeat({ now, latest }));
      return;
    }
    if (published(now, latest)) {
      console.log(`issue ${latest} is public`);
      return;
    }
    const problem = diagnose({ now, latest, lastIssue: await ledgerLastIssue() });
    console.log(problem);
    await telegram(env, problem);
  },
};
