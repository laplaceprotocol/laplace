# watch

Laplace Watch, an off-box dead-man's switch for the hourly record: a Cloudflare Worker with two cron triggers.

- **Every hour at :20 UTC** it checks that the current hour's issue is public
  (`https://data.laplaceprotocol.com/v1/latest.json`). If not, it reads `lastIssue()` from `LaplaceLedger` to tell
  "not committed" from "committed but not published", and messages the operator on Telegram. Because it runs on
  Cloudflare, it alerts even when the machine that runs the engine is down.
- **Mondays at 08:00 UTC** it reports that it is armed, so its own silence would be noticed.

The engine alerts on everything else itself (failed commits, gaps, conflicts, failed forecasters, a low key balance)
and sends a daily digest; see `../../engine/README.md`.

```bash
npm test                                     # node --test, no dependencies
CLOUDFLARE_ACCOUNT_ID=… CLOUDFLARE_API_TOKEN=… npx wrangler deploy
printf %s "$TOKEN" | npx wrangler secret put TG_TOKEN   # and TG_CHAT; never in this repository
```
