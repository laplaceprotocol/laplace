# Security

Laplace's claim is that every forecast was committed before its outcome and never changed afterwards. Anything that
breaks that claim, or lets someone else's forecasts or prices pass as ours, is a vulnerability.

## Reporting
Report privately through GitHub: **Security → Report a vulnerability** on this repository. Please don't open a public
issue. We acknowledge reports within 72 hours, keep you informed, and credit you when the fix is published unless you
prefer otherwise. There is no paid bounty yet.

## In scope
- **`contracts/`:** `LaplaceLedger`. Ways to commit as someone other than the committer, outside an issue's
  15-minute window, out of order, or to make `verifyForecast` accept a forecast that is not under a committed root.
- **`engine/`:** sealing, commitments, publication, settlement and verification. Ways to publish files that don't
  rebuild their committed root, to commit a root other than the sealed one, to settle a question on prices other
  than the rule's, or to make the verifier pass a forged issue.
- **`research/` sandbox:** ways for a forecasting program to see a candle after its forecast time, reach the file
  system, network or host, or escape its restricted namespace.
- **Published data** at `https://data.laplaceprotocol.com/v1/` and the off-box monitor in `ops/watch`.

## Out of scope
- Forecast quality: that is what the public record measures.
- Third-party services the record depends on (Coinbase market data, Robinhood Chain and its RPC endpoints,
  Cloudflare, GitHub).
- Findings that need a compromised operator machine or key. These are real risks, and they are described below.

## How the system limits damage
- **The ledger has no owner and no admin functions.** Its only privileged key, the committer, can only add one
  commitment per hour, inside that hour's window, after the last one. A leaked committer key could spam or block
  future commitments, but it could not change any past one. The response would be a new ledger and a public
  announcement; the old record would stand as it is.
- **The commit key holds gas money only.** Deployer and treasury keys are never kept on servers.
- **Every published issue is independently checkable** from its files and the chain alone
  ([how](docs/design/live-record.md#verifying-an-issue)).
- **CodeQL, Dependabot and secret scanning with push protection** run on this repository. Every change lands through
  a pull request with CI.
