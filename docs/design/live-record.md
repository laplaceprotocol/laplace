# The live record

Every hour the engine *seals an issue*: each forecaster in the current release forecasts every question, the
forecasts are rounded to basis points and hashed into a Merkle tree, and the tree's root is committed to
`LaplaceLedger` on Robinhood Chain within the first 15 minutes of the hour, before the shortest question can
settle. The files that describe the issue are then published, so anyone can rebuild the root, check it against the
chain, and score the forecasts once they settle. Code: `engine/` ([README](../../engine/README.md)).

## An hour
1. **:07 UTC.** The engine fetches the last 744 hourly candles of every asset and waits until the candle that
   closed at the top of the hour is published for all of them and the exchange has opened the new hour (at most until
   :10). If the exchange's candle feed lags past :05, its own clock vouches that the hour has closed, and the candles
   available then are used: the committed last candle is the definitive starting price. The run starts at :07, not
   at the top of the hour, because Coinbase re-aggregates a just-closed hourly candle a little over five minutes later
   (the close and volume of the less liquid assets move by up to 0.2%); sealing after that commits the exchange's
   final candle. Only candles that closed by the top of the hour are ever read.
2. **Forecasts.** Every program of the release runs once per question in a fresh process that only ever reads the
   720 candles before the issue time (the same sandbox and past-only reader as in research). A program that fails
   is listed in the manifest and has no forecasts that hour; a pool needs all its members.
3. **Sealing.** Forecasts are rounded to basis points, pools are mixed from their members' rounded forecasts, and
   the leaves are hashed into a tree. The manifest and the forecasts file are written to disk before anything is sent.
4. **Commitment.** One transaction: `commit(issue, root, keccak256(manifest), leaves)`. The ledger accepts it only
   from the committer, only for an issue after the last one, and only inside the issue's 15-minute window, so the
   block timestamp proves when the forecasts were sealed. The signed transaction's hash is journaled before it is
   sent, transient failures are retried until the window is about to close, and every commitment is read back from
   the chain's own event. A missed hour stays visible as a gap.
5. **Publication.** The manifest, the forecasts and the commitment's receipt go to
   `https://data.laplaceprotocol.com/v1/issues/<issue>/` (`manifest.json`, `forecasts.json`, `commitment.json`).
6. **Settlement.** Every question that has become due is scored (`settlement-<h>h.json` next to its issue), and
   `v1/scoreboard.json` is rebuilt from all settlements.

Every run also reconciles the past: an issue whose commitment reached the chain without being recorded locally (a lost
reply, a crash) is recovered from the chain and published; one that never reached it is marked as a gap. Each step is
isolated, so a failure in one (an exchange outage during settlement, say) never stops the others, and anything left
undone is retried on the next run, however old. Files are only published if they rebuild exactly the committed root
and manifest hash. Inputs from outside degrade instead of failing the hour. A sense whose source fails is unknown (a
source that is unreachable or keeps throttling is not asked again that hour), and if the senses together take longer
than two minutes, the hour is sealed with every sense unknown. A calendar that cannot be read means programs see no
events, which the manifest says.

The operator is alerted on Telegram whenever a run needs attention, and receives a daily digest; an off-box
dead-man's switch on Cloudflare (`ops/watch`) checks the public record every hour, so even a dead engine host is
noticed within the hour.

An issue is numbered by its hour: hours since the Unix epoch (issue 498,019 is 2026-10-24 19:00 UTC).

## Questions
For each of seven assets (BTC, ETH, SOL, XRP, DOGE, AVAX, LINK; an append-only registry, so an asset keeps its index)
and each horizon h of 1, 4 and 24 hours: the probabilities of ten bins for the log return from the close of the last
input candle (the hour before the issue) to the close of the hourly candle that ends h hours after the issue time.
The bins are symmetric around zero with edges at ±0.3, 0.7, 1.5 and 3% (1 hour), ±0.6, 1.4, 3 and 6% (4 hours) and
±1.5, 3.5, 7 and 14% (24 hours) of log return; the outer bins are open. Prices are Coinbase Exchange hourly candles
for `<asset>-USD`. The manifest carries the last input candle of every asset, so the starting price of every question
is part of the commitment.

Forecasts are scored with the binned log score on the rounded forecast (`bps / 10,000`). Every bin holds at least one
basis point, so no single outcome can cost more than log(10,000) ≈ 9.2 nats.

## Settlement and scores
A question of issue N at horizon h settles on the close of the candle that ends at the start of issue N + h, and that
candle is issue N + h's last input candle: its close is in N + h's manifest, committed on-chain. So scores normally
follow from committed files alone, and anyone can recompute them. If issue N + h was never committed, the question
settles on the exchange's public candle instead, so a missing hour can never hide an outcome; the settlement file
names its source either way. The committed prices themselves can be checked against the exchange's public candles.

Release r1 is judged by [a pre-registered rule](../judging/r1.md): the Demon against GARCH and HAR on the 1- and
4-hour questions, with looks at 90, 180 and 365 days; the scoreboard shows its statistic as "running, unresolved"
between looks.

The scoreboard lists each forecaster's mean log score (overall and by horizon) and its paired gain over the naive,
GARCH and HAR baselines on the questions both answered, issues weighted equally, with a 95% circular block-bootstrap
interval over issues (72-hour blocks). Until two blocks of issues have settled it shows no interval at all: a few days
of a small edge are a direction, not a verdict.

## Forecasters and releases
A release ([`engine/releases/r1/release.json`](../../engine/releases/r1/release.json)) fixes the forecasters:
- **Programs** are identified by the keccak256 of their source. Baselines are public and live in the release
  directory; evolved champions are private until revealed, and their identity is published from the first issue, so a
  revealed program can be matched to its whole record. Before every run the engine checks each source against the
  release.
- **Pools** are linear pools of the release's programs with integer weights in parts per million, either one set for
  every horizon (`linear-pool`) or one set per horizon (`linear-pool-by-horizon`). A pool's identity is the keccak256
  of its canonical specification (members and weights), and its forecast is mixed from the members' *rounded*
  forecasts in integer arithmetic, so anyone can recompute it exactly from the published record. No two forecasters
  of a release may share an identity.

Release r1 is frozen from research run `evolve-poc2` exactly as its report judged it: the eight island champions,
the Demon (the pool of champions, weights fitted on validation: seasonality 44.8%, ranges 24.5%, regimes 11.5%,
clustering 10.8%, spillover 4.5%, tails 3.8%), and the baseline ladder (naive, EWMA, GARCH, HAR) with parameters
fitted on the search window. See [the PoC 2 note](../research/2026-10-24-poc2-evolution.md) for what that
configuration did on the holdout.

Release r1.1 (live from 2026-10-25) keeps every r1 forecaster, identity and record unchanged, and adds shadow
forecasters, exploratory until a rule of their own judges them: **the monkey** (equal odds on every bin, the floor
every forecaster must clear) and **demon-h**, a pool of all thirteen programs, baselines and monkey included, with its
own EM weights at each horizon (fitted on PoC 2's validation window; GARCH gets 5.6% at 24 hours). A single-weight
pool of the same programs fitted to exactly the Demon's weights (EM gives the baselines and the monkey nothing), so it
was left out: two forecasters with one identity would collide. The fit's holdout screening is kept with the release
([`screening.json`](../../engine/releases/r1.1/screening.json)).

Release r1.2 (live from 2026-10-25) adds data beyond candles. Every issue now carries senses (Deribit implied
volatility for BTC and ETH, 5-minute realized variance, perpetual funding), and two classical baselines use them:
**RV-HAR** (HAR on 5-minute realized variance) and **IV-HAR** (plus the variance implied by DVOL; the other assets
use BTC's index). Two more pools of every program include them: **demon-plus-s** (one set of weights) and
**demon-hs** (weights per horizon; IV-HAR gets 17–19% at every horizon). r1.1's forecasters, identities and records
are unchanged. On PoC 2's holdout IV-HAR nearly doubles HAR's gain at 1 hour but is weaker at 24 hours, RV-HAR adds
nothing, and demon-hs is the best pool so far on the 1- and 4-hour questions (+0.0238 nats per question over GARCH,
against the Demon's +0.0210): development evidence, kept in
[`screening.json`](../../engine/releases/r1.2/screening.json).

Programs also see when markets are scheduled to move (`ctx.events`, since 2026-10-25): FOMC decisions, CPI and
jobs-report releases, and the open and close of the NYSE regular session, from 24 hours before to 48 hours after the
forecast time. The dates are copied from the publishers' calendars into
[`events.toml`](../../research/src/laplace_research/data/events.toml), which names each source, and converted from
US Eastern Time with daylight saving. Every issue lists the events its programs saw and the calendar's hash, and the
engine alerts 60 days before the calendar runs out. Nothing about what an event announces is ever an input.

Release r1.3 (live from 2026-10-25) adds the first forecaster that uses them. **EV-HAR** scales IV-HAR's standard
deviation for the hours known to move: weekends, the UTC hour, the US session and its open (placed by the calendar,
so daylight saving is exact), and FOMC, CPI and jobs releases inside the window or just before or after it. Its
coefficients are fitted by binned log score on PoC 2's search window. Two more pools of every program include it:
**demon-plus-e** and **demon-he** (EV-HAR gets 27% of demon-he at 1 hour, 49% at 4 hours and 47% at 24 hours).
r1.2's forecasters, identities and records are unchanged. On PoC 2's holdout, EV-HAR alone beats GARCH by +0.0185
nats per 1- and 4-hour question, almost nine-tenths of the Demon's +0.0210, and demon-he is the best pool so far
(+0.0276): development evidence, in [`screening.json`](../../engine/releases/r1.3/screening.json) and
[the research note](../research/2026-10-25-scheduled-events.md).

The baselines run as ordinary programs on the same 720-hour window, so their recursions start from the window
rather than from the start of the research panel, and HAR's 720-hour mean covers the window's 719 returns. Rerun over
the PoC 2 holdout (11,487 questions), the naive, EWMA and GARCH programs match the research baselines to within
0.0002 basis points; HAR differs by 0.13 basis points on average (at most 35, when an outlier hour sits at the edge
of the window), which changes its mean log score by 0.00002 nats per question.

## Formats
**Leaf:** `keccak256(keccak256(abi.encode(uint64 issue, bytes32 forecaster, uint8 asset, uint16 horizon,
uint16[10] bps)))`. Inner nodes hash the sorted pair of their children (OpenZeppelin's `MerkleProof` convention); an
odd node at the end of a level is carried up unchanged. `LaplaceLedger.leafHash` computes the same leaf, and
`verifyForecast(issue, forecaster, asset, horizon, bps, proof)` checks a forecast against the committed root, so a
contract can use a forecast without trusting anyone. Shared test vectors keep the Python and Solidity
implementations identical.

**Manifest** (canonical JSON: sorted keys, no whitespace, ASCII; its keccak256 is committed): the issue and its time,
the ledger, the engine version and git commit, the release, the feed definition, the inputs (the hash of the 720
hours of candles and senses the programs saw, as little-endian float64 ordered hour, asset, field; the last candle and
senses of every asset in full; and the scheduled events around the issue time, with the hash of the calendar they
come from), the forecasters with their identities, any failures, the pre-registered judging rules in force (each one's
path and keccak256, so the chain dates the rule before the results it judges), the leaf encoding, the number of leaves
and the root.

**Forecasts:** every leaf as `{forecaster, asset, horizon, bps}`, plus the issue, the root and the manifest's hash.

## Verifying an issue
1. `keccak256(manifest.json)` equals the manifest hash in the issue's `Committed` event, the root equals
   `rootOf(issue)`, and the commitment's block falls inside the issue's window.
2. The leaves rebuilt from `forecasts.json` give that root, and every pool's forecasts are its members' forecasts
   mixed with its weights. `laplace-engine verify --issue N` does steps 1 and 2 from the public files and the chain.
3. Public programs rerun on the inputs the manifest identifies give the published forecasts (identical with the
   locked dependencies; floating-point differences between library versions can move a rounding by a basis point).
4. Once a private program is revealed, its source hashes to its identity, and rerunning it reproduces its record.

## What it does not claim
The record proves *when* each forecast was made and that it was not changed afterwards. It does not prove that
nothing else was forecast and discarded: the release fixes the forecasters in advance and every hour lists all of
them, failures included, which is what makes selective reporting visible. The committer key is the ledger's only
trust assumption; if it were compromised, a new ledger would be deployed and announced, and the old record would
stand as it is.

## Deployments
| contract | chain | address | status |
|---|---|---|---|
| LaplaceLedger v2 (stores roots, `verifyForecast`) | Robinhood Chain (4663) | `0x15AaE63A0EF391559153bCF9c56E5E2BE916b2BE` | current |
| LaplaceLedger v1 (events only) | Robinhood Chain (4663) | `0x0E461fFCb4D0169eB5D87eceb309066ca1f95b26` | superseded before its first commit |

Both are verified on Sourcify; the committer is `0x227c3530c600bdFdCB33bfbe5C8c86471f8dd7e2`.
