# Laplace

> *"We may regard the present state of the universe as the effect of its past and the cause of its future."*
> (Pierre-Simon Laplace, 1814)

Laplace evolves forecasting programs. Language models write and mutate small Python programs that put probabilities
on what markets do next; reality judges them with a proper scoring rule, and the survivors are pooled into a single
forecaster, the Demon. **Every forecast is committed on-chain before its outcome is known**, next to classical
baselines that answer the same questions.

Laplace's demon is impossible: chaos, quantum uncertainty and the limits of computation all say so. We don't claim the
demon. We measure how close evolution can get.

## Status
**The hourly record is live** (since 2026-10-24 20:00 UTC). Every hour, 13 forecasters (eight evolved champions, the
Demon and four classical baselines) answer 21 questions: the distribution of the 1-, 4- and 24-hour log return of BTC,
ETH, SOL, XRP, DOGE, AVAX and LINK over ten bins. The hour's Merkle root is committed to
[`LaplaceLedger`](docs/design/live-record.md#deployments) on Robinhood Chain within 15 minutes of the hour.

- Latest issue: <https://data.laplaceprotocol.com/v1/latest.json>; scores: <https://data.laplaceprotocol.com/v1/scoreboard.json>
- Check any issue yourself: `cd engine && uv sync && uv run laplace-engine verify --issue <n>`
- How it will be judged, fixed in advance: [the pre-registered rule](docs/judging/r1.md) (the Demon against GARCH and
  HAR on the 1- and 4-hour questions; scheduled looks at 90, 180 and 365 days)
- How it works: [the live record](docs/design/live-record.md), [the evolution engine](docs/design/evolution-engine.md)
- Research so far: [PoC 1](docs/research/2026-10-23-poc1-majors.md) (language-model agents against baselines),
  [PoC 2](docs/research/2026-10-24-poc2-evolution.md) (evolved programs against GARCH and HAR)

## Principles
- **Pre-committed:** forecasts are hashed into an hourly Merkle root on-chain before they settle. Nobody can backdate a call.
- **Open scoring:** the scoring rules and a reference verifier live in this repo. Anyone can recompute any forecaster's record.
- **Honest baselines:** naive volatility, EWMA, GARCH and HAR face the same questions, with public code.
- **Corrections in the open:** when we find a mistake, the fix and the numbers it replaces stay side by side.

## Repository
| | |
|---|---|
| [`research/`](research) | the evolution engine, the baseline ladder and the experiments |
| [`engine/`](engine) | the live side: releases, hourly issues, commitments, publication, settlement and verification |
| [`contracts/`](contracts) | `LaplaceLedger`, the on-chain record |
| [`docs/`](docs) | design notes and dated research notes |

## No token yet
There is no Laplace token. Anything claiming to be one before an announcement on
[@laplaceproto](https://x.com/laplaceproto) is not ours.
