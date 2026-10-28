# Corrections

When we find a mistake, the fix and the numbers it replaces stay side by side, in the note or pull request that made
the fix. This log lists them, newest first. The live page at laplaceprotocol.com shows the same entries
(`site/public/index.html`; its tests check that the two agree).

## 2026-10-27 · Research returns as log price ratios
Research computed 4- and 24-hour returns as sums of hourly returns, while programs use log price ratios; the two
differed by about 1e-15 where a price came back to the same value. Rerun on ratios, EV-NET's gain over EV-HAR stays
+0.0084 nats per question (1 hour: +0.0169 instead of +0.0167), and no month moves by more than 0.0003.
[The learned-layer note](research/2026-10-27-learned-layer.md).

## 2026-10-25 · The first day's committed candles were provisional
Coinbase re-aggregates a just-closed hourly candle a little over five minutes after the hour, and the engine sealed at
:00:30. Of the first day's 168 committed last candles, 9 differ from the exchange's final candles, all in the close, by
up to 0.19% (LINK, issue 498039). Those issues stand as committed, and settlement uses the committed closes, so scores
stay consistent with the record. Since issue 498048 the engine seals at :07, after the re-aggregation.
[PR #27](https://github.com/laplaceprotocol/laplace/pull/27).

## 2026-10-25 · The event study's comparison hours
The first event study compared the hours around scheduled releases with all hours, which mostly measures the US
session's clock. Against the same UTC hours on weekdays without releases, CPI's next three hours are 1.2x, 0.9x and
1.4x the usual variance, not 2.1x to 2.6x, and the jobs report's 1.0x to 1.3x. The quiet hour before both holds
(0.45x and 0.34x). [PR #23](https://github.com/laplaceprotocol/laplace/pull/23),
[the scheduled-events note](research/2026-10-25-scheduled-events.md).

## 2026-10-24 · The GARCH baseline was mis-fitted
GARCH's Student-t likelihood had an extra 1/ν in its normalizing constant, which pushed ν towards infinity and left its
volatility dynamics arbitrary. Fixed and rerun on the same programs and windows: GARCH moved from −0.0016 to +0.0297
over naive, and the evolved pool's edge over it from +0.0426 to +0.0113 (−0.0151 to +0.0312). The claim that the pool
clearly beats GARCH was withdrawn. [PR #9](https://github.com/laplaceprotocol/laplace/pull/9),
[the PoC 2 note](research/2026-10-24-poc2-evolution.md#corrections).

## 2026-10-24 · PoC 2, after an independent review
Future candles were reachable from a program's context (no program used them); the pool was smoothed twice when
scored, which flattered it by about 0.0006 nats per question, and its weights were not fully fitted; window boundaries
leaked; intervals were too narrow; and the run cost about $2.9–3.5, not the $2.35 recorded. Each was fixed with
regression tests and the report rerun; the claim that the pool's edge over HAR was just clear of zero was withdrawn.
[PR #4](https://github.com/laplaceprotocol/laplace/pull/4),
[the PoC 2 note](research/2026-10-24-poc2-evolution.md#corrections).
