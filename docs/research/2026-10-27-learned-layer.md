# A learned layer on EV-HAR, tested month by month

2026-10-27 · development evidence (no fold was used to choose a variant; the live record is the test)

## Question
EV-HAR, a classical model that knows the US session, weekends and scheduled releases, matches the evolved Demon on
PoC 2's holdout. Can a learned correction on top of EV-HAR add edge that survives an honest out-of-sample test?

## The model: EV-NET
EV-NET keeps EV-HAR's distribution and adds a correction to its log-probabilities, one small network per horizon:

    logits = log EV-HAR(bins) + W2 tanh(W1 standardize(x) + b1) + b2      (27 inputs, 8 hidden units, 10 bins)

The 27 inputs (`laplace_research.evolve.learned.NAMES`) are all things a program can compute from its context:
- realized variance over 1, 4, 24 and 719 hours, against the past week's;
- the variance DVOL implies;
- recent returns and their size;
- the high-low range;
- BTC's last hour;
- funding;
- the hour and weekday as cycles;
- EV-HAR's calendar inputs.

The network starts as no correction at all and is fitted on the binned log score, with a light L2 penalty, on every
hour of its fit window. It keeps the weights that score best on a later window.

The weights are stored as 16-bit integers, and the research forecasts use exactly those. The frozen program
(`EV_NET` in `engine/src/laplace_engine/baselines.py`) is about 15 KB, inside the sandbox's 16 KB limit. Given the
same EV-HAR spread, it computes the same inputs as research; its forecasts differ only where EV-HAR's own program does.

## The test: rolling origin
Each fold tests one calendar month: April to October 2026, seven folds, 4,784 test hours.
- **Fitting.** GARCH, IV-HAR, EV-HAR and the network are fitted on every hour from 2025-02-01 to the month before
  the test month.
- **Stopping.** The network stops early on that month before.
- **Scoring.** Every hour of the test month is scored, after a 24-hour gap so no label crosses a boundary.

Gains are per forecast hour, averaged over the seven assets, in nats per question. Reproduce with
`laplace-research evolve rolling --run-id senses2`.

**Correction (2026-10-27, later).** Research now computes 4- and 24-hour returns as log price ratios, as programs
do, rather than as sums of hourly returns (they differed by 1e-15 where a price came back to the same value).
Rerun, EV-NET's pooled gain over EV-HAR is the same +0.0084 (1h +0.0169 instead of +0.0167) and no fold moves by
more than 0.0003; EV-HAR and GARCH are unchanged. The numbers below are the original run's
([the EV-GBM note](2026-10-27-ev-gbm.md) has the rerun).

### Results
Gains in nats per question at 1 and 4 hours pooled, per fold (the month tested):

| test month | hours | EV-NET vs EV-HAR | EV-HAR vs GARCH | EV-NET vs GARCH |
|---|---|---|---|---|
| 2026-04 | 696 | +0.0149 | -0.0009 | +0.0140 |
| 2026-05 | 720 | +0.0097 | +0.0187 | +0.0284 |
| 2026-06 | 696 | +0.0020 | +0.0211 | +0.0230 |
| 2026-07 | 720 | +0.0065 | +0.0214 | +0.0279 |
| 2026-08 | 720 | +0.0137 | -0.0119 | +0.0018 |
| 2026-09 | 696 | +0.0063 | +0.0277 | +0.0339 |
| 2026-10 | 536 | +0.0045 | +0.0493 | +0.0538 |

Pooled over every test hour, with 95% intervals from a circular block bootstrap over 72-hour blocks:

| | 1h | 4h | 24h | 1h + 4h | months ahead |
|---|---|---|---|---|---|
| EV-NET vs EV-HAR | +0.0167 (+0.0113 to +0.0230) | +0.0000 (-0.0061 to +0.0064) | -0.0133 (-0.0254 to -0.0031) | +0.0084 (+0.0038 to +0.0135) | 7 of 7 |
| EV-HAR vs GARCH | +0.0099 (-0.0010 to +0.0201) | +0.0236 (+0.0111 to +0.0363) | +0.0247 (+0.0049 to +0.0425) | +0.0167 (+0.0060 to +0.0269) | 5 of 7 |
| EV-NET vs GARCH | +0.0266 (+0.0173 to +0.0354) | +0.0236 (+0.0102 to +0.0371) | +0.0114 (-0.0168 to +0.0344) | +0.0251 (+0.0149 to +0.0355) | 7 of 7 |

- **EV-HAR beats GARCH out of sample.** It is ahead in 5 of 7 months, by +0.017 at 1 and 4 hours: the
  calendar-aware classical bar is real.
- **The learned layer adds to it in every month.** That is +0.008 at 1 and 4 hours, almost all at 1 hour. Against
  GARCH, EV-NET is ahead in all seven months, by +0.025.
- **At 24 hours the network does harm** (-0.013). One validation month of heavily overlapping 24-hour labels is too
  noisy to stop it in time. 24 hours is outside the judged scope; pools weight it per horizon.

## Which learned model
An exploratory comparison used the same folds, from a scratch script that is not in the repository. It boosted
LightGBM from EV-HAR's log-probabilities on the same inputs: +0.0166 at 1 and 4 hours (+0.0115 to +0.0220), ahead
in 7 of 7 months, and +0.004 at 24 hours. That is about twice the small network's gain, and it did not lose at 24
hours. Splitting the 5-minute returns into downside, jumps and late-hour bursts added nothing to LightGBM
(+0.0165 with them).

LightGBM's trees are far beyond a program's 16 KB, so it ships through public parameter files: EV-GBM, a later
release. It will be measured with this same harness before it is judged.

The network ships now as EV-NET, because it fits inside a program as it is.

## The shipped instance
Release r1.4 carries one EV-NET, fitted like every r1.x program on PoC 2's windows: the network on every hour of the
search window (to 2026-06-01), stopped on every hour of the validation window (to 2026-08-16). Run as a program, it
scores exactly as the research model on the holdout (to four decimals; probabilities differ by at most 0.002, where
EV-HAR's own program does).

On PoC 2's holdout it adds much less than the rolling test found:

| EV-NET over EV-HAR | 1h | 4h | 24h | 1h + 4h |
|---|---|---|---|---|
| the holdout's 547 forecast hours | +0.0012 | +0.0001 | +0.0014 | +0.0007 |
| every hour of the holdout window (1,639) | +0.0070 | +0.0004 | +0.0015 | +0.0037 |
| the validation window it stopped on, every hour | +0.0257 | -0.0001 | -0.0003 | +0.0128 |

- **The holdout's clock.** The holdout only starts at 01, 04, …, 22 UTC (the
  [scheduled-events note](2026-10-25-scheduled-events.md) found this). On those eight hours EV-NET gains +0.0012 at
  1 hour, on the other sixteen +0.0099. It loses most at 13:00 and 16:00 (-0.055 and -0.048 at 1 hour, 68 days
  each), in the US session, and the holdout samples both.
- **Its age.** Over the same months, the monthly refits of the rolling test gained +0.0137 (August), +0.0063
  (September) and +0.0045 (October). This instance was fitted through May. How fast a fitted layer fades is the next
  question.
- **The pools.** It gains +0.0128 on the validation window it stopped on, and the r1.4 pools are fitted on that
  window, so they may lean on it more than an independent window would justify. EV-NET gets 47% of demon-hn at
  1 hour, and almost nothing at 4 and 24 hours.

## Caveats
- **Fold boundaries.** The folds are calendar months, chosen before the run; nothing was selected on them.
- **The same seven months.** They overlap the PoC 2 validation and holdout, so this is still development evidence.
  The live record, and the rule that will judge the learned layer, are the test.
- **Instance or procedure.** The rolling test measures a procedure that refits every month; a release freezes one
  fit. What the live record tests depends on how often releases refit.
