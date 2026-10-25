# Scheduled events and hourly crypto volatility

2026-10-25 · development evidence (the holdout has been inspected before; the live record is the test)

## Question
Do scheduled US events move hourly crypto volatility enough that a forecaster should know about them? And what is
knowing them worth under our score?

## Data
- **Prices.** PoC 2's panel: Coinbase hourly candles for BTC, ETH, SOL, XRP, DOGE, AVAX and LINK, 2025-01-01 to
  2026-10-24.
- **Calendar.** [`events.toml`](../../research/src/laplace_research/data/events.toml) holds FOMC decisions, CPI and
  jobs-report releases, and NYSE holidays and early closes, copied from the publishers' calendars, which it names. It
  lists the dates releases actually happened, including those the 2025 lapse in appropriations delayed. Times are US
  Eastern and are converted with daylight saving.

## How much events move volatility
Reproduce with `laplace-research evolve events --run-id senses1`. Each ratio is a mean squared hourly log return.
- **Event hour:** the candle that contains the event, against weekday candles at the same UTC hours that contain no
  event of that kind, as a range over the seven assets.
- **Hours before and after:** against all hours, as a mean over assets.

| event | n | event hour (range over assets) | hour before | next hours |
|---|---|---|---|---|
| FOMC | 14 | 1.43x to 2.39x | 1.01x | 2.05x, 0.82x, 0.49x |
| CPI | 21 | 2.05x to 5.19x | 0.37x | 2.26x, 2.11x, 2.64x |
| jobs report | 21 | 1.24x to 1.79x | 0.28x | 1.99x, 3.10x, 2.35x |
| US open | 454 | 1.46x to 2.56x | 1.18x | 2.67x, 1.86x, 1.24x |

- **CPI** moves crypto the most. In the release hour, variance is 2 to 5 times that of the same hour on other
  weekdays, and it stays at about twice the average for the next three hours. The hour before is unusually quiet
  (0.37 times the average): the market waits.
- **The jobs report** moves less in its own hour, more in the next three (2 to 3 times), and its hour before is the
  quietest of all.
- **FOMC statements** (2:00 p.m. ET; the press conference follows at 2:30) double variance for two hours, then
  calm falls below average.
- **The US open** raises its hour by 1.5 to 2.6 times over other weekday hours, and the next hour runs 2.7 times the
  average. A program that sees only the UTC hour cannot place the open, which moves between 13:30 and 14:30 UTC with
  daylight saving, nor any release.

Programs now see these events as `ctx.events`: every event from 24 hours before to 48 hours after the forecast time,
as (hours, kind). That is when each event is scheduled, never what it announces. Every issue records the events its
programs saw.

## A baseline that knows them: EV-HAR
EV-HAR scales IV-HAR's standard deviation by `exp(x·β/2)`. The inputs `x` are:
- the share of the window that falls on a weekend;
- the UTC hour (1- and 4-hour questions only);
- FOMC, CPI and jobs releases inside the window, in the 3 hours before it, and in the 2 hours after it;
- NYSE opens inside the window, and the share of the window the session is open.

The coefficients β are fitted per horizon on every hour of the search window (2025-02-01 to 2026-06-01), all assets
pooled, at the maximum of the binned log score under a Gaussian prior (sd 1 in log variance per unit of input, scaled
to the number of questions); tails are then chosen again. The frozen program computes exactly the research inputs
(tested to 1e-12).

Fitted effects on variance (`exp(β)`, per unit of input) for the inputs that do not depend on the clock alone:

| input | 1h | 4h | 24h |
|---|---|---|---|
| weekend (whole window) | 0.67x | 0.63x | 0.66x |
| release inside the window | 2.40x | 1.00x | 0.85x |
| release in the 3 hours before | 1.11x | 1.21x | 1.12x |
| release in the 2 hours after | 0.69x | 0.70x | 0.74x |
| NYSE open inside the window | 1.32x | 1.40x | 1.41x |
| session (whole window) | 1.31x | 1.23x | 0.64x |

At 24 hours the weekend, open and session inputs are nearly collinear: a weekday window holds one open and about 6.5
hours of session. Read them together there.

### Gains
**Every hour of validation and holdout** (2026-06-01 to 2026-10-23, 3,463 forecast hours × 7 assets). Gain over
IV-HAR in nats per question, 95% intervals from a circular block bootstrap over 72-hour blocks
(`laplace-research evolve events`):

| 1h | 4h | 24h | 1h + 4h | all |
|---|---|---|---|---|
| +0.0135 (+0.0068 to +0.0201) | +0.0193 (+0.0094 to +0.0287) | +0.0186 (+0.0072 to +0.0302) | +0.0164 (+0.0089 to +0.0237) | +0.0171 (+0.0101 to +0.0241) |

**The PoC 2 holdout** (2026-08-16 to 2026-10-24, every third hour), from release r1.3's screening
([`screening.json`](../../engine/releases/r1.3/screening.json)), 1- and 4-hour questions pooled:

| forecaster | vs GARCH | vs IV-HAR |
|---|---|---|
| EV-HAR | +0.0185 (+0.0010 to +0.0361) | +0.0092 (-0.0027 to +0.0201) |
| the Demon (r1) | +0.0210 (+0.0087 to +0.0335) | +0.0116 (+0.0013 to +0.0232) |
| demon-he (every program, weights per horizon) | +0.0276 (+0.0139 to +0.0420) | +0.0183 (+0.0078 to +0.0286) |

EV-HAR's 1-hour gain over IV-HAR is +0.0011 on this holdout but +0.0135 on every hour. The holdout samples every
third hour, and 3 divides 24, so its 1-hour questions only ever start at 00, 03, …, 21 UTC. The windows that start
at 13:00 and 14:00 UTC, where the open and the releases fall, are never scored at 1 hour. The search (every 23
hours) and validation (every 5) windows rotate through the clock; future holdouts should too.

What this means:
- A classical model given the same information gets most of the way. EV-HAR alone holds almost nine-tenths of the
  Demon's edge over GARCH at 1 and 4 hours. The Demon still beats IV-HAR, but the bar for the next evolution round
  (PoC 3, with senses and events in every program's context) is EV-HAR, not GARCH.
- Combined, the two are better than either. In demon-he, EV-HAR takes 27% of the weight at 1 hour, 49% at 4 hours
  and 47% at 24 hours, next to the evolved programs. It is the best pool so far, and with demon-plus-e the first
  whose 24-hour mean is above GARCH's (well inside the noise).

## How the fit was settled
Two choices were corrected before anything was released:
- **The prior.** A fixed ridge (1e-3 on the mean score) shrank the rare release inputs to almost nothing (1.16x for a
  release inside a 1-hour window). A prior scaled to the number of questions shrinks an input by how rarely it
  occurs, not by a fixed amount.
- **Every hour.** The run's fit sample takes every second hour, so the first fits saw only half the hours of the
  day, and the other half kept their starting coefficients. Fitted on every hour, the gain over IV-HAR at 1 and 4
  hours rose from +0.0129 to +0.0164, and a release inside a 1-hour window from 1.86x to 2.40x.

## Caveats
- **Few events.** The panel holds 21 CPI releases, 21 jobs reports and 14 FOMC decisions, so the release
  coefficients are the least certain. The session and weekend inputs are seen every week.
- **Short holdout.** The holdout holds only a handful of releases, so it measures the session and weekend inputs far
  better than the release inputs.
- **Rescheduling.** The calendar lists the dates releases actually happened. If a release was rescheduled less than
  48 hours before it happened, a backtest context would have known a few hours too much. The live record cannot have
  this problem.
- **Maintenance.** The calendar covers 2025 to 2027. The engine alerts 60 days before it runs out, and it must be
  corrected whenever a publisher reschedules.
