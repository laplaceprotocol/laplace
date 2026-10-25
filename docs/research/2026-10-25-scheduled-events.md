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
Reproduce with `laplace-research evolve events --run-id senses1`. Each cell compares candles at one offset from an
event (the candle that contains it, the one before, the three after) with weekday candles at the same UTC hours that
are not at that offset from such an event; for releases, days with any release are left out of the comparison.
Matching the UTC hour matters: hours near the US session are busier anyway, so comparing with all hours would credit
the clock to the event. Ratios are of mean squared hourly log returns, per asset and then averaged over assets, with
95% intervals from resampling the events.

| event | n | hour before | event hour | +1 h | +2 h | +3 h |
|---|---|---|---|---|---|---|
| FOMC | 14 | 0.73x (0.44 to 1.16) | 1.96x (0.90 to 3.25) | 1.93x (0.81 to 3.36) | 0.85x (0.34 to 1.66) | 0.47x (0.29 to 0.69) |
| CPI | 21 | 0.45x (0.20 to 0.85) | 3.60x (1.61 to 6.27) | 1.17x (0.48 to 2.13) | 0.90x (0.52 to 1.33) | 1.42x (0.61 to 2.71) |
| jobs report | 21 | 0.34x (0.16 to 0.59) | 1.71x (0.77 to 3.00) | 1.01x (0.61 to 1.50) | 1.30x (0.73 to 2.09) | 1.31x (0.39 to 2.58) |
| US open | 454 | 0.84x (0.69 to 1.02) | 1.05x (0.86 to 1.26) | 1.36x (1.13 to 1.63) | 1.06x (0.89 to 1.24) | 0.73x (0.62 to 0.86) |

- **CPI** moves crypto the most. Its release hour carries 3.6 times the variance of the same hour on other weekdays,
  every asset more than twice (2.3x to 5.1x). The hour before is quiet (0.45x), and after the release hour nothing
  is clearly different from an ordinary hour.
- **The jobs report** moves less (1.7x, the interval reaches below 1), and the hour before it is the quietest of
  all (0.34x).
- **FOMC statements** (2:00 p.m. ET; the press conference follows at 2:30) about double variance for two hours,
  though with fourteen decisions both intervals reach below 1; by the third hour, variance falls below the usual
  (0.47x).
- **The US open** is harder to read this way: the same UTC hour falls in another phase of the session in the other
  half of the year, so its row compares session phases rather than session against no session. The first full hour
  of the session runs 1.36 times the same UTC hour in the other season. That shift, one hour twice a year, is what a
  program that sees only the UTC hour cannot place.

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

Fitted effects on variance (`exp(β)`, per unit of input) for the inputs other than the UTC hour:

| input | 1h | 4h | 24h |
|---|---|---|---|
| weekend (whole window) | 0.67x | 0.63x | 0.66x |
| release inside the window | 2.40x | 1.00x | 0.85x |
| release in the 3 hours before the window | 1.11x | 1.21x | 1.12x |
| release in the 2 hours after the window | 0.69x | 0.70x | 0.74x |
| NYSE open inside the window | 1.32x | 1.40x | 1.41x |
| session (whole window) | 1.31x | 1.23x | 0.64x |

At 24 hours the weekend, open and session inputs are nearly collinear: a weekday window holds one open and about 6.5
hours of session. Read them together there.

### Gains
**Every hour of validation and holdout** (2026-06-01 to 2026-10-23, 3,464 forecast hours × 7 assets). Gain over
IV-HAR in nats per question, 95% intervals from a circular block bootstrap over 72-hour blocks
(`laplace-research evolve events`):

| 1h | 4h | 24h | 1h + 4h | all |
|---|---|---|---|---|
| +0.0135 (+0.0068 to +0.0201) | +0.0193 (+0.0097 to +0.0289) | +0.0186 (+0.0077 to +0.0304) | +0.0164 (+0.0089 to +0.0241) | +0.0171 (+0.0101 to +0.0246) |

**The PoC 2 holdout** (2026-08-16 to 2026-10-24, every third hour), from release r1.3's screening
([`screening.json`](../../engine/releases/r1.3/screening.json)), 1- and 4-hour questions pooled:

| forecaster | vs GARCH | vs IV-HAR |
|---|---|---|
| EV-HAR | +0.0185 (+0.0010 to +0.0361) | +0.0092 (-0.0027 to +0.0201) |
| the Demon (r1) | +0.0210 (+0.0087 to +0.0335) | +0.0116 (+0.0013 to +0.0232) |
| demon-he (every program, weights per horizon) | +0.0276 (+0.0139 to +0.0420) | +0.0183 (+0.0078 to +0.0286) |

EV-HAR's 1-hour gain over IV-HAR is +0.0011 on this holdout but +0.0135 on every hour. On every hour, the 1-hour
gain sits in the windows that start at 12:00 to 15:00 UTC (+0.03, +0.09, +0.13 and +0.07 nats): the releases at
12:30 and the session's first hours, in daylight saving time. The holdout samples every third hour from 01:00, and 3
divides 24, so its 1-hour questions only ever start at 01, 04, …, 22 UTC: of those four windows it scores only the
one at 13:00. The search (every 23 hours) and validation (every 5) windows rotate through the clock; future holdouts
should too.

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
