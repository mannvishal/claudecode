# tqsq — trading the 1/2/3 exhaustion marks on TQQQ and SQQQ

A Python port of the 0DTE composite regime score from **Vishal's Upper Pane v6.40**,
a three-tranche backtest of the trade the marks suggest, and a Robinhood decision
bridge for running it live.

**The measured answer to "can it profit me": no. It loses money, and after eight
years of 1-minute data that is a measurement rather than an absence of evidence.**
Per-trade expectancy is negative with a confidence interval that excludes zero, in
both sessions, in every one of nine calendar years, and out of sample after fitting
the parameter grid. The numbers are below.

---

## The strategy under test

Taken from the request, stated exactly:

| Mark | Meaning | Trade |
|---|---|---|
| Exhaustion **1/2/3** at the **bottom** | the down-stretch is tiring | buy **TQQQ** |
| Exhaustion **1/2/3** at the **top** | the up-stretch is tiring | buy **SQQQ** |

One tranche per tier, so a full 1 → 2 → 3 episode ends three tranches deep, plus a
stop loss. Exits are the stop, an optional target, the score returning to the
neutral ±4 band, and a session-end flat.

It is worth naming the structure plainly: **this is a martingale.** Tier 2 fires
because tier 1 is losing, and tier 3 because tier 2 is losing. That turns out to be
the single most damaging thing about it — see *The ladder is the problem* below.

---

## Data

**TQQQ and SQQQ 1-minute bars, XNAS.ITCH via Databento, 2018-08-08 → 2026-08-07 —
1,984 sessions.** Signal computed on TQQQ, matching the charted symbol.

Databento's `ohlcv-1m` is **as-traded**, and over this window TQQQ split 2:1 three
times while SQQQ reverse-split five times. Left raw, each is a several-hundred-percent
overnight bar the score reads as a real move. `bars.detect_splits` finds them and
snaps to the nearest standard ratio rather than the observed gap — the observed gap
is the true ratio times that night's real return, so adjusting by it would erase the
move. SQQQ's 2019-05-24 gap of 3.890 is a 1-for-4 on a −2.7% night, not a 1-for-5 on
a −22% one. After adjustment the largest residual overnight gap in either series is
17.5%, on 2020-03-13 — the COVID crash, behaving as a 3× ETF should.

---

## Results

### Headline

| | RTH (09:30–16:00) | ETH (04:00–20:00) |
|---|---|---|
| sessions | 1,984 | 1,984 |
| exhaustion fires | 4,422 (2.2/session) | 10,021 (5.1/session) |
| trades | **2,037** | **4,523** |
| slippage assumed | 5 bps | 15 bps |
| win rate | 56.7% | 34.6% |
| **mean trade** | **−11.5 bps, CI [−16.0, −7.1]** | **−29.7 bps, CI [−32.1, −27.4]** |
| profit factor | 0.60 | 0.25 |
| buy & hold TQQQ | +1,678.7% | +786.5% |

Both confidence intervals lie entirely below zero. This is not "no edge detected";
it is a **negative** edge, measured on thousands of trades.

> **On the aggregate return figures.** Cumulative P&L is −$178k (RTH) and −$766k
> (ETH) against $100k of starting capital, because the engine sizes every tranche
> off the *initial* capital and never compounds or halts. Read those as "the sum of
> trade P&Ls came to 1.8× / 7.7× the starting stake," not as a survivable drawdown —
> in reality the account is gone long before. The honest headline is the per-trade
> expectancy above.

### Negative in every single year

RTH, mean trade in basis points:

| year | trades | win rate | mean trade | 95% CI |
|---|---|---|---|---|
| 2018 | 97 | 49% | −27.7 | [−53.6, −2.1] |
| 2019 | 261 | 59% | −6.3 | [−16.1, +3.1] |
| 2020 | 252 | 53% | −20.3 | [−34.6, −5.6] |
| 2021 | 229 | 55% | −14.4 | [−27.2, −2.5] |
| 2022 | 271 | 58% | −14.3 | [−29.3, +0.8] |
| 2023 | 263 | 54% | −8.4 | [−19.9, +2.7] |
| 2024 | 255 | 61% | −9.0 | [−20.6, +2.2] |
| 2025 | 256 | 59% | −4.6 | [−16.5, +6.8] |
| 2026 | 153 | 56% | −8.1 | [−24.9, +8.0] |

Nine for nine, through a bull market, a COVID crash, a rate-hike bear market and the
recovery. ETH is the same shape and worse: every year between −22 and −35 bps, with
win rates of 26–47%.

### It does not beat entering at random

| | strategy | random entry, same duration (95% CI) | beats random? |
|---|---|---|---|
| RTH | −11.5 bps | [−3.3, +8.9] | **no** |
| ETH | −29.7 bps | [−1.9, +4.3] | **no** |

Random entry is mildly positive — it is a leveraged ETF that rose 16× over the
window. The strategy is well below it in both sessions.

### Walk-forward: even the best-fitted parameters lose

Fit the whole 30-cell stop × target grid on the first half, then score that single
choice on the second half it never saw:

| | RTH | ETH |
|---|---|---|
| best in-sample cell | stop 0.5%, target 0.5% | stop 0.5%, target 0.5% |
| in-sample P&L | −$45,732 | −$308,506 |
| **out-of-sample mean trade** | **−9.0 bps, CI [−11.8, −6.2]** | **−31.2 bps, CI [−32.9, −29.6]** |

The best cell the grid could find was still losing money *in sample*. There was no
overfit to detect, because there was nothing positive to overfit to.

### The ladder is the problem

RTH, by how deep the episode ran:

| side | tranches | n | win rate | mean return | total P&L |
|---|---|---|---|---|---|
| TQQQ | 1 | 621 | 65% | −0.03% | −$5,489 |
| TQQQ | 2 | 176 | 53% | −0.28% | −$32,272 |
| TQQQ | 3 | 71 | **23%** | **−0.79%** | −$56,184 |
| SQQQ | 1 | 787 | 61% | −0.02% | −$4,269 |
| SQQQ | 2 | 268 | 45% | −0.23% | −$41,788 |
| SQQQ | 3 | 114 | **35%** | **−0.33%** | −$38,106 |

This is the clearest result in the study. **A single tranche is roughly break-even
before costs; the adds are what destroy it.** Win rate falls monotonically with
depth — 65% → 53% → 23% on the long side — because reaching tier 3 *is* the news
that the first two entries were wrong. Averaging down into a 3× leveraged ETF
converts a coin-flip into a reliable loser.

### Every quality filter makes it worse

The two knobs meant to improve signal quality both reduce it:

| filter | RTH mean trade | ETH mean trade |
|---|---|---|
| all fires (tier 1+) | −11.5 bps | −29.7 bps |
| tier 2+ only | −20.0 bps | −34.8 bps |
| tier 3+ only | −25.0 bps | −37.4 bps |
| + amplifier required | −18.8 bps | −27.8 bps |

Requiring a *deeper* exhaustion makes the trade worse, not better. Deeper is rarer,
not better — which is exactly what the indicator's own tooltip warns: *"nothing in
this project has established that deeper fires are BETTER, only rarer."*

### The P&L shape

ETH exits:

| reason | n | mean return | total P&L |
|---|---|---|---|
| score returned to neutral | 3,895 | −0.08% | −$214,984 |
| **stop** | **497** | **−2.17%** | **−$558,669** |
| session end | 88 | −0.52% | −$24,454 |
| target | 43 | +1.87% | +$31,660 |

In RTH the neutral-band exits are mildly positive (+0.26% over 1,558 trades) and the
324 stops at −2.05% overwhelm them. In ETH even the neutral exits are negative. The
worst single trade was −5.9%, well through the 2% stop — that is a gap, and gaps run
against a martingale.

---

## This reproduces the indicator author's own finding

The Pine source header is explicit, and this study lands in the same place from
independent data:

> the +/-8.5 reversal entry is dead unconditionally at decade scale
> (N=3,307, 30-minute mean **+0.0 bps**, REVERSAL_STUDY_REPORT.md); exhaustion is an
> extreme MARKER at about 2x time-matched no-skill, explicitly not an entry

> **THIS IS A MARKER LOCATION, NOT AN ENTRY.** … Trading it as a reversal is the
> thing that has already been measured and rejected.

The marks do their job — they mark stretch and exhaustion. That is a different claim
from "entering there makes money after costs."

## Why the two-month pilot looked positive

An earlier run on 43 sessions (2026-06 to 2026-08) showed ETH at **+4.36%** with a
74% win rate. The eight-year answer for the same configuration is **−29.7 bps per
trade across 4,523 trades.** The pilot's confidence interval included zero at the
time and it was labelled underpowered — this is what that caveat was for. Forty-three
sessions of a high-win-rate, negative-skew strategy will show a profit most of the
time, because the rare 2% stops have not arrived yet.

---

## Install and use

```bash
pip install -e ".[databento,dev]"
pytest                      # 26 tests

tqsq cost   --start 2018-05-01 --end 2026-08-08     # free, prices the pull (~$1.92)
tqsq fetch  --start 2018-05-01 --end 2026-08-08     # chunked, newest first, resumable
tqsq backtest --session rth --stop 2 --target 2 --slippage 5
tqsq sweep    --session rth --slippage 5            # every robustness table
```

`fetch` chunks by year newest-first and writes each chunk as it lands, so an
interrupted pull keeps the most recent history rather than nothing.
`metadata.get_cost` prices the *request*, not your balance — a cheap estimate can
still return `402 account_insufficient_funds`.

---

## Running it live through Robinhood

The Robinhood connector is a set of MCP tools, callable by an agent session and not
by a long-running Python process. So this package is deliberately **not a daemon**.
It is a pure function of `(bars, state) → order intents`, with JSON on both sides:

```
agent  --get_equity_historicals-->  bars.json
tqsq decide --bars-json bars.json --state state.json --quotes '{"TQQQ":74.1,"SQQQ":42.3}'
                                -->  intents.json
agent  --review_equity_order, then place_equity_order-->  broker
agent  --get_equity_positions-->  state.json
```

Every real-money call stays with the agent, where the review step and your
confirmation live. `decide` emits diagnostics on every invocation — score, tier,
climax, divergence, window — so a quiet minute reports *why* nothing fired.

Risk gates in `RiskLimits`, checked before any intent is emitted: `max_position_usd`,
`max_daily_loss_usd` (breaching it **halts and flattens**, it does not size down),
`max_tranches`, and `allow_live`, which is off by default so nothing places without
being set explicitly.

**This is wired and tested, and the measurement says do not point it at real money in
this configuration.** It is here because the port is the reusable part: the score,
the machines and the diagnostics are faithful, and they are worth having whether or
not you trade this particular rule.

**On "24×7":** TQQQ and SQQQ do not trade around the clock, and the extended-hours
variant is the worse of the two by a factor of nearly three per trade.

---

## What is faithful in the port, and what is not

Verified by 26 tests (`tests/test_engine.py`):

- higher-timeframe bars anchor to the **session open**, not the wall clock — a
  60-minute bar runs 09:30–10:30
- leg state comes from **completed** bars only; the v6.37 `[1]` shift is reproduced,
  and skipping it makes the score lead the truth in a trend
- the score saturates at exactly ±10 and is **monotone in price**, which is what makes
  the isoline solver well defined
- exhaustion tiers are **independent one-shot machines**, re-arming only in the
  neutral band; the RSI confirmation genuinely vetoes the pullback branch
- full precision in the decision path (the v6.36 rounding leak), rounding only for display
- entries fill at the **next bar's open**, never the signal bar's close
- splits are detected, snapped to standard ratios, and back-adjusted; a real 17.5%
  gap is *not* mistaken for one

Known approximations:

- `ta.ema`/`ta.rma` are seeded from the first value rather than an SMA over the first
  `length` bars. Both converge, and eight years dwarfs the warmup.
- Volume comes from XNAS.ITCH (Nasdaq only, not consolidated), so the volume-climax
  z-score sees a fraction of true volume. It feeds only the optional amplifier filter,
  which the sweep shows makes results worse anyway.
- Stops fill **at** the level when a bar trades through it, except on a gap, which
  fills at the open. `gap_fill_share` reports how often this flattered the result:
  0.5% (RTH) and 0.7% (ETH).

---

## Layout

```
tqsq/bars.py       session handling, HTF bucketing, split detection/adjustment
tqsq/score.py      the composite score: leg state + f_legAt
tqsq/signals.py    zone (A/B/C) and exhaustion (1/2/3) machines
tqsq/backtest.py   three-tranche engine, stops, exits, costs
tqsq/metrics.py    stats with bootstrap CIs, random-entry baseline
tqsq/sweep.py      slippage, stop x target grid, by-year, walk-forward, split-half
tqsq/data.py       Databento fetch with a cost gate and chunked cache
tqsq/live.py       decision engine, risk gates, JSON bridge
tqsq/cli.py        cost | fetch | backtest | sweep | decide
results/           REPORT.md, grids, trade lists
```

---

## Bottom line

Across 1,984 sessions and 6,560 trades, entering TQQQ and SQQQ on the 1/2/3
exhaustion marks with three tranches and a stop **lost money in both sessions, in
every calendar year, out of sample, and against a random-entry baseline.** The
tranche ladder is the main mechanism: a single tranche is near break-even, and win
rate collapses to 23% by the third add, because reaching tier 3 is itself the
evidence that the first two entries were wrong.

If any part of this is worth keeping, it is the marks as *context* — which is what
the indicator says they are — and the single-tranche version as the thing to study
further, without the martingale attached to it.

*Not financial advice. This is a measurement of a specific rule on a specific sample.*
