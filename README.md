# tqsq — trading the 1/2/3 exhaustion marks on TQQQ and SQQQ

A Python port of the 0DTE composite regime score from **Vishal's Upper Pane v6.40**,
a three-tranche backtest of the trade the marks suggest, and a Robinhood decision
bridge for running it live.

**The measured answer to "can it profit me": no, not as specified.** At realistic
transaction costs the strategy loses money in both the regular session and the
extended session, and it never beats entering at random for the same holding
period. The numbers, the caveats and the size of the sample are all below —
they matter more than the headline.

---

## The strategy under test

Taken from the request, stated exactly:

| Mark | Meaning | Trade |
|---|---|---|
| Exhaustion **1/2/3** at the **bottom** | the down-stretch is tiring | buy **TQQQ** |
| Exhaustion **1/2/3** at the **top** | the up-stretch is tiring | buy **SQQQ** |

One tranche per tier, so a full 1 → 2 → 3 episode ends three tranches deep, plus
a stop loss. Exits are the stop, an optional target, the score returning to the
neutral ±4 band, and a session-end flat.

It is worth naming the structure plainly: **this is a martingale.** Tier 2 fires
because tier 1 is losing, and tier 3 because tier 2 is losing. On a 3× leveraged
ETF the stop is the only thing between the ladder and an unbounded loss, which is
why every result below is quoted per stop level rather than averaged over them.

---

## Results

Data: **TQQQ and SQQQ 1-minute bars, XNAS.ITCH via Databento, 2026-06-08 to
2026-08-07 — 43 sessions.** Signal computed on TQQQ, matching the charted symbol.

### Headline

| | RTH (09:30–16:00) | ETH (04:00–20:00) |
|---|---|---|
| exhaustion fires | 95 (2.2/session) | 258 (6.0/session) |
| trades | 42 | 117 |
| slippage assumed | 5 bps | 15 bps |
| **total return** | **−2.34%** | **−10.9%** |
| win rate | 59.5% | 46.9% |
| mean trade | +1.1 bps, CI **[−36.7, +37.4]** | −17.9 bps, CI **[−31.8, −5.2]** |
| profit factor | 0.77 | 0.43 |
| max drawdown | −4.74% | −11.51% |
| buy & hold TQQQ | −3.50% | −0.19% |

### Slippage is what decides it

The single most important table in the project. The strategy takes liquidity on
entry and on every stop.

**Regular session**

| slippage (bps) | total return | mean trade (bps) | 95% CI | profit factor |
|---|---|---|---|---|
| 0 | −0.70% | +9.7 | [−28.5, +46.4] | 0.93 |
| 2 | −1.36% | +6.3 | [−31.7, +42.8] | 0.86 |
| **5** | **−2.34%** | **+1.1** | [−36.7, +37.4] | 0.77 |
| 10 | −3.98% | −7.5 | [−44.8, +28.4] | 0.63 |
| 20 | −7.28% | −24.9 | [−61.2, +10.1] | 0.41 |

**Extended session**

| slippage (bps) | total return | mean trade (bps) | 95% CI | profit factor |
|---|---|---|---|---|
| 0 | +7.05% | +13.9 | [+0.1, +26.5] | 1.61 |
| 2 | +4.36% | +8.5 | [−5.8, +21.5] | 1.35 |
| 5 | +1.11% | +2.8 | [−11.4, +15.6] | 1.08 |
| **10** | **−5.54%** | **−8.4** | [−22.5, +4.5] | 0.67 |
| 30 | −25.51% | −46.5 | [−59.6, −34.3] | 0.10 |

The extended-hours result is the one that looks attractive, and it is the one
least worth believing. TQQQ quotes about a penny wide at $74 in the regular
session — roughly 1.4 bps — but ten to thirty times that at 04:00. The edge dies
somewhere between 5 and 10 bps, which is *inside* the range of a realistic
overnight spread. **The ETH profit is a statement about the fill assumption, not
about the signal.**

### It does not beat entering at random

Random entries of the same 33/41-minute duration over the same tape:

| | strategy | random entry (95% CI) | beats random? |
|---|---|---|---|
| RTH | +1.1 bps | [−44.0, +43.9] | **no** |
| ETH | −17.9 bps | [−18.7, +20.1] | **no** |

The strategy mean sits comfortably inside the random band in both cases. Whatever
the marks are selecting for, it is not something that shows up as return over this
sample.

### The tranche ladder makes it worse

ETH, by how deep the episode ran (total P&L, $):

| side | tranches | n | win rate | total P&L |
|---|---|---|---|---|
| SQQQ | 1 | 39 | 36% | −3,815 |
| SQQQ | 2 | 7 | 14% | −876 |
| SQQQ | 3 | 6 | 17% | −684 |
| TQQQ | 1 | 38 | 66% | +168 |
| TQQQ | 2 | 20 | 50% | −4,663 |
| TQQQ | 3 | 7 | 57% | −1,056 |

Adding to the loser lost money in five of six buckets. Note also the asymmetry:
the **SQQQ (short) side is where most of the damage is**, which is what you would
expect from a mean-reversion entry fighting an instrument with a structural
downward drift.

### The shape of the P&L is the real warning

ETH exits:

| reason | n | mean return | total P&L |
|---|---|---|---|
| score returned to neutral | 105 | +0.03% | +1,375 |
| **stop** | **11** | **−2.15%** | **−12,166** |
| session end | 1 | −0.40% | −135 |

105 small winners against 11 large losers. A 74%-win-rate strategy that loses
money is picking up pennies, and the 3× leverage plus the martingale add means the
tail is doing all the work. Widening the stop *improves* the backtest at every
step of the grid (0.5% → −1.8%, 5% → +8% at 2 bps) — which is the classic
signature of no edge: the only thing helping is giving losers more room, and the
limit of that is no stop at all.

### Stability

The RTH sample flips sign between halves (−28 bps → +40 bps mean trade). On 42
trades that is what noise looks like.

---

## This reproduces the indicator author's own finding

The Pine source header is explicit, and this backtest independently lands in the
same place:

> the +/-8.5 reversal entry is dead unconditionally at decade scale
> (N=3,307, 30-minute mean **+0.0 bps**, REVERSAL_STUDY_REPORT.md); exhaustion is
> an extreme MARKER at about 2x time-matched no-skill, explicitly not an entry

and, on the exhaustion-fire price level:

> **THIS IS A MARKER LOCATION, NOT AN ENTRY.** … Trading it as a reversal is the
> thing that has already been measured and rejected.

The marks are doing their job — they mark stretch and exhaustion. That is a
different claim from "entering there makes money after costs."

---

## The sample is small, and here is exactly why

The Databento account ran out of budget mid-project.

- The full 2018-05 → 2026-08 pull across TQQQ/SQQQ/QQQ prices at **$2.96**.
- The account balance covered about **$0.05**, so the pull returned
  `402 account_insufficient_funds`.
- What was salvageable was **2 months of TQQQ + SQQQ**, which is what everything
  above is measured on.

`metadata.get_cost` prices the *request*, not your remaining balance, so a cheap
estimate can still 402. `tqsq fetch` handles this: it chunks by year **newest
first** and writes each chunk as it lands, so a mid-way 402 leaves the most recent
history on disk instead of nothing.

**To run the full eight-year test**, top up at
<https://databento.com/portal/billing> (about $3 covers it) and:

```bash
tqsq cost   --start 2018-05-01 --end 2026-08-08          # free, prices the pull
tqsq fetch  --start 2018-05-01 --end 2026-08-08
tqsq sweep  --session rth --slippage 5
```

Nothing else changes — the loaders read whatever is cached. Forty-three sessions
cannot settle this question; eight years can. My expectation, given both the
author's decade-scale null and the cost curves above, is that more data makes the
result *more* clearly negative, not less — but that is a prediction, not a
measurement, and it is the reason the code is written to run at that scale.

---

## Install and use

```bash
pip install -e ".[databento,dev]"
pytest                      # 21 tests
```

```bash
# One configuration
tqsq backtest --session rth --stop 2 --target 2 --slippage 5

# Every robustness table: slippage curve, stop x target grid,
# tier/amplifier filter, random baseline, split-half
tqsq sweep --session rth --slippage 5
```

---

## Running it live through Robinhood

The Robinhood connector here is a set of MCP tools, callable by an agent session
and not by a long-running Python process. So this package is deliberately **not a
daemon**. It is a pure function of `(bars, state) → order intents`, with JSON on
both sides:

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

Risk gates in `RiskLimits`, checked before any intent is emitted:

- `max_position_usd` — total exposure cap per side
- `max_daily_loss_usd` — breaching it **halts and flattens**, it does not size down
- `max_tranches` — hard ceiling on the ladder
- `allow_live` — off by default; nothing places without it being set explicitly

**On "24×7":** TQQQ and SQQQ do not trade around the clock. Robinhood's overnight
session covers a subset of tickers, and the measurement above says the extended-hours
edge is an artefact of assuming regular-session spreads overnight. Scheduling this
every minute of the night would be trading the widest spreads of the day on the
weakest evidence in the study.

---

## What is faithful in the port, and what is not

Verified by 21 tests (`tests/test_engine.py`):

- higher-timeframe bars anchor to the **session open**, not the wall clock — a
  60-minute bar runs 09:30–10:30
- leg state comes from **completed** bars only; the v6.37 `[1]` shift is
  reproduced, and skipping it makes the score lead the truth in a trend
- the score saturates at exactly ±10 and is **monotone in price**, which is what
  makes the isoline solver well defined
- exhaustion tiers are **independent one-shot machines**, re-arming only in the
  neutral band; the RSI confirmation genuinely vetoes the pullback branch
- full precision in the decision path (the v6.36 rounding leak), rounding only for
  display
- entries fill at the **next bar's open**, never the signal bar's close

Known approximations:

- `ta.ema`/`ta.rma` are seeded from the first value rather than an SMA over the
  first `length` bars. Both converge; discard early bars if it matters.
- Volume comes from XNAS.ITCH (Nasdaq only, not consolidated), so the volume-climax
  z-score is computed on a fraction of true volume. It only feeds the optional
  amplifier filter, which the sweep shows does not help.
- Stops fill **at** the level when a bar trades through it, except on a gap, which
  fills at the open. `gap_fill_share` in the output reports how often this flattered
  the result — it was 0% in the runs above.

---

## Layout

```
tqsq/bars.py       session handling, session-anchored HTF bucketing
tqsq/score.py      the composite score: leg state + f_legAt
tqsq/signals.py    zone (A/B/C) and exhaustion (1/2/3) machines
tqsq/backtest.py   three-tranche engine, stops, exits, costs
tqsq/metrics.py    stats with bootstrap CIs, random-entry baseline
tqsq/sweep.py      slippage curve, stop x target grid, split-half
tqsq/data.py       Databento fetch with a cost gate and chunked cache
tqsq/live.py       decision engine, risk gates, JSON bridge
tqsq/cli.py        cost | fetch | backtest | sweep | decide
results/           REPORT.md, grids, trade lists
```

---

## Bottom line

Over 43 sessions, entering TQQQ and SQQQ on the 1/2/3 exhaustion marks with three
tranches and a stop **did not make money after realistic costs, did not beat random
entry, and did not survive the slippage its extended-hours variant depends on.**
The tranche ladder amplified losses rather than improving the average price, and
the P&L shape — many small wins against rare 2% stops on a 3× leveraged ETF — is
the profile that ends badly when the tail arrives.

That is a small sample and it deserves the eight-year test before anyone calls it
settled. The path to that test is one `tqsq fetch` away and costs about $3. But it
would be starting from a prior that both this study and the indicator author's own
decade-scale measurement point the same way.

*Not financial advice. This is a measurement of a specific rule on a specific
sample.*
