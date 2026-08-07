# spreadscout

A screener for SPX 0DTE credit spreads and iron condors. It reads live chains
from Tradier, recomputes the greeks itself, sizes positions against your account
equity, and prints order tickets for you to review.

**It never places an order.** There is no order-placement code path anywhere in
this repository — see [`tradier.py`](spreadscout/tradier.py).

---

## Read this before the install instructions

This tool was built to answer "can I make 1% a day selling credit spreads?" The
honest answer is no, and the tool is designed to keep showing you why rather
than to hide it.

**1%/day compounds to roughly +1,100%/year.** Nothing in this strategy produces
that. What the strategy does produce is a very high win rate, which feels like
the same thing and is not.

Here is the arithmetic the tool is built around:

> A credit spread's expected value under the market's own implied distribution
> is **exactly zero before costs**, and negative after them.

That is not a modelling opinion. `spreadscout` solves each leg's implied vol
from its own live mid, so the model reprices the market exactly; selling at the
mid and valuing at the mid is a wash, and commissions and slippage come straight
off the top. This is asserted as an executable test — see
`TestExpectancyIdentity` in [`tests/test_risk.py`](tests/test_risk.py).

Run against a real SPX 0DTE chain, with `vol_multiplier: 1.00`:

```
At vol_multiplier=1.0,  candidates with positive EV after costs:  0 of 39
At vol_multiplier=0.85, candidates with positive EV after costs: 39 of 39
```

Every dollar of expected profit in this strategy comes from that second line —
from *your forecast* that realized volatility will come in below implied. It
does not come from the strike selection, the delta band, the width, or the
screener. Those change your risk profile. They do not create edge.

So the tool makes you state the forecast explicitly, in `beliefs.vol_multiplier`,
and reports expectancy under both measures side by side. At the default of 1.00
it will recommend nothing at all. **That is correct behaviour, not a bug.**

### Why the win rate is the trap

The top-ranked candidate from that live chain — a 7770/7780 call credit spread —
collects **$101** and risks **$899**, with an **87.8%** chance of keeping the
credit. That feels like an edge. It isn't:

```
0.878 × $101  −  0.122 × $899  =  −$21.0 per trade
```

You need to win **89.9%** of the time (`899 / (101 + 899)`) just to break even
before costs, and the market is offering you 87.8%. The high
probability of profit is not evidence of an edge — it is the price you are being
paid for carrying the tail. Traders who set a fixed daily percentage target end
up doing the two things that convert this from break-even to ruinous: sizing up
after wins, and refusing to close losers because booking the loss ruins the
day's number. One gap erases months.

`spreadscout` therefore ships with `require_positive_ev: true`, a per-trade risk
cap, a portfolio risk cap, and a daily-loss circuit breaker that refuses to
recommend anything once you're down for the day.

### What the tool is actually good for

Systematic strike selection, honest expectancy accounting, correct position
sizing, and hard limits — the things discretionary spread traders do worst by
hand. If you are going to trade this, trading it with these guard rails is
meaningfully better than trading it without them. That is the real value on
offer, and it is not 1% a day.

---

## Two things it gets right that most screeners don't

**1. It ignores the vendor's greeks.** Tradier's chain carries greeks from a
periodic ORATS batch. On a live SPX 0DTE chain the timestamps were from the
previous evening — older than the entire remaining life of the contracts. The
ATM implied vol it reported was **14.3%** against **29.5%** recomputed from the
live mid. Screening 0DTE strikes on those deltas would be selecting a different
option than the one you trade. `spreadscout` backs implied vol out of the
current mid and rebuilds every greek from it.

SPX options are European-exercise and cash-settled, so Black-Scholes is the
*exact* model here, not an approximation — there is no early-exercise premium
and no discrete dividend. (For American-style equity options it would be an
approximation, which is one reason this defaults to SPX.)

**2. It knows 0DTE is a different scale, not just a shorter one.** One standard
deviation of SPX at 0DTE is about 25-55 index points depending on vol. Filters
and skew models carried over from 45-DTE intuition are off by an order of
magnitude. Skew is fitted in standard deviations, not percent; settlement time
is read from `root_symbol` (SPXW settles PM, monthly SPX settles AM — a full
trading day of time value); and iron condor margin uses the wider wing rather
than the sum, which is correct only because European cash settlement means one
side cannot be assigned while the other is still open.

---

## Install

```bash
pip install -e ".[dev]"
cp .env.example .env          # add your Tradier token
cp config.example.yaml spreadscout.yaml
pytest                        # 103 tests
```

Get a token at <https://dash.tradier.com/settings/api>. Options chains with
greeks are free with a funded brokerage account. `.env` and `spreadscout.yaml`
are both gitignored.

> **On Databento:** you don't need it for this. Databento's *equities* feed
> contains no options data at all, and its OPRA feed is a subscription plan plus
> exchange license fees. Tradier already provides the chains, greeks, and quotes
> this tool requires at no additional cost. Databento would only be worth it if
> you later want tick-level underlying data for a serious intraday backtest.

## Use

```bash
spreadscout scan                      # screen today's expiry, print sized tickets
spreadscout scan --top 10 --show-rejected
spreadscout explain                   # the expectancy arithmetic, worked through
spreadscout account                   # equity, open risk, daily stop status
spreadscout backtest --start 2024-01-01
spreadscout scan --vol-multiplier 0.85   # state a forecast and see what changes
```

`scan` warns when the market is closed — quotes outside session hours are stale
closing prints and the modelled fills are not achievable against them.

## About the backtest

It is a **model**, not a record. There is no intraday historical options data
wired in, and for 0DTE that gap is fatal: an end-of-day chain snapshot on expiry
day is taken *at* settlement, so it cannot tell you what you'd have been filled
at in the morning. Rather than invent entry prices and call the result
validated, `backtest.py` is explicit that it models entries — strikes by
Black-Scholes delta, credits from a smile calibrated off a live chain today and
assumed stationary, with the vol level driven by each day's VIX close.
Settlement is the underlying's real close.

Every error runs the same direction: real fills are worse, real chains gap, and
no model here can halt trading or blow through a strike between prints. Treat
the output as an **upper bound**. An upper bound that already looks unattractive
is still worth knowing.

## Layout

| File | Role |
|---|---|
| `pricing.py` | Black-Scholes, IV solving, probability measures |
| `models.py` | Contracts, verticals, iron condors |
| `tradier.py` | REST client — read-only by construction |
| `screener.py` | Candidate construction, liquidity filters, ranking |
| `risk.py` | Expectancy identity, sizing caps, daily-loss guard |
| `backtest.py` | Modelled historical replay |
| `cli.py` | Commands |

## If you want it to trade

Adding execution is a small change — Tradier's multileg endpoint takes exactly
the tickets this already prints. Before doing it, decide deliberately: the guard
rails in `risk.py` are advisory when a human reads a ticket and load-bearing
when a loop submits one. An unattended bug in a 0DTE strategy has until 4pm to
compound.

## Licence

MIT. No warranty. Nothing here is financial advice, and you can lose more than
these numbers suggest.
