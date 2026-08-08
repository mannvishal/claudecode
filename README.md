# spreadscout

Watches SPX 0DTE credit spreads and iron condors through the Tradier API. It
measures whether conditions actually favour selling premium, alerts you when they
do, monitors the positions you already hold, and sizes everything against your
account equity.

**It never places an order.** There is no order-placement code path anywhere in
this repository — the broker client in [`tradier.py`](spreadscout/tradier.py) is
read-only by construction. The only outbound POST in the package goes to a
webhook URL you supply.

---

```bash
spreadscout watch     # monitor open positions + alert when conditions permit entry
spreadscout regime    # what conditions look like right now, and why
spreadscout calendar  # upcoming FOMC/NFP and whether any source has gone stale
```

---

## When is "the right time"?

There is one defensible answer, and it is measurable rather than a matter of
taste: **only when implied volatility is genuinely rich relative to what the
index is actually realizing.** That gap — the variance risk premium — is the
only thing a premium seller harvests. Strike selection, delta bands and widths
shape your risk; they do not create return.

So `spreadscout` measures it rather than asking you to guess. `spreadscout
regime` reports what it sees:

```
realized (o->c)      10.07%   <- horizon-matched for 0DTE
realized (c->c)      13.72%   <- includes overnight gaps you never hold
ATM implied vol      15.20%
variance risk prem.  +10.8% (conservative reading)

ENTRY BLOCKED
  - variance risk premium is +10.8% against close-to-close realized vol, below
    the +15.0% floor (the open-to-close reading is +51.0%)
```

Those are real numbers from 67 SPX sessions. Note the spread between the two
readings — roughly a third of the index's variance arrives overnight, which a
0DTE seller never carries. Picking the denominator moves the measured premium by
5×, so **both are always reported and the gate names which one it used.** A tool
that showed you only the number that blocked would just be teaching you to
disable it.

### Entry gates

All must pass. Each is a reason *not* to trade, and the default posture is no:

| Gate | Default | Why |
|---|---|---|
| Variance risk premium | ≥ +15% | Below this there is nothing to harvest |
| VIX percentile | ≥ 20th | Don't sell vol that is already on the floor |
| Today's move | ≤ 1.25σ | Trend days are when short strikes get run over |
| Time window | 10:00–14:00 ET | Early quotes are wide; late leaves no time to manage |
| Scheduled events | High impact | See below |
| Blackout dates | — | Manual one-offs on top of the calendar |

### The economic calendar

**FOMC at 14:00 ET is the event that matters.** It lands mid-session, inside
your entry window, while the position is open. The 08:30 releases (CPI, NFP)
have already printed by the time entries open — with the default −120/+90
window an 08:30 event blocks 06:30–10:00, clear before the entry window even
starts. They matter mainly as a signal that the session will realize more
volatility than the trailing window implies.

```
$ spreadscout calendar
sources: derived, bundled-seed

  BLOCK  2026-08-07 08:30 ET -- Employment Situation (NFP, derived) <- TODAY
  BLOCK  2026-09-16 14:00 ET -- FOMC rate decision (with projections)

  + Employment Situation at 08:30 ET today; entries blocked 06:30-10:00
  no event blocks in force right now.
```

Three design rules, because the error costs here are wildly asymmetric — a
false positive costs one skipped session, a false negative puts you in a 0DTE
condor through a rate decision:

- **Over-block when unsure.** An event with no known release time blocks the
  whole day rather than being guessed at.
- **Fail closed.** If no source can confirm today is clear, entry is blocked.
  A data source returning nothing looks exactly like a genuinely empty day.
- **Hardcoded dates announce their own expiry.** The bundled seed carries a
  `verified_through` date and reports itself *stale* past it — which blocks —
  rather than implying next year has no FOMC meetings.

| Source | Coverage | Notes |
|---|---|---|
| `derived` | NFP, first Friday 08:30 | A calendar rule; never goes stale |
| `user-file` | Whatever you add | Put your CPI/PPI/PCE dates here |
| `bundled-seed` | FOMC 2026 | **Written from memory — verify it** (see below) |
| `fmp` | Full macro calendar | Needs FMP Starter tier; adapter **unverified** |

Staleness is split into *critical* (the files that carry FOMC dates — going
stale blocks) and *supplementary* (an optional vendor feed — an outage warns).
Fail-closed on a source you never depended on just teaches you to switch the
gate off, and a gate that gets switched off protects nobody.

> **Verify the seeded FOMC dates.** They were written from model memory at build
> time, not fetched — `federalreserve.gov` was unreachable from the build
> environment, and shipping an untested scraper would have been worse. Check
> them against
> <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm> and correct
> `spreadscout/data/econ_events.yaml` before you rely on it. CPI is deliberately
> *not* seeded: it lands somewhere between roughly the 10th and 15th with no
> rule tight enough to derive, and a wrong guess would block an arbitrary quiet
> day while leaving the real one open — the exact failure this gate exists to
> prevent.

### Position monitoring matters more

A missed entry costs nothing. A breached 0DTE short strike costs the width. So
positions are checked **first on every pass**, before the chain is even fetched —
a breach warning must never queue behind a slow request. There's a test that
asserts that ordering.

| Alert | Trigger |
|---|---|
| `CRITICAL` breach | Spot is through a short strike |
| `CRITICAL` delta | Short leg delta ≥ 0.45 |
| `WARN` delta | Short leg delta ≥ 0.33 |
| `WARN` proximity | Spot within 15 pts of a short strike |
| `WARN`/`CRITICAL` loss | Loss ≥ 2× the credit taken in |
| `CRITICAL` guard | Daily loss limit hit — entries off for the session |

Positions are marked at the side you'd have to cross to get out (pay the ask to
close a short, hit the bid to close a long), because the cost of escaping is the
one number you don't want flattered when deciding whether to escape.

The daily-loss guard stops new entries but **keeps monitoring what you hold** —
being down for the day is exactly when you most need the watcher and least need
another position.

Alerts go to console, a JSONL log, and optionally a Slack or Discord webhook.
Repeats inside a cooldown are suppressed, with `CRITICAL` alerts repeating 4×
sooner. A broken webhook can never silence the console — there's a test for that
too, because a watcher you believe is running and isn't is worse than none.

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

So the tool never lets that forecast be invisible. By default (`beliefs.source:
measured`) it derives the multiplier from the measured variance risk premium
described above and prints its provenance on every ticket — *"measured
realized/implied = 0.90, +0.05 haircut → 0.95"*. Set `source: manual` and it uses
your number verbatim; at the 1.00 default that means it recommends nothing at
all. **That is correct behaviour, not a bug.**

The haircut only ever makes the assumption worse, and can never push the
multiplier above 1.00 — that would be a reason to *buy* premium, not sell it.

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
pytest                        # 246 tests
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
spreadscout watch                     # the main loop: monitor positions, alert on entries
spreadscout watch --once              # a single pass, for testing your config
spreadscout regime                    # measured conditions and whether they permit entry
spreadscout calendar                  # upcoming economic events and source freshness
spreadscout scan                      # screen today's expiry, print sized tickets
spreadscout account                   # equity, open risk, daily stop status
spreadscout explain                   # the expectancy arithmetic, worked through
spreadscout backtest --start 2024-01-01
spreadscout scan --vol-multiplier 0.85   # override the measurement with your own number
```

`watch` polls every 60s by default. Run it under `screen`/`tmux` or as a systemd
unit; it survives unexpected errors rather than dying silently, because a
watcher you believe is running and isn't is worse than no watcher at all.

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

## Running the backtest against live Databento

`backtest/` needs the Databento **Python SDK**, not an MCP server — MCP tools can
only be called by an agent inside a live session, so nothing you run unattended
can use one. There is also no Databento connector in the MCP registry; I checked.

```bash
pip install -e ".[databento]"
export DATABENTO_API_KEY=db-...
spreadscout-backtest cost  --start 2026-07-01 --end 2026-07-31   # price it first
spreadscout-backtest smoke --date 2026-07-15                     # one session
spreadscout-backtest run   --start 2026-07-01 --end 2026-07-31 --out trades.csv
```

**From a Claude Code web/remote session** this needs an environment whose
network policy allows `hist.databento.com` — see
[docs/databento-setup.md](docs/databento-setup.md) for the exact steps, the real
per-session costs, and the three request bugs that only a live response could
expose. On your own machine none of the sandbox setup applies.

`api.tradier.com` is blocked by the same default policy, which is why the live
tool's Tradier access runs through an MCP server rather than direct HTTP — MCP
servers execute outside the sandbox.

## The intraday advisor

The question "have we bottomed or peaked today" is a question about the
**underlying**, not the option chain — and ES continuous minute bars cost about
$0.0014 a session against ~$17.69 for the SPXW quote chain. Signal research
therefore happens on bars; OPRA money is spent only to confirm that a validated
signal picks spreads somebody was actually willing to pay for.

```bash
# Fit the range model and prove its confidence means something, out of sample.
spreadscout-backtest calibrate --start 2023-01-01 --end 2026-08-07

# What it says about one session at one moment. Reads no bar after --at.
spreadscout-backtest advise --date 2026-08-06 --at 11:00 --train-start 2023-01-01

# Only once calibrated: check those strikes against real option quotes.
spreadscout-backtest validate --train-start 2023-01-01 \
    --start 2026-02-01 --end 2026-08-07 --days 20
```

The model does **not** predict the turn. A point call on the intraday high or
low is the least reliable thing to ask of this data and the easiest to overfit —
any rule can be tuned to catch the turns in a sample you have already seen. What
a credit spread actually needs is the distribution of the remaining-day move,
which is what picks the strike. "The low is probably in" then falls out of the
excursion distribution as a consequence rather than being asserted, and `advise`
prints it as a probability rather than a verdict.

The only claim made is a calibration claim: when the model says 95%, it should
be right 95% of the time on sessions it was not fitted on. That is falsifiable
in a way "did it call the low" is not, and `calibrate` reports it whether or not
it flatters the model.

### How far the adapter is verified

| Checked | How |
|---|---|
| `metadata.get_cost` argument set | Bound against the real signature, SDK 0.83.0 |
| `timeseries.get_range` argument set | Same |
| `to_df()` indexes on `ts_recv` | Confirmed — hence `reset_index()` |
| `to_df()` returns UTC | Confirmed — what `to_eastern` converts from |
| Float prices, `symbol` column | Pinned explicitly rather than left to defaults |
| Dataset and schemas exist | Asked of the server; free metadata calls |
| Live bar responses | Confirmed — GLBX minute bars pulled and parsed |
| **Live OPRA quote responses** | **Not yet** — no bulk chain request has been made |

`TestSdkConformance` binds argument shapes whenever the SDK is installed.
`TestVendorAgreesWithOurConstants` asks the server the questions binding cannot
answer — that is the gap that let a request for `mbp-1`, a schema OPRA does not
offer, pass 334 tests.

## Layout

| File | Role |
|---|---|
| `pricing.py` | Black-Scholes, IV solving, probability measures |
| `models.py` | Contracts, verticals, iron condors |
| `tradier.py` | REST client — read-only by construction |
| `screener.py` | Candidate construction, liquidity filters, ranking |
| `risk.py` | Expectancy identity, sizing caps, daily-loss guard |
| `regime.py` | Realized-vol measurement, variance risk premium, entry gates |
| `monitor.py` | OCC parsing, position grouping, breach and loss alerts |
| `alerts.py` | Routing, de-duplication, console/file/webhook sinks |
| `events.py` | Economic calendar, providers, fail-closed event gate |
| `watch.py` | The polling loop and its ordering guarantees |
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
