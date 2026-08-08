# Databento access and live-data notes

The backtest harness (`backtest/`) reaches Databento's historical API directly
through the official SDK. This document covers the sandbox setup that makes that
possible, and — more usefully — what the first live requests actually returned.

## Status

Live access is **working**. The first authenticated request was made on
2026-08-08: `metadata.list_datasets()` returned 29 datasets including
`OPRA.PILLAR`. The cost gate has been exercised against real responses.

## Environment setup (one time)

The environment selector is the cloud pill **above the message box** on a new
task at [claude.ai/code](https://claude.ai/code). It is not on the session info
popover under the session title, and there is no settings page or direct URL for
it. If the menu you opened has no **Add cloud environment** option, it is not the
selector.

Click **Add cloud environment** and fill in:

| Field | Value |
|---|---|
| Name | `databento` |
| Network access | **Custom** |
| Allowed domains | `hist.databento.com` |
| Also include default list of common package managers | **ticked** |
| Environment variables | `DATABENTO_API_KEY=db-...` |

Two things that will bite otherwise:

- **Tick the package-managers box.** Custom allows *only* what you list, so
  leaving it unchecked blocks PyPI and `pip install -e ".[databento]"` fails --
  trading one blocker for a worse one.
- **Creating a new environment beats editing `Default`.** Every other session
  uses `Default`; a separate environment keeps paid-API reach scoped to sessions
  that need it.

A session's policy is fixed when its VM is provisioned, so editing the
environment does not affect a session already running -- the change applies to
the **next** session.

Add `*.databento.com` only if you later want the live gateway. The historical API
is the sole host this harness touches.

Environment variables are visible to anyone using the environment. Fine for a
personal one; do not put the key in a shared environment.

### Confirming the policy applied

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://hist.databento.com/v0/metadata.list_datasets
```

`401` is the healthy answer: the CONNECT tunnel opened and Databento itself
replied, rejecting a curl that carries no auth header. A **`CONNECT tunnel
failed`** or a proxy `403` means the policy did not apply — that is an
environment problem, not a harness one, and nothing in the code can route
around it.

## What the first live requests revealed

Three bugs that no amount of offline testing could have caught, because each one
produces well-formed arguments that only the server can reject.

**`mbp-1` does not exist on OPRA.** It is a real Databento schema, valid on
single-venue equity datasets, and `OPRA.PILLAR` answers a request for it with
`422 dataset_schema_not_supported`. Because OPRA is a *consolidated* feed, its
top-of-book schema is **`cmbp-1`**. The book columns the quote parser reads
(`bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`) are present in both, so the
switch is confined to the constant, now `SCHEMA_QUOTES` in `config.py`.

**`SPX.OPT` and `SPXW.OPT` are different parents.** They are disjoint, not
nested: `SPX.OPT` is the AM-settled monthly root and contains none of the
weeklies that carry 0DTE. The config previously defaulted `parent_symbol` to
`SPX.OPT` while filtering the result for root `SPXW`, which would have billed a
definition pull and then discarded every row, reporting the session as "no
contracts" — indistinguishable from a market-data gap. `validate()` now rejects
the mismatch.

**Definitions must be requested over the whole UTC day.** They are a snapshot
stamped at the start of the UTC day, not an intraday stream. Asking for them
over the trading session (13:45 UTC onward) returns only definitions restated
later in the day, so the chain comes back short with no error. The SDK emits a
`BentoWarning` about this and it is correct; `definition_bounds()` handles it.

The chain pull also asked for `ALL_SYMBOLS` rather than one parent, pricing at
~$3.46 a session against ~$0.036 for SPXW alone.

## Real costs

`metadata.get_cost` is free to call. Measured for one session
(2026-08-06, 9:45–16:00 ET):

| Data | $/session | ~1yr (250d) |
|---|---|---|
| OPRA `ohlcv-1m`, full chain | $67.48 | $16,870 |
| OPRA `cbbo-1m`, full chain | $61.24 | $15,310 |
| OPRA `cmbp-1` / `cbbo-1s`, full chain | *estimate endpoint times out* | — |
| OPRA `cmbp-1`, SPXW only | $17.69 | $4,422 |
| OPRA `cbbo-1s`, SPXW only | $29.85 | $7,463 |
| OPRA `cbbo-1m`, SPXW only | **$1.13** | $283 |
| OPRA `definition`, SPXW only | $0.036 | $9 |
| CME ES `mbp-1` (`ES.c.0`) | $0.83 | $208 |
| CME ES `ohlcv-1m` | **$0.0014** | $0.35 |
| SPY `ohlcv-1m` (XNAS.BASIC) | $0.0002 | $0.05 |

Two counterintuitive entries worth internalising:

- **`cbbo-1s` costs more than `cmbp-1`** ($29.85 vs $17.69). One-second
  snapshots emit a row per contract per second whether or not anything changed;
  across thousands of SPXW strikes that exceeds the count of genuine quote
  updates. It also sits above the default `$25` ceiling.
- **The full chain is large enough that pricing it times out.** A 504 from
  `get_cost` is itself the answer: do not request that.

There is **no SPX index feed** among the 29 entitled datasets. The underlying
comes from ES futures, SPY, or `spot_from_parity` on the chain itself.

## Cost discipline

`cost` prices the two requests the engine actually issues per session — the
definition file for one parent, and the quote pull for the full root. The quote
figure is an **upper bound**: real pulls are narrowed to a strike band whose
size is unknowable until the definitions are in hand.

An estimate above the `$25` ceiling is the gate working, not a failure;
`--ceiling` raises it deliberately. `cost.require_estimate` (default on) refuses
to pull anything the endpoint could not price, because an unknown cost is not a
zero cost. A session that could not be priced is reported separately and
excluded from the total rather than silently counted as free.

Research the *timing* signal on ES minute bars, not OPRA. "Have we bottomed
today" is a question about the underlying; at $0.0014 a session, several years
of it costs about a dollar. Spend on OPRA only to confirm that the spreads a
validated signal selects were genuinely sellable.

## What is verified, and what is not

`TestSdkConformance` binds the adapter's arguments against the installed SDK.
`TestVendorAgreesWithOurConstants` asks the server the questions binding cannot
answer — that the dataset exists and that both schemas are offered on it. Those
are free metadata calls, skipped without an API key.

Still **not** verified: the shape of a real bulk response. Row counts under live
OPRA data, whether `cmbp-1` frames parse cleanly through `QuoteBook`, and
whether the strike-band pull is affordable in practice. No bulk `timeseries`
request has been made yet. Treat the first one as an experiment, not a backtest.

## On your own machine

None of the sandbox setup applies:

```bash
pip install -e ".[databento]"
export DATABENTO_API_KEY=db-...
spreadscout-backtest cost --start 2026-07-01 --end 2026-07-31
```

The SDK is the right data layer regardless -- an MCP tool can only be called by
an agent inside a live session, so nothing running unattended could use one.
There is also no Databento connector in the MCP registry; that was checked
directly.
