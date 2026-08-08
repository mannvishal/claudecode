# Enabling live Databento access

The backtest harness (`backtest/`) is complete and tested, but has **never made a
live request**. Outbound access to `hist.databento.com` is refused by the cloud
sandbox's egress proxy:

```
{"kind": "connect_rejected", "host": "hist.databento.com:443",
 "detail": "gateway answered 403 to CONNECT (policy denial...)"}
```

This is an environment network policy, not a bug in the harness. A session's
policy is fixed when its VM is provisioned, so editing the environment does not
affect a session already running -- the change applies to the **next** session.

## Setup (one time)

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

Add `*.databento.com` only if you later want the live gateway. The historical API
is the sole host this harness touches.

Environment variables are visible to anyone using the environment. Fine for a
personal one; do not put the key in a shared environment.

## Verification order for the next session

Do these in order. Do not skip to `run`.

```bash
# 1. Did the policy actually apply? Expect 2xx/4xx, NOT "CONNECT tunnel failed".
curl -sS -o /dev/null -w '%{http_code}\n' https://hist.databento.com/v0/metadata.list_datasets

# 2. First live exercise of the adapter. Free -- get_cost bills nothing.
pip install -e ".[databento]"
python -c "
import databento as db
print(db.Historical().metadata.list_datasets()[:5])
"

# 3. Price before pulling. This is the first time the cost gate meets a real response.
spreadscout-backtest cost --start 2026-07-01 --end 2026-07-31

# 4. Only then, one session end to end.
spreadscout-backtest smoke --date 2026-07-15
```

Step 3 matters more than it looks. `cost` prices the **unfiltered** chain, and
full-chain OPRA `mbp-1` is enormous -- the engine narrows each real pull to a
strike band around the money, which is far cheaper, but the unfiltered number is
the one that tells you whether your OPRA entitlement makes this affordable at
all. An estimate above the `$25` default ceiling is the gate working, not a
failure; `--ceiling` raises it deliberately.

## What is already verified, and what is not

`TestSdkConformance` binds the adapter's arguments against the real SDK (0.83.0)
and runs whenever `databento` is installed:

- `metadata.get_cost` and `timeseries.get_range` accept the argument set built by
  `DatabentoFetcher.request_kwargs`
- `to_df()` indexes on `ts_recv`, hence the `reset_index()`
- `to_df()` returns UTC, which is what `to_eastern` converts from
- float prices and the `symbol` column are pinned explicitly rather than relying
  on SDK defaults

Not verified: **anything about a live response.** Row counts, schema drift under
real OPRA data, whether `get_cost` prices a symbol-filtered request the way the
engine assumes, and whether the strike-band pull is actually affordable. Treat
the first live run as an experiment, not a backtest.

## If you would rather not touch the sandbox

None of this applies on your own machine:

```bash
pip install -e ".[databento]"
export DATABENTO_API_KEY=db-...
spreadscout-backtest cost --start 2026-07-01 --end 2026-07-31
```

The SDK is the right data layer for the harness regardless -- an MCP tool can
only be called by an agent inside a live session, so nothing running unattended
could use one. There is also no Databento connector in the MCP registry; that was
checked directly.
