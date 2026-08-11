"""Tests for the invariants that make the port trustworthy.

These are deliberately about *fidelity and safety*, not about performance. A test
that pins a return number would just freeze whatever the last sweep produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tqsq.backtest import BacktestConfig, run_backtest
from tqsq.bars import SESSIONS, aggregate, bucket_ids
from tqsq.score import LEG_TFS, Weights, compute_score, leg_score_at, leg_state
from tqsq.signals import SignalConfig, build_signals


def synth(n_days: int = 6, seed: int = 3, session: str = "rth") -> pd.DataFrame:
    """A synthetic 1-minute tape with real session boundaries."""
    lo, hi = SESSIONS[session]
    rng = np.random.default_rng(seed)
    rows = []
    price = 70.0
    for d in range(n_days):
        day = pd.Timestamp("2026-06-01", tz="America/New_York") + pd.Timedelta(days=d)
        for m in range(lo, hi):
            price *= 1 + rng.normal(0, 0.0006)
            hi_p = price * (1 + abs(rng.normal(0, 0.0004)))
            lo_p = price * (1 - abs(rng.normal(0, 0.0004)))
            rows.append(
                {
                    "ts": day + pd.Timedelta(minutes=m),
                    "date": day,
                    "minute": m,
                    "open": price,
                    "high": hi_p,
                    "low": lo_p,
                    "close": price,
                    "volume": float(rng.integers(1_000, 50_000)),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["anchor"] = lo
    return df


# --------------------------------------------------------------------------
# Bucketing: TradingView anchors higher-timeframe bars to the SESSION open.
# --------------------------------------------------------------------------

def test_buckets_anchor_to_session_open_not_the_clock():
    df = synth(1)
    b = bucket_ids(df, 60, anchor=SESSIONS["rth"][0])
    first_boundary = int(np.argmax(b > b[0]))
    # 09:30 + 60 minutes = 10:30, i.e. 60 rows in -- not at the top of the hour.
    assert df["minute"].iloc[first_boundary] == 10 * 60 + 30


def test_buckets_are_monotonic_and_restart_each_day():
    df = synth(3)
    b = bucket_ids(df, 15, anchor=SESSIONS["rth"][0])
    assert np.all(np.diff(b) >= 0)
    per_day = df.groupby("date").size().iloc[0]
    assert len(set(b[:per_day])) == per_day / 15


# --------------------------------------------------------------------------
# Splits. Databento's ohlcv-1m is as-traded, and both instruments have split --
# TQQQ 2:1 three times, SQQQ 1-for-5 (and once 1-for-4) five times. Left raw,
# each one is a several-hundred-percent overnight bar the strategy would trade.
# --------------------------------------------------------------------------

def _with_split(df: pd.DataFrame, on_day: int, ratio: float) -> pd.DataFrame:
    """Un-adjust a clean frame: divide everything from `on_day` onward."""
    df = df.copy()
    days = sorted(df["date"].unique())
    later = df["date"] >= days[on_day]
    for col in ("open", "high", "low", "close"):
        df.loc[later, col] = df.loc[later, col] / ratio
    return df


def test_detects_a_forward_split():
    from tqsq.bars import detect_splits

    raw = _with_split(synth(6), on_day=3, ratio=2.0)  # TQQQ-style 2:1
    hits = detect_splits(raw)
    assert len(hits) == 1
    assert hits["ratio"].iloc[0] == pytest.approx(0.5)


def test_detects_a_reverse_split():
    from tqsq.bars import detect_splits

    raw = _with_split(synth(6), on_day=3, ratio=1 / 5)  # SQQQ-style 1-for-5
    hits = detect_splits(raw)
    assert len(hits) == 1
    assert hits["ratio"].iloc[0] == pytest.approx(5.0)


def test_split_ratio_snaps_to_a_standard_value():
    """The observed gap carries that night's real move; snapping keeps the move
    in the tape instead of burying it in the adjustment factor."""
    from tqsq.bars import detect_splits

    # A 1-for-4 reverse split on a -2.7% night, which is SQQQ 2019-05-24.
    raw = _with_split(synth(6), on_day=3, ratio=1 / 4)
    days = sorted(raw["date"].unique())
    raw.loc[raw["date"] >= days[3], ["open", "high", "low", "close"]] *= 0.973
    hits = detect_splits(raw)
    assert hits["ratio"].iloc[0] == pytest.approx(4.0), "must not snap to 5"
    assert hits["observed"].iloc[0] != pytest.approx(4.0)


def test_adjustment_removes_the_artificial_gap():
    from tqsq.bars import apply_split_adjustment, detect_splits

    clean = synth(6)
    raw = _with_split(clean, on_day=3, ratio=2.0)
    fixed = apply_split_adjustment(raw, detect_splits(raw))
    assert detect_splits(fixed).empty
    # Returns are what the score reads, and they must match the unsplit tape.
    np.testing.assert_allclose(
        fixed["close"].pct_change().dropna().to_numpy(),
        clean["close"].pct_change().dropna().to_numpy(),
        atol=1e-9,
    )


def test_a_violent_but_real_gap_is_not_treated_as_a_split():
    """A 3x ETF gapped 17.5% on 2020-03-13. The threshold must clear that."""
    from tqsq.bars import detect_splits

    raw = _with_split(synth(6), on_day=3, ratio=1 / 1.175)
    assert detect_splits(raw).empty


def test_aggregate_preserves_ohlc_semantics():
    df = synth(1)
    b = bucket_ids(df, 5, anchor=SESSIONS["rth"][0])
    htf = aggregate(df, b)
    assert htf["high"].iloc[0] == df["high"].iloc[:5].max()
    assert htf["low"].iloc[0] == df["low"].iloc[:5].min()
    assert htf["close"].iloc[0] == df["close"].iloc[4]
    assert htf["volume"].iloc[0] == df["volume"].iloc[:5].sum()


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------

def test_score_is_bounded_by_ten():
    sf = compute_score(synth(8))
    s = sf.avg[np.isfinite(sf.avg)]
    assert len(s) > 0
    assert s.min() >= -10.0001 and s.max() <= 10.0001


def test_score_saturates_at_the_extremes():
    """Every component agreeing must give exactly +/-10, since the weights sum
    to the divisor. This is what makes 8.5 a meaningful fraction of the range."""
    df = synth(8)
    st = leg_state(df, 5, SESSIONS["rth"][0])
    i = len(df) - 1
    one = type(st)(**{k: np.asarray([getattr(st, k)[i]]) for k in st.__dataclass_fields__})
    assert leg_score_at(np.asarray([1e6]), one)[0] == pytest.approx(10.0)
    assert leg_score_at(np.asarray([1e-6]), one)[0] == pytest.approx(-10.0)


def test_score_is_monotone_in_price():
    """The isoline solver bisects on this. If it were not monotone the level
    would be ambiguous and the cloud meaningless."""
    df = synth(8)
    st = leg_state(df, 15, SESSIONS["rth"][0])
    i = len(df) - 1
    one = type(st)(**{k: np.asarray([getattr(st, k)[i]]) for k in st.__dataclass_fields__})
    grid = np.linspace(df["close"].iloc[i] * 0.95, df["close"].iloc[i] * 1.05, 60)
    vals = [leg_score_at(np.asarray([p]), one)[0] for p in grid]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_leg_state_uses_only_completed_bars():
    """The first row of each higher-timeframe bucket must not see its own bar."""
    df = synth(4)
    st = leg_state(df, 60, SESSIONS["rth"][0])
    b = bucket_ids(df, 60, SESSIONS["rth"][0])
    # Row 0 is in bucket 0, which has no predecessor: state must be undefined.
    assert not np.isfinite(st.close[0])
    first_of_second = int(np.argmax(b == 1))
    htf = aggregate(df, b)
    assert st.close[first_of_second] == pytest.approx(htf["close"].iloc[0])


def test_weights_change_the_score():
    df = synth(6)
    a = compute_score(df, weights=Weights())
    b = compute_score(df, weights=Weights(rsi=0.0))
    assert not np.allclose(np.nan_to_num(a.avg), np.nan_to_num(b.avg))


# --------------------------------------------------------------------------
# Machines
# --------------------------------------------------------------------------

def _machine_inputs(n=400):
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "ts": ts,
            "date": ts.normalize(),
            "minute": ts.hour * 60 + ts.minute,
            "open": 70.0, "high": 70.1, "low": 69.9, "close": 70.0, "volume": 1000.0,
        }
    )


def test_exhaustion_is_one_shot_per_episode():
    """A tier must fire at most once between neutral-band resets."""
    df = _machine_inputs()
    n = len(df)
    score = np.full(n, 9.0)          # parked deep in the top band ...
    score[:20] = 0.0                 # ... after starting neutral
    rsi = np.full(n, 85.0)
    sg = build_signals(df, score, rsi, SignalConfig())
    assert sg.exh_top[1].sum() == 1, "stall branch must latch after one fire"


def test_exhaustion_rearms_only_after_neutral():
    df = _machine_inputs()
    n = len(df)
    score = np.zeros(n)
    score[20:120] = 9.0    # episode 1
    score[120:200] = 0.0   # back to neutral -> re-arm
    score[200:300] = 9.0   # episode 2
    sg = build_signals(df, score, np.full(n, 85.0), SignalConfig())
    assert sg.exh_top[1].sum() == 2


def test_pullback_branch_needs_rsi_confirmation():
    df = _machine_inputs()
    n = len(df)
    score = np.zeros(n)
    score[20:40] = 9.0
    score[40:60] = 7.0   # a 2.0 pullback, larger than rev_delta 1.5
    hot = build_signals(df, score, np.full(n, 85.0), SignalConfig(hold_bars=10_000))
    cold = build_signals(df, score, np.full(n, 50.0), SignalConfig(hold_bars=10_000))
    assert hot.exh_top[1].sum() == 1
    assert cold.exh_top[1].sum() == 0, "RSI below the OB level must veto the fire"


def test_tiers_are_independent_machines():
    df = _machine_inputs()
    n = len(df)
    score = np.zeros(n)
    score[20:200] = 8.6   # deep enough for all three tiers
    sg = build_signals(df, score, np.full(n, 85.0), SignalConfig())
    tiers = sg.exhaustion_tier("top")
    assert sg.exh_top[1].sum() == 1 and sg.exh_top[3].sum() == 1
    assert tiers.max() == 3, "the reported tier is the deepest firing on the bar"


def test_window_gate_blocks_marks_outside_the_session():
    ts = pd.date_range("2026-06-01 02:00", periods=120, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {"ts": ts, "date": ts.normalize(), "minute": ts.hour * 60 + ts.minute,
         "open": 70.0, "high": 70.1, "low": 69.9, "close": 70.0, "volume": 1000.0}
    )
    score = np.full(len(df), 9.0)
    score[:5] = 0.0
    rsi = np.full(len(df), 85.0)
    assert build_signals(df, score, rsi, SignalConfig(eth_marks=False)).exh_top[1].sum() == 0
    assert build_signals(df, score, rsi, SignalConfig(eth_marks=True)).exh_top[1].sum() == 1


# --------------------------------------------------------------------------
# Backtest safety
# --------------------------------------------------------------------------

def test_entry_fills_on_the_next_bar_open_not_the_signal_close():
    """No-lookahead. A fire on bar i must never transact at bar i's price."""
    df = synth(6)
    sf = compute_score(df)
    sg = build_signals(df, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=False))
    res = run_backtest(df, sg, df, df, BacktestConfig(slippage_bps=0.0))
    fires = np.flatnonzero((sg.exhaustion_tier("bot") > 0) | (sg.exhaustion_tier("top") > 0))
    if not len(fires) or not res.trades:
        pytest.skip("synthetic tape produced no fires")

    # Each entry must sit exactly one bar after a fire, and must transact at
    # that later bar's open. Comparing timestamp *sets* would not catch a
    # lookahead, because a fire can land on the same bar another fire fills on.
    ts_to_row = {t: i for i, t in enumerate(df["ts"])}
    fire_rows = set(fires.tolist())
    for t in res.trades:
        row = ts_to_row[t.entry_ts]
        assert row - 1 in fire_rows, "entry is not one bar after its fire"
        if t.tranches == 1:
            # Multi-tranche trades blend several fills, so only a single-tranche
            # position pins directly to one bar's open.
            assert t.avg_cost == pytest.approx(df["open"].iloc[row]), "filled at the wrong bar"


def test_stop_caps_the_loss_on_every_trade():
    df = synth(10)
    sf = compute_score(df)
    sg = build_signals(df, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=False))
    stop = 0.015
    res = run_backtest(df, sg, df, df, BacktestConfig(stop_pct=stop, slippage_bps=0.0))
    for t in res.trades:
        if t.reason == "stop" and not t.gap_fill:
            assert t.ret >= -stop - 1e-6


def test_never_exceeds_max_tranches():
    df = synth(10)
    sf = compute_score(df)
    sg = build_signals(df, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=False))
    res = run_backtest(df, sg, df, df, BacktestConfig(max_tranches=3))
    assert all(t.tranches <= 3 for t in res.trades)


def test_min_tier_filter_only_removes_trades():
    df = synth(12)
    sf = compute_score(df)
    sg = build_signals(df, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=False))
    n1 = len(run_backtest(df, sg, df, df, BacktestConfig(min_tier=1)).trades)
    n3 = len(run_backtest(df, sg, df, df, BacktestConfig(min_tier=3)).trades)
    assert n3 <= n1


def test_slippage_monotonically_reduces_pnl():
    df = synth(12)
    sf = compute_score(df)
    sg = build_signals(df, sf.avg, sf.avg_rsi, SignalConfig(eth_marks=False))
    pnls = [
        sum(t.pnl for t in run_backtest(df, sg, df, df, BacktestConfig(slippage_bps=b)).trades)
        for b in (0.0, 5.0, 20.0)
    ]
    assert pnls[0] >= pnls[1] >= pnls[2]


# --------------------------------------------------------------------------
# Live risk gates
# --------------------------------------------------------------------------

def test_daily_loss_limit_halts_and_flattens():
    from tqsq.live import EngineState, PositionState, RiskLimits, decide

    df = synth(6)
    state = EngineState(long=PositionState("TQQQ", shares=10, avg_cost=70.0), realized_pnl_today=-600.0)
    intents, diag = decide(df, state, RiskLimits(max_daily_loss_usd=500.0),
                           quotes={"TQQQ": 70.0, "SQQQ": 42.0})
    assert diag["halted"] is True
    assert [i.action for i in intents] == ["close"]


def test_no_live_intent_without_a_quote_for_that_leg():
    """A missing SQQQ quote must skip the order and say so, never size off TQQQ."""
    from tqsq.live import EngineState, RiskLimits, decide

    df = synth(6)
    intents, diag = decide(df, EngineState(), RiskLimits(), quotes={})
    assert all(i.symbol != "SQQQ" for i in intents)
    if diag["exhaustion_top_tier"] > 0:
        assert any("SQQQ" in w for w in diag.get("warnings", []))


def test_live_position_cap_is_respected():
    from tqsq.live import EngineState, PositionState, RiskLimits, decide

    df = synth(6)
    full = EngineState(long=PositionState("TQQQ", shares=200, avg_cost=50.0, tiers_filled=[1]))
    intents, _ = decide(df, full, RiskLimits(max_position_usd=10_000.0),
                        quotes={"TQQQ": 50.0, "SQQQ": 42.0})
    assert all(not (i.symbol == "TQQQ" and i.side == "buy") for i in intents)
