"""Backtest harness for SPX 0DTE credit spreads, sourced from Databento.

Layers are deliberately separate (rule 6):

    source.py / cache.py / data.py   DATA        fetching, cost gate, parquet, tz
    signals.py                       SIGNAL      strike selection, exit rules
    fills.py                         EXECUTION   slippage, commissions, marks

A different fill assumption means a different ``FillModel``. It does not mean
touching a signal.
"""

__all__ = []
