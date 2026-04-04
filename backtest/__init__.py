"""
Unified Backtest Package
=========================
Usage:
    from backtest import BacktestEngine, DataBundle, run_variant, calc_stats
    from backtest.runner import C12_KWARGS, V3_REGIME

Backward compatibility:
    Old code doing `from backtest import Position, TradeRecord, load_csv, _aggregate_daily_pnl`
    continues to work — these are re-exported here.
"""
from backtest.engine import BacktestEngine, Position, TradeRecord
from backtest.stats import (
    calc_stats, aggregate_daily_pnl, print_comparison, print_ranked,
    probabilistic_sharpe, deflated_sharpe, compute_pbo,
)
from backtest.runner import (
    DataBundle, run_variant, run_variants, run_kfold,
    load_csv, load_m15, load_h1_aligned, add_atr_percentile,
    prepare_indicators_custom, sanitize_for_json,
    C12_KWARGS, V3_REGIME, TRUE_BASELINE_KWARGS,
    M15_CSV_PATH, H1_CSV_PATH,
)

# Backward compatibility alias for old code that imports _aggregate_daily_pnl
_aggregate_daily_pnl = aggregate_daily_pnl
