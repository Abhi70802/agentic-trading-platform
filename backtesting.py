"""Leakage-conscious expanding-window backtest orchestration."""

from __future__ import annotations

from collections.abc import Callable

from core import Candle, RiskEngine, SimulationConfig, Strategy, run_simulation


def run_walk_forward(
    candles: list[Candle],
    strategy_factory: Callable[[], Strategy],
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    config: SimulationConfig | None = None,
    risk_engine: RiskEngine | None = None,
) -> dict:
    if train_bars < 2 or test_bars < 2:
        raise ValueError("Walk-forward training and test windows must contain at least two bars")
    step = test_bars if step_bars is None else step_bars
    if step < test_bars:
        raise ValueError("Walk-forward steps must not overlap out-of-sample test windows")
    if len(candles) < train_bars + 2:
        raise ValueError("Not enough candles for the requested training and test windows")
    if any(candles[index].timestamp >= candles[index + 1].timestamp for index in range(len(candles) - 1)):
        raise ValueError("Candles must be strictly chronological")
    if len({candle.symbol for candle in candles}) != 1:
        raise ValueError("Walk-forward replay accepts exactly one instrument")
    if len({candle.timeframe for candle in candles}) != 1:
        raise ValueError("Walk-forward replay accepts exactly one timeframe")

    folds = []
    fold_number = 0
    test_start = train_bars
    while test_start < len(candles) - 1:
        test_end = min(test_start + test_bars, len(candles))
        if test_end - test_start < 2:
            break
        replay = run_simulation(
            candles[:test_end],
            strategy_factory(),
            config=config,
            risk_engine=risk_engine,
            mode="backtest",
            evaluation_start_index=test_start,
            liquidate_at_end=True,
        )
        fold_number += 1
        folds.append({
            "fold": fold_number,
            "training_bars": test_start,
            "test_bars": test_end - test_start,
            "training_end": candles[test_start - 1].timestamp,
            "test_start": candles[test_start].timestamp,
            "test_end": candles[test_end - 1].timestamp,
            "return_pct": replay["return_pct"],
            "max_drawdown_pct": replay["max_drawdown_pct"],
            "closed_trades": replay["closed_trades"],
            "risk_rejections": replay["risk_rejections"],
            "partial_fills": replay["partial_fills"],
            "liquidity_rejections": replay["liquidity_rejections"],
            "open_quantity": replay["open_quantity"],
            "cost_model": replay["cost_model"],
            "slippage_bps": replay["slippage_bps"],
            "market_impact_bps": replay["market_impact_bps"],
            "execution_delay_bars": replay["execution_delay_bars"],
            "ending_equity": replay["ending_equity"],
            "trades": replay["trades"],
        })
        test_start += step

    if not folds:
        raise ValueError("No complete out-of-sample fold could be constructed")
    return {
        "symbol": candles[0].symbol,
        "timeframe": candles[0].timeframe,
        "method": "EXPANDING_WINDOW_WALK_FORWARD",
        "training_bars_initial": train_bars,
        "test_bars_requested": test_bars,
        "step_bars": step,
        "fold_count": len(folds),
        "total_test_bars": sum(fold["test_bars"] for fold in folds),
        "total_closed_trades": sum(fold["closed_trades"] for fold in folds),
        "total_partial_fills": sum(fold["partial_fills"] for fold in folds),
        "total_liquidity_rejections": sum(fold["liquidity_rejections"] for fold in folds),
        "mean_fold_return_pct": round(sum(fold["return_pct"] for fold in folds) / len(folds), 6),
        "worst_fold_drawdown_pct": max(fold["max_drawdown_pct"] for fold in folds),
        "folds": folds,
        "compounds_fold_returns": False,
    }
