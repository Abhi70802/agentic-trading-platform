"""Quantitative historical analogue search with strictly forward outcome windows."""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import mean, median

from core import Candle
from features import feature_snapshot


def historical_similarity(
    candles: list[Candle],
    *,
    horizon_bars: int = 5,
    top_k: int = 20,
    max_distance: float = 0.35,
    min_history_bars: int = 50,
) -> dict:
    if horizon_bars < 1 or top_k < 1 or min_history_bars < 50:
        raise ValueError("Horizon and result limit must be positive; at least 50 history bars are required")
    if not isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Maximum feature distance must be finite and positive")
    if not candles:
        raise ValueError("At least one candle is required")

    ordered = sorted(candles, key=lambda candle: candle.timestamp)
    if len({candle.symbol for candle in ordered}) != 1:
        raise ValueError("Historical similarity accepts exactly one instrument")
    if len({candle.timeframe for candle in ordered}) != 1:
        raise ValueError("Historical similarity accepts exactly one timeframe")
    if len({candle.timestamp for candle in ordered}) != len(ordered):
        raise ValueError("Historical similarity requires unique candle timestamps")

    current_snapshot = feature_snapshot(ordered)
    current_vector = _feature_vector(current_snapshot)
    if current_vector is None:
        raise ValueError("Current feature history is insufficient for similarity analysis")

    candidates = []
    final_candidate_index = len(ordered) - horizon_bars - 1
    for index in range(min_history_bars - 1, final_candidate_index + 1):
        observed_history = ordered[max(0, index - 49):index + 1]
        snapshot = feature_snapshot(observed_history)
        historical_vector = _feature_vector(snapshot)
        if historical_vector is None:
            continue
        distance = _distance(current_vector, historical_vector)
        if distance > max_distance:
            continue

        entry = ordered[index].close
        future = ordered[index + 1:index + horizon_bars + 1]
        future_return_pct = (future[-1].close / entry - 1) * 100
        mae_pct = (min(candle.low for candle in future) / entry - 1) * 100
        mfe_pct = (max(candle.high for candle in future) / entry - 1) * 100
        candidates.append({
            "reference_timestamp": ordered[index].timestamp,
            "feature_distance": round(distance, 6),
            "forward_return_pct": round(future_return_pct, 6),
            "maximum_adverse_excursion_pct": round(mae_pct, 6),
            "maximum_favorable_excursion_pct": round(mfe_pct, 6),
            "holding_period_bars": horizon_bars,
            "regime": snapshot["regime"]["trend"],
        })

    candidates.sort(key=lambda item: (item["feature_distance"], item["reference_timestamp"]))
    matches = candidates[:top_k]
    returns = [item["forward_return_pct"] for item in matches]
    maes = [item["maximum_adverse_excursion_pct"] for item in matches]
    mfes = [item["maximum_favorable_excursion_pct"] for item in matches]
    return {
        "symbol": ordered[-1].symbol,
        "timeframe": ordered[-1].timeframe,
        "as_of": ordered[-1].timestamp,
        "horizon_bars": horizon_bars,
        "top_k": top_k,
        "max_feature_distance": max_distance,
        "sample_count": len(matches),
        "win_rate_pct": round(sum(value > 0 for value in returns) / len(returns) * 100, 4) if returns else None,
        "average_return_pct": round(mean(returns), 6) if returns else None,
        "median_return_pct": round(median(returns), 6) if returns else None,
        "return_distribution_pct": _distribution(returns),
        "average_mae_pct": round(mean(maes), 6) if maes else None,
        "average_mfe_pct": round(mean(mfes), 6) if mfes else None,
        "average_holding_period_bars": horizon_bars if matches else None,
        "matches": matches,
        "source": "deterministic_ohlcv_feature_similarity",
        "is_calibrated_probability": False,
        "execution_authority": False,
    }


def _feature_vector(snapshot: dict) -> dict | None:
    features = snapshot["features"]
    keys = ("rsi_14", "atr_pct", "momentum_20_pct", "volume_zscore_20")
    values = {key: features.get(key) for key in keys}
    if any(value is None or not isfinite(value) for value in values.values()):
        return None
    return {
        "rsi": values["rsi_14"] / 100,
        "atr_pct": min(max(values["atr_pct"], 0), 10) / 10,
        "momentum_pct": min(max(values["momentum_20_pct"], -20), 20) / 40,
        "volume_zscore": min(max(values["volume_zscore_20"], -5), 5) / 10,
        "trend": snapshot["regime"]["trend"],
    }


def _distance(left: dict, right: dict) -> float:
    numeric_keys = ("rsi", "atr_pct", "momentum_pct", "volume_zscore")
    squared = sum((left[key] - right[key]) ** 2 for key in numeric_keys)
    trend_penalty = 0.5 if left["trend"] != right["trend"] else 0.0
    return sqrt((squared + trend_penalty**2) / (len(numeric_keys) + 1))


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return round(ordered[index], 6)

    return {
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
    }