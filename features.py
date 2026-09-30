"""Deterministic quantitative features computed from normalized OHLCV bars."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from uuid import uuid4

from core import Candle


def feature_snapshot(candles: list[Candle]) -> dict:
    if not candles:
        raise ValueError("At least one candle is required")
    ordered = sorted(candles, key=lambda item: item.timestamp)
    if len({item.symbol for item in ordered}) != 1:
        raise ValueError("Feature snapshots accept exactly one instrument")
    closes = [item.close for item in ordered]
    volumes = [item.volume for item in ordered]
    current = ordered[-1]
    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    macd = ema_12 - ema_26 if ema_12 is not None and ema_26 is not None else None
    atr_14 = _atr(ordered, 14)
    bands = _bollinger(closes, 20)
    returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes)) if closes[index - 1] > 0]
    volatility = pstdev(returns[-20:]) * sqrt(252) * 100 if len(returns) >= 2 else None
    volume_z = _zscore(volumes[-20:]) if len(volumes) >= 2 else None
    vwap_window = ordered[-20:]
    vwap_volume = sum(item.volume for item in vwap_window)
    vwap = (
        sum(((item.high + item.low + item.close) / 3) * item.volume for item in vwap_window) / vwap_volume
        if vwap_volume > 0
        else None
    )
    trend = "INSUFFICIENT_DATA"
    if sma_20 is not None and sma_50 is not None:
        relative_gap = (sma_20 - sma_50) / current.close
        trend = "UPTREND" if relative_gap > 0.003 else "DOWNTREND" if relative_gap < -0.003 else "RANGE_BOUND"
    atr_pct = atr_14 / current.close * 100 if atr_14 is not None else None
    volatility_regime = "INSUFFICIENT_DATA" if atr_pct is None else "HIGH_VOLATILITY" if atr_pct >= 3 else "LOW_VOLATILITY" if atr_pct <= 1 else "NORMAL_VOLATILITY"
    momentum_20 = current.close / closes[-21] - 1 if len(closes) >= 21 else None
    return {
        "snapshot_id": str(uuid4()),
        "symbol": current.symbol,
        "timestamp": current.timestamp,
        "bar_count": len(ordered),
        "source": "ohlcv_feature_engine",
        "features": {
            "sma_20": _round(sma_20),
            "sma_50": _round(sma_50),
            "ema_12": _round(ema_12),
            "ema_26": _round(ema_26),
            "rsi_14": _round(_rsi(closes, 14)),
            "macd": _round(macd),
            "atr_14": _round(atr_14),
            "atr_pct": _round(atr_pct),
            "bollinger_upper": _round(bands[0]) if bands else None,
            "bollinger_middle": _round(bands[1]) if bands else None,
            "bollinger_lower": _round(bands[2]) if bands else None,
            "momentum_20_pct": _round(momentum_20 * 100) if momentum_20 is not None else None,
            "annualized_volatility_pct": _round(volatility),
            "volume_zscore_20": _round(volume_z),
            "vwap_20": _round(vwap),
        },
        "regime": {
            "trend": trend,
            "volatility": volatility_regime,
            "risk_posture": "RISK_ON" if momentum_20 is not None and momentum_20 > 0 else "RISK_OFF" if momentum_20 is not None else "UNKNOWN",
        },
    }


def _sma(values: list[float], window: int) -> float | None:
    return mean(values[-window:]) if len(values) >= window else None


def _ema(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    alpha = 2 / (window + 1)
    result = mean(values[:window])
    for value in values[window:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _rsi(values: list[float], window: int) -> float | None:
    if len(values) < window + 1:
        return None
    changes = [values[index] - values[index - 1] for index in range(len(values) - window, len(values))]
    gains = mean(max(change, 0) for change in changes)
    losses = mean(max(-change, 0) for change in changes)
    if losses == 0:
        return 100.0 if gains else 50.0
    return 100 - 100 / (1 + gains / losses)


def _atr(candles: list[Candle], window: int) -> float | None:
    if len(candles) < window + 1:
        return None
    ranges = []
    for index in range(len(candles) - window, len(candles)):
        current = candles[index]
        previous_close = candles[index - 1].close
        ranges.append(max(current.high - current.low, abs(current.high - previous_close), abs(current.low - previous_close)))
    return mean(ranges)


def _bollinger(values: list[float], window: int) -> tuple[float, float, float] | None:
    if len(values) < window:
        return None
    recent = values[-window:]
    middle = mean(recent)
    deviation = pstdev(recent)
    return middle + 2 * deviation, middle, middle - 2 * deviation


def _zscore(values: list[float]) -> float | None:
    deviation = pstdev(values)
    return (values[-1] - mean(values)) / deviation if deviation else 0.0


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None