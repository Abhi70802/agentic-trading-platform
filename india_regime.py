"""Deterministic India-market regime classification from normalized index candles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from core import Candle
from features import feature_snapshot


REQUIRED_INDICES = ("NIFTY 50", "NIFTY BANK")


@dataclass(frozen=True)
class IndiaRegimeConfig:
    strong_momentum_threshold_pct: float = 2.0

    def __post_init__(self) -> None:
        if not isfinite(self.strong_momentum_threshold_pct) or self.strong_momentum_threshold_pct <= 0:
            raise ValueError("Strong momentum threshold must be a finite positive percentage")


class IndiaMarketRegimeEngine:
    def __init__(self, config: IndiaRegimeConfig | None = None):
        self.config = config or IndiaRegimeConfig()

    def classify(
        self,
        candles_by_index: dict[str, list[Candle]],
        *,
        timeframe: str = "1d",
        event_driven: bool = False,
        expiry_driven: bool = False,
    ) -> dict:
        missing = [symbol for symbol in REQUIRED_INDICES if not candles_by_index.get(symbol)]
        if missing:
            return self._insufficient(timeframe, "required_index_history_missing", missing)

        snapshots = {symbol: feature_snapshot(candles_by_index[symbol]) for symbol in REQUIRED_INDICES}
        if any(snapshot["timeframe"] != timeframe for snapshot in snapshots.values()):
            return self._insufficient(timeframe, "index_timeframe_mismatch", [])
        if len({snapshot["timestamp"] for snapshot in snapshots.values()}) != 1:
            return self._insufficient(timeframe, "index_timestamps_not_aligned", [])

        index_features = {symbol: snapshot["features"] for symbol, snapshot in snapshots.items()}
        trends = [snapshot["regime"]["trend"] for snapshot in snapshots.values()]
        momentums = [features["momentum_20_pct"] for features in index_features.values()]
        if any(value is None for value in momentums) or any(trend == "INSUFFICIENT_DATA" for trend in trends):
            return self._insufficient(timeframe, "index_history_insufficient", [])

        average_momentum = sum(momentums) / len(momentums)
        high_volatility = any(snapshot["regime"]["volatility"] == "HIGH_VOLATILITY" for snapshot in snapshots.values())
        if event_driven:
            regime, risk_posture = "EVENT_DRIVEN", "UNKNOWN"
            reason = "event risk was explicitly flagged by the caller"
        elif expiry_driven:
            regime, risk_posture = "EXPIRY_DRIVEN", "RISK_OFF"
            reason = "expiry risk was explicitly flagged by the caller"
        elif high_volatility:
            regime, risk_posture = "HIGH_VOLATILITY", "RISK_OFF"
            reason = "one or more index feature snapshots report high volatility"
        elif trends == ["UPTREND", "UPTREND"]:
            strong = average_momentum >= self.config.strong_momentum_threshold_pct
            regime = "STRONG_BULL_TREND" if strong else "WEAK_BULL_TREND"
            risk_posture = "RISK_ON"
            reason = "NIFTY 50 and NIFTY BANK trends are aligned upward"
        elif trends == ["DOWNTREND", "DOWNTREND"]:
            strong = average_momentum <= -self.config.strong_momentum_threshold_pct
            regime = "STRONG_BEAR_TREND" if strong else "WEAK_BEAR_TREND"
            risk_posture = "RISK_OFF"
            reason = "NIFTY 50 and NIFTY BANK trends are aligned downward"
        elif trends == ["RANGE_BOUND", "RANGE_BOUND"]:
            regime, risk_posture = "RANGE_BOUND", "UNKNOWN"
            reason = "both index trends are range-bound"
        else:
            regime, risk_posture = "MIXED", "RISK_OFF"
            reason = "NIFTY 50 and NIFTY BANK trends are not aligned"

        return {
            "regime": regime,
            "risk_posture": risk_posture,
            "timeframe": timeframe,
            "as_of": snapshots[REQUIRED_INDICES[0]]["timestamp"],
            "index_snapshots": snapshots,
            "average_momentum_20_pct": round(average_momentum, 6),
            "strong_momentum_threshold_pct": self.config.strong_momentum_threshold_pct,
            "high_volatility": high_volatility,
            "event_driven": event_driven,
            "expiry_driven": expiry_driven,
            "reason": reason,
            "execution_authority": False,
        }

    @staticmethod
    def _insufficient(timeframe: str, reason: str, missing_indices: list[str]) -> dict:
        return {
            "regime": "INSUFFICIENT_DATA",
            "risk_posture": "UNKNOWN",
            "timeframe": timeframe,
            "as_of": None,
            "index_snapshots": {},
            "missing_indices": missing_indices,
            "reason": reason,
            "execution_authority": False,
        }