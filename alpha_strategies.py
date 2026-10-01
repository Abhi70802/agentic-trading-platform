"""Small deterministic alpha strategy set and conflict-aware aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Mapping


@dataclass(frozen=True)
class AlphaSignal:
    strategy: str
    symbol: str
    score: float
    direction: str
    confidence: float
    evidence: tuple[str, ...]


class SmaAlpha:
    name = "sma_cross"

    def generate(self, symbol: str, candles: list, *, evidence_id: str) -> AlphaSignal:
        closes = [float(candle.close) for candle in candles]
        if len(closes) < 21:
            return AlphaSignal(self.name, symbol, 0.0, "HOLD", 0.0, (evidence_id,))
        fast = mean(closes[-8:])
        slow = mean(closes[-21:])
        score = max(-1.0, min(1.0, (fast - slow) / slow)) if slow else 0.0
        direction = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
        return AlphaSignal(self.name, symbol, score, direction, min(1.0, abs(score) * 20), (evidence_id,))


class MomentumAlpha:
    name = "momentum"

    def generate(self, symbol: str, candles: list, *, evidence_id: str) -> AlphaSignal:
        closes = [float(candle.close) for candle in candles]
        if len(closes) < 6 or closes[-6] <= 0:
            return AlphaSignal(self.name, symbol, 0.0, "HOLD", 0.0, (evidence_id,))
        score = max(-1.0, min(1.0, (closes[-1] / closes[-6]) - 1.0))
        direction = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
        return AlphaSignal(self.name, symbol, score, direction, min(1.0, abs(score) * 10), (evidence_id,))


class AlphaAggregator:
    def __init__(
        self,
        *,
        strategy_weights: Mapping[str, float] | None = None,
        regime_weights: Mapping[str, Mapping[str, float]] | None = None,
    ):
        self.strategy_weights = _normalize_weights(strategy_weights or {}, "strategy")
        self.regime_weights = {
            str(regime).strip().upper(): _normalize_weights(weights, "regime")
            for regime, weights in (regime_weights or {}).items()
        }

    def aggregate(self, signals: list[AlphaSignal], *, regime: str | None = None) -> dict:
        if not signals:
            return {"direction": "HOLD", "score": 0.0, "confidence": 0.0, "signals": [], "evidence_ids": []}

        regime_weights = self.regime_weights.get(str(regime).strip().upper(), {}) if regime else {}
        weighted_signals = [
            (
                signal,
                self.strategy_weights.get(signal.strategy.casefold(), 1.0)
                * regime_weights.get(signal.strategy.casefold(), 1.0),
            )
            for signal in signals
        ]
        total_weight = sum(weight for _, weight in weighted_signals)
        score = (
            sum(signal.score * weight for signal, weight in weighted_signals) / total_weight
            if total_weight
            else 0.0
        )
        confidence = (
            sum(signal.confidence * weight for signal, weight in weighted_signals) / total_weight
            if total_weight
            else 0.0
        )
        direction = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
        return {
            "direction": direction,
            "score": round(score, 6),
            "confidence": round(confidence, 6),
            "signals": [signal.__dict__ for signal in signals],
            "evidence_ids": sorted({evidence for signal in signals for evidence in signal.evidence}),
        }


def _normalize_weights(weights: Mapping[str, float], label: str) -> dict[str, float]:
    normalized = {}
    for strategy, weight in weights.items():
        key = str(strategy).strip().casefold()
        value = float(weight)
        if not key or not isfinite(value) or value < 0:
            raise ValueError(f"{label.title()} weights must have names and finite non-negative values")
        normalized[key] = value
    return normalized
