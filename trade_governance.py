"""Trade-level evidence contract for measurable and reproducible decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite


@dataclass(frozen=True)
class TradeTrace:
    trade_id: str
    symbol: str
    decision_timestamp: str
    context_hash: str
    strategy_names: tuple[str, ...]
    alpha_score: float
    calibrated_probability: str
    expected_net_value: str
    evidence_ids: tuple[str, ...]
    explanation: str
    historical_validation: dict
    portfolio_impact: dict
    risk_controls: dict
    compliance: dict
    audit_event_ids: tuple[str, ...]
    backtest_reference: str

    def __post_init__(self) -> None:
        try:
            calibrated_probability = Decimal(self.calibrated_probability)
            expected_net_value = Decimal(self.expected_net_value)
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("Trade trace measurements must be finite numbers") from error
        if (
            isinstance(self.alpha_score, bool)
            or not isinstance(self.alpha_score, (int, float))
            or not isfinite(self.alpha_score)
            or not calibrated_probability.is_finite()
            or not Decimal("0") <= calibrated_probability <= Decimal("1")
            or not expected_net_value.is_finite()
        ):
            raise ValueError("Trade trace measurements must be finite; probability must be in [0, 1]")
        if not self.trade_id.strip() or not self.symbol.strip() or len(self.context_hash) != 64:
            raise ValueError("Trade trace identity and context hash are required")
        if not self.evidence_ids or not self.explanation.strip():
            raise ValueError("Trade trace requires evidence and explanation")
        if not self.strategy_names or not self.backtest_reference.strip():
            raise ValueError("Trade trace requires strategy and backtest references")
        if not self.audit_event_ids:
            raise ValueError("Trade trace requires audit event ids")
        if not isinstance(self.risk_controls, dict) or not isinstance(self.compliance, dict):
            raise ValueError("Risk and compliance decisions are required")

    @property
    def reproducibility_hash(self) -> str:
        payload = {
            "symbol": self.symbol,
            "context_hash": self.context_hash,
            "strategies": self.strategy_names,
            "evidence_ids": self.evidence_ids,
            "backtest_reference": self.backtest_reference,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "decision_timestamp": self.decision_timestamp,
            "context_hash": self.context_hash,
            "reproducibility_hash": self.reproducibility_hash,
            "strategy_names": list(self.strategy_names),
            "measurements": {
                "alpha_score": self.alpha_score,
                "calibrated_probability": self.calibrated_probability,
                "expected_net_value": self.expected_net_value,
            },
            "explainability": {"evidence_ids": list(self.evidence_ids), "explanation": self.explanation},
            "historical_validation": _jsonable(self.historical_validation),
            "portfolio_impact": _jsonable(self.portfolio_impact),
            "risk_controls": _jsonable(self.risk_controls),
            "compliance": _jsonable(self.compliance),
            "audit_event_ids": list(self.audit_event_ids),
            "backtest_reference": self.backtest_reference,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_eng_string"):
        return str(value)
    return value
