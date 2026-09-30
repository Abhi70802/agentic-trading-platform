"""Configurable market-specific order policy; no exchange rules are assumed implicitly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ComplianceDecision:
    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MarketCompliancePolicy:
    market: str
    timezone_name: str
    session_open: time
    session_close: time
    allowed_order_types: frozenset[str] = frozenset({"MARKET", "LIMIT"})
    weekdays: frozenset[int] = frozenset({0, 1, 2, 3, 4})
    holidays: frozenset[str] = frozenset()
    max_order_value: float = 1_000_000.0
    max_order_quantity: int = 100_000
    require_exchange_calendar: bool = True

    def evaluate(
        self,
        *,
        now: datetime,
        order_type: str,
        quantity: int,
        notional: float,
        calendar_session: bool | None,
    ) -> ComplianceDecision:
        reasons: list[str] = []
        local = now.astimezone(ZoneInfo(self.timezone_name))
        if self.require_exchange_calendar and calendar_session is None:
            reasons.append("Exchange calendar status is unavailable")
        elif calendar_session is False:
            reasons.append("Exchange calendar reports the market closed")
        elif local.weekday() not in self.weekdays or local.date().isoformat() in self.holidays:
            reasons.append("Market date is not an allowed trading session")
        elif not self.session_open <= local.time().replace(tzinfo=None) <= self.session_close:
            reasons.append("Market is outside configured trading hours")
        if order_type.upper() not in self.allowed_order_types:
            reasons.append("Order type is not permitted by this market policy")
        if quantity <= 0 or quantity > self.max_order_quantity:
            reasons.append("Order quantity violates the configured market limit")
        if notional <= 0 or notional > self.max_order_value:
            reasons.append("Order value violates the configured market limit")
        return ComplianceDecision(not reasons, tuple(reasons))


@dataclass
class ComplianceRegistry:
    policies: dict[str, MarketCompliancePolicy] = field(default_factory=dict)

    def register(self, policy: MarketCompliancePolicy) -> None:
        self.policies[policy.market.upper()] = policy

    def get(self, market: str) -> MarketCompliancePolicy:
        try:
            return self.policies[market.upper()]
        except KeyError as error:
            raise ValueError(f"No compliance policy configured for market {market}") from error

    def markets(self) -> list[str]:
        return sorted(self.policies)