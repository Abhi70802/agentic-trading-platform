"""Configurable market-specific order policy; no exchange rules are assumed implicitly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from math import isfinite
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


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

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Market policy timezone must be a valid IANA timezone") from error
        if self.session_open.tzinfo is not None or self.session_close.tzinfo is not None:
            raise ValueError("Configured session hours must be local wall-clock times")
        if self.session_open >= self.session_close:
            raise ValueError("Configured session close must follow session open")
        if not self.weekdays or any(day not in range(7) for day in self.weekdays):
            raise ValueError("Configured trading weekdays must be explicit and valid")
        if not _finite_number(self.max_order_value) or self.max_order_value <= 0:
            raise ValueError("Maximum order value must be finite and positive")
        if (
            not _finite_number(self.max_order_quantity)
            or isinstance(self.max_order_quantity, bool)
            or not isinstance(self.max_order_quantity, int)
            or self.max_order_quantity < 1
        ):
            raise ValueError("Maximum order quantity must be a positive integer")
        try:
            for holiday in self.holidays:
                date.fromisoformat(holiday)
        except (TypeError, ValueError) as error:
            raise ValueError("Configured holidays must use ISO-8601 dates") from error

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
        if (
            not _finite_number(quantity)
            or isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
            or quantity > self.max_order_quantity
        ):
            reasons.append("Order quantity violates the configured market limit")
        if not _finite_number(notional) or notional <= 0 or notional > self.max_order_value:
            reasons.append("Order value violates the configured market limit")
        return ComplianceDecision(not reasons, tuple(reasons))


def _finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


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