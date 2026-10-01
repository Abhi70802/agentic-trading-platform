"""Fail-closed controls for a future controlled live-trading adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Callable

from execution import Order, OrderIntent
from marketdata import MarketDataQualityGate
from portfolio import PortfolioSnapshot


@dataclass(frozen=True)
class ControlledLiveConfig:
    enabled: bool = False
    require_human_approval: bool = True
    max_capital_allocation: float = 10_000.0
    max_order_value: float = 1_000.0
    max_daily_loss: float = 100.0

    def __post_init__(self) -> None:
        limits = (self.max_capital_allocation, self.max_order_value, self.max_daily_loss)
        if any(not _finite_number(value) or value <= 0 for value in limits):
            raise ValueError("Controlled live limits must be finite and positive")
        if self.max_order_value > self.max_capital_allocation:
            raise ValueError("Maximum order value cannot exceed capital allocation")


@dataclass(frozen=True)
class LiveApproval:
    approval_id: str
    approved_by: str
    approved_at: str
    expires_at: str


class ControlledLiveGate:
    def __init__(self, config: ControlledLiveConfig, *, audit: Callable[[str, dict], None]):
        self.config = config
        self.audit = audit
        self.kill_switch = True
        self._approval: LiveApproval | None = None

    def set_kill_switch(self, active: bool, *, reason: str) -> None:
        self.kill_switch = active
        self.audit("live.kill_switch_changed", {"active": active, "reason": reason})

    def approve(self, approval: LiveApproval) -> None:
        if not approval.approval_id.strip() or not approval.approved_by.strip():
            raise ValueError("Approval id and approver are required")
        self._approval = approval
        self.audit("live.human_approval_granted", {
            "approval_id": approval.approval_id,
            "approved_by": approval.approved_by,
            "approved_at": approval.approved_at,
            "expires_at": approval.expires_at,
        })

    def evaluate(
        self,
        *,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
        quantity: int,
        data_is_fresh: bool,
        daily_realized_loss: float,
        now: datetime | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        reasons: list[str] = []
        valid_quantity = _finite_number(quantity) and quantity > 0
        valid_price = _finite_number(intent.entry_price) and intent.entry_price > 0
        notional = quantity * intent.entry_price if valid_quantity and valid_price else None
        if not self.config.enabled:
            reasons.append("Controlled live trading is disabled")
        if self.kill_switch:
            reasons.append("Live trading kill switch is active")
        if self.config.require_human_approval and self._approval is None:
            reasons.append("Current human approval is required")
        timestamp_is_fresh = intent.market_data_timestamp is not None and MarketDataQualityGate.is_fresh(
            intent.market_data_timestamp,
            now=current,
        )
        if not data_is_fresh or not timestamp_is_fresh:
            reasons.append("Market data is stale or its timestamp is missing")
        if not valid_quantity or not isinstance(quantity, int) or isinstance(quantity, bool):
            reasons.append("Order quantity must be a finite positive integer")
        if not valid_price:
            reasons.append("Order price must be finite and positive")
        if notional is not None and (not isfinite(notional) or notional > self.config.max_order_value):
            reasons.append("Order exceeds the strict live order limit")
        portfolio_capital = portfolio.cash + sum(position.market_value for position in portfolio.positions)
        if not isfinite(portfolio_capital) or portfolio_capital > self.config.max_capital_allocation:
            reasons.append("Portfolio capital exceeds the controlled allocation")
        if not _finite_number(daily_realized_loss) or daily_realized_loss > self.config.max_daily_loss:
            reasons.append("Daily loss limit has been exceeded")
        if self._approval is not None:
            expires = datetime.fromisoformat(self._approval.expires_at.replace("Z", "+00:00"))
            if expires.tzinfo is None or expires.astimezone(timezone.utc) <= current:
                reasons.append("Human approval has expired")
        allowed = not reasons
        self.audit("live.order_gate_evaluated", {
            "client_order_id": intent.client_order_id,
            "allowed": allowed,
            "reasons": reasons,
            "quantity": quantity,
            "notional": notional,
        })
        return allowed, tuple(reasons)


def _finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


class ControlledLiveExecutor:
    """Submits only after every live gate passes; adapter ownership stays explicit."""

    def __init__(self, *, broker_adapter, gate: ControlledLiveGate):
        self.broker_adapter = broker_adapter
        self.gate = gate

    def submit(
        self,
        order: Order,
        *,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
        data_is_fresh: bool,
        daily_realized_loss: float,
        now: datetime | None = None,
    ) -> Order:
        allowed, reasons = self.gate.evaluate(
            intent=intent,
            portfolio=portfolio,
            quantity=order.quantity,
            data_is_fresh=data_is_fresh,
            daily_realized_loss=daily_realized_loss,
            now=now,
        )
        if not allowed:
            return Order(**{**order.__dict__, "status": "REJECTED", "reason": "; ".join(reasons)})
        self.gate.audit("live.order_submission_started", {"client_order_id": order.client_order_id})
        try:
            submitted = self.broker_adapter.submit(order, price=intent.entry_price)
        except Exception as error:
            self.gate.audit("live.order_submission_failed", {
                "client_order_id": order.client_order_id,
                "error_type": type(error).__name__,
            })
            raise
        self.gate.audit("live.order_submission_completed", {
            "client_order_id": order.client_order_id,
            "status": submitted.status,
            "filled_quantity": submitted.filled_quantity,
        })
        return submitted
