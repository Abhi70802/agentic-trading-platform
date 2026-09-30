"""Broker-independent OMS contracts with a deterministic, idempotent paper adapter only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

from compliance import MarketCompliancePolicy
from core import RiskEngine
from portfolio import PortfolioSnapshot, PositionSizer


OrderSide = Literal["BUY", "SELL"]
OrderStatus = Literal["REJECTED", "PENDING_APPROVAL", "FILLED", "PARTIALLY_FILLED"]


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    symbol: str
    market: str
    side: OrderSide
    order_type: str
    entry_price: float
    stop_price: float


@dataclass(frozen=True)
class Order:
    order_id: str
    client_order_id: str
    symbol: str
    market: str
    side: OrderSide
    order_type: str
    quantity: int
    status: OrderStatus
    created_at: str
    requested_price: float | None = None
    filled_quantity: int = 0
    average_fill_price: float | None = None
    reason: str | None = None


class BrokerAdapter(Protocol):
    def submit(self, order: Order, *, price: float, liquidity_quantity: int | None = None) -> Order: ...


class PaperBrokerAdapter:
    def __init__(self, slippage_bps: float = 5.0):
        if slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        self.slippage_bps = slippage_bps
        self._orders_by_client_id: dict[str, Order] = {}

    def submit(self, order: Order, *, price: float, liquidity_quantity: int | None = None) -> Order:
        prior = self._orders_by_client_id.get(order.client_order_id)
        if prior:
            same_intent = all(getattr(prior, field) == getattr(order, field) for field in (
                "symbol", "market", "side", "order_type", "quantity", "requested_price",
            ))
            if same_intent:
                return prior
            return Order(**{
                **order.__dict__,
                "status": "REJECTED",
                "reason": "Idempotency key was reused for a different order intent",
            })
        if price <= 0:
            return self._save(Order(**{**order.__dict__, "status": "REJECTED", "reason": "Invalid fill price"}))
        fill_quantity = order.quantity if liquidity_quantity is None else min(order.quantity, max(0, liquidity_quantity))
        if fill_quantity == 0:
            return self._save(Order(**{**order.__dict__, "status": "REJECTED", "reason": "No available simulated liquidity"}))
        direction = 1 if order.side == "BUY" else -1
        fill_price = price * (1 + direction * self.slippage_bps / 10_000)
        status: OrderStatus = "FILLED" if fill_quantity == order.quantity else "PARTIALLY_FILLED"
        return self._save(Order(**{
            **order.__dict__,
            "status": status,
            "filled_quantity": fill_quantity,
            "average_fill_price": round(fill_price, 6),
        }))

    def get(self, client_order_id: str) -> Order | None:
        return self._orders_by_client_id.get(client_order_id)

    def _save(self, order: Order) -> Order:
        self._orders_by_client_id[order.client_order_id] = order
        return order


class ExecutionService:
    def __init__(
        self,
        *,
        risk_engine: RiskEngine,
        position_sizer: PositionSizer,
        paper_broker: PaperBrokerAdapter,
    ):
        self.risk_engine = risk_engine
        self.position_sizer = position_sizer
        self.paper_broker = paper_broker
        if paper_broker.slippage_bps > risk_engine.config.max_slippage_bps:
            raise ValueError("Paper broker slippage exceeds the risk engine limit")

    def submit_paper(
        self,
        intent: OrderIntent,
        *,
        portfolio: PortfolioSnapshot,
        policy: MarketCompliancePolicy,
        now: datetime | None = None,
        calendar_session: bool | None = None,
        data_is_fresh: bool,
        execution_mode: Literal["ANALYSIS_ONLY", "HUMAN_APPROVAL", "AUTOMATIC"] = "ANALYSIS_ONLY",
        human_approved: bool = False,
        liquidity_quantity: int | None = None,
    ) -> Order:
        created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        order = Order(
            order_id=str(uuid4()),
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            market=intent.market,
            side=intent.side,
            order_type=intent.order_type.upper(),
            quantity=0,
            status="REJECTED",
            created_at=created,
            requested_price=intent.entry_price,
        )
        if execution_mode == "ANALYSIS_ONLY":
            return Order(**{**order.__dict__, "status": "REJECTED", "reason": "Analysis-only mode does not submit orders"})
        if execution_mode == "HUMAN_APPROVAL" and not human_approved:
            return Order(**{**order.__dict__, "status": "PENDING_APPROVAL", "reason": "Human approval is required"})
        if execution_mode == "AUTOMATIC":
            return Order(**{**order.__dict__, "reason": "Autonomous execution is disabled in this build"})
        if not data_is_fresh:
            return Order(**{**order.__dict__, "reason": "Market data is stale"})
        if intent.side != "BUY":
            return Order(**{**order.__dict__, "reason": "This paper adapter only supports long entries"})
        snapshot = portfolio.as_dict()
        risk = self.risk_engine.size_entry(
            price=intent.entry_price,
            equity=snapshot["equity"],
            current_exposure=snapshot["gross_exposure"],
            data_is_fresh=data_is_fresh,
        )
        sized_quantity = self.position_sizer.size_long(
            equity=snapshot["equity"],
            entry_price=intent.entry_price,
            stop_price=intent.stop_price,
            existing_exposure=snapshot["gross_exposure"],
        )
        quantity = min(risk.quantity, sized_quantity) if risk.approved else 0
        if quantity <= 0:
            return Order(**{**order.__dict__, "reason": risk.reason if not risk.approved else "Position sizing permits no shares"})
        compliance = policy.evaluate(
            now=now or datetime.now(timezone.utc),
            order_type=intent.order_type,
            quantity=quantity,
            notional=quantity * intent.entry_price,
            calendar_session=calendar_session,
        )
        if not compliance.allowed:
            return Order(**{**order.__dict__, "reason": "; ".join(compliance.reasons)})
        approved_order = Order(**{**order.__dict__, "quantity": quantity, "status": "FILLED"})
        return self.paper_broker.submit(approved_order, price=intent.entry_price, liquidity_quantity=liquidity_quantity)