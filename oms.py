"""Local OMS state and fail-closed broker order reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from execution import Order, OrderIntent, ExecutionService
from compliance import MarketCompliancePolicy
from portfolio import PortfolioSnapshot


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    client_order_id: str
    broker_order_id: str | None
    status: str
    filled_quantity: int
    average_fill_price: float | None = None


class OrderManagementSystem:
    def __init__(self, execution: ExecutionService):
        self.execution = execution
        self._orders: dict[str, Order] = {}
        self._intents: dict[str, OrderIntent] = {}

    def submit_paper(
        self,
        intent: OrderIntent,
        *,
        portfolio: PortfolioSnapshot,
        policy: MarketCompliancePolicy,
        now: datetime | None = None,
        calendar_session: bool | None = None,
        data_is_fresh: bool,
        human_approved: bool = False,
        liquidity_quantity: int | None = None,
    ) -> Order:
        existing = self._orders.get(intent.client_order_id)
        if existing is not None:
            original_intent = self._intents[intent.client_order_id]
            if _intent_identity(original_intent) == _intent_identity(intent):
                return existing
            return Order(
                order_id=str(uuid4()),
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                market=intent.market,
                side=intent.side,
                order_type=intent.order_type.upper(),
                quantity=0,
                status="REJECTED",
                created_at=(now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
                requested_price=intent.entry_price,
                reason="Idempotency key was reused for a different order intent",
            )
        order = self.execution.submit_paper(
            intent,
            portfolio=portfolio,
            policy=policy,
            now=now or datetime.now(timezone.utc),
            calendar_session=calendar_session,
            data_is_fresh=data_is_fresh,
            execution_mode="HUMAN_APPROVAL",
            human_approved=human_approved,
            liquidity_quantity=liquidity_quantity,
        )
        self._orders[intent.client_order_id] = order
        self._intents[intent.client_order_id] = intent
        return order

    def orders(self) -> list[Order]:
        return list(self._orders.values())


def _intent_identity(intent: OrderIntent) -> tuple:
    return (
        intent.symbol,
        intent.market,
        intent.side,
        intent.order_type.upper(),
        intent.entry_price,
        intent.stop_price,
    )


class OrderReconciler:
    """Compares local OMS state with broker snapshots without hiding discrepancies."""

    def reconcile(
        self,
        local_orders: list[Order],
        broker_orders: list[BrokerOrderSnapshot],
    ) -> dict:
        local_by_client = {order.client_order_id: order for order in local_orders}
        broker_by_client = {order.client_order_id: order for order in broker_orders}
        discrepancies = []
        for client_order_id in sorted(set(local_by_client) | set(broker_by_client)):
            local = local_by_client.get(client_order_id)
            broker = broker_by_client.get(client_order_id)
            if local is None:
                discrepancies.append({"client_order_id": client_order_id, "type": "broker_order_without_local_order"})
                continue
            if broker is None:
                discrepancies.append({"client_order_id": client_order_id, "type": "local_order_missing_at_broker"})
                continue
            if local.status != broker.status:
                discrepancies.append({"client_order_id": client_order_id, "type": "status_mismatch", "local": local.status, "broker": broker.status})
            if local.filled_quantity != broker.filled_quantity:
                discrepancies.append({"client_order_id": client_order_id, "type": "filled_quantity_mismatch", "local": local.filled_quantity, "broker": broker.filled_quantity})
            if (
                local.average_fill_price is not None
                and broker.average_fill_price is not None
                and abs(local.average_fill_price - broker.average_fill_price) > 1e-6
            ):
                discrepancies.append({"client_order_id": client_order_id, "type": "average_fill_price_mismatch"})
        return {
            "status": "MATCHED" if not discrepancies else "DISCREPANCY",
            "orders_checked": len(set(local_by_client) | set(broker_by_client)),
            "discrepancies": discrepancies,
            "execution_authority": False,
        }
