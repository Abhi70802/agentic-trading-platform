"""Live-data to full-decision to simulated-execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

from compliance import MarketCompliancePolicy
from execution import ExecutionService, Order, OrderIntent
from portfolio import PortfolioSnapshot


NewsRacePolicy = Literal["CANCEL", "CONTINUE"]


@dataclass(frozen=True)
class PaperTradeDecision:
    client_order_id: str
    symbol: str
    market: str
    entry_price: float
    stop_price: float
    approved_for_risk_review: bool
    decision_id: str
    reason: str = ""


class LivePaperTradingSession:
    """Consumes live normalized ticks and can submit only simulated paper orders."""

    def __init__(
        self,
        *,
        execution: ExecutionService,
        portfolio: PortfolioSnapshot,
        policy: MarketCompliancePolicy,
        paper_mode_enabled: bool = False,
        human_approval: bool = False,
        max_tick_age_seconds: int = 30,
        news_provider: Callable[[str], list[dict]] | None = None,
        news_race_policy: NewsRacePolicy = "CANCEL",
    ):
        if max_tick_age_seconds < 1:
            raise ValueError("max_tick_age_seconds must be positive")
        if news_race_policy not in {"CANCEL", "CONTINUE"}:
            raise ValueError("news_race_policy must be CANCEL or CONTINUE")
        self.execution = execution
        self.portfolio = portfolio
        self.policy = policy
        self.paper_mode_enabled = paper_mode_enabled
        self.human_approval = human_approval
        self.max_tick_age_seconds = max_tick_age_seconds
        self.news_provider = news_provider
        self.news_race_policy = news_race_policy
        self.latest_tick = None
        self.last_decision: PaperTradeDecision | None = None
        self.last_order: Order | None = None
        self._pending_intent: OrderIntent | None = None
        self._pending_decision_id: str | None = None

    def on_tick(
        self,
        tick,
        *,
        decision_factory: Callable[[object, PortfolioSnapshot], PaperTradeDecision | None],
        now: datetime | None = None,
        calendar_session: bool | None = None,
        liquidity_quantity: int | None = None,
    ) -> dict:
        observed = datetime.fromisoformat(tick.timestamp.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (current - observed.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > self.max_tick_age_seconds:
            return {"status": "REJECTED", "reason": "Live market data is stale or from the future", "execution": False}
        self.latest_tick = tick
        decision = decision_factory(tick, self.portfolio)
        if decision is None:
            return {"status": "NO_DECISION", "execution": False}
        self.last_decision = decision
        if not decision.approved_for_risk_review:
            return {"status": "DECISION_REJECTED", "decision_id": decision.decision_id, "reason": decision.reason, "execution": False}
        if not self.paper_mode_enabled:
            return {"status": "PAPER_MODE_DISABLED", "decision_id": decision.decision_id, "execution": False}
        intent = OrderIntent(
            client_order_id=decision.client_order_id,
            symbol=decision.symbol,
            market=decision.market,
            side="BUY",
            order_type="LIMIT",
            entry_price=decision.entry_price,
            stop_price=decision.stop_price,
            market_data_timestamp=tick.timestamp,
        )
        news_race_reason = self._news_race_reason(intent, current)
        if news_race_reason:
            return {
                "status": "CANCELLED",
                "decision_id": decision.decision_id,
                "reason": news_race_reason,
                "execution": False,
            }
        order = self.execution.submit_paper(
            intent,
            portfolio=self.portfolio,
            policy=self.policy,
            now=current,
            calendar_session=calendar_session,
            data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL",
            human_approved=self.human_approval,
            liquidity_quantity=liquidity_quantity,
        )
        self.last_order = order
        if order.status == "PENDING_APPROVAL":
            self._pending_intent = intent
            self._pending_decision_id = decision.decision_id
        else:
            self._pending_intent = None
            self._pending_decision_id = None
        return {
            "status": order.status,
            "decision_id": decision.decision_id,
            "client_order_id": order.client_order_id,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "reason": order.reason,
            "execution": order.status in {"FILLED", "PARTIALLY_FILLED"},
            "broker": "paper_only",
        }

    def approve_pending(
        self,
        *,
        now: datetime | None = None,
        calendar_session: bool | None = None,
        liquidity_quantity: int | None = None,
    ) -> dict:
        intent = self._pending_intent
        decision_id = self._pending_decision_id
        if intent is None or decision_id is None:
            return {"status": "NO_PENDING_ORDER", "execution": False}

        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        news_race_reason = self._news_race_reason(intent, current)
        if news_race_reason:
            if self.last_order is not None:
                self.last_order = Order(**{
                    **self.last_order.__dict__,
                    "status": "REJECTED",
                    "reason": news_race_reason,
                })
            self._pending_intent = None
            self._pending_decision_id = None
            return {
                "status": "CANCELLED",
                "decision_id": decision_id,
                "reason": news_race_reason,
                "execution": False,
            }

        order = self.execution.submit_paper(
            intent,
            portfolio=self.portfolio,
            policy=self.policy,
            now=current,
            calendar_session=calendar_session,
            data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL",
            human_approved=True,
            liquidity_quantity=liquidity_quantity,
        )
        self.last_order = order
        self._pending_intent = None
        self._pending_decision_id = None
        return {
            "status": order.status,
            "decision_id": decision_id,
            "client_order_id": order.client_order_id,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "reason": order.reason,
            "execution": order.status in {"FILLED", "PARTIALLY_FILLED"},
            "broker": "paper_only",
        }

    def _news_race_reason(self, intent: OrderIntent, now: datetime) -> str | None:
        if self.news_race_policy == "CONTINUE":
            return None
        if self.news_provider is None:
            return "Execution cancelled because current news could not be revalidated"
        try:
            events = self.news_provider(intent.symbol)
        except Exception:
            return "Execution cancelled because current news could not be revalidated"
        if not isinstance(events, (list, tuple)):
            return "Execution cancelled because current news could not be revalidated"

        signal_time = _utc_datetime(intent.market_data_timestamp)
        current = now.astimezone(timezone.utc)
        for record in events:
            if not isinstance(record, dict):
                continue
            event = record.get("intelligence", record)
            if not isinstance(event, dict) or event.get("is_stale") is True:
                continue
            if str(event.get("severity", "")).upper() != "HIGH":
                continue
            instruments = event.get("instruments", ())
            if intent.symbol.upper() not in {str(symbol).upper() for symbol in instruments}:
                continue
            received_at = record.get("ingested_at") or event.get("ingested_at")
            if received_at is None:
                evidence = event.get("evidence", ())
                received_at = next((item.get("published_at") for item in evidence if isinstance(item, dict)), None)
            try:
                received_time = _utc_datetime(received_at)
            except (AttributeError, TypeError, ValueError):
                return "Execution cancelled because news timing could not be revalidated"
            if signal_time < received_time <= current:
                return "Execution cancelled because new high-severity news arrived after the signal"
        return None


def _utc_datetime(value: str | None) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be ISO-8601 text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
