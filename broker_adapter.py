"""Broker boundaries for paper execution and read-only live integration."""

from __future__ import annotations

from typing import Protocol

from execution import Order


class BrokerOrderAdapter(Protocol):
    name: str

    def submit(self, order: Order, *, price: float, liquidity_quantity: int | None = None) -> Order: ...


class DisabledLiveOrderAdapter:
    """Explicit fail-closed placeholder until a live-order authorization exists."""

    name = "live_orders_disabled"

    def submit(self, order: Order, *, price: float, liquidity_quantity: int | None = None) -> Order:
        raise RuntimeError("Live broker order submission is disabled")


class AngelOneReadOnlyBrokerAdapter:
    """Exposes portfolio reconciliation only; it has no order submission method."""

    name = "angelone_read_only"

    def __init__(self, market_adapter):
        self._market_adapter = market_adapter

    def portfolio_snapshot(self) -> dict:
        return self._market_adapter.read_portfolio_snapshot()
