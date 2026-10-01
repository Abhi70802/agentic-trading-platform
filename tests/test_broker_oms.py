import unittest
from datetime import datetime, time, timezone

import pytest

from broker_adapter import DisabledLiveOrderAdapter
from compliance import MarketCompliancePolicy
from core import RiskConfig, RiskEngine
from execution import ExecutionService, OrderIntent, PaperBrokerAdapter
from oms import BrokerOrderSnapshot, OrderManagementSystem, OrderReconciler
from portfolio import PortfolioSnapshot, PositionSizer, SizingConfig


@pytest.mark.component
class BrokerOmsTests(unittest.TestCase):
    def service(self):
        return ExecutionService(
            risk_engine=RiskEngine(RiskConfig(max_position_fraction=0.2, max_exposure_fraction=0.5)),
            position_sizer=PositionSizer(SizingConfig(max_loss_fraction=0.01)),
            paper_broker=PaperBrokerAdapter(0),
        )

    def policy(self):
        return MarketCompliancePolicy(
            market="NSE", timezone_name="Asia/Kolkata", session_open=time(9, 15), session_close=time(15, 30),
            require_exchange_calendar=True,
        )

    @pytest.mark.chaos
    def test_live_order_adapter_fails_closed(self):
        order = OrderIntent("live-1", "TEST", "NSE", "BUY", "LIMIT", 100, 90)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            DisabledLiveOrderAdapter().submit(
                type("Order", (), {"client_order_id": order.client_order_id})(), price=100,
            )

    def test_oms_is_idempotent_and_reconciles_fills(self):
        oms = OrderManagementSystem(self.service())
        intent = OrderIntent(
            "paper-1", "TEST", "NSE", "BUY", "LIMIT", 100, 90,
            market_data_timestamp="2026-10-01T10:00:00+00:00",
        )
        kwargs = dict(
            portfolio=PortfolioSnapshot(100_000, ()), policy=self.policy(),
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc), calendar_session=True,
            data_is_fresh=True, human_approved=True,
        )
        first = oms.submit_paper(intent, **kwargs)
        second = oms.submit_paper(intent, **kwargs)
        self.assertEqual(first, second)
        matched = OrderReconciler().reconcile(oms.orders(), [BrokerOrderSnapshot("paper-1", "broker-1", "FILLED", first.filled_quantity, first.average_fill_price)])
        self.assertEqual(matched["status"], "MATCHED")
        mismatch = OrderReconciler().reconcile(oms.orders(), [BrokerOrderSnapshot("paper-1", "broker-1", "PARTIALLY_FILLED", 1, first.average_fill_price)])
        self.assertEqual(mismatch["status"], "DISCREPANCY")
        self.assertFalse(mismatch["execution_authority"])

    @pytest.mark.chaos
    def test_conflicting_intent_cannot_reuse_an_existing_client_order_id(self):
        execution = self.service()
        oms = OrderManagementSystem(execution)
        common = dict(
            portfolio=PortfolioSnapshot(100_000, ()), policy=self.policy(),
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc), calendar_session=True,
            data_is_fresh=True, human_approved=True,
        )
        original_intent = OrderIntent(
            "stable-id", "TEST", "NSE", "BUY", "LIMIT", 100, 90,
            market_data_timestamp="2026-10-01T10:00:00+00:00",
        )
        original = oms.submit_paper(original_intent, **common)

        for conflicting_intent in (
            OrderIntent("stable-id", "TEST", "NSE", "BUY", "LIMIT", 101, 91),
            OrderIntent("stable-id", "TEST", "NSE", "BUY", "LIMIT", 100, 89),
        ):
            with self.subTest(intent=conflicting_intent):
                conflicting = oms.submit_paper(conflicting_intent, **common)
                self.assertEqual(conflicting.status, "REJECTED")
                self.assertIn("reused", conflicting.reason)
                self.assertEqual(conflicting.requested_price, conflicting_intent.entry_price)
                self.assertNotEqual(conflicting, original)
                self.assertEqual(oms.orders(), [original])
                self.assertEqual(execution.paper_broker.get("stable-id"), original)

    def test_reconciliation_reports_missing_unknown_and_fill_mismatches(self):
        oms = OrderManagementSystem(self.service())
        common = dict(
            portfolio=PortfolioSnapshot(100_000, ()), policy=self.policy(),
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc), calendar_session=True,
            data_is_fresh=True, human_approved=True,
        )
        first = oms.submit_paper(
            OrderIntent(
                "local-1", "TEST", "NSE", "BUY", "LIMIT", 100, 90,
                market_data_timestamp="2026-10-01T10:00:00+00:00",
            ), **common,
        )
        oms.submit_paper(
            OrderIntent(
                "local-2", "TEST2", "NSE", "BUY", "LIMIT", 100, 90,
                market_data_timestamp="2026-10-01T10:00:00+00:00",
            ), **common,
        )
        result = OrderReconciler().reconcile(
            oms.orders(),
            [
                BrokerOrderSnapshot("local-1", "broker-1", "PARTIALLY_FILLED", 1, first.average_fill_price + 1),
                BrokerOrderSnapshot("broker-only", "broker-3", "FILLED", 1, 100),
            ],
        )
        discrepancy_types = {item["type"] for item in result["discrepancies"]}
        self.assertEqual(result["status"], "DISCREPANCY")
        self.assertEqual(result["orders_checked"], 3)
        self.assertTrue({
            "status_mismatch",
            "filled_quantity_mismatch",
            "average_fill_price_mismatch",
            "local_order_missing_at_broker",
            "broker_order_without_local_order",
        } <= discrepancy_types)
        self.assertFalse(result["execution_authority"])


if __name__ == "__main__":
    unittest.main()
