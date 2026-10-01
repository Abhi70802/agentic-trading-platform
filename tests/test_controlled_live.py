import unittest
from datetime import datetime, timedelta, timezone
import math

import pytest

from controlled_live import ControlledLiveConfig, ControlledLiveExecutor, ControlledLiveGate, LiveApproval
from execution import Order, OrderIntent, PaperBrokerAdapter
from portfolio import PortfolioSnapshot


@pytest.mark.component
class ControlledLiveTests(unittest.TestCase):
    def setUp(self):
        self.audit_events = []
        self.gate = ControlledLiveGate(
            ControlledLiveConfig(enabled=True, max_capital_allocation=10_000, max_order_value=1_000, max_daily_loss=100),
            audit=lambda event, payload: self.audit_events.append((event, payload)),
        )
        self.gate.set_kill_switch(False, reason="test")
        now = datetime.now(timezone.utc)
        self.gate.approve(LiveApproval("approval-1", "operator", now.isoformat(), (now + timedelta(minutes=5)).isoformat()))

    def intent(self, *, market_data_timestamp=None):
        return OrderIntent(
            "live-1", "TEST", "NSE", "BUY", "LIMIT", 100, 90,
            market_data_timestamp=market_data_timestamp or datetime.now(timezone.utc).isoformat(),
        )

    def test_gate_allows_only_small_fresh_approved_orders(self):
        allowed, reasons = self.gate.evaluate(
            intent=self.intent(), portfolio=PortfolioSnapshot(5_000, ()), quantity=5,
            data_is_fresh=True, daily_realized_loss=0,
        )
        self.assertTrue(allowed, reasons)
        self.assertTrue(any(event == "live.order_gate_evaluated" for event, _ in self.audit_events))

    def test_live_gate_revalidates_market_data_timestamp_at_arrival(self):
        t3 = datetime.now(timezone.utc)
        t1 = t3 - timedelta(seconds=31)
        intent = OrderIntent(
            "live-stale-signal", "TEST", "NSE", "BUY", "LIMIT", 100, 90,
            market_data_timestamp=t1.isoformat(),
        )

        allowed, reasons = self.gate.evaluate(
            intent=intent,
            portfolio=PortfolioSnapshot(5_000, ()),
            quantity=5,
            data_is_fresh=True,
            daily_realized_loss=0,
            now=t3,
        )

        self.assertFalse(allowed)
        self.assertTrue(any("stale" in reason for reason in reasons))

        missing_timestamp, missing_reasons = self.gate.evaluate(
            intent=OrderIntent("live-missing-time", "TEST", "NSE", "BUY", "LIMIT", 100, 90),
            portfolio=PortfolioSnapshot(5_000, ()),
            quantity=5,
            data_is_fresh=True,
            daily_realized_loss=0,
            now=t3,
        )
        self.assertFalse(missing_timestamp)
        self.assertTrue(any("timestamp is missing" in reason for reason in missing_reasons))

    def test_live_gate_accepts_exact_limits_and_rejects_each_excess(self):
        gate = ControlledLiveGate(
            ControlledLiveConfig(
                enabled=True,
                max_capital_allocation=1_000,
                max_order_value=500,
                max_daily_loss=100,
            ),
            audit=lambda *_: None,
        )
        gate.set_kill_switch(False, reason="test")
        now = datetime(2026, 10, 1, 10, tzinfo=timezone.utc)
        gate.approve(LiveApproval(
            "boundary-approval", "operator", now.isoformat(), (now + timedelta(minutes=5)).isoformat(),
        ))
        portfolio = PortfolioSnapshot(1_000, ())

        allowed, reasons = gate.evaluate(
            intent=self.intent(market_data_timestamp=(now - timedelta(seconds=1)).isoformat()),
            portfolio=portfolio, quantity=5,
            data_is_fresh=True, daily_realized_loss=100, now=now,
        )
        self.assertTrue(allowed, reasons)

        cases = (
            (portfolio, 6, 0, "Order exceeds the strict live order limit"),
            (PortfolioSnapshot(1_000.01, ()), 5, 0, "Portfolio capital exceeds the controlled allocation"),
            (portfolio, 5, 100.01, "Daily loss limit has been exceeded"),
        )
        for candidate_portfolio, quantity, daily_loss, expected_reason in cases:
            with self.subTest(quantity=quantity, daily_loss=daily_loss, cash=candidate_portfolio.cash):
                allowed, reasons = gate.evaluate(
                    intent=self.intent(market_data_timestamp=(now - timedelta(seconds=1)).isoformat()),
                    portfolio=candidate_portfolio, quantity=quantity,
                    data_is_fresh=True, daily_realized_loss=daily_loss, now=now,
                )
                self.assertFalse(allowed)
                self.assertIn(expected_reason, reasons)

    @pytest.mark.chaos
    def test_kill_switch_and_loss_limit_fail_closed(self):
        self.gate.set_kill_switch(True, reason="operator emergency")
        allowed, reasons = self.gate.evaluate(
            intent=self.intent(),
            portfolio=PortfolioSnapshot(5_000, ()), quantity=5,
            data_is_fresh=True, daily_realized_loss=101,
        )
        self.assertFalse(allowed)
        self.assertIn("Live trading kill switch is active", reasons)
        self.assertIn("Daily loss limit has been exceeded", reasons)

    def test_executor_records_submission_and_returns_adapter_result(self):
        executor = ControlledLiveExecutor(broker_adapter=PaperBrokerAdapter(0), gate=self.gate)
        order = Order("order-1", "live-1", "TEST", "NSE", "BUY", "LIMIT", 5, "FILLED", datetime.now(timezone.utc).isoformat(), requested_price=100)
        result = executor.submit(
            order, intent=self.intent(), portfolio=PortfolioSnapshot(5_000, ()),
            data_is_fresh=True, daily_realized_loss=0,
        )
        self.assertEqual(result.status, "FILLED")
        self.assertTrue(any(event == "live.order_submission_completed" for event, _ in self.audit_events))

    def test_default_configuration_is_not_live_enabled(self):
        gate = ControlledLiveGate(ControlledLiveConfig(), audit=lambda *_: None)
        gate.set_kill_switch(False, reason="test")
        allowed, reasons = gate.evaluate(
            intent=self.intent(), portfolio=PortfolioSnapshot(5_000, ()), quantity=5,
            data_is_fresh=True, daily_realized_loss=0,
        )
        self.assertFalse(allowed)
        self.assertIn("Controlled live trading is disabled", reasons)

    @pytest.mark.chaos
    def test_non_finite_live_limits_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            for field in ("max_capital_allocation", "max_order_value", "max_daily_loss"):
                with self.subTest(field=field, value=invalid):
                    values = {
                        "max_capital_allocation": 10_000,
                        "max_order_value": 1_000,
                        "max_daily_loss": 100,
                    }
                    values[field] = invalid
                    with self.assertRaisesRegex(ValueError, "finite"):
                        ControlledLiveConfig(enabled=True, **values)

    @pytest.mark.chaos
    def test_live_gate_rejects_non_finite_quantity_and_daily_loss(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(quantity=invalid):
                allowed, reasons = self.gate.evaluate(
                    intent=self.intent(), portfolio=PortfolioSnapshot(5_000, ()), quantity=invalid,
                    data_is_fresh=True, daily_realized_loss=0,
                )
                self.assertFalse(allowed)
                self.assertTrue(any("quantity" in reason for reason in reasons))
            with self.subTest(daily_loss=invalid):
                allowed, reasons = self.gate.evaluate(
                    intent=self.intent(), portfolio=PortfolioSnapshot(5_000, ()), quantity=5,
                    data_is_fresh=True, daily_realized_loss=invalid,
                )
                self.assertFalse(allowed)
                self.assertIn("Daily loss limit has been exceeded", reasons)

    @pytest.mark.chaos
    def test_expired_approval_is_rejected_at_the_expiry_instant(self):
        now = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
        self.gate.approve(LiveApproval(
            "approval-expiring", "operator", (now - timedelta(minutes=5)).isoformat(), now.isoformat(),
        ))
        allowed, reasons = self.gate.evaluate(
            intent=self.intent(market_data_timestamp=(now - timedelta(seconds=1)).isoformat()),
            portfolio=PortfolioSnapshot(5_000, ()), quantity=5,
            data_is_fresh=True, daily_realized_loss=0, now=now,
        )
        self.assertFalse(allowed)
        self.assertIn("Human approval has expired", reasons)


if __name__ == "__main__":
    unittest.main()
