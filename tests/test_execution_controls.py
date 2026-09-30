import unittest
from datetime import datetime, time, timezone

from compliance import ComplianceRegistry, MarketCompliancePolicy
from core import RiskConfig, RiskEngine
from execution import ExecutionService, OrderIntent, PaperBrokerAdapter
from llm_gateway import LLMGateway, validate_hypothesis_output
from portfolio import PortfolioSnapshot, Position, PositionSizer, SizingConfig


def policy(*, require_calendar=False, max_order_value=5000):
    return MarketCompliancePolicy(
        market="TEST",
        timezone_name="UTC",
        session_open=time(0, 0),
        session_close=time(23, 59),
        max_order_value=max_order_value,
        require_exchange_calendar=require_calendar,
    )


def service(broker=None):
    risk = RiskEngine(RiskConfig(max_position_fraction=0.2, max_exposure_fraction=0.5))
    return ExecutionService(
        risk_engine=risk,
        position_sizer=PositionSizer(SizingConfig(max_loss_fraction=0.005)),
        paper_broker=broker or PaperBrokerAdapter(5),
    )


class FakeProvider:
    name = "test-model-provider"

    def __init__(self, output):
        self.output = output

    def generate_structured(self, *, model, prompt, context):
        return self.output


def model_output():
    return {
        "instrument": "TEST",
        "market": "TEST",
        "direction": "LONG",
        "time_horizon": "days",
        "entry_range": {"low": 99, "high": 101},
        "stop_loss": 95,
        "take_profit": 110,
        "thesis": "Evidence-backed candidate.",
        "supporting_evidence": [{"source_id": "bar-1", "claim": "Signal"}],
        "contradictory_evidence": [{"source_id": "news-1", "claim": "Risk"}],
        "invalidating_conditions": ["Close below stop"],
        "confidence": 0.7,
        "data_timestamp": "2026-09-30T10:00:00+00:00",
        "strategy": "test",
    }


class ExecutionControlTests(unittest.TestCase):
    def test_portfolio_snapshot_tracks_exposure_pnl_and_drawdown(self):
        snapshot = PortfolioSnapshot(
            cash=5000,
            positions=(Position("AAA", 10, 100, 110, sector="TECH"),),
            realized_pnl=50,
            high_water_mark=7000,
        ).as_dict()
        self.assertEqual(snapshot["equity"], 6100)
        self.assertEqual(snapshot["gross_exposure"], 1100)
        self.assertEqual(snapshot["unrealized_pnl"], 100)
        self.assertEqual(snapshot["sector_exposure"]["TECH"], 1100)
        self.assertGreater(snapshot["drawdown_pct"], 0)

    def test_position_sizer_obeys_stop_risk_and_exposure_caps(self):
        sizer = PositionSizer(SizingConfig(max_loss_fraction=0.01, max_position_fraction=0.2))
        quantity = sizer.size_long(equity=10_000, entry_price=100, stop_price=90, existing_exposure=0)
        self.assertEqual(quantity, 10)
        self.assertEqual(sizer.size_long(equity=10_000, entry_price=100, stop_price=90, existing_exposure=5000), 0)
        self.assertEqual(sizer.size_long(equity=10_000, entry_price=90, stop_price=100, existing_exposure=0), 0)

    def test_compliance_fails_closed_without_exchange_calendar(self):
        decision = policy(require_calendar=True).evaluate(
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            order_type="LIMIT", quantity=1, notional=100, calendar_session=None,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Exchange calendar status is unavailable", decision.reasons)

    def test_compliance_registry_requires_explicit_market_policy(self):
        registry = ComplianceRegistry()
        registry.register(policy())
        self.assertEqual(registry.markets(), ["TEST"])
        with self.assertRaisesRegex(ValueError, "No compliance policy configured"):
            registry.get("UNKNOWN")

    def test_analysis_mode_and_unapproved_human_mode_do_not_submit(self):
        engine = service()
        intent = OrderIntent("client-1", "TEST", "TEST", "BUY", "LIMIT", 100, 90)
        portfolio = PortfolioSnapshot(10_000, ())
        common = dict(
            portfolio=portfolio, policy=policy(),
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            calendar_session=True, data_is_fresh=True,
        )
        analysis = engine.submit_paper(intent, **common)
        pending = engine.submit_paper(intent, **common, execution_mode="HUMAN_APPROVAL")
        self.assertEqual(analysis.status, "REJECTED")
        self.assertEqual(pending.status, "PENDING_APPROVAL")
        self.assertIsNone(engine.paper_broker.get("client-1"))

    def test_human_approved_paper_order_is_sized_and_idempotent(self):
        engine = service()
        intent = OrderIntent("client-2", "TEST", "TEST", "BUY", "LIMIT", 100, 90)
        common = dict(
            portfolio=PortfolioSnapshot(10_000, ()), policy=policy(),
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            calendar_session=True, data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL", human_approved=True,
        )
        partial = engine.submit_paper(intent, **common, liquidity_quantity=3)
        duplicate = engine.submit_paper(intent, **common, liquidity_quantity=10)
        conflicting = engine.submit_paper(
            OrderIntent("client-2", "TEST", "TEST", "BUY", "LIMIT", 101, 91),
            **common,
            liquidity_quantity=10,
        )
        self.assertEqual(partial.status, "PARTIALLY_FILLED")
        self.assertEqual(partial.quantity, 5)
        self.assertEqual(partial.filled_quantity, 3)
        self.assertEqual(duplicate, partial)
        self.assertEqual(conflicting.status, "REJECTED")
        self.assertIn("reused", conflicting.reason)

    def test_execution_rejects_stale_data_and_automatic_mode(self):
        engine = service()
        intent = OrderIntent("client-3", "TEST", "TEST", "BUY", "LIMIT", 100, 90)
        common = dict(
            portfolio=PortfolioSnapshot(10_000, ()), policy=policy(),
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc), calendar_session=True,
            data_is_fresh=False, execution_mode="HUMAN_APPROVAL", human_approved=True,
        )
        stale = engine.submit_paper(intent, **common)
        automatic = engine.submit_paper(
            intent,
            **{**common, "data_is_fresh": True, "execution_mode": "AUTOMATIC"},
        )
        self.assertIn("stale", stale.reason)
        self.assertIn("disabled", automatic.reason)

    def test_paper_adapter_rejects_excessive_slippage_vs_risk_policy(self):
        with self.assertRaisesRegex(ValueError, "slippage exceeds"):
            service(PaperBrokerAdapter(60))

    def test_llm_gateway_requires_configured_provider(self):
        with self.assertRaisesRegex(RuntimeError, "No LLM provider configured"):
            LLMGateway().propose(model="unset", context={})

    def test_llm_guard_rejects_unreferenced_claims_and_model_quantity(self):
        output = model_output()
        context = {
            "instruments": ["TEST"],
            "evidence_ids": ["bar-1"],
            "data_timestamp": output["data_timestamp"],
        }
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_hypothesis_output(output, context)
        output = model_output()
        output["quantity"] = 500
        with self.assertRaisesRegex(ValueError, "must not determine"):
            validate_hypothesis_output(output, {**context, "evidence_ids": ["bar-1", "news-1"]})

    def test_llm_gateway_records_provider_and_reproducible_context_hash(self):
        output = model_output()
        context = {
            "instruments": ["TEST"],
            "evidence_ids": ["bar-1", "news-1"],
            "data_timestamp": output["data_timestamp"],
        }
        gateway = LLMGateway(FakeProvider(output))
        decision = gateway.propose(model="fake-v1", context=context)
        self.assertEqual(decision.provider, "test-model-provider")
        self.assertEqual(decision.model, "fake-v1")
        self.assertEqual(len(decision.context_hash), 64)
        self.assertEqual(decision.output["instrument"], "TEST")


if __name__ == "__main__":
    unittest.main()