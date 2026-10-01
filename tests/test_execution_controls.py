import unittest
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
import math
from zoneinfo import ZoneInfo

import pytest

from compliance import ComplianceRegistry, MarketCompliancePolicy
from core import RiskConfig, RiskEngine
from execution import ExecutionService, Order, OrderIntent, PaperBrokerAdapter
from llm_gateway import LLMGateway, validate_hypothesis_output
from portfolio import (
    PortfolioOptimizationLimits,
    PortfolioOptimizer,
    PortfolioSnapshot,
    Position,
    PositionSizer,
    SizingConfig,
)


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


@pytest.mark.component
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

    @pytest.mark.chaos
    def test_portfolio_and_sizing_reject_non_finite_inputs_and_metrics(self):
        sizer = PositionSizer()
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Position("AAA", 1, 100, invalid)
                with self.assertRaisesRegex(ValueError, "finite"):
                    PortfolioSnapshot(invalid, ())
                with self.assertRaisesRegex(ValueError, "finite"):
                    PortfolioSnapshot(1_000, (), realized_pnl=invalid)
                with self.assertRaisesRegex(ValueError, "finite"):
                    SizingConfig(max_loss_fraction=invalid)
                with self.assertRaisesRegex(ValueError, "finite"):
                    sizer.size_long(
                        equity=10_000,
                        entry_price=invalid,
                        stop_price=90,
                        existing_exposure=0,
                    )

        with self.assertRaisesRegex(ValueError, "quantity"):
            Position("AAA", math.nan, 100, 100)
        with self.assertRaisesRegex(ValueError, "exposure.*P&L"):
            PortfolioSnapshot(1_000, (Position("AAA", 1, 100, 1e308), Position("BBB", 1, 100, 1e308)))

    def test_position_sizer_obeys_stop_risk_and_exposure_caps(self):
        sizer = PositionSizer(SizingConfig(max_loss_fraction=0.01, max_position_fraction=0.2))
        quantity = sizer.size_long(equity=10_000, entry_price=100, stop_price=90, existing_exposure=0)
        self.assertEqual(quantity, 10)
        self.assertEqual(sizer.size_long(equity=10_000, entry_price=100, stop_price=90, existing_exposure=5000), 0)
        self.assertEqual(sizer.size_long(equity=10_000, entry_price=90, stop_price=100, existing_exposure=0), 0)

    def test_position_sizer_uses_exact_decimal_arithmetic_at_quantity_boundary(self):
        sizer = PositionSizer(SizingConfig(
            max_loss_fraction=0.1,
            max_position_fraction=1.0,
            max_exposure_fraction=1.0,
        ))

        quantity = sizer.size_long(
            equity=0.7,
            entry_price=0.1,
            stop_price=0.09,
            existing_exposure=0.0,
        )

        self.assertEqual(quantity, 7)

    def test_position_sizer_scenarios_respect_risk_and_exposure_budgets(self):
        config = SizingConfig(max_loss_fraction=0.01, max_position_fraction=0.2, max_exposure_fraction=0.5)
        sizer = PositionSizer(config)
        cases = (
            ("low volatility", 10_000, 100, 95, 0, 20),
            ("high volatility", 10_000, 100, 80, 0, 5),
            ("small capital", 1_000, 100, 90, 0, 1),
            ("large capital", 100_000, 100, 90, 0, 100),
            ("high existing exposure", 10_000, 100, 90, 4_800, 2),
        )
        for name, equity, entry, stop, exposure, expected_quantity in cases:
            with self.subTest(scenario=name):
                quantity = sizer.size_long(
                    equity=equity,
                    entry_price=entry,
                    stop_price=stop,
                    existing_exposure=exposure,
                )
                self.assertEqual(quantity, expected_quantity)
                self.assertLessEqual(quantity * (entry - stop), equity * config.max_loss_fraction)
                self.assertLessEqual(quantity * entry, equity * config.max_position_fraction)
                self.assertLessEqual(
                    exposure + quantity * entry,
                    equity * config.max_exposure_fraction,
                )

    def test_portfolio_optimizer_allows_candidate_within_limits_and_positive_net_ev(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.30"),
            max_symbol_exposure_fraction=Decimal("0.20"),
            max_sector_exposure_fraction=Decimal("0.30"),
            max_asset_class_exposure_fraction=Decimal("0.50"),
            max_correlation_weighted_exposure_fraction=Decimal("0.30"),
        ))
        decision = optimizer.evaluate_addition(
            PortfolioSnapshot(90_000, (Position("AAA", 100, 90, 100, sector="BANKS"),)),
            symbol="BBB",
            sector="BANKS",
            asset_class="EQUITY",
            notional=Decimal("10_000"),
            expected_net_value=Decimal("125"),
            correlations_to_positions={"AAA": Decimal("0.5")},
        )
        self.assertTrue(decision["eligible_for_risk_review"], decision["reasons"])
        self.assertEqual(decision["projected_exposure_fractions"]["gross_exposure_fraction"], "0.2")
        self.assertFalse(decision["execution_authority"])

    def test_portfolio_optimizer_rejects_sector_limit_missing_correlation_and_negative_ev(self):
        limits = PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.50"),
            max_symbol_exposure_fraction=Decimal("0.20"),
            max_sector_exposure_fraction=Decimal("0.15"),
            max_asset_class_exposure_fraction=Decimal("0.50"),
            max_correlation_weighted_exposure_fraction=Decimal("0.50"),
        )
        optimizer = PortfolioOptimizer(limits)
        portfolio = PortfolioSnapshot(90_000, (Position("AAA", 100, 90, 100, sector="BANKS"),))
        common = dict(
            portfolio=portfolio,
            symbol="BBB",
            sector="BANKS",
            asset_class="EQUITY",
            notional=Decimal("10_000"),
            expected_net_value=Decimal("100"),
        )
        concentrated = optimizer.evaluate_addition(
            **common, correlations_to_positions={"AAA": Decimal("0.1")}
        )
        missing_correlation = optimizer.evaluate_addition(**common, correlations_to_positions={})
        negative_edge = optimizer.evaluate_addition(
            **{**common, "expected_net_value": Decimal("-1")},
            correlations_to_positions={"AAA": Decimal("0.1")},
        )
        self.assertIn("Maximum sector exposure would be exceeded", concentrated["reasons"])
        self.assertIn("Correlation data is missing for AAA", missing_correlation["reasons"])
        self.assertIn("Expected net value is not positive", negative_edge["reasons"])

    def test_portfolio_optimizer_allows_empty_portfolio_and_enforces_gross_boundaries(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.30"),
            max_symbol_exposure_fraction=Decimal("0.80"),
            max_sector_exposure_fraction=Decimal("0.80"),
            max_asset_class_exposure_fraction=Decimal("0.80"),
            max_correlation_weighted_exposure_fraction=Decimal("0.80"),
        ))
        empty = optimizer.evaluate_addition(
            PortfolioSnapshot(100_000, ()),
            symbol="CCC",
            sector="HEALTH",
            asset_class="EQUITY",
            notional=Decimal("10000"),
            expected_net_value=Decimal("100"),
            correlations_to_positions={},
        )
        self.assertTrue(empty["eligible_for_risk_review"], empty["reasons"])

        portfolio = PortfolioSnapshot(80_000, (
            Position("AAA", 100, 100, 100, sector="TECH"),
            Position("BBB", 100, 100, 100, sector="BANKS"),
        ))
        for notional, expected_allowed in (
            (Decimal("9999.99"), True),
            (Decimal("10000"), True),
            (Decimal("10000.01"), False),
            (Decimal("40000"), False),
        ):
            with self.subTest(notional=notional):
                decision = optimizer.evaluate_addition(
                    portfolio,
                    symbol="CCC",
                    sector="HEALTH",
                    asset_class="EQUITY",
                    notional=notional,
                    expected_net_value=Decimal("100"),
                    correlations_to_positions={"AAA": Decimal("0"), "BBB": Decimal("0")},
                )
                self.assertEqual(decision["eligible_for_risk_review"], expected_allowed)
                self.assertEqual(
                    "Maximum gross exposure would be exceeded" in decision["reasons"],
                    not expected_allowed,
                )

    def test_portfolio_optimizer_compounds_correlated_holdings_for_candidate(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.80"),
            max_symbol_exposure_fraction=Decimal("0.50"),
            max_sector_exposure_fraction=Decimal("0.80"),
            max_asset_class_exposure_fraction=Decimal("0.80"),
            max_correlation_weighted_exposure_fraction=Decimal("0.25"),
        ))
        portfolio = PortfolioSnapshot(80_000, (
            Position("AAA", 100, 100, 100, sector="TECH"),
            Position("BBB", 100, 100, 100, sector="BANKS"),
        ))
        candidate = dict(
            portfolio=portfolio,
            symbol="CCC",
            sector="HEALTH",
            asset_class="EQUITY",
            notional=Decimal("10000"),
            expected_net_value=Decimal("100"),
        )

        highly_correlated = optimizer.evaluate_addition(
            **candidate,
            correlations_to_positions={"AAA": Decimal("0.9"), "BBB": Decimal("0.9")},
        )
        weakly_correlated = optimizer.evaluate_addition(
            **candidate,
            correlations_to_positions={"AAA": Decimal("0.2"), "BBB": Decimal("0.2")},
        )

        self.assertFalse(highly_correlated["eligible_for_risk_review"])
        self.assertIn("Maximum correlation-weighted exposure would be exceeded", highly_correlated["reasons"])
        self.assertTrue(weakly_correlated["eligible_for_risk_review"], weakly_correlated["reasons"])

    def test_portfolio_optimizer_rejects_high_leverage_as_gross_exposure(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("1.00"),
            max_symbol_exposure_fraction=Decimal("0.50"),
            max_sector_exposure_fraction=Decimal("0.80"),
            max_asset_class_exposure_fraction=Decimal("0.50"),
            max_correlation_weighted_exposure_fraction=Decimal("0.80"),
        ))
        portfolio = PortfolioSnapshot(
            -50_000,
            (Position("AAA", 1500, 100, 100, sector="BONDS", asset_class="BONDS"),),
        )

        decision = optimizer.evaluate_addition(
            portfolio,
            symbol="CCC",
            sector="HEALTH",
            asset_class="EQUITY",
            notional=Decimal("1000"),
            expected_net_value=Decimal("100"),
            correlations_to_positions={"AAA": Decimal("0")},
        )

        self.assertEqual(portfolio.as_dict()["leverage"], 1.5)
        self.assertFalse(decision["eligible_for_risk_review"])
        self.assertIn("Maximum gross exposure would be exceeded", decision["reasons"])

    def test_portfolio_optimizer_sector_exposure_boundary_matrix(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.80"),
            max_symbol_exposure_fraction=Decimal("0.50"),
            max_sector_exposure_fraction=Decimal("0.25"),
            max_asset_class_exposure_fraction=Decimal("0.80"),
            max_correlation_weighted_exposure_fraction=Decimal("0.80"),
        ))
        portfolio = PortfolioSnapshot(55_000, (Position("AAA", 50, 100, 100, sector="TECH"),))
        for notional, expected_allowed in (
            (Decimal("9999.99"), True),
            (Decimal("10000"), True),
            (Decimal("10000.01"), False),
            (Decimal("20000"), False),
        ):
            with self.subTest(notional=notional):
                decision = optimizer.evaluate_addition(
                    portfolio,
                    symbol="BBB",
                    sector="TECH",
                    asset_class="EQUITY",
                    notional=notional,
                    expected_net_value=Decimal("100"),
                    correlations_to_positions={"AAA": Decimal("0")},
                )
                self.assertEqual(decision["eligible_for_risk_review"], expected_allowed)
                self.assertEqual(
                    "Maximum sector exposure would be exceeded" in decision["reasons"],
                    not expected_allowed,
                )

    def test_portfolio_optimizer_concentration_boundaries_for_existing_symbol(self):
        optimizer = PortfolioOptimizer(PortfolioOptimizationLimits(
            max_gross_exposure_fraction=Decimal("0.80"),
            max_symbol_exposure_fraction=Decimal("0.50"),
            max_sector_exposure_fraction=Decimal("0.50"),
            max_asset_class_exposure_fraction=Decimal("0.50"),
            max_correlation_weighted_exposure_fraction=Decimal("0.80"),
        ))
        portfolio = PortfolioSnapshot(60_000, (
            Position("AAA", 400, 100, 100, sector="TECH", asset_class="EQUITY"),
        ))
        for notional, expected_allowed in (
            (Decimal("9999.99"), True),
            (Decimal("10000"), True),
            (Decimal("10000.01"), False),
        ):
            with self.subTest(notional=notional):
                decision = optimizer.evaluate_addition(
                    portfolio,
                    symbol="AAA",
                    sector="TECH",
                    asset_class="EQUITY",
                    notional=notional,
                    expected_net_value=Decimal("100"),
                    correlations_to_positions={},
                )
                self.assertEqual(decision["eligible_for_risk_review"], expected_allowed)
                if not expected_allowed:
                    self.assertIn("Maximum symbol exposure would be exceeded", decision["reasons"])
                    self.assertIn("Maximum sector exposure would be exceeded", decision["reasons"])
                    self.assertIn("Maximum asset-class exposure would be exceeded", decision["reasons"])

    def test_compliance_accepts_exact_order_limits_and_rejects_excess(self):
        market_policy = MarketCompliancePolicy(
            market="TEST",
            timezone_name="UTC",
            session_open=time(0, 0),
            session_close=time(23, 59),
            max_order_value=1_000,
            max_order_quantity=10,
            require_exchange_calendar=True,
        )
        cases = (
            (10, 1_000, True, None),
            (11, 1_000, False, "Order quantity violates the configured market limit"),
            (10, 1_000.01, False, "Order value violates the configured market limit"),
        )
        for quantity, notional, expected_allowed, expected_reason in cases:
            with self.subTest(quantity=quantity, notional=notional):
                decision = market_policy.evaluate(
                    now=datetime(2026, 10, 1, 12, tzinfo=timezone.utc),
                    order_type="LIMIT",
                    quantity=quantity,
                    notional=notional,
                    calendar_session=True,
                )
                self.assertEqual(decision.allowed, expected_allowed)
                if expected_reason:
                    self.assertIn(expected_reason, decision.reasons)

    def test_compliance_fails_closed_without_exchange_calendar(self):
        decision = policy(require_calendar=True).evaluate(
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            order_type="LIMIT", quantity=1, notional=100, calendar_session=None,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Exchange calendar status is unavailable", decision.reasons)

    @pytest.mark.chaos
    def test_compliance_rejects_non_finite_order_notional(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(notional=invalid):
                decision = policy().evaluate(
                    now=datetime(2026, 10, 1, 12, tzinfo=timezone.utc),
                    order_type="LIMIT", quantity=1, notional=invalid, calendar_session=True,
                )
                self.assertFalse(decision.allowed)
                self.assertIn("Order value violates", decision.reasons[0])
            with self.subTest(quantity=invalid):
                decision = policy().evaluate(
                    now=datetime(2026, 10, 1, 12, tzinfo=timezone.utc),
                    order_type="LIMIT", quantity=invalid, notional=100, calendar_session=True,
                )
                self.assertFalse(decision.allowed)
                self.assertTrue(any("quantity" in reason for reason in decision.reasons))

    @pytest.mark.chaos
    def test_order_and_compliance_limits_reject_non_finite_values(self):
        for invalid in (math.nan, math.inf, -math.inf):
            for field in ("entry_price", "stop_price"):
                with self.subTest(field=field, value=invalid):
                    values = {"entry_price": 100, "stop_price": 90}
                    values[field] = invalid
                    with self.assertRaisesRegex(ValueError, "finite"):
                        OrderIntent("finite-test", "TEST", "TEST", "BUY", "LIMIT", **values)
            with self.subTest(quantity=invalid):
                with self.assertRaisesRegex(ValueError, "quantity"):
                    Order(
                        "finite-order", "finite-test", "TEST", "TEST", "BUY", "LIMIT", invalid,
                        "REJECTED", "2026-10-01T10:00:00+00:00", requested_price=100,
                    )
            with self.subTest(max_order_value=invalid):
                with self.assertRaisesRegex(ValueError, "Maximum order value"):
                    MarketCompliancePolicy(
                        market="TEST", timezone_name="UTC", session_open=time(0), session_close=time(23, 59),
                        max_order_value=invalid,
                    )

    def test_nse_session_boundaries_use_india_local_time_and_configured_calendar(self):
        timezone_name = "Asia/Kolkata"
        india_policy = MarketCompliancePolicy(
            market="NSE",
            timezone_name=timezone_name,
            session_open=time(9, 15),
            session_close=time(15, 30),
            holidays=frozenset({"2026-10-02"}),
            require_exchange_calendar=True,
        )
        india_timezone = ZoneInfo(timezone_name)
        session_date = datetime(2026, 10, 1).date()
        local_open = datetime.combine(session_date, india_policy.session_open, tzinfo=india_timezone)
        local_close = datetime.combine(session_date, india_policy.session_close, tzinfo=india_timezone)

        def evaluate(local_time):
            return india_policy.evaluate(
                now=local_time.astimezone(timezone.utc),
                order_type="LIMIT", quantity=1, notional=100, calendar_session=True,
            )

        self.assertFalse(evaluate(local_open - timedelta(seconds=1)).allowed)
        self.assertTrue(evaluate(local_open).allowed)
        self.assertTrue(evaluate(local_open + (local_close - local_open) / 2).allowed)
        self.assertTrue(evaluate(local_close - timedelta(seconds=1)).allowed)
        self.assertTrue(evaluate(local_close).allowed)
        self.assertFalse(evaluate(local_close + timedelta(seconds=1)).allowed)
        self.assertFalse(evaluate(datetime(2026, 10, 3, 12, tzinfo=india_timezone)).allowed)
        holiday = evaluate(datetime(2026, 10, 2, 12, tzinfo=india_timezone))
        self.assertFalse(holiday.allowed)
        self.assertIn("Market date is not an allowed trading session", holiday.reasons)
        prior_day_late = evaluate(datetime(2026, 10, 1, 23, 59, 59, tzinfo=india_timezone))
        self.assertFalse(prior_day_late.allowed)
        holiday_midnight = evaluate(datetime(2026, 10, 2, 0, 0, tzinfo=india_timezone))
        self.assertFalse(holiday_midnight.allowed)
        self.assertIn("Market date is not an allowed trading session", holiday_midnight.reasons)
        next_session_open = datetime.combine(
            datetime(2026, 10, 5).date(), india_policy.session_open, tzinfo=india_timezone,
        )
        self.assertTrue(evaluate(next_session_open).allowed)

    def test_market_policy_rejects_an_invalid_trading_session(self):
        with self.assertRaisesRegex(ValueError, "session close"):
            MarketCompliancePolicy(
                market="NSE",
                timezone_name="Asia/Kolkata",
                session_open=time(15, 30),
                session_close=time(9, 15),
            )

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
        intent = OrderIntent(
            "client-2", "TEST", "TEST", "BUY", "LIMIT", 100, 90,
            market_data_timestamp="2026-09-30T12:00:00+00:00",
        )
        common = dict(
            portfolio=PortfolioSnapshot(10_000, ()), policy=policy(),
            now=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            calendar_session=True, data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL", human_approved=True,
        )
        partial = engine.submit_paper(intent, **common, liquidity_quantity=3)
        duplicate = engine.submit_paper(intent, **common, liquidity_quantity=10)
        conflicting = engine.submit_paper(
            OrderIntent(
                "client-2", "TEST", "TEST", "BUY", "LIMIT", 101, 91,
                market_data_timestamp="2026-09-30T12:00:00+00:00",
            ),
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

    def test_stale_t1_signal_is_revalidated_at_t3_after_t2_price_change(self):
        engine = service(PaperBrokerAdapter(0))
        t1 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 10, 1, 10, 0, 35, tzinfo=timezone.utc)
        t3 = datetime(2026, 10, 1, 10, 0, 45, tzinfo=timezone.utc)
        market_at_t1 = {"timestamp": t1.isoformat(), "price": 100}
        market_at_t2 = {"timestamp": t2.isoformat(), "price": 101}
        signal = OrderIntent(
            "race-condition-1", "TEST", "TEST", "BUY", "LIMIT", market_at_t1["price"], 90,
            market_data_timestamp=market_at_t1["timestamp"],
        )

        self.assertGreater(market_at_t2["price"], market_at_t1["price"])
        order = engine.submit_paper(
            signal,
            portfolio=PortfolioSnapshot(10_000, ()),
            policy=policy(),
            now=t3,
            calendar_session=True,
            data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL",
            human_approved=True,
        )

        self.assertEqual(order.status, "REJECTED")
        self.assertIn("stale", order.reason)
        self.assertIsNone(engine.paper_broker.get("race-condition-1"))

    def test_human_approved_order_without_quote_timestamp_fails_closed(self):
        engine = service(PaperBrokerAdapter(0))
        intent = OrderIntent("missing-quote-time", "TEST", "TEST", "BUY", "LIMIT", 100, 90)
        now = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)

        order = engine.submit_paper(
            intent,
            portfolio=PortfolioSnapshot(10_000, ()),
            policy=policy(),
            now=now,
            calendar_session=True,
            data_is_fresh=True,
            execution_mode="HUMAN_APPROVAL",
            human_approved=True,
        )

        self.assertEqual(order.status, "REJECTED")
        self.assertIn("timestamp is missing", order.reason)
        self.assertIsNone(engine.paper_broker.get("missing-quote-time"))

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