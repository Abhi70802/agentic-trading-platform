import unittest
from datetime import datetime, time, timezone
from decimal import Decimal

import pytest

from compliance import ComplianceRegistry, MarketCompliancePolicy
from decision_engine import DecisionEngine
from india_economics import IndiaExpectedValueEngine, IndiaProductChargeRates, IndiaProductType, IndiaTransactionCostSchedule
from portfolio import PortfolioOptimizationLimits, PortfolioOptimizer, PortfolioSnapshot, PositionSizer, SizingConfig


@pytest.mark.component
class DecisionEngineTests(unittest.TestCase):
    def engine(self):
        rates = IndiaProductChargeRates(
            brokerage_rate=Decimal("0.0001"), stt_buy_rate=Decimal("0"), stt_sell_rate=Decimal("0.0001"),
            exchange_transaction_rate=Decimal("0.0001"), sebi_turnover_rate=Decimal("0.00001"),
            gst_rate=Decimal("0.18"), stamp_duty_buy_rate=Decimal("0.0001"), other_turnover_rate=Decimal("0"),
            gst_base_components=("brokerage", "exchange_charges", "sebi_charges"), brokerage_cap_per_order=None,
        )
        registry = ComplianceRegistry()
        registry.register(MarketCompliancePolicy(
            market="NSE", timezone_name="Asia/Kolkata", session_open=time(9, 15), session_close=time(15, 30),
            require_exchange_calendar=True,
        ))
        return DecisionEngine(
            expected_value_engine=IndiaExpectedValueEngine(IndiaTransactionCostSchedule({IndiaProductType.EQUITY_INTRADAY: rates})),
            portfolio_optimizer=PortfolioOptimizer(PortfolioOptimizationLimits(
                Decimal("0.5"), Decimal("0.2"), Decimal("0.5"), Decimal("0.8"), Decimal("0.8"),
            )),
            position_sizer=PositionSizer(SizingConfig(max_loss_fraction=0.01, max_position_fraction=0.2, max_exposure_fraction=0.5)),
            compliance_registry=registry,
        )

    def arguments(self):
        return dict(
            portfolio=PortfolioSnapshot(100_000, ()), market="NSE", product=IndiaProductType.EQUITY_INTRADAY,
            symbol="TEST", sector="TECH", asset_class="EQUITY", entry_price=Decimal("100"), stop_price=Decimal("90"),
            average_win=Decimal("25"), average_loss=Decimal("10"),
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7"), "sample_count": 50},
            buy_turnover=Decimal("10000"), sell_turnover=Decimal("10000"),
            slippage_bps_per_side=Decimal("1"), market_impact_bps_per_side=Decimal("1"),
            correlations_to_positions={}, now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
            order_type="LIMIT", calendar_session=True,
        )

    def test_decision_engine_composes_calibration_ev_sizing_portfolio_and_compliance(self):
        result = self.engine().evaluate_long_candidate(**self.arguments())
        self.assertTrue(result["eligible_for_risk_review"], result)
        self.assertEqual(result["position_sizing"]["quantity"], 100)
        self.assertTrue(result["expected_value"]["edge_positive"])
        self.assertTrue(result["compliance"]["allowed"])
        self.assertFalse(result["execution_authority"])

    def test_decision_engine_withholds_uncalibrated_probability(self):
        arguments = self.arguments()
        arguments["calibration"] = {"status": "INSUFFICIENT_SAMPLES", "calibrated_probability": None}
        result = self.engine().evaluate_long_candidate(**arguments)
        self.assertFalse(result["eligible_for_risk_review"])
        self.assertIn("calibrated probability", result["reasons"][0])
        self.assertFalse(result["execution_authority"])

    def test_high_win_probability_does_not_override_negative_net_expected_value(self):
        arguments = self.arguments()
        arguments["calibration"] = {
            "status": "CALIBRATED",
            "calibrated_probability": Decimal("0.95"),
            "sample_count": 50,
        }
        arguments["average_win"] = Decimal("10")
        arguments["average_loss"] = Decimal("300")

        result = self.engine().evaluate_long_candidate(**arguments)

        self.assertLess(Decimal(result["expected_value"]["expected_net_value"]), Decimal("0"))
        self.assertFalse(result["eligible_for_risk_review"])
        self.assertIn("Expected net value is not positive", result["reasons"])

    @pytest.mark.chaos
    def test_decision_engine_rejects_non_finite_probability_and_money_inputs(self):
        for invalid in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(probability=invalid):
                arguments = self.arguments()
                arguments["calibration"] = {"status": "CALIBRATED", "calibrated_probability": invalid}
                result = self.engine().evaluate_long_candidate(**arguments)
                self.assertFalse(result["eligible_for_risk_review"])
                self.assertIn("finite calibrated probability", result["reasons"][0])

            with self.subTest(entry_price=invalid):
                arguments = self.arguments()
                arguments["entry_price"] = invalid
                result = self.engine().evaluate_long_candidate(**arguments)
                self.assertFalse(result["eligible_for_risk_review"])
                self.assertIn("must be finite decimals", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
