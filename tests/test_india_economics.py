import unittest
from decimal import Decimal
from typing import cast

from hypothesis import given, settings, strategies as st
import pytest

from india_economics import (
    ExpectedValueResult,
    IndiaExpectedValueEngine,
    IndiaProductChargeRates,
    IndiaProductType,
    IndiaTransactionCostSchedule,
    RoundTripCost,
)
from probability_calibration import ProbabilityCalibrator, ProbabilityObservation


def cost_schedule():
    rates = IndiaProductChargeRates(
        brokerage_rate=Decimal("0.001"),
        stt_buy_rate=Decimal("0"),
        stt_sell_rate=Decimal("0.001"),
        exchange_transaction_rate=Decimal("0.0001"),
        sebi_turnover_rate=Decimal("0.00001"),
        gst_rate=Decimal("0.18"),
        stamp_duty_buy_rate=Decimal("0.00002"),
        other_turnover_rate=Decimal("0"),
        gst_base_components=("brokerage", "exchange_charges", "sebi_charges", "other_charges"),
        brokerage_cap_per_order=None,
    )
    return IndiaTransactionCostSchedule({IndiaProductType.EQUITY_INTRADAY: rates})


def expected_value_example(
    win_probability: Decimal,
    average_win: Decimal,
    average_loss: Decimal,
    fees: Decimal,
    slippage_bps_per_side: Decimal,
) -> tuple[ExpectedValueResult, RoundTripCost]:
    rates = IndiaProductChargeRates(
        brokerage_rate=fees / Decimal("10000"),
        stt_buy_rate=Decimal("0"),
        stt_sell_rate=Decimal("0"),
        exchange_transaction_rate=Decimal("0"),
        sebi_turnover_rate=Decimal("0"),
        gst_rate=Decimal("0"),
        stamp_duty_buy_rate=Decimal("0"),
        other_turnover_rate=Decimal("0"),
        gst_base_components=("brokerage",),
        brokerage_cap_per_order=None,
    )
    engine = IndiaExpectedValueEngine(
        IndiaTransactionCostSchedule({IndiaProductType.EQUITY_INTRADAY: rates})
    )
    costs = engine.round_trip_cost(
        product=IndiaProductType.EQUITY_INTRADAY,
        buy_turnover=Decimal("5000"),
        sell_turnover=Decimal("5000"),
        slippage_bps_per_side=slippage_bps_per_side,
    )
    result = engine.expected_value(
        win_probability=win_probability,
        average_win=average_win,
        average_loss=average_loss,
        round_trip_cost=costs,
    )
    return result, costs


class IndiaEconomicsTests(unittest.TestCase):
    def test_cost_schedule_requires_product_rates_and_uses_decimal_breakdown(self):
        engine = IndiaExpectedValueEngine(cost_schedule())
        costs = engine.round_trip_cost(
            product=IndiaProductType.EQUITY_INTRADAY,
            buy_turnover=Decimal("10000"),
            sell_turnover=Decimal("11000"),
            slippage_bps_per_side=Decimal("5"),
            market_impact_bps_per_side=Decimal("2"),
        )
        self.assertEqual(costs.brokerage, Decimal("21.000"))
        self.assertEqual(costs.stt, Decimal("11.000"))
        self.assertEqual(costs.exchange_charges, Decimal("2.10000"))
        self.assertEqual(costs.sebi_charges, Decimal("0.210000"))
        self.assertEqual(costs.gst, Decimal("4.19580000"))
        self.assertEqual(costs.stamp_duty, Decimal("0.20000"))
        self.assertEqual(costs.slippage, Decimal("10.5000"))
        self.assertEqual(costs.market_impact, Decimal("4.2000"))
        self.assertEqual(costs.total_cost, Decimal("53.40580000"))
        with self.assertRaisesRegex(ValueError, "No transaction-charge schedule"):
            engine.round_trip_cost(
                product=IndiaProductType.OPTIONS,
                buy_turnover=Decimal("1000"),
                sell_turnover=Decimal("1000"),
            )

    def test_expected_value_subtracts_configured_round_trip_costs(self):
        engine = IndiaExpectedValueEngine(cost_schedule())
        costs = engine.round_trip_cost(
            product=IndiaProductType.EQUITY_INTRADAY,
            buy_turnover=Decimal("10000"),
            sell_turnover=Decimal("11000"),
            slippage_bps_per_side=Decimal("5"),
            market_impact_bps_per_side=Decimal("2"),
        )
        result = engine.expected_value(
            win_probability=Decimal("0.6"),
            average_win=Decimal("200"),
            average_loss=Decimal("100"),
            round_trip_cost=costs,
        )
        self.assertEqual(result.expected_gross_value, Decimal("80.0"))
        self.assertEqual(result.expected_net_value, Decimal("26.59420000"))
        self.assertTrue(result.edge_positive)
        self.assertEqual(result.risk_reward_ratio, Decimal("2"))

    def test_hand_calculated_expected_value_scenarios(self):
        cases = (
            ("provided example", "0.60", "100", "70", "5", "3", "32", "24", True),
            ("zero net value", "0.60", "100", "70", "32", "0", "32", "0", False),
            ("negative net value", "0.50", "100", "100", "5", "3", "0", "-8", False),
            ("high win probability, poor payoff", "0.95", "10", "300", "5", "3", "-5.50", "-13.50", False),
            ("low win probability, strong payoff", "0.10", "500", "10", "5", "3", "41", "33", True),
            ("costs erase gross edge", "0.90", "100", "20", "100", "0", "88", "-12", False),
        )
        for name, probability, average_win, average_loss, fees, slippage, gross, net, positive in cases:
            with self.subTest(scenario=name):
                result, costs = expected_value_example(
                    win_probability=Decimal(probability),
                    average_win=Decimal(average_win),
                    average_loss=Decimal(average_loss),
                    fees=Decimal(fees),
                    slippage_bps_per_side=Decimal(slippage),
                )
                self.assertEqual(result.expected_gross_value, Decimal(gross))
                self.assertEqual(costs.brokerage, Decimal(fees))
                self.assertEqual(costs.slippage, Decimal(slippage))
                self.assertEqual(result.transaction_cost, Decimal(fees) + Decimal(slippage))
                self.assertEqual(result.expected_net_value, Decimal(net))
                self.assertEqual(result.edge_positive, positive)

    def test_round_trip_cost_applies_brokerage_cap_per_side_and_india_charge_sides(self):
        rates = IndiaProductChargeRates(
            brokerage_rate=Decimal("0.01"),
            stt_buy_rate=Decimal("0.0001"),
            stt_sell_rate=Decimal("0.0002"),
            exchange_transaction_rate=Decimal("0.00005"),
            sebi_turnover_rate=Decimal("0.000001"),
            gst_rate=Decimal("0.18"),
            stamp_duty_buy_rate=Decimal("0.00003"),
            other_turnover_rate=Decimal("0"),
            gst_base_components=("brokerage", "exchange_charges", "sebi_charges"),
            brokerage_cap_per_order=Decimal("20"),
        )
        engine = IndiaExpectedValueEngine(IndiaTransactionCostSchedule({IndiaProductType.EQUITY_INTRADAY: rates}))
        costs = engine.round_trip_cost(
            product=IndiaProductType.EQUITY_INTRADAY,
            buy_turnover=Decimal("100000"),
            sell_turnover=Decimal("120000"),
        )
        self.assertEqual(costs.brokerage, Decimal("40"))
        self.assertEqual(costs.stt, Decimal("34.0000"))
        self.assertEqual(costs.stamp_duty, Decimal("3.00000"))
        self.assertEqual(costs.gst, Decimal("9.21960000"))
        self.assertEqual(costs.total_cost, Decimal("97.43960000"))

    def test_invalid_money_or_probability_values_fail_closed(self):
        for invalid in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(rate=invalid), self.assertRaisesRegex(ValueError, "finite decimal rate"):
                IndiaProductChargeRates(
                    brokerage_rate=invalid, stt_buy_rate=Decimal("0"), stt_sell_rate=Decimal("0"),
                    exchange_transaction_rate=Decimal("0"), sebi_turnover_rate=Decimal("0"),
                    gst_rate=Decimal("0"), stamp_duty_buy_rate=Decimal("0"), other_turnover_rate=Decimal("0"),
                    gst_base_components=("brokerage",),
                    brokerage_cap_per_order=None,
                )
        engine = IndiaExpectedValueEngine(cost_schedule())
        costs = engine.round_trip_cost(
            product=IndiaProductType.EQUITY_INTRADAY,
            buy_turnover=Decimal("100"), sell_turnover=Decimal("100"),
        )
        for invalid in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with self.subTest(probability=invalid), self.assertRaisesRegex(ValueError, "probability"):
                engine.expected_value(
                    win_probability=invalid,
                    average_win=Decimal("10"),
                    average_loss=Decimal("10"),
                    round_trip_cost=costs,
                )
        with self.assertRaisesRegex(ValueError, "probability"):
            engine.expected_value(
                win_probability=Decimal("1.1"),
                average_win=Decimal("10"),
                average_loss=Decimal("10"),
                round_trip_cost=costs,
            )

    def test_calibrator_withholds_probability_until_minimum_samples(self):
        calibrator = ProbabilityCalibrator(minimum_samples=3)
        observations = [
            ProbabilityObservation(Decimal("0.71"), True),
            ProbabilityObservation(Decimal("0.72"), False),
        ]
        insufficient = calibrator.calibrate(Decimal("0.75"), observations)
        self.assertIsNone(insufficient["calibrated_probability"])
        self.assertEqual(insufficient["status"], "INSUFFICIENT_SAMPLES")

        calibrated = calibrator.calibrate(
            Decimal("0.75"),
            observations + [ProbabilityObservation(Decimal("0.79"), True)],
        )
        self.assertEqual(calibrated["sample_count"], 3)
        self.assertEqual(calibrated["calibrated_probability"], Decimal(2) / Decimal(3))
        self.assertEqual(calibrated["status"], "CALIBRATED")


@pytest.mark.parametrize("probability", [Decimal("0"), Decimal("1")])
def test_probability_calibrator_accepts_zero_and_one(probability):
    outcome = probability == Decimal("1")
    result = ProbabilityCalibrator(minimum_samples=1).calibrate(
        probability,
        [ProbabilityObservation(probability, outcome)],
    )

    assert result["status"] == "CALIBRATED"
    assert result["calibrated_probability"] == probability


@pytest.mark.parametrize("probability", [Decimal("-0.01"), Decimal("1.01")])
def test_probability_engine_rejects_values_outside_zero_and_one(probability):
    with pytest.raises(ValueError, match="between zero and one"):
        ProbabilityObservation(probability, False)

    with pytest.raises(ValueError, match="between zero and one"):
        ProbabilityCalibrator().calibrate(probability, [])


def test_probability_engine_rejects_missing_probability():
    missing_probability = cast(Decimal, None)

    with pytest.raises(ValueError, match="between zero and one"):
        ProbabilityObservation(missing_probability, False)

    with pytest.raises(ValueError, match="between zero and one"):
        ProbabilityCalibrator().calibrate(missing_probability, [])


def test_calibrator_matches_controlled_seventy_percent_forecasts():
    observations = [
        ProbabilityObservation(Decimal("0.7"), index < 70)
        for index in range(100)
    ]

    result = ProbabilityCalibrator(minimum_samples=20).calibrate(Decimal("0.7"), observations)

    assert result["sample_count"] == 100
    assert result["calibrated_probability"] == Decimal("0.7")
    assert result["brier_score"] == Decimal("0.21")
    assert result["status"] == "CALIBRATED"


def test_calibrator_exposes_badly_calibrated_seventy_percent_forecasts():
    observations = [
        ProbabilityObservation(Decimal("0.7"), index < 40)
        for index in range(100)
    ]

    result = ProbabilityCalibrator(minimum_samples=20).calibrate(Decimal("0.7"), observations)

    assert result["sample_count"] == 100
    assert result["calibrated_probability"] == Decimal("0.4")
    assert result["brier_score"] == Decimal("0.33")
    assert result["status"] == "CALIBRATED"


@settings(max_examples=100, derandomize=True)
@given(outcomes=st.lists(st.booleans(), min_size=1, max_size=100))
def test_calibrated_probability_is_bounded_and_matches_observed_frequency(outcomes):
    observations = [ProbabilityObservation(Decimal("0.5"), outcome) for outcome in outcomes]
    result = ProbabilityCalibrator(minimum_samples=1).calibrate(Decimal("0.5"), observations)
    expected = Decimal(sum(outcomes)) / Decimal(len(outcomes))
    assert Decimal("0") <= result["calibrated_probability"] <= Decimal("1")
    assert result["calibrated_probability"] == expected


@settings(max_examples=100, derandomize=True)
@given(
    buy_turnover=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000000"), places=2),
    sell_turnover=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000000"), places=2),
    slippage=st.decimals(min_value=Decimal("0"), max_value=Decimal("100"), places=3),
    impact=st.decimals(min_value=Decimal("0"), max_value=Decimal("100"), places=3),
)
def test_round_trip_costs_are_nonnegative_and_reconcile_to_charge_components(
    buy_turnover, sell_turnover, slippage, impact,
):
    costs = IndiaExpectedValueEngine(cost_schedule()).round_trip_cost(
        product=IndiaProductType.EQUITY_INTRADAY,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        slippage_bps_per_side=slippage,
        market_impact_bps_per_side=impact,
    )
    components = (
        costs.brokerage, costs.stt, costs.exchange_charges, costs.sebi_charges,
        costs.other_charges, costs.gst, costs.stamp_duty, costs.slippage, costs.market_impact,
    )
    assert all(component >= 0 for component in components)
    assert costs.total_cost == sum(components, Decimal("0"))
    assert costs.currency == "INR"


@settings(max_examples=100, derandomize=True)
@given(
    first_probability=st.decimals(min_value=Decimal("0"), max_value=Decimal("1"), places=4),
    second_probability=st.decimals(min_value=Decimal("0"), max_value=Decimal("1"), places=4),
)
def test_expected_net_value_is_monotone_in_calibrated_win_probability(first_probability, second_probability):
    engine = IndiaExpectedValueEngine(cost_schedule())
    costs = engine.round_trip_cost(
        product=IndiaProductType.EQUITY_INTRADAY,
        buy_turnover=Decimal("10000"),
        sell_turnover=Decimal("11000"),
    )
    lower, higher = sorted((first_probability, second_probability))
    lower_ev = engine.expected_value(
        win_probability=lower,
        average_win=Decimal("200"),
        average_loss=Decimal("100"),
        round_trip_cost=costs,
    )
    higher_ev = engine.expected_value(
        win_probability=higher,
        average_win=Decimal("200"),
        average_loss=Decimal("100"),
        round_trip_cost=costs,
    )
    assert higher_ev.expected_net_value >= lower_ev.expected_net_value


if __name__ == "__main__":
    unittest.main()