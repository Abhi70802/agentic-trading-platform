import unittest
from datetime import datetime, timedelta, timezone

from core import Candle
from india_regime import IndiaMarketRegimeEngine, IndiaRegimeConfig


class IndiaMarketRegimeTests(unittest.TestCase):
    def setUp(self):
        self.engine = IndiaMarketRegimeEngine(IndiaRegimeConfig(strong_momentum_threshold_pct=2.0))

    def candles(self, symbol, direction="bull", *, spread=0.25, offset_seconds=0, offset_days=0):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset_days, seconds=offset_seconds)
        rows = []
        for index in range(60):
            if direction == "bull":
                close = 100 + index
            elif direction == "weak_bull":
                close = 100 + index * 0.05
            elif direction == "bear":
                close = 200 - index
            else:
                close = 100 + (0.1 if index % 2 else -0.1)
            rows.append(Candle(
                symbol,
                (start + timedelta(days=index)).isoformat(),
                close,
                close + spread,
                close - spread,
                close,
                1000 + index,
                timeframe="1d",
            ))
        return rows

    def inputs(self, nifty="bull", bank="bull", *, spread=0.25, bank_offset=0, offset_days=0):
        return {
            "NIFTY 50": self.candles("NIFTY 50", nifty, spread=spread, offset_days=offset_days),
            "NIFTY BANK": self.candles(
                "NIFTY BANK", bank, spread=spread, offset_seconds=bank_offset, offset_days=offset_days
            ),
        }

    def test_aligned_index_trends_produce_strong_bull_regime(self):
        result = self.engine.classify(self.inputs())
        self.assertEqual(result["regime"], "STRONG_BULL_TREND")
        self.assertEqual(result["risk_posture"], "RISK_ON")
        self.assertFalse(result["execution_authority"])

    def test_weak_bull_regime_uses_configured_momentum_threshold(self):
        result = self.engine.classify(self.inputs("weak_bull", "weak_bull"))
        self.assertEqual(result["regime"], "WEAK_BULL_TREND")

    def test_mixed_index_trends_fail_toward_risk_off(self):
        result = self.engine.classify(self.inputs("bull", "bear"))
        self.assertEqual(result["regime"], "MIXED")
        self.assertEqual(result["risk_posture"], "RISK_OFF")

    def test_high_volatility_overrides_trend_and_reduces_risk_posture(self):
        result = self.engine.classify(self.inputs(spread=5.0))
        self.assertEqual(result["regime"], "HIGH_VOLATILITY")
        self.assertEqual(result["risk_posture"], "RISK_OFF")

    def test_event_and_expiry_overrides_are_explicit_inputs(self):
        event = self.engine.classify(self.inputs(), event_driven=True)
        expiry = self.engine.classify(self.inputs(), expiry_driven=True)
        self.assertEqual(event["regime"], "EVENT_DRIVEN")
        self.assertEqual(expiry["regime"], "EXPIRY_DRIVEN")

    def test_missing_or_misaligned_index_history_fails_closed(self):
        missing = self.engine.classify({"NIFTY 50": self.candles("NIFTY 50")})
        misaligned = self.engine.classify(self.inputs(bank_offset=1))
        self.assertEqual(missing["regime"], "INSUFFICIENT_DATA")
        self.assertEqual(missing["missing_indices"], ["NIFTY BANK"])
        self.assertEqual(misaligned["reason"], "index_timestamps_not_aligned")

    def test_synthetic_scenario_matrix_matches_regime_engine_rules(self):
        scenarios = (
            ("strong bull trend", self.inputs(), {}, "STRONG_BULL_TREND", "RISK_ON", "LOW_VOLATILITY"),
            ("strong bear trend", self.inputs("bear", "bear"), {}, "STRONG_BEAR_TREND", "RISK_OFF", "LOW_VOLATILITY"),
            ("range bound", self.inputs("range", "range"), {}, "RANGE_BOUND", "UNKNOWN", "LOW_VOLATILITY"),
            ("high volatility", self.inputs(spread=5.0), {}, "HIGH_VOLATILITY", "RISK_OFF", "HIGH_VOLATILITY"),
            (
                "low volatility",
                self.inputs("range", "range", spread=0.05),
                {},
                "RANGE_BOUND",
                "UNKNOWN",
                "LOW_VOLATILITY",
            ),
            ("risk-on", self.inputs(), {}, "STRONG_BULL_TREND", "RISK_ON", "LOW_VOLATILITY"),
            ("risk-off", self.inputs("bear", "bear"), {}, "STRONG_BEAR_TREND", "RISK_OFF", "LOW_VOLATILITY"),
            (
                "event driven",
                self.inputs(),
                {"event_driven": True},
                "EVENT_DRIVEN",
                "UNKNOWN",
                "LOW_VOLATILITY",
            ),
        )

        for name, candles_by_index, options, expected_regime, expected_posture, expected_volatility in scenarios:
            with self.subTest(scenario=name):
                result = self.engine.classify(candles_by_index, **options)

                self.assertEqual(result["regime"], expected_regime)
                self.assertEqual(result["risk_posture"], expected_posture)
                self.assertEqual(
                    {snapshot["regime"]["volatility"] for snapshot in result["index_snapshots"].values()},
                    {expected_volatility},
                )

    def test_regime_transitions_follow_new_candle_windows(self):
        sequence = (
            (self.inputs(offset_days=0), "STRONG_BULL_TREND"),
            (self.inputs(spread=5.0, offset_days=60), "HIGH_VOLATILITY"),
            (self.inputs("range", "range", spread=0.05, offset_days=120), "RANGE_BOUND"),
            (self.inputs("bear", "bear", offset_days=180), "STRONG_BEAR_TREND"),
        )
        previous_as_of = None

        for candles_by_index, expected_regime in sequence:
            result = self.engine.classify(candles_by_index)

            self.assertEqual(result["regime"], expected_regime)
            current_as_of = datetime.fromisoformat(result["as_of"])
            if previous_as_of is not None:
                self.assertGreater(current_as_of, previous_as_of)
            previous_as_of = current_as_of

    def test_insufficient_current_history_does_not_reuse_previous_regime(self):
        previous = self.engine.classify(self.inputs())
        unavailable = self.engine.classify({"NIFTY 50": self.candles("NIFTY 50")})

        self.assertEqual(previous["regime"], "STRONG_BULL_TREND")
        self.assertEqual(unavailable["regime"], "INSUFFICIENT_DATA")
        self.assertIsNone(unavailable["as_of"])
        self.assertEqual(unavailable["index_snapshots"], {})


if __name__ == "__main__":
    unittest.main()
