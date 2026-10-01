import unittest
from types import SimpleNamespace

from alpha_strategies import AlphaAggregator, AlphaSignal, MomentumAlpha, SmaAlpha
from core import SmaCrossStrategy


def candles(closes):
    return [SimpleNamespace(close=close) for close in closes]


class AlphaStrategyTests(unittest.TestCase):
    def test_momentum_threshold_is_zero_return(self):
        strategy = MomentumAlpha()
        cases = (
            ([100, 100, 100, 100, 100, 100], "HOLD", 0.0),
            ([100, 100, 100, 100, 100, 100.000001], "BUY", 1e-8),
            ([100, 100, 100, 100, 100, 99.999999], "SELL", -1e-8),
        )

        for closes, direction, score in cases:
            with self.subTest(direction=direction, final_close=closes[-1]):
                signal = strategy.generate("TEST", candles(closes), evidence_id="bar:1")
                self.assertEqual(signal.direction, direction)
                self.assertAlmostEqual(signal.score, score, places=9)
                self.assertEqual(signal.evidence, ("bar:1",))

    def test_momentum_holds_with_insufficient_history(self):
        signal = MomentumAlpha().generate("TEST", candles([100] * 5), evidence_id="bar:short")

        self.assertEqual(signal.direction, "HOLD")
        self.assertEqual(signal.score, 0)
        self.assertEqual(signal.confidence, 0)

    def test_sma_trend_threshold_is_exactly_zero_gap(self):
        strategy = SmaAlpha()
        equal = [100.0] * 21
        above = [*equal]
        above[-1] += 1e-8
        below = [*equal]
        below[-1] -= 1e-8

        cases = ((equal, "HOLD"), (above, "BUY"), (below, "SELL"))
        for closes, direction in cases:
            with self.subTest(direction=direction, final_close=closes[-1]):
                signal = strategy.generate("TEST", candles(closes), evidence_id="bar:2")
                self.assertEqual(signal.direction, direction)
                self.assertEqual(signal.evidence, ("bar:2",))

    def test_sma_trend_holds_with_insufficient_history(self):
        signal = SmaAlpha().generate("TEST", candles([100] * 20), evidence_id="bar:short")

        self.assertEqual(signal.direction, "HOLD")
        self.assertEqual(signal.score, 0)

    def test_sma_cross_equality_and_adjacent_values(self):
        strategy = SmaCrossStrategy(fast_window=2, slow_window=3)
        cases = (
            ([101, 100, 99, 100.999999], "HOLD"),
            ([101, 100, 99, 101], "HOLD"),
            ([101, 100, 99, 101.000001], "BUY"),
            ([99, 100, 101, 99.000001], "HOLD"),
            ([99, 100, 101, 99], "HOLD"),
            ([99, 100, 101, 98.999999], "SELL"),
            ([100, 101, 102, 103], "HOLD"),
        )

        for closes, expected in cases:
            with self.subTest(closes=closes):
                self.assertEqual(strategy.action(candles(closes)), expected)

    def test_sma_cross_holds_until_slow_window_has_a_previous_value(self):
        self.assertEqual(SmaCrossStrategy(2, 3).action(candles([100, 99, 98])), "HOLD")


def signal(strategy, score, confidence=0.5):
    direction = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
    return AlphaSignal(strategy, "TEST", score, direction, confidence, (f"source:{strategy}",))


class AlphaAggregatorTests(unittest.TestCase):
    def test_unanimous_bullish_and_bearish_signals_keep_their_direction(self):
        aggregator = AlphaAggregator()

        bullish = aggregator.aggregate([signal("momentum", 0.4), signal("sma_cross", 0.2)])
        bearish = aggregator.aggregate([signal("momentum", -0.4), signal("sma_cross", -0.2)])

        self.assertEqual(bullish["direction"], "BUY")
        self.assertEqual(bullish["score"], 0.3)
        self.assertEqual(bearish["direction"], "SELL")
        self.assertEqual(bearish["score"], -0.3)

    def test_equal_opposing_signals_produce_hold(self):
        result = AlphaAggregator().aggregate([signal("momentum", 0.5), signal("sma_cross", -0.5)])

        self.assertEqual(result["direction"], "HOLD")
        self.assertEqual(result["score"], 0)

    def test_strong_signal_outweighs_a_weak_opposing_signal(self):
        result = AlphaAggregator().aggregate([signal("momentum", 0.6), signal("sma_cross", -0.1)])

        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["score"], 0.25)

    def test_high_confidence_alone_does_not_override_bearish_score_consensus(self):
        result = AlphaAggregator().aggregate([
            signal("confident_buy", 0.05, confidence=1.0),
            signal("seller_a", -0.2, confidence=0.1),
            signal("seller_b", -0.2, confidence=0.1),
        ])

        self.assertEqual(result["direction"], "SELL")
        self.assertEqual(result["score"], -0.116667)

    def test_empty_and_all_hold_signals_produce_no_direction(self):
        aggregator = AlphaAggregator()

        empty = aggregator.aggregate([])
        holds = aggregator.aggregate([signal("momentum", 0), signal("sma_cross", 0)])

        self.assertEqual(empty["direction"], "HOLD")
        self.assertEqual(empty["score"], 0)
        self.assertEqual(holds["direction"], "HOLD")
        self.assertEqual(holds["score"], 0)

    def test_configured_strategy_weight_can_dominate_lower_weight_conflicts(self):
        aggregator = AlphaAggregator(strategy_weights={"breakout": 5, "mean_reversion": 1})
        result = aggregator.aggregate([
            signal("breakout", 0.2),
            signal("mean_reversion", -0.3),
            signal("mean_reversion", -0.3),
        ])

        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["score"], 0.057143)

    def test_regime_specific_weights_change_only_the_regime_weighted_vote(self):
        signals = [signal("momentum", 0.4), signal("sma_cross", -0.4)]
        aggregator = AlphaAggregator(regime_weights={
            "BULLISH": {"momentum": 3},
            "BEARISH": {"sma_cross": 3},
        })

        neutral = aggregator.aggregate(signals)
        bullish = aggregator.aggregate(signals, regime="BULLISH")
        bearish = aggregator.aggregate(signals, regime="BEARISH")

        self.assertEqual(neutral["direction"], "HOLD")
        self.assertEqual(bullish["direction"], "BUY")
        self.assertEqual(bullish["score"], 0.2)
        self.assertEqual(bearish["direction"], "SELL")
        self.assertEqual(bearish["score"], -0.2)


if __name__ == "__main__":
    unittest.main()