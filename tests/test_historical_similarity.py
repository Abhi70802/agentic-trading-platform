import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from core import Candle
from historical_similarity import historical_similarity


def series(count=80, *, timeframe="1d"):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(count):
        change = 0.25 + ((index % 7) - 3) * 0.04
        close = price + change
        rows.append(Candle(
            "TEST",
            (start + timedelta(days=index)).isoformat(),
            price,
            max(price, close) + 0.5,
            min(price, close) - 0.5,
            close,
            1000 + (index % 9) * 100,
            timeframe=timeframe,
        ))
        price = close
    return rows


def state_closes(direction=1):
    changes = (0.25, -0.08, 0.18, 0.12, 0.30, -0.03, 0.15)
    closes = [100.0]
    for index in range(49):
        closes.append(closes[-1] + direction * changes[index % len(changes)])
    return closes


def state_segment(start_index, closes):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index else close
        rows.append(Candle(
            "CONTROLLED",
            (start + timedelta(days=start_index + index)).isoformat(),
            open_price,
            max(open_price, close) + 0.2,
            min(open_price, close) - 0.2,
            close,
            1000 + (index % 7) * 50,
            timeframe="1d",
        ))
    return rows


def forward_segment(start_index, entry, returns_pct):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    previous_close = entry
    for index, return_pct in enumerate(returns_pct):
        close = entry * (1 + return_pct / 100)
        rows.append(Candle(
            "CONTROLLED",
            (start + timedelta(days=start_index + index)).isoformat(),
            previous_close,
            max(previous_close, close) + 0.2,
            min(previous_close, close) - 0.2,
            close,
            1000,
            timeframe="1d",
        ))
        previous_close = close
    return rows


def controlled_history(analogues, *, current_closes=None):
    candles = []
    cursor = 0
    for closes, outcomes in analogues:
        candles.extend(state_segment(cursor, closes))
        cursor += len(closes)
        candles.extend(forward_segment(cursor, closes[-1], outcomes))
        cursor += len(outcomes)
    candles.extend(state_segment(cursor, current_closes or state_closes()))
    return candles


class HistoricalSimilarityTests(unittest.TestCase):
    def test_similarity_returns_only_past_states_with_forward_outcomes(self):
        bars = series()
        result = historical_similarity(bars, horizon_bars=5, top_k=4, max_distance=2.0)

        self.assertEqual(result["symbol"], "TEST")
        self.assertEqual(result["timeframe"], "1d")
        self.assertEqual(result["sample_count"], 4)
        self.assertTrue(all(match["reference_timestamp"] < result["as_of"] for match in result["matches"]))
        self.assertTrue(all(match["holding_period_bars"] == 5 for match in result["matches"]))
        self.assertIsNotNone(result["average_return_pct"])
        self.assertIsNotNone(result["average_mae_pct"])
        self.assertIsNotNone(result["average_mfe_pct"])
        self.assertEqual(set(result["return_distribution_pct"]), {"p10", "p25", "p50", "p75", "p90"})
        self.assertFalse(result["is_calibrated_probability"])
        self.assertFalse(result["execution_authority"])

    def test_similarity_returns_explicit_empty_statistics_without_matches(self):
        result = historical_similarity(series(50), horizon_bars=5, max_distance=1e-12)
        self.assertEqual(result["sample_count"], 0)
        self.assertIsNone(result["win_rate_pct"])
        self.assertIsNone(result["return_distribution_pct"])

    def test_exact_match_reports_known_forward_outcome(self):
        closes = state_closes()
        bars = controlled_history([(closes, (1, 2, 3))])
        result = historical_similarity(bars, horizon_bars=3, top_k=5, max_distance=1e-12)

        exact = [match for match in result["matches"] if match["reference_timestamp"] == bars[49].timestamp]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["feature_distance"], 0)
        self.assertAlmostEqual(exact[0]["forward_return_pct"], 3.0, places=6)
        self.assertEqual(exact[0]["holding_period_bars"], 3)

    def test_near_match_is_ranked_within_distance_threshold(self):
        closes = state_closes()
        bars = controlled_history([(closes, (1, 2, 3))])
        bars[-1] = replace(bars[-1], volume=bars[-1].volume * 1.01)
        result = historical_similarity(bars, horizon_bars=3, max_distance=0.05)

        reference_match = next(match for match in result["matches"] if match["reference_timestamp"] == bars[49].timestamp)
        self.assertGreater(reference_match["feature_distance"], 0)
        self.assertLess(reference_match["feature_distance"], 0.05)

    def test_dissimilar_current_state_returns_no_match(self):
        bars = controlled_history(
            [(state_closes(1), (1, 2, 3))],
            current_closes=state_closes(-1),
        )
        result = historical_similarity(bars, horizon_bars=3, max_distance=1e-9)

        self.assertEqual(result["sample_count"], 0)
        self.assertIsNone(result["average_return_pct"])
        self.assertIsNone(result["return_distribution_pct"])

    def test_single_analogue_is_reported_as_insufficient_uncalibrated_sample(self):
        bars = controlled_history([(state_closes(), (1, 2, 3))])
        result = historical_similarity(bars, horizon_bars=3, top_k=20, max_distance=1e-12)

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["win_rate_pct"], 100.0)
        self.assertFalse(result["is_calibrated_probability"])

    def test_conflicting_exact_analogues_preserve_both_outcomes_in_statistics(self):
        closes = state_closes()
        bars = controlled_history([
            (closes, (2, 3, 4)),
            (closes, (-2, -3, -4)),
        ])
        result = historical_similarity(bars, horizon_bars=3, top_k=10, max_distance=1e-12)

        exact_matches = [match for match in result["matches"] if match["feature_distance"] == 0]
        self.assertEqual(len(exact_matches), 2)
        self.assertEqual({round(match["forward_return_pct"], 6) for match in exact_matches}, {4.0, -4.0})
        self.assertEqual(result["win_rate_pct"], 50.0)
        self.assertEqual(result["average_return_pct"], 0.0)

    def test_matches_expose_regimes_and_rank_same_regime_first(self):
        bars = controlled_history([
            (state_closes(1), (2, 3, 4)),
            (state_closes(-1), (-2, -3, -4)),
        ])
        result = historical_similarity(bars, horizon_bars=3, top_k=100, max_distance=2.0)

        self.assertEqual(result["matches"][0]["regime"], "UPTREND")
        self.assertEqual(result["matches"][0]["feature_distance"], 0)
        self.assertIn("DOWNTREND", {match["regime"] for match in result["matches"]})
        bearish = next(match for match in result["matches"] if match["regime"] == "DOWNTREND")
        self.assertGreater(bearish["feature_distance"], 0)

    def test_forward_outcomes_end_no_later_than_current_as_of(self):
        bars = controlled_history([
            (state_closes(), (1, 2, 3)),
            (state_closes(), (-1, -2, -3)),
        ])
        result = historical_similarity(bars, horizon_bars=3, top_k=10, max_distance=2.0)
        index_by_timestamp = {candle.timestamp: index for index, candle in enumerate(bars)}
        as_of_index = index_by_timestamp[result["as_of"]]

        for match in result["matches"]:
            reference_index = index_by_timestamp[match["reference_timestamp"]]
            outcome_end_index = reference_index + result["horizon_bars"]
            self.assertLessEqual(outcome_end_index, as_of_index)
            self.assertLessEqual(bars[outcome_end_index].timestamp, result["as_of"])
            entry = bars[reference_index].close
            expected_return = (bars[outcome_end_index].close / entry - 1) * 100
            self.assertAlmostEqual(match["forward_return_pct"], expected_return, places=6)

    def test_similarity_rejects_mixed_instruments_and_timeframes(self):
        bars = series(60)
        with self.assertRaisesRegex(ValueError, "one instrument"):
            historical_similarity([*bars[:-1], Candle(
                "OTHER", bars[-1].timestamp, 1, 2, 1, 1, 1, timeframe="1d",
            )])
        with self.assertRaisesRegex(ValueError, "one timeframe"):
            historical_similarity([*bars[:-1], Candle(
                "TEST", bars[-1].timestamp, bars[-1].open, bars[-1].high,
                bars[-1].low, bars[-1].close, bars[-1].volume, timeframe="1h",
            )])


if __name__ == "__main__":
    unittest.main()
