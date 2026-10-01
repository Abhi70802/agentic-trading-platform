import unittest
from datetime import datetime, timedelta, timezone

import pytest

from backtesting import run_walk_forward
from core import Candle


class AlwaysBuy:
    def action(self, history):
        return "BUY"


def bars(count=40, timeframe="1d"):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(
            "TEST",
            (start + timedelta(days=index)).isoformat(),
            100 + index - 0.5,
            100 + index + 1,
            100 + index - 1,
            100 + index,
            1000,
            timeframe=timeframe,
        )
        for index in range(count)
    ]


@pytest.mark.replay
class WalkForwardTests(unittest.TestCase):
    def test_expanding_walk_forward_scores_only_future_test_windows(self):
        history = bars()
        result = run_walk_forward(
            history,
            AlwaysBuy,
            train_bars=20,
            test_bars=10,
        )
        self.assertEqual(result["method"], "EXPANDING_WINDOW_WALK_FORWARD")
        self.assertEqual(result["fold_count"], 2)
        self.assertEqual(result["total_test_bars"], 20)
        self.assertFalse(result["compounds_fold_returns"])
        first_fold = result["folds"][0]
        self.assertEqual(first_fold["training_end"], history[19].timestamp)
        self.assertEqual(first_fold["test_start"], history[20].timestamp)
        self.assertEqual(first_fold["trades"][0]["entry_time"], history[20].timestamp)
        self.assertEqual(first_fold["trades"][0]["exit_time"], history[29].timestamp)
        self.assertEqual(first_fold["trades"][0]["exit_reason"], "evaluation_window_end")

    def test_walk_forward_rejects_overlapping_or_mixed_timeframe_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            run_walk_forward(bars(), AlwaysBuy, train_bars=20, test_bars=10, step_bars=5)
        mixed = bars()
        mixed[-1] = Candle(
            mixed[-1].symbol, mixed[-1].timestamp, mixed[-1].open,
            mixed[-1].high, mixed[-1].low, mixed[-1].close,
            mixed[-1].volume, timeframe="1h",
        )
        with self.assertRaisesRegex(ValueError, "one timeframe"):
            run_walk_forward(mixed, AlwaysBuy, train_bars=20, test_bars=10)


if __name__ == "__main__":
    unittest.main()
