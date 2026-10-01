import unittest
from datetime import datetime, timedelta, timezone

from core import Candle
from breadth import MarketBreadthEngine


def series(symbol, direction, *, count=60, time_offset=0, current_volume=100):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=time_offset)
    rows = []
    for index in range(count):
        if direction == "up":
            close = 100 + index
        else:
            close = 300 - index
        rows.append(Candle(
            symbol,
            (start + timedelta(days=index)).isoformat(),
            close - 0.1,
            close + 0.5,
            close - 0.5,
            close,
            current_volume if index == count - 1 else 100,
            timeframe="1d",
        ))
    return rows


class MarketBreadthTests(unittest.TestCase):
    def test_breadth_metrics_and_coverage_use_explicit_universe(self):
        engine = MarketBreadthEngine()
        breadth = engine.calculate(
            symbols=["UP-A", "UP-B", "DOWN", "MISSING", "OLD"],
            candles_by_symbol={
                "UP-A": series("UP-A", "up"),
                "UP-B": series("UP-B", "up"),
                "DOWN": series("DOWN", "down"),
                "OLD": series("OLD", "up", time_offset=-24),
            },
            timeframe="1d",
            sector_by_symbol={"UP-A": "BANKS", "UP-B": "BANKS", "DOWN": "IT"},
            index_symbols=("UP-A", "DOWN"),
        )

        self.assertEqual((breadth["advances"], breadth["declines"], breadth["unchanged"]), (2, 1, 0))
        self.assertEqual(breadth["advance_decline_ratio"], 2.0)
        self.assertEqual(breadth["percent_advancing"], 66.6667)
        self.assertEqual(breadth["percent_declining"], 33.3333)
        self.assertEqual(breadth["sample_count"], 3)
        self.assertEqual(breadth["missing_symbols"], ["MISSING"])
        self.assertEqual(breadth["excluded_symbols"], {"OLD": "latest_bar_not_aligned"})
        self.assertEqual(breadth["above_moving_averages"]["sma_20"]["above_count"], 2)
        self.assertEqual(breadth["above_moving_averages"]["sma_50"]["evaluated_count"], 3)
        self.assertEqual(breadth["sector_breadth"]["BANKS"]["advances"], 2)
        self.assertEqual(breadth["sector_breadth"]["IT"]["declines"], 1)
        self.assertEqual(breadth["index_participation"], {"DOWN": "DECLINING", "UP-A": "ADVANCING"})
        self.assertEqual(breadth["new_highs_lookback"]["evaluated_count"], 0)
        self.assertFalse(breadth["execution_authority"])

    def test_breadth_requires_unique_universe_and_requested_timeframe(self):
        engine = MarketBreadthEngine()
        with self.assertRaisesRegex(ValueError, "unique"):
            engine.calculate(
                symbols=["UP-A", "UP-A"],
                candles_by_symbol={"UP-A": series("UP-A", "up")},
                timeframe="1d",
            )
        mismatch = engine.calculate(
            symbols=["UP-A"],
            candles_by_symbol={"UP-A": [
                Candle("UP-A", "2026-01-01T00:00:00+00:00", 100, 101, 99, 100, 10, timeframe="1h"),
                Candle("UP-A", "2026-01-02T00:00:00+00:00", 101, 102, 100, 101, 10, timeframe="1h"),
            ]},
            timeframe="1d",
        )
        self.assertEqual(mismatch["sample_count"], 0)
        self.assertEqual(mismatch["excluded_symbols"], {"UP-A": "timeframe_mismatch"})
        self.assertIsNone(mismatch["percent_advancing"])
        self.assertIsNone(mismatch["percent_declining"])

    def test_new_high_low_and_volume_breadth(self):
        engine = MarketBreadthEngine(high_low_lookback=3, volume_lookback=2)
        breadth = engine.calculate(
            symbols=["NEW-HIGH", "NEW-LOW", "NORMAL-UP"],
            candles_by_symbol={
                "NEW-HIGH": series("NEW-HIGH", "up", count=4, current_volume=300),
                "NEW-LOW": series("NEW-LOW", "down", count=4, current_volume=200),
                "NORMAL-UP": series("NORMAL-UP", "up", count=4),
            },
            timeframe="1d",
        )

        self.assertEqual(breadth["new_highs_lookback"], {"count": 2, "evaluated_count": 3, "lookback": 3})
        self.assertEqual(breadth["new_lows_lookback"], {"count": 1, "evaluated_count": 3, "lookback": 3})
        self.assertEqual(breadth["volume_breadth"]["advancing_volume"], 400)
        self.assertEqual(breadth["volume_breadth"]["declining_volume"], 200)
        self.assertEqual(breadth["volume_breadth"]["net_advancing_volume"], 200)
        self.assertEqual(breadth["volume_breadth"]["high_relative_volume_count"], 2)
        self.assertEqual(breadth["volume_breadth"]["relative_volume_evaluated_count"], 3)

    def test_advancing_index_does_not_override_declining_constituent_majority(self):
        symbols = ["NIFTY 50", "STOCK-UP", "STOCK-DOWN-A", "STOCK-DOWN-B", "STOCK-DOWN-C"]
        breadth = MarketBreadthEngine().calculate(
            symbols=symbols,
            candles_by_symbol={
                "NIFTY 50": series("NIFTY 50", "up", count=2),
                "STOCK-UP": series("STOCK-UP", "up", count=2),
                "STOCK-DOWN-A": series("STOCK-DOWN-A", "down", count=2),
                "STOCK-DOWN-B": series("STOCK-DOWN-B", "down", count=2),
                "STOCK-DOWN-C": series("STOCK-DOWN-C", "down", count=2),
            },
            timeframe="1d",
            index_symbols=("NIFTY 50",),
        )

        self.assertEqual(breadth["index_participation"]["NIFTY 50"], "ADVANCING")
        self.assertEqual(breadth["advances"], 2)
        self.assertEqual(breadth["declines"], 3)
        self.assertEqual(breadth["percent_advancing"], 40.0)
        self.assertEqual(breadth["percent_declining"], 60.0)
        self.assertLess(breadth["percent_advancing"], 50)
        self.assertGreater(breadth["percent_declining"], 50)


if __name__ == "__main__":
    unittest.main()
