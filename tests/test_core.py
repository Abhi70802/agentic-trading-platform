import tempfile
import unittest
from pathlib import Path

from core import Candle, RiskConfig, RiskEngine, SimulationConfig, SmaCrossStrategy, demo_candles, run_simulation
from store import EventStore


def candle(index: int, close: float, open_price: float | None = None) -> Candle:
    open_price = close if open_price is None else open_price
    return Candle(
        "TEST",
        f"2025-01-{index + 1:02d}T00:00:00+00:00",
        open_price,
        max(open_price, close) + 1,
        min(open_price, close) - 1,
        close,
        1000,
    )


class FixedStrategy:
    def __init__(self, action: str):
        self.next_action = action
        self.histories = []

    def action(self, history):
        self.histories.append(list(history))
        return self.next_action


class CoreTests(unittest.TestCase):
    def test_sma_strategy_rejects_invalid_windows(self):
        with self.assertRaises(ValueError):
            SmaCrossStrategy(8, 8)

    def test_risk_engine_blocks_kill_switch_and_stale_data(self):
        halted = RiskEngine(RiskConfig(kill_switch=True)).size_entry(
            price=100, equity=10_000, current_exposure=0
        )
        stale = RiskEngine().size_entry(
            price=100, equity=10_000, current_exposure=0, data_is_fresh=False
        )
        self.assertFalse(halted.approved)
        self.assertIn("kill switch", halted.reason)
        self.assertFalse(stale.approved)

    def test_risk_engine_caps_order_and_exposure(self):
        engine = RiskEngine(RiskConfig(max_position_fraction=0.1, max_exposure_fraction=0.2))
        first = engine.size_entry(price=10, equity=1000, current_exposure=0)
        second = engine.size_entry(price=10, equity=1000, current_exposure=195)
        self.assertEqual(first.quantity, 10)
        self.assertEqual(second.quantity, 0)
        self.assertFalse(second.approved)

    def test_simulation_rejects_slippage_above_risk_limit(self):
        with self.assertRaisesRegex(ValueError, "slippage exceeds"):
            run_simulation(
                [candle(0, 10), candle(1, 11)],
                FixedStrategy("HOLD"),
                config=SimulationConfig(slippage_bps=51),
            )

    def test_signal_fills_at_next_bar_open(self):
        bars = [candle(0, 10), candle(1, 11, 12), candle(2, 12)]
        strategy = FixedStrategy("BUY")
        result = run_simulation(bars, strategy, config=SimulationConfig(1000, 0, 0))
        fill = next(event for event in result["events"] if event["event_type"] == "order.filled")
        self.assertEqual(fill["timestamp"], bars[1].timestamp)
        self.assertEqual(fill["payload"]["price"], bars[1].open)
        self.assertEqual(strategy.histories[0], bars[:1])

    def test_kill_switch_rejects_entries_inside_simulation(self):
        bars = [candle(0, 10), candle(1, 11), candle(2, 12)]
        result = run_simulation(
            bars,
            FixedStrategy("BUY"),
            risk_engine=RiskEngine(RiskConfig(kill_switch=True)),
        )
        self.assertEqual(result["open_quantity"], 0)
        self.assertEqual(result["risk_rejections"], 2)
        self.assertTrue(any(event["event_type"] == "risk.rejected" for event in result["events"]))

    def test_paper_and_backtest_share_identical_simulation_results(self):
        bars = demo_candles()
        backtest = run_simulation(bars, SmaCrossStrategy(), mode="backtest")
        paper = run_simulation(bars, SmaCrossStrategy(), mode="paper")
        for field in ("ending_equity", "return_pct", "max_drawdown_pct", "trades", "equity_curve"):
            self.assertEqual(backtest[field], paper[field])

    def test_out_of_order_bars_are_rejected(self):
        bars = [candle(1, 10), candle(0, 11)]
        with self.assertRaises(ValueError):
            run_simulation(bars, FixedStrategy("HOLD"))

    def test_event_and_ohlcv_storage_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "test.db")
            bars = demo_candles()[:3]
            bars[1] = Candle(
                bars[1].symbol, bars[1].timestamp, bars[1].open, bars[1].high,
                bars[1].low, bars[1].close, bars[1].volume, 1234,
            )
            store.save_candles(bars)
            self.assertEqual(store.read_candles("DEMO"), bars)
            self.assertEqual(store.list_instruments()[0]["symbol"], "DEMO")
            event = {
                "event_id": "event-1",
                "event_type": "test.completed",
                "timestamp": bars[-1].timestamp,
                "correlation_id": "trace-1",
                "source": "unit_test",
                "version": 1,
                "schema_version": "1.0",
                "payload": {"ok": True},
            }
            store.append_events([event])
            self.assertEqual(store.read_events()[0], event)


if __name__ == "__main__":
    unittest.main()