import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from decimal import Decimal
import math
from pathlib import Path
from threading import Barrier

import pytest

from core import Candle, RiskConfig, RiskEngine, SimulationConfig, SmaCrossStrategy, demo_candles, run_simulation
from india_economics import IndiaProductChargeRates, IndiaProductType
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


class SequenceStrategy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def action(self, history):
        return next(self.actions, "HOLD")


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

    def test_risk_engine_exact_position_and_exposure_boundaries(self):
        engine = RiskEngine(RiskConfig(max_position_fraction=0.1, max_exposure_fraction=0.2))
        cases = (
            (0, 10, True),
            (190, 1, True),
            (191, 0, False),
            (200, 0, False),
        )
        for exposure, expected_quantity, expected_approved in cases:
            with self.subTest(current_exposure=exposure):
                decision = engine.size_entry(price=10, equity=1000, current_exposure=exposure)
                self.assertEqual(decision.quantity, expected_quantity)
                self.assertEqual(decision.approved, expected_approved)
                if decision.approved:
                    self.assertLessEqual(decision.quantity * 10, 100)
                    self.assertLessEqual(exposure + decision.quantity * 10, 200)
                else:
                    self.assertIn("Risk limits leave no permissible quantity", decision.reason)

    @pytest.mark.chaos
    def test_risk_and_simulation_limits_reject_non_finite_inputs(self):
        for invalid in (math.nan, math.inf, -math.inf):
            for field in ("max_position_fraction", "max_exposure_fraction", "max_slippage_bps"):
                with self.subTest(config="risk", field=field, value=invalid):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        RiskConfig(**{field: invalid})
            for field in ("starting_cash", "fee_bps", "slippage_bps", "market_impact_bps", "max_volume_participation"):
                with self.subTest(config="simulation", field=field, value=invalid):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        SimulationConfig(**{field: invalid})
            with self.subTest(config="execution_delay", value=invalid):
                with self.assertRaises(ValueError):
                    SimulationConfig(execution_delay_bars=invalid)

        engine = RiskEngine()
        for field in ("price", "equity", "current_exposure"):
            for invalid in (math.nan, math.inf, -math.inf):
                with self.subTest(risk_input=field, value=invalid):
                    values = {"price": 100, "equity": 10_000, "current_exposure": 0}
                    values[field] = invalid
                    result = engine.size_entry(**values)
                    self.assertFalse(result.approved)
                    self.assertEqual(result.quantity, 0)

    def test_simulation_rejects_slippage_above_risk_limit(self):
        with self.assertRaisesRegex(ValueError, "slippage exceeds"):
            run_simulation(
                [candle(0, 10), candle(1, 11)],
                FixedStrategy("HOLD"),
                config=SimulationConfig(slippage_bps=51),
            )

    @pytest.mark.replay
    def test_signal_fills_at_next_bar_open(self):
        bars = [candle(0, 10), candle(1, 11, 12), candle(2, 12)]
        strategy = FixedStrategy("BUY")
        result = run_simulation(bars, strategy, config=SimulationConfig(1000, 0, 0))
        fill = next(event for event in result["events"] if event["event_type"] == "order.filled")
        self.assertEqual(fill["timestamp"], bars[1].timestamp)
        self.assertEqual(fill["payload"]["price"], bars[1].open)
        self.assertEqual(strategy.histories[0], bars[:1])

    @pytest.mark.replay
    def test_execution_delay_uses_older_information_and_fills_later(self):
        bars = [candle(index, 10 + index) for index in range(5)]
        result = run_simulation(
            bars,
            FixedStrategy("BUY"),
            config=SimulationConfig(1000, 0, 0, market_impact_bps=10, execution_delay_bars=2),
        )
        fill = next(event for event in result["events"] if event["event_type"] == "order.filled")
        signal = next(event for event in result["events"] if event["event_type"] == "signal.generated")
        self.assertEqual(fill["timestamp"], bars[3].timestamp)
        self.assertEqual(fill["payload"]["price"], bars[3].open * 1.001)
        self.assertEqual(signal["payload"]["based_on"], bars[0].timestamp)

    @pytest.mark.replay
    def test_volume_participation_models_partial_fill_and_cancelled_remainder(self):
        bars = [
            Candle("TEST", f"2026-01-0{index + 1}T00:00:00+00:00", 10, 11, 9, 10, 3)
            for index in range(3)
        ]
        result = run_simulation(
            bars,
            FixedStrategy("BUY"),
            config=SimulationConfig(1000, 0, 0, max_volume_participation=0.5),
        )
        partial = next(event for event in result["events"] if event["event_type"] == "order.partially_filled")
        self.assertEqual(partial["payload"]["quantity"], 1)
        self.assertGreater(partial["payload"]["unfilled_quantity"], 0)
        self.assertEqual(partial["payload"]["remainder_action"], "CANCELLED")
        self.assertEqual(result["partial_fills"], 1)

    @pytest.mark.replay
    def test_volume_limited_sell_keeps_unfilled_position_open(self):
        bars = [
            Candle("TEST", f"2026-01-0{index + 1}T00:00:00+00:00", opening, max(opening, close) + 1, min(opening, close) - 1, close, volume, timeframe="1d")
            for index, (opening, close, volume) in enumerate(((10, 10, 1000), (10, 11, 1000), (12, 12, 20), (12, 12, 20)))
        ]
        result = run_simulation(
            bars,
            SequenceStrategy(["BUY", "SELL"]),
            config=SimulationConfig(1000, 0, 0, max_volume_participation=0.1),
        )
        self.assertEqual(result["partial_fills"], 1)
        self.assertEqual(result["open_quantity"], 8)
        self.assertEqual(result["liquidity_rejections"], 0)

    @pytest.mark.replay
    def test_india_cost_schedule_is_applied_to_simulated_round_trip(self):
        bars = [
            Candle("TEST", f"2026-01-0{index + 1}T00:00:00+00:00", opening, max(opening, close) + 1, min(opening, close) - 1, close, 1000, timeframe="1d")
            for index, (opening, close) in enumerate(((10, 10), (10, 11), (12, 12)))
        ]
        rates = IndiaProductChargeRates(
            brokerage_rate=Decimal("0.01"),
            brokerage_cap_per_order=None,
            stt_buy_rate=Decimal("0"),
            stt_sell_rate=Decimal("0"),
            exchange_transaction_rate=Decimal("0"),
            sebi_turnover_rate=Decimal("0"),
            gst_rate=Decimal("0"),
            stamp_duty_buy_rate=Decimal("0"),
            other_turnover_rate=Decimal("0"),
            gst_base_components=("brokerage",),
        )
        result = run_simulation(
            bars,
            SequenceStrategy(["BUY", "SELL"]),
            config=SimulationConfig(
                starting_cash=1000,
                fee_bps=0,
                slippage_bps=0,
                india_product=IndiaProductType.EQUITY_INTRADAY,
                india_charge_rates=rates,
            ),
        )
        self.assertEqual(result["cost_model"], "india_configured")
        self.assertEqual(result["trades"][0]["pnl"], 17.8)

    @pytest.mark.replay
    def test_configured_india_transaction_charges_change_backtest_net_pnl(self):
        bars = [
            Candle("TEST", f"2026-01-0{index + 1}T00:00:00+00:00", 10, 12, 9, close, 1000)
            for index, close in enumerate((10, 11, 12))
        ]
        rates = IndiaProductChargeRates(
            brokerage_rate=Decimal("0.01"),
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
        result = run_simulation(
            bars,
            SequenceStrategy(["BUY", "SELL"]),
            config=SimulationConfig(
                starting_cash=1000,
                fee_bps=0,
                slippage_bps=0,
                india_product=IndiaProductType.EQUITY_INTRADAY,
                india_charge_rates=rates,
            ),
        )
        self.assertEqual(result["cost_model"], "india_configured")
        self.assertLess(result["trades"][0]["pnl"], 20)

    @pytest.mark.chaos
    @pytest.mark.replay
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

    @pytest.mark.replay
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

    def test_concurrent_duplicate_audit_events_are_idempotent_and_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concurrent-events.db"
            stores = [EventStore(path) for _ in range(8)]
            barrier = Barrier(len(stores))
            event = {
                "event_id": "concurrent-event-1",
                "event_type": "market.tick_received",
                "timestamp": "2026-10-01T04:30:00+00:00",
                "correlation_id": "nse-feed-correlation-1",
                "source": "fixture_feed",
                "version": 1,
                "schema_version": "1.0",
                "payload": {"symbol": "NIFTY 50", "last_price": 22_010.0},
            }

            def append_concurrently(store):
                barrier.wait(timeout=5)
                store.append_events([event])

            with ThreadPoolExecutor(max_workers=len(stores)) as executor:
                futures = [executor.submit(append_concurrently, store) for store in stores]
                for future in futures:
                    future.result(timeout=10)

            reopened = EventStore(path)
            self.assertEqual(reopened.read_events(), [event])

    def test_paper_events_follow_run_insertion_order_and_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "paper-events.db")

            def event(event_id, event_type, correlation_id, timestamp, payload):
                return {
                    "event_id": event_id,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "correlation_id": correlation_id,
                    "source": "test",
                    "version": 1,
                    "schema_version": "1.0",
                    "payload": payload,
                }

            store.append_events([
                event("paper-fill", "order.filled", "paper-run", "2025-01-01T00:00:00+00:00", {"mode": "paper"}),
                event("paper-complete", "simulation.completed", "paper-run", "2025-01-01T00:00:01+00:00", {"mode": "paper"}),
                event("live-noise", "market.tick_received", "feed", "2026-10-01T00:00:00+00:00", {}),
                event("backtest-complete", "simulation.completed", "backtest-run", "2026-10-01T00:00:01+00:00", {"mode": "backtest"}),
            ])

            self.assertEqual(
                {event["event_id"] for event in store.read_simulation_events("paper")},
                {"paper-fill", "paper-complete"},
            )

    def test_store_keeps_same_timestamp_candles_in_separate_timeframes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "timeframes.db")
            timestamp = "2026-10-01T09:15:00+00:00"
            minute = Candle("NIFTY", timestamp, 100, 101, 99, 100, 10, timeframe="1m")
            daily = Candle("NIFTY", timestamp, 90, 110, 89, 105, 1000, timeframe="1d")
            self.assertEqual(store.save_candles([minute, daily]), 2)
            self.assertEqual(store.read_candles("NIFTY", timeframe="1m"), [minute])
            self.assertEqual(store.read_candles("NIFTY", timeframe="1d"), [daily])

    def test_store_deduplicates_candles_and_reads_them_in_timestamp_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "ordered-candles.db")
            earlier = Candle("NIFTY", "2026-10-01T09:15:00+05:30", 100, 102, 99, 101, 10, timeframe="1m")
            later = Candle("NIFTY", "2026-10-01T09:16:00+05:30", 101, 103, 100, 102, 12, timeframe="1m")
            duplicate = Candle("NIFTY", earlier.timestamp, 100, 105, 98, 104, 50, timeframe="1m")

            inserted = store.save_candles([later, earlier, duplicate])
            repeated_inserted = store.save_candles([earlier])
            loaded = store.read_candles("NIFTY", timeframe="1m")

        self.assertEqual(inserted, 2)
        self.assertEqual(repeated_inserted, 0)
        self.assertEqual(loaded, [earlier, later])
        self.assertEqual(loaded[0].timestamp, earlier.timestamp)
        self.assertEqual(loaded[0].close, 101)

    def test_store_keeps_missing_candle_gaps_visible_without_filling_them(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "candle-gap.db")
            first = Candle("NIFTY", "2026-10-01T09:15:00+05:30", 100, 101, 99, 100, 10, timeframe="1m")
            third = Candle("NIFTY", "2026-10-01T09:17:00+05:30", 102, 103, 101, 102, 12, timeframe="1m")
            store.save_candles([first, third])
            loaded = store.read_candles("NIFTY", timeframe="1m")

        timestamps = [datetime.fromisoformat(candle.timestamp) for candle in loaded]
        gap = [
            left_time + timedelta(minutes=1)
            for left_time, right_time in zip(timestamps, timestamps[1:])
            if right_time - left_time > timedelta(minutes=1)
        ]
        self.assertEqual(len(loaded), 2)
        self.assertEqual(gap, [datetime.fromisoformat("2026-10-01T09:16:00+05:30")])

    def test_store_migrates_old_candles_to_unknown_legacy_timeframe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE candles (
                    symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
                    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                    close REAL NOT NULL, volume REAL NOT NULL, open_interest REAL,
                    PRIMARY KEY (symbol, timestamp)
                )"""
            )
            connection.execute(
                "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("OLD", "2025-01-01T00:00:00+00:00", 1, 2, 1, 2, 3, None),
            )
            connection.execute("CREATE INDEX candles_symbol_time ON candles(symbol, timestamp)")
            connection.commit()
            connection.close()

            store = EventStore(path)
            legacy = store.read_candles("OLD", timeframe="legacy")
            minute = Candle("OLD", legacy[0].timestamp, 1, 2, 1, 2, 3, timeframe="1m")
            self.assertEqual(legacy[0].timeframe, "legacy")
            self.assertEqual(store.save_candles([minute]), 1)
            self.assertEqual(store.read_candles("OLD", timeframe="1m"), [minute])
            self.assertEqual(store.read_candles("OLD", timeframe="legacy"), legacy)


if __name__ == "__main__":
    unittest.main()