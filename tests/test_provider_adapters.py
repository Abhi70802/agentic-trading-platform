import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from kite_adapter import (
    KiteCredentials,
    KiteMarketDataAdapter,
    MAX_KITE_SUBSCRIPTIONS,
    OptionContract,
    select_nearest_option_contracts,
)
from core import Candle
from india_market import Exchange, IndiaMarketPolicy, TradingHoliday, TradingSession
from llm_gateway import COPILOT_HYPOTHESIS_SCHEMA, CopilotModelProvider, OpenAIModelProvider, REQUIRED_HYPOTHESIS_FIELDS
from marketdata import MarketTickQualityGate, MinuteCandleAggregator, SessionCandleResampler
from store import EventStore


def option_master():
    result = []
    token = 1
    expiries = (date(2026, 10, 1), date(2026, 10, 8))
    for underlying, center, step in (("NIFTY", 22_000, 50), ("BANKNIFTY", 48_000, 100)):
        for expiry in expiries:
            for offset in range(-6, 7):
                strike = center + offset * step
                for option_type in ("CE", "PE"):
                    result.append({
                        "instrument_token": token,
                        "exchange": "NFO",
                        "tradingsymbol": f"{underlying}{expiry:%y%m%d}{strike}{option_type}",
                        "name": underlying,
                        "expiry": expiry,
                        "strike": strike,
                        "instrument_type": option_type,
                        "segment": "NFO-OPT",
                    })
                    token += 1
    result.append({
        "instrument_token": token,
        "exchange": "NFO",
        "tradingsymbol": "NIFTY-FUT",
        "name": "NIFTY",
        "expiry": expiries[0],
        "strike": 0,
        "instrument_type": "FUT",
        "segment": "NFO-FUT",
    })
    return result


class FakeKiteClient:
    def instruments(self, exchange):
        if exchange == "NFO":
            return option_master()
        return [
            {"exchange": exchange, "tradingsymbol": "NIFTY 50", "segment": "INDICES", "instrument_token": 7001},
            {"exchange": exchange, "tradingsymbol": "NIFTY BANK", "segment": "INDICES", "instrument_token": 7002},
        ]

    def quote(self, *instruments):
        values = {"NSE:NIFTY 50": 22_010, "NSE:NIFTY BANK": 48_040}
        return {
            symbol: {"timestamp": datetime.now(timezone.utc), "last_price": values[symbol]}
            for symbol in instruments
        }

    def historical_data(self, *args, **kwargs):
        self.history_call = (args, kwargs)
        return [{
            "date": datetime(2025, 1, 1, 9, 15),
            "open": 100, "high": 102, "low": 99, "close": 101,
            "volume": 500, "oi": 250,
        }]


class FakeTicker:
    MODE_FULL = 3
    MODE_QUOTE = 2

    def __init__(self, api_key, access_token, **kwargs):
        self.api_key = api_key
        self.access_token = access_token
        self.kwargs = kwargs
        self.calls = []

    def subscribe(self, tokens):
        self.calls.append(("subscribe", tokens))

    def set_mode(self, mode, tokens):
        self.calls.append(("mode", mode, tokens))

    def connect(self, *, threaded):
        self.threaded = threaded
        self.on_connect(self, {})
        self.on_ticks(self, [{
            "instrument_token": 17,
            "last_price": 100.25,
            "volume_traded": 2000,
            "oi": 300,
            "exchange_timestamp": datetime(2026, 9, 30, 10, 0),
            "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99.5},
            "depth": {"buy": [{"price": 100, "quantity": 10, "orders": 2}], "sell": []},
        }])


class FakeOpenAIClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCopilotSession:
    session_id = "fake-session"

    def __init__(self, content, tool_requests=None):
        self.content = content
        self.tool_requests = tool_requests or []
        self.calls = []

    async def send_and_wait(self, prompt, *, timeout, response_schema=None):
        self.calls.append((prompt, timeout, response_schema))
        return SimpleNamespace(data=SimpleNamespace(content=self.content, tool_requests=self.tool_requests))


class FakeCopilotClient:
    def __init__(self, session, **options):
        self.session = session
        self.options = options
        self.session_options = None
        self.deleted_sessions = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def create_session(self, **options):
        self.session_options = options
        return self.session

    async def delete_session(self, session_id):
        self.deleted_sessions.append(session_id)


@pytest.mark.component
class ProviderAdapterTests(unittest.TestCase):
    @staticmethod
    def _session_resampler(holidays=()):
        policy = IndiaMarketPolicy(
            exchanges=(Exchange("NSE-TEST", "Synthetic NSE", "IN", "Asia/Kolkata"),),
            sessions=(TradingSession(
                exchange_code="NSE-TEST",
                segment="SYNTHETIC-TEST",
                opens_at=time(9, 15),
                closes_at=time(9, 30),
                weekdays=frozenset({0, 1, 2, 3, 4}),
            ),),
            holidays=tuple(
                TradingHoliday("NSE-TEST", session_date, "test holiday", "SYNTHETIC-TEST")
                for session_date in holidays
            ),
        )
        return SessionCandleResampler(policy, exchange_code="NSE-TEST", segment="SYNTHETIC-TEST")

    @staticmethod
    def _one_minute_bars(missing=(), session_date=date(2026, 10, 1)):
        session_open = datetime.combine(session_date, time(9, 15), tzinfo=ZoneInfo("Asia/Kolkata"))
        candles = []
        for index in range(15):
            if index in missing:
                continue
            opening = 100 + index
            candles.append(Candle(
                "NIFTY",
                (session_open + timedelta(minutes=index)).astimezone(timezone.utc).isoformat(),
                opening,
                opening + 2,
                opening - 1,
                opening + 1,
                index + 1,
                1000 + index,
                "1m",
            ))
        return candles

    def test_kite_credentials_are_read_only_and_redacted(self):
        credentials = KiteCredentials("private-api-key", "private-access-token")
        self.assertNotIn("private-api-key", repr(credentials))
        self.assertNotIn("private-access-token", repr(credentials))

    def test_option_selector_uses_nearest_expiry_and_atm_plus_five(self):
        contracts = select_nearest_option_contracts(
            option_master(),
            spot_prices={"NIFTY": 22_010, "BANKNIFTY": 48_040},
            as_of=date(2026, 9, 30),
        )
        self.assertEqual(len(contracts), 44)
        self.assertEqual({contract.underlying for contract in contracts}, {"NIFTY", "BANKNIFTY"})
        self.assertEqual({contract.expiry for contract in contracts}, {date(2026, 10, 1)})
        for underlying in ("NIFTY", "BANKNIFTY"):
            selected = [contract for contract in contracts if contract.underlying == underlying]
            self.assertEqual(len({contract.strike for contract in selected}), 11)
            self.assertEqual({contract.option_type for contract in selected}, {"CE", "PE"})

    def test_option_selector_handles_near_expiry_expiry_day_and_rollover(self):
        cases = (
            (date(2026, 9, 30), date(2026, 10, 1)),
            (date(2026, 10, 1), date(2026, 10, 1)),
            (date(2026, 10, 2), date(2026, 10, 8)),
        )
        for as_of, expected_expiry in cases:
            with self.subTest(as_of=as_of):
                contracts = select_nearest_option_contracts(
                    option_master(),
                    spot_prices={"NIFTY": 22_000},
                    underlyings=("NIFTY",),
                    strikes_each_side=0,
                    as_of=as_of,
                )
                self.assertEqual({contract.expiry for contract in contracts}, {expected_expiry})
                self.assertEqual({contract.option_type for contract in contracts}, {"CE", "PE"})
                self.assertEqual({contract.strike for contract in contracts}, {22_000})

    def test_option_selector_discards_boolean_and_fractional_instrument_tokens(self):
        rows = [
            {
                "instrument_token": True,
                "exchange": "NFO",
                "tradingsymbol": "NIFTY-INVALID-CE",
                "name": "NIFTY",
                "expiry": date(2026, 10, 1),
                "strike": 22_000,
                "instrument_type": "CE",
                "segment": "NFO-OPT",
            },
            {
                "instrument_token": 1.5,
                "exchange": "NFO",
                "tradingsymbol": "NIFTY-INVALID-PE",
                "name": "NIFTY",
                "expiry": date(2026, 10, 1),
                "strike": 22_000,
                "instrument_type": "PE",
                "segment": "NFO-OPT",
            },
        ]

        contracts = select_nearest_option_contracts(
            rows,
            spot_prices={"NIFTY": 22_000},
            underlyings=("NIFTY",),
            strikes_each_side=0,
            as_of=date(2026, 9, 30),
        )

        self.assertEqual(contracts, [])

    def test_option_contract_rejects_invalid_direct_metadata(self):
        valid = {
            "instrument_token": 123,
            "exchange": "NFO",
            "trading_symbol": "NIFTY26OCT22000CE",
            "underlying": "NIFTY",
            "expiry": date(2026, 10, 1),
            "strike": 22_000.0,
            "option_type": "CE",
        }
        invalid = (
            ("boolean token", {"instrument_token": True}, "token"),
            ("zero token", {"instrument_token": 0}, "token"),
            ("datetime expiry", {"expiry": datetime(2026, 10, 1)}, "expiry"),
            ("non-finite strike", {"strike": float("nan")}, "strike"),
            ("unsupported side", {"option_type": "FUT"}, "type"),
        )
        for name, overrides, message in invalid:
            with self.subTest(metadata=name):
                with self.assertRaisesRegex(ValueError, message):
                    OptionContract(**{**valid, **overrides})

    def test_option_selector_requires_spot_prices_and_rejects_invalid_range(self):
        with self.assertRaisesRegex(ValueError, "spot price"):
            select_nearest_option_contracts(option_master(), spot_prices={"NIFTY": 22_000})
        with self.assertRaisesRegex(ValueError, "strikes_each_side"):
            select_nearest_option_contracts(
                option_master(), spot_prices={"NIFTY": 22_000, "BANKNIFTY": 48_000}, strikes_each_side=-1
            )

    def test_initial_watchlist_includes_index_quotes_and_selected_option_contracts(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        watchlist, spots = adapter.initial_option_watchlist(as_of=date(2026, 9, 30))
        self.assertEqual(len(watchlist), 46)
        self.assertEqual(set(spots), {"NIFTY", "BANKNIFTY"})
        self.assertEqual({row["tradingsymbol"] for row in watchlist[:2]}, {"NIFTY 50", "NIFTY BANK"})
        self.assertTrue(all(row.get("instrument_type", "") in {"CE", "PE", ""} for row in watchlist))

    def test_initial_watchlist_rejects_stale_index_spot_quotes(self):
        client = FakeKiteClient()
        client.quote = lambda *symbols: {
            symbol: {
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=2),
                "last_price": 22_000 if "50" in symbol else 48_000,
            }
            for symbol in symbols
        }
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=client)
        with self.assertRaisesRegex(ValueError, "stale"):
            adapter.initial_option_watchlist(as_of=date(2026, 9, 30))

    def test_historical_candles_normalize_kite_local_timestamps_and_oi(self):
        client = FakeKiteClient()
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=client)
        rows = adapter.historical_candles(
            instrument_token=10,
            from_date="2025-01-01",
            to_date="2025-01-02",
            interval="minute",
            symbol="nifty",
            include_open_interest=True,
        )
        self.assertEqual(rows[0]["symbol"], "NIFTY")
        self.assertEqual(rows[0]["timestamp"], "2025-01-01T03:45:00+00:00")
        self.assertEqual(rows[0]["open_interest"], 250)
        self.assertTrue(client.history_call[1]["oi"])

    def test_kite_tick_normalizes_quote_depth_and_event_envelope(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        tick = adapter.normalize_tick(
            {
                "instrument_token": 17,
                "last_price": 100.25,
                "volume_traded": 2000,
                "oi": 300,
                "exchange_timestamp": datetime(2026, 9, 30, 10, 0),
                "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99.5},
                "depth": {"buy": [{"price": 100, "quantity": 10, "orders": 2}], "sell": []},
            },
            {"tradingsymbol": "NIFTY26OCT22000CE", "exchange": "NFO"},
        )
        event = tick.as_event()
        self.assertEqual(tick.timestamp, "2026-09-30T04:30:00+00:00")
        self.assertEqual(tick.open_interest, 300)
        self.assertEqual(tick.bids[0]["price"], 100)
        self.assertEqual(event["event_type"], "market.tick_received")

    def test_kite_tick_normalization_rejects_non_positive_prices(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        instrument = {"tradingsymbol": "NIFTY", "exchange": "NSE"}

        for price in (-100, 0):
            with self.subTest(price=price), self.assertRaisesRegex(ValueError, "finite positive last price"):
                adapter.normalize_tick({
                    "instrument_token": 17,
                    "last_price": price,
                    "exchange_timestamp": datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc),
                }, instrument)

    @pytest.mark.chaos
    def test_tick_quality_rejects_duplicates_stale_and_out_of_order_ticks(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        instrument = {"tradingsymbol": "NIFTY", "exchange": "NSE"}
        now = datetime.now(timezone.utc)
        fresh = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 100,
            "exchange_timestamp": now, "ohlc": {}, "depth": {},
        }, instrument)
        stale = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 99,
            "exchange_timestamp": now - timedelta(minutes=1), "ohlc": {}, "depth": {},
        }, instrument)
        older = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 98,
            "exchange_timestamp": now - timedelta(seconds=1), "ohlc": {}, "depth": {},
        }, instrument)
        gate = MarketTickQualityGate()
        self.assertIsNone(gate.review(fresh, now=now))
        self.assertEqual(gate.review(fresh, now=now), "duplicate_tick")
        self.assertEqual(gate.review(stale, now=now), "stale_tick")
        # A new event at the same market timestamp is permitted, but an older one is not.
        equal_time = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 100.1,
            "exchange_timestamp": now, "ohlc": {}, "depth": {},
        }, instrument)
        self.assertIsNone(gate.review(equal_time, now=now))
        self.assertEqual(gate.review(older, now=now), "out_of_order_tick")

    @pytest.mark.chaos
    def test_out_of_order_sequence_is_dropped_instead_of_reordered(self):
        gate = MarketTickQualityGate(max_age_seconds=30)
        latest = datetime(2026, 10, 1, 10, 15, 3, tzinfo=timezone.utc)

        def tick(event_id, second):
            timestamp = latest.replace(second=second).isoformat()
            return SimpleNamespace(
                event_id=event_id,
                timestamp=timestamp,
                source="offline_fixture",
                symbol="NIFTY",
            )

        self.assertIsNone(gate.review(tick("tick-03", 3), now=latest))
        self.assertEqual(gate.review(tick("tick-01", 1), now=latest), "out_of_order_tick")
        self.assertEqual(gate.review(tick("tick-02", 2), now=latest), "out_of_order_tick")
        self.assertEqual(gate.status()["accepted"], 1)
        self.assertEqual(gate.status()["out_of_order"], 2)

    def test_tick_rejection_audit_is_rate_limited_per_symbol_and_reason(self):
        gate = MarketTickQualityGate()
        now = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
        self.assertTrue(gate.should_audit_rejection("NIFTY", "stale_tick", now=now))
        self.assertFalse(gate.should_audit_rejection("NIFTY", "stale_tick", now=now + timedelta(seconds=30)))
        self.assertTrue(gate.should_audit_rejection("BANKNIFTY", "stale_tick", now=now + timedelta(seconds=30)))
        self.assertTrue(gate.should_audit_rejection("NIFTY", "stale_tick", now=now + timedelta(seconds=61)))

    def test_tick_store_persists_numerical_fields_and_rejects_duplicate_event(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        tick = adapter.normalize_tick({
            "instrument_token": 17,
            "last_price": 100.25,
            "volume_traded": 2000,
            "oi": 300,
            "exchange_timestamp": datetime.now(timezone.utc),
            "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99.5},
            "depth": {"buy": [{"price": 100, "quantity": 10, "orders": 2}], "sell": []},
        }, {"tradingsymbol": "NIFTY26OCT22000CE", "exchange": "NFO"})
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "ticks.db")
            self.assertTrue(store.save_market_tick(tick.as_event()))
            self.assertFalse(store.save_market_tick(tick.as_event()))
            stored = store.read_market_ticks("NIFTY26OCT22000CE")[0]
            self.assertEqual(stored["last_price"], 100.25)
            self.assertEqual(stored["open_interest"], 300)
            self.assertEqual(stored["bids"][0]["quantity"], 10)

    def test_tick_aggregator_builds_one_minute_bars_and_rolls_forward(self):
        adapter = KiteMarketDataAdapter(api_key="key", access_token="token", kite_client=FakeKiteClient())
        instrument = {"tradingsymbol": "NIFTY", "exchange": "NSE"}
        aggregator = MinuteCandleAggregator()
        first = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 100, "last_traded_quantity": 4,
            "exchange_timestamp": datetime(2026, 9, 30, 10, 0, 10, tzinfo=timezone.utc),
            "ohlc": {}, "depth": {},
        }, instrument)
        same_minute = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 103, "last_traded_quantity": 6,
            "exchange_timestamp": datetime(2026, 9, 30, 10, 0, 50, tzinfo=timezone.utc),
            "ohlc": {}, "depth": {},
        }, instrument)
        next_minute = adapter.normalize_tick({
            "instrument_token": 17, "last_price": 102, "last_traded_quantity": 2,
            "exchange_timestamp": datetime(2026, 9, 30, 10, 1, 5, tzinfo=timezone.utc),
            "ohlc": {}, "depth": {},
        }, instrument)
        first_bar = aggregator.add(first)
        current_bar = aggregator.add(same_minute)
        next_bar = aggregator.add(next_minute)
        self.assertEqual(first_bar.timestamp, "2026-09-30T10:00:00+00:00")
        self.assertEqual(current_bar.open, 100)
        self.assertEqual(current_bar.high, 103)
        self.assertEqual(current_bar.close, 103)
        self.assertEqual(current_bar.volume, 10)
        self.assertEqual(next_bar.timestamp, "2026-09-30T10:01:00+00:00")
        self.assertEqual(next_bar.volume, 2)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            aggregator.add(first)

    def test_minute_aggregator_closes_complete_bars_and_keeps_latest_bar_provisional(self):
        aggregator = MinuteCandleAggregator()
        start = datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)

        def tick(minute, second, price, quantity):
            timestamp = (start + timedelta(minutes=minute, seconds=second)).isoformat()
            return SimpleNamespace(
                symbol="NIFTY",
                timestamp=timestamp,
                last_price=price,
                last_trade_quantity=quantity,
            )

        aggregator.add(tick(0, 10, 100, 3))
        aggregator.add(tick(0, 50, 103, 4))
        self.assertEqual(aggregator.take_closed(), [])

        aggregator.add(tick(1, 10, 99, 5))
        first_closed = aggregator.take_closed()
        self.assertEqual(len(first_closed), 1)
        self.assertEqual(
            (first_closed[0].open, first_closed[0].high, first_closed[0].low,
             first_closed[0].close, first_closed[0].volume),
            (100, 103, 100, 103, 7),
        )

        aggregator.add(tick(2, 10, 102, 6))
        second_closed = aggregator.take_closed()
        self.assertEqual(len(second_closed), 1)
        self.assertEqual(
            (second_closed[0].open, second_closed[0].high, second_closed[0].low,
             second_closed[0].close, second_closed[0].volume),
            (99, 99, 99, 99, 5),
        )
        self.assertEqual(aggregator.take_closed(), [])

    def test_minute_aggregator_resets_bar_metrics_at_day_rollover_and_preserves_history(self):
        aggregator = MinuteCandleAggregator()
        prior_session_tick = SimpleNamespace(
            symbol="NIFTY",
            timestamp="2026-10-01T09:59:30+00:00",
            last_price=100,
            last_trade_quantity=7,
        )
        current_session_tick = SimpleNamespace(
            symbol="NIFTY",
            timestamp="2026-10-02T03:45:15+00:00",
            last_price=110,
            last_trade_quantity=2,
        )

        prior_bar = aggregator.add(prior_session_tick)
        current_bar = aggregator.add(current_session_tick)
        closed_bars = aggregator.take_closed()

        self.assertEqual(closed_bars, [prior_bar])
        self.assertEqual((prior_bar.open, prior_bar.close, prior_bar.volume), (100, 100, 7))
        self.assertEqual(
            (current_bar.open, current_bar.high, current_bar.low, current_bar.close, current_bar.volume),
            (110, 110, 110, 110, 2),
        )
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            aggregator.add(prior_session_tick)

        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "session-rollover.db")
            self.assertEqual(store.save_candles([*closed_bars, current_bar]), 2)
            history = store.read_candles("NIFTY", timeframe="1m")

        self.assertEqual(history, [prior_bar, current_bar])

    def test_session_resampler_derives_complete_timeframes_from_canonical_minutes(self):
        resampler = self._session_resampler()
        results = resampler.resample_all(self._one_minute_bars())

        five_minute = results["5m"].candles
        fifteen_minute = results["15m"].candles
        daily = results["1d"].candles
        self.assertEqual(len(five_minute), 3)
        self.assertEqual(
            (five_minute[0].timestamp, five_minute[0].open, five_minute[0].high,
             five_minute[0].low, five_minute[0].close, five_minute[0].volume,
             five_minute[0].open_interest),
            ("2026-10-01T03:45:00+00:00", 100, 106, 99, 105, 15, 1004),
        )
        self.assertEqual(len(fifteen_minute), 1)
        self.assertEqual((fifteen_minute[0].close, fifteen_minute[0].volume), (115, 120))
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily[0].timeframe, "1d")
        self.assertEqual(results["1h"].candles, ())
        self.assertEqual(results["1h"].incomplete_buckets[0].reasons, ("partial_session_bucket",))

    def test_session_resampler_skips_and_reports_buckets_with_missing_minutes(self):
        result = self._session_resampler().resample(self._one_minute_bars(missing={2}), "5m")

        self.assertEqual(len(result.candles), 2)
        self.assertEqual(len(result.incomplete_buckets), 1)
        incomplete = result.incomplete_buckets[0]
        self.assertEqual(incomplete.timestamp, "2026-10-01T03:45:00+00:00")
        self.assertEqual(incomplete.reasons, ("missing_minutes",))
        self.assertEqual(incomplete.missing_minutes, ("2026-10-01T03:47:00+00:00",))

    def test_session_resampler_ignores_future_minutes_and_marks_current_bucket_in_progress(self):
        resampler = self._session_resampler()
        candles = self._one_minute_bars()
        in_progress = resampler.resample(
            candles,
            "5m",
            as_of=datetime(2026, 10, 1, 9, 18, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        )
        completed = resampler.resample(
            candles,
            "5m",
            as_of=datetime(2026, 10, 1, 9, 20, tzinfo=ZoneInfo("Asia/Kolkata")),
        )

        self.assertEqual(in_progress.candles, ())
        self.assertEqual(len(in_progress.incomplete_buckets), 1)
        self.assertEqual(in_progress.incomplete_buckets[0].reasons, ("in_progress",))
        self.assertEqual(in_progress.incomplete_buckets[0].missing_minutes, ())
        self.assertEqual(len(completed.candles), 1)
        self.assertEqual(len(completed.incomplete_buckets), 1)
        self.assertEqual(completed.incomplete_buckets[0].reasons, ("in_progress",))

    def test_session_resampler_rejects_late_duplicate_and_holiday_candles(self):
        resampler = self._session_resampler()
        candles = self._one_minute_bars()
        duplicate = Candle(
            candles[0].symbol, candles[0].timestamp, candles[0].open, candles[0].high,
            candles[0].low, candles[0].close, candles[0].volume, candles[0].open_interest, "1m",
        )
        with self.assertRaisesRegex(ValueError, "late or out-of-order"):
            resampler.resample(list(reversed(candles)), "5m")
        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            resampler.resample([candles[0], duplicate], "5m")

        holiday_resampler = self._session_resampler(holidays=(date(2026, 10, 1),))
        with self.assertRaisesRegex(ValueError, "configured holiday"):
            holiday_resampler.resample(candles, "5m")

    def test_session_resampler_reports_missing_sessions_but_skips_holidays_and_weekends(self):
        resampler = self._session_resampler(holidays=(date(2026, 10, 2),))
        candles = [
            *self._one_minute_bars(session_date=date(2026, 10, 1)),
            *self._one_minute_bars(session_date=date(2026, 10, 6)),
        ]

        result = resampler.resample(
            candles,
            "1d",
            as_of=datetime(2026, 10, 6, 10, tzinfo=ZoneInfo("Asia/Kolkata")),
        )

        self.assertEqual(len(result.candles), 2)
        self.assertEqual(result.missing_sessions, (date(2026, 10, 5),))

    def test_stream_subscribes_full_mode_and_emits_normalized_ticks(self):
        emitted = []
        statuses = []
        adapter = KiteMarketDataAdapter(
            api_key="key", access_token="token", kite_client=FakeKiteClient(), ticker_factory=FakeTicker
        )
        ticker = adapter.start_stream(
            [{"instrument_token": 17, "tradingsymbol": "NIFTY26OCT22000CE", "exchange": "NFO"}],
            on_tick=emitted.append,
            on_status=lambda name, payload: statuses.append((name, payload)),
        )
        self.assertTrue(ticker.threaded)
        self.assertEqual(ticker.calls[0], ("subscribe", [17]))
        self.assertEqual(ticker.calls[1], ("mode", FakeTicker.MODE_FULL, [17]))
        self.assertEqual(emitted[0].symbol, "NIFTY26OCT22000CE")
        self.assertEqual(statuses[0][0], "market.connected")

    def test_stream_rejects_more_than_kites_connection_limit(self):
        adapter = KiteMarketDataAdapter(
            api_key="key", access_token="token", kite_client=FakeKiteClient(),
            ticker_factory=FakeTicker, max_subscriptions=MAX_KITE_SUBSCRIPTIONS + 1,
        )
        instruments = [{"instrument_token": token, "tradingsymbol": str(token)} for token in range(MAX_KITE_SUBSCRIPTIONS + 1)]
        with self.assertRaisesRegex(ValueError, "subscription limit"):
            adapter.start_stream(instruments, on_tick=lambda _tick: None)

    def test_openai_provider_uses_json_mode_and_output_budget(self):
        output = {"direction": "NONE"}
        fake = FakeOpenAIClient(json.dumps(output))
        provider = OpenAIModelProvider(api_key="test-only", client=fake, max_output_tokens=700)
        result = provider.generate_structured(model="fake-model", prompt="Return JSON", context={"safe": True})
        self.assertEqual(result, output)
        self.assertEqual(fake.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fake.calls[0]["max_completion_tokens"], 700)

    def test_openai_provider_rejects_oversized_context_and_malformed_json(self):
        provider = OpenAIModelProvider(api_key="test-only", client=FakeOpenAIClient("{}"), max_context_chars=5)
        with self.assertRaisesRegex(ValueError, "character budget"):
            provider.generate_structured(model="fake", prompt="prompt", context={"payload": "too long"})
        malformed = OpenAIModelProvider(api_key="test-only", client=FakeOpenAIClient("not-json"))
        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            malformed.generate_structured(model="fake", prompt="prompt", context={})

    def test_copilot_provider_disables_all_tools_and_deletes_session(self):
        session = FakeCopilotSession('{"direction":"NONE"}')
        clients = []

        def client_factory(**options):
            client = FakeCopilotClient(session, **options)
            clients.append(client)
            return client

        provider = CopilotModelProvider(
            client_factory=client_factory,
            permission_denied_factory=lambda: "denied",
        )
        result = provider.generate_structured(model="auto", prompt="Return JSON", context={"safe": True})

        self.assertEqual(result, {"direction": "NONE"})
        self.assertEqual(clients[0].session_options["available_tools"], [])
        self.assertEqual(clients[0].session_options["excluded_tools"], [])
        self.assertEqual(clients[0].session_options["tools"], [])
        self.assertEqual(clients[0].session_options["mcp_servers"], {})
        self.assertIsNotNone(clients[0].session_options["on_permission_request"]())
        self.assertEqual(clients[0].deleted_sessions, ["fake-session"])
        self.assertIn('"safe":true', session.calls[0][0])
        self.assertEqual(set(COPILOT_HYPOTHESIS_SCHEMA["required"]), REQUIRED_HYPOTHESIS_FIELDS)
        self.assertFalse(COPILOT_HYPOTHESIS_SCHEMA["additionalProperties"])
        self.assertEqual(session.calls[0][2], COPILOT_HYPOTHESIS_SCHEMA)

    def test_copilot_provider_rejects_tool_calls_and_invalid_responses(self):
        tool_call = CopilotModelProvider(
            client_factory=lambda **options: FakeCopilotClient(
                FakeCopilotSession('{"direction":"NONE"}', ["read_file"]), **options
            ),
            permission_denied_factory=lambda: "denied",
        )
        with self.assertRaisesRegex(ValueError, "tool request"):
            tool_call.generate_structured(model="auto", prompt="prompt", context={})

        oversized = CopilotModelProvider(
            client_factory=lambda **options: FakeCopilotClient(FakeCopilotSession("{}"), **options),
            permission_denied_factory=lambda: "denied",
            max_context_chars=3,
        )
        with self.assertRaisesRegex(ValueError, "character budget"):
            oversized.generate_structured(model="auto", prompt="prompt", context={"long": True})

        malformed = CopilotModelProvider(
            client_factory=lambda **options: FakeCopilotClient(FakeCopilotSession("not-json"), **options),
            permission_denied_factory=lambda: "denied",
        )
        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            malformed.generate_structured(model="auto", prompt="prompt", context={})


if __name__ == "__main__":
    unittest.main()