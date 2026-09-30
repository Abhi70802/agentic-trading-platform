import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from kite_adapter import (
    KiteCredentials,
    KiteMarketDataAdapter,
    MAX_KITE_SUBSCRIPTIONS,
    select_nearest_option_contracts,
)
from llm_gateway import CopilotModelProvider, OpenAIModelProvider
from marketdata import MarketTickQualityGate, MinuteCandleAggregator
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

    async def send_and_wait(self, prompt, *, timeout):
        self.calls.append((prompt, timeout))
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


class ProviderAdapterTests(unittest.TestCase):
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