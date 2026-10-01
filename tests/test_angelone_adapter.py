import unittest
import threading
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from angelone_adapter import (
    AngelOneCredentials,
    AngelOneMarketDataAdapter,
    AngelOneRestClient,
    _verified_websocket_class,
    normalize_angelone_instrument_master,
)


def master_rows():
    rows = []
    token = 1000
    for name, center, step in (("NIFTY", 22_000, 50), ("BANKNIFTY", 48_000, 100)):
        for expiry in ("01OCT2026", "08OCT2026"):
            for offset in range(-6, 7):
                strike = (center + offset * step) * 100
                for side in ("CE", "PE"):
                    rows.append({
                        "token": str(token),
                        "symbol": f"{name}{expiry}{center + offset * step}{side}",
                        "name": name,
                        "expiry": expiry,
                        "strike": str(strike),
                        "instrumenttype": "OPTIDX",
                        "exch_seg": "NFO",
                        "lotsize": "25",
                        "tick_size": "5.000000",
                    })
                    token += 1
    return rows


class FakeSmartApi:
    def __init__(self, *, login_response=None, quote_timestamp=None):
        self.login_response = login_response or {
            "status": True,
            "data": {"jwtToken": "jwt-secret", "refreshToken": "refresh-secret", "feedToken": "feed-secret"},
        }
        self.quote_timestamp = quote_timestamp or datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.calls = []

    def generateSession(self, client_code, pin, totp):
        self.calls.append((client_code, pin, totp))
        return self.login_response

    def getMarketData(self, mode, exchange_tokens):
        self.calls.append((mode, exchange_tokens))
        values = {"99926000": 22_010, "99926009": 48_040}
        fetched = [
            {"symbolToken": token, "ltp": values[token], "exchFeedTime": self.quote_timestamp}
            for token in exchange_tokens["NSE"]
        ]
        return {"status": True, "data": {"fetched": fetched}}

    def getCandleData(self, _params):
        return {"status": True, "data": [["2026-09-29T09:15:00+05:30", 100, 102, 99, 101, 2000]]}

    def getOIData(self, _params):
        return {"status": True, "data": [{"time": "2026-09-29T09:15:00+05:30", "oi": 350}]}

    def getPosition(self):
        return getattr(self, "positions_response", {"status": True, "data": {"net": [], "day": []}})

    def getHolding(self):
        return getattr(self, "holdings_response", {"status": True, "data": []})

    def terminateSession(self, _client_code):
        return {"status": True}


class FakeWebSocket:
    SNAP_QUOTE = 3

    def __init__(self, auth_token, api_key, client_code, feed_token):
        self.auth = (auth_token, api_key, client_code, feed_token)
        self.subscriptions = []

    def subscribe(self, correlation_id, mode, token_list):
        self.subscriptions.append((correlation_id, mode, token_list))

    def connect(self):
        self.on_open(self)
        self.on_data(self, {
            "token": "99926000",
            "exchange_type": 1,
            "sequence_number": 12,
            "exchange_timestamp": int(datetime.now().timestamp() * 1000),
            "last_traded_price": 2_201_000,
            "last_traded_quantity": 1,
            "volume_trade_for_the_day": 1000,
            "open_interest": 0,
            "open_price_of_the_day": 2_190_000,
            "high_price_of_the_day": 2_205_000,
            "low_price_of_the_day": 2_185_000,
            "closed_price": 2_195_000,
            "best_5_buy_data": [{"price": 2_200_900, "quantity": 10, "no of orders": 2}],
            "best_5_sell_data": [{"price": 2_201_100, "quantity": 12, "no of orders": 3}],
        })

    def close_connection(self):
        self.closed = True


class FakeHttpResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeHttpSession:
    def __init__(self):
        self.calls = []
        self.get_responses = {}

    def post(self, url, *, json, headers, timeout):
        self.calls.append((url, json, headers, timeout))
        if url.endswith("loginByPassword"):
            return FakeHttpResponse({"status": True, "data": {
                "jwtToken": "jwt-secret", "refreshToken": "refresh-secret", "feedToken": "feed-secret",
            }})
        return FakeHttpResponse({"status": True, "data": {"fetched": []}})

    def get(self, url, *, headers, timeout):
        self.calls.append((url, None, headers, timeout))
        return FakeHttpResponse(self.get_responses.get(url, {"status": True, "data": []}))


class FakeWebSocketClient:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs

    def run_forever(self, **kwargs):
        self.run_options = kwargs


class AngelOneAdapterTests(unittest.TestCase):
    def adapter(self, smart_api=None):
        result = AngelOneMarketDataAdapter(
            api_key="api-key",
            smart_api=smart_api or FakeSmartApi(),
            websocket_factory=FakeWebSocket,
            instrument_loader=master_rows,
        )
        result.login(client_code="AB1234", pin="7391", totp="123456")
        return result

    def test_credential_repr_redacts_api_key(self):
        self.assertNotIn("do-not-show", repr(AngelOneCredentials("do-not-show")))

    def test_quiet_rest_client_redacts_auth_failures_and_authenticates_feed_calls(self):
        http = FakeHttpSession()
        client = AngelOneRestClient("private-api-key", http_session=http)
        client._public_ip = "203.0.113.4"
        result = client.generateSession("AB1234", "7391", "123456")
        self.assertTrue(result["status"])
        url, body, headers, timeout = http.calls[0]
        self.assertTrue(url.endswith("loginByPassword"))
        self.assertEqual(body, {"clientcode": "AB1234", "password": "7391", "totp": "123456"})
        self.assertEqual(headers["X-PrivateKey"], "private-api-key")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 10)
        client.getMarketData("FULL", {"NSE": ["99926000"]})
        self.assertEqual(http.calls[1][2]["Authorization"], "Bearer jwt-secret")

    def test_read_only_portfolio_gets_use_authenticated_documented_paths(self):
        http = FakeHttpSession()
        client = AngelOneRestClient("private-api-key", http_session=http)
        client._public_ip = "203.0.113.4"
        client.generateSession("AB1234", "7391", "123456")
        position_url = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/getPosition"
        holding_url = "https://apiconnect.angelone.in/rest/secure/angelbroking/portfolio/v1/getHolding"
        http.get_responses = {
            position_url: {"status": True, "data": {"net": [], "day": []}},
            holding_url: {"status": True, "data": []},
        }

        self.assertTrue(client.getPosition()["status"])
        self.assertTrue(client.getHolding()["status"])
        self.assertEqual([call[0] for call in http.calls[1:]], [position_url, holding_url])
        self.assertTrue(all(call[2]["Authorization"] == "Bearer jwt-secret" for call in http.calls[1:]))
        self.assertTrue(all(call[3] == 10 for call in http.calls[1:]))

    def test_portfolio_snapshot_keeps_holdings_separate_and_rejects_invalid_rows(self):
        broker = FakeSmartApi()
        broker.positions_response = {"status": True, "data": {
            "net": [
                {"tradingsymbol": "RELIANCE-EQ", "exchange": "NSE", "symboltoken": "2885",
                 "producttype": "CARRYFORWARD", "netqty": "5", "avgnetprice": "2500.50", "ltp": "2510"},
                {"tradingsymbol": "BROKEN", "exchange": "NSE", "symboltoken": "1", "netqty": "NaN"},
            ],
            "day": [{"tradingsymbol": "IGNORED", "exchange": "NSE", "symboltoken": "2", "netqty": "9"}],
        }}
        broker.holdings_response = {"status": True, "data": [{
            "tradingsymbol": "TCS-EQ", "exchange": "NSE", "symboltoken": "11536",
            "quantity": "2", "averageprice": "3200", "ltp": "3250",
        }]}
        snapshot = self.adapter(broker).read_portfolio_snapshot()

        self.assertEqual(len(snapshot["positions"]), 1)
        self.assertEqual(snapshot["positions"][0]["quantity"], 5)
        self.assertEqual(snapshot["positions"][0]["market_value"], "12550")
        self.assertEqual(snapshot["holdings"][0]["quantity"], 2)
        self.assertEqual(snapshot["rejected_rows"], {"positions": 1, "holdings": 0})
        self.assertFalse(snapshot["orders_enabled"])
        self.assertNotIn("day", str(snapshot))

    def test_portfolio_snapshot_does_not_treat_broker_failure_as_empty_account(self):
        broker = FakeSmartApi()
        broker.positions_response = {"status": False, "message": "private broker detail"}
        adapter = self.adapter(broker)

        with self.assertRaisesRegex(RuntimeError, "response is invalid") as error:
            adapter.read_portfolio_snapshot()
        self.assertNotIn("private broker detail", str(error.exception))

    def test_verified_websocket_wrapper_does_not_disable_tls_verification(self):
        class BaseSocket:
            ROOT_URI = "wss://example.invalid/feed"
            HEART_BEAT_INTERVAL = 10

            def _on_open(self, *_args): pass
            def _on_message(self, *_args): pass
            def _on_error(self, *_args): pass
            def _on_close(self, *_args): pass
            def _on_data(self, *_args): pass
            def _on_ping(self, *_args): pass
            def _on_pong(self, *_args): pass

        class FakeWebSocketModule:
            WebSocketApp = FakeWebSocketClient

        websocket_class = _verified_websocket_class(BaseSocket, FakeWebSocketModule)
        socket = websocket_class()
        socket.auth_token = "auth"
        socket.api_key = "api"
        socket.client_code = "client"
        socket.feed_token = "feed"
        pongs = []
        socket.on_pong = lambda *_args: pongs.append(True)
        socket.connect()
        self.assertNotIn("sslopt", socket.wsapp.run_options)
        self.assertEqual(socket.wsapp.run_options["ping_interval"], 30)
        self.assertEqual(socket.wsapp.run_options["ping_timeout"], 10)
        socket.wsapp.kwargs["on_pong"](socket.wsapp, "ping")
        self.assertEqual(pongs, [True])

    def test_manual_login_keeps_tokens_redacted_and_pin_totp_not_on_session(self):
        client = FakeSmartApi()
        adapter = self.adapter(client)
        session = adapter.session
        self.assertEqual(client.calls[0], ("AB1234", "7391", "123456"))
        self.assertNotIn("jwt-secret", repr(session))
        self.assertNotIn("pin", session.__dict__)
        self.assertNotIn("totp", session.__dict__)
        self.assertTrue(session.is_current())

    def test_login_rejects_invalid_totp_and_auth_failures_without_echoing_secrets(self):
        adapter = AngelOneMarketDataAdapter(api_key="api-key", smart_api=FakeSmartApi())
        with self.assertRaisesRegex(ValueError, "six-digit TOTP"):
            adapter.login(client_code="AB1234", pin="7391", totp="not-a-code")
        failure = AngelOneMarketDataAdapter(
            api_key="api-key",
            smart_api=FakeSmartApi(login_response={"status": False, "message": "secret echoed by broker"}),
        )
        with self.assertRaisesRegex(RuntimeError, "authentication failed") as error:
            failure.login(client_code="AB1234", pin="7391", totp="123456")
        self.assertNotIn("secret echoed", str(error.exception))

    def test_option_watchlist_selects_nearest_expiry_atm_plus_five(self):
        adapter = self.adapter()
        watchlist, spots = adapter.initial_option_watchlist(as_of=date(2026, 9, 30))
        self.assertEqual(len(watchlist), 46)
        self.assertEqual(set(spots), {"NIFTY", "BANKNIFTY"})
        options = [row for row in watchlist if row["instrument_type"] in {"CE", "PE"}]
        self.assertEqual(len(options), 44)
        self.assertTrue(all(row["expiry"] == "2026-10-01" for row in options))
        for name in ("NIFTY", "BANKNIFTY"):
            contracts = [row for row in options if row["name"] == name]
            self.assertEqual(len({row["strike"] for row in contracts}), 11)

    def test_instrument_master_normalizes_options_into_india_reference_policy(self):
        policy, rejected = normalize_angelone_instrument_master(
            master_rows(),
            price_scale=Decimal("100"),
            as_of=date(2026, 10, 1),
        )

        self.assertEqual(rejected, {})
        self.assertEqual({exchange.code for exchange in policy.exchanges}, {"NSE"})
        self.assertEqual({item.symbol for item in policy.instruments}, {"NIFTY", "BANKNIFTY"})
        self.assertEqual(len(policy.contracts), 104)
        option = policy.contracts[0]
        self.assertEqual(option.instrument.exchange_code, "NSE")
        self.assertEqual(option.instrument.segment, "NFO")
        self.assertEqual(option.instrument.asset_type.value, "OPTION")
        self.assertEqual(option.tick_size.value, Decimal("0.05"))
        self.assertEqual(option.lot_size.quantity, 25)
        self.assertEqual(len(policy.expiry_calendars), 2)
        self.assertEqual(policy.sessions, ())
        self.assertEqual(policy.holidays, ())

    def test_instrument_master_skips_unknown_segments_and_malformed_contracts(self):
        rows = [
            None,
            {"symbol": "UNKNOWN", "exch_seg": "UNKNOWN", "instrumenttype": "OPTIDX"},
            {"symbol": "EMPTY-EQ", "name": "EMPTY", "token": None, "exch_seg": "NSE", "instrumenttype": ""},
            {
                "symbol": "NIFTY26OCT22000CE",
                "name": "NIFTY",
                "expiry": "01OCT2026",
                "strike": "2200000",
                "instrumenttype": "OPTIDX",
                "exch_seg": "NFO",
                "lotsize": "25",
                "token": "1001",
            },
        ]
        policy, rejected = normalize_angelone_instrument_master(
            rows,
            price_scale=Decimal("100"),
            as_of=date(2026, 10, 1),
        )
        self.assertEqual(policy.contracts, ())
        self.assertEqual(rejected["invalid_master_row"], 1)
        self.assertEqual(rejected["unsupported_exchange_segment"], 1)
        self.assertEqual(rejected["missing_symbol_or_token"], 1)
        self.assertEqual(rejected["invalid_derivative_reference_fields"], 1)

    def test_instrument_master_rejects_boolean_and_fractional_lot_sizes(self):
        rows = [dict(master_rows()[index]) for index in range(2)]
        rows[0]["lotsize"] = True
        rows[1]["lotsize"] = 25.5

        policy, rejected = normalize_angelone_instrument_master(
            rows,
            price_scale=Decimal("100"),
            as_of=date(2026, 10, 1),
        )

        self.assertEqual(policy.contracts, ())
        self.assertEqual(rejected["invalid_derivative_reference_fields"], 2)

    def test_instrument_master_requires_explicit_positive_price_scale(self):
        with self.assertRaisesRegex(ValueError, "price scale"):
            normalize_angelone_instrument_master(master_rows(), price_scale=Decimal("0"))

    def test_option_watchlist_rejects_stale_spot_quote(self):
        client = FakeSmartApi(quote_timestamp=(datetime.now() - timedelta(minutes=2)).strftime("%d-%b-%Y %H:%M:%S"))
        adapter = self.adapter(client)
        with self.assertRaisesRegex(ValueError, "stale"):
            adapter.initial_option_watchlist(as_of=date(2026, 9, 30))

    def test_subscription_only_watchlist_allows_stale_reference_but_marks_it(self):
        client = FakeSmartApi(quote_timestamp=(datetime.now() - timedelta(minutes=2)).strftime("%d-%b-%Y %H:%M:%S"))
        adapter = self.adapter(client)
        watchlist, _spots = adapter.initial_option_watchlist(
            as_of=date(2026, 9, 30),
            allow_stale_quotes=True,
        )
        self.assertEqual(len(watchlist), 46)
        self.assertTrue(adapter.spot_reference_stale)
        self.assertTrue(all(age > 30 for age in adapter.spot_reference_age_seconds.values()))

    def test_subscription_only_watchlist_marks_future_clock_skew_as_provisional(self):
        client = FakeSmartApi(quote_timestamp=(datetime.now() + timedelta(minutes=2)).strftime("%d-%b-%Y %H:%M:%S"))
        adapter = self.adapter(client)
        watchlist, _spots = adapter.initial_option_watchlist(
            as_of=date(2026, 9, 30),
            allow_stale_quotes=True,
        )
        self.assertEqual(len(watchlist), 46)
        self.assertTrue(adapter.spot_reference_stale)
        self.assertTrue(all(age < 0 for age in adapter.spot_reference_age_seconds.values()))

    def test_history_import_normalizes_timezone_and_joins_oi(self):
        adapter = self.adapter()
        rows = adapter.historical_candles(
            instrument_token=12345,
            symbol="NIFTY26OCT22000CE",
            exchange="NFO",
            interval="minute",
            from_date=date(2026, 9, 29),
            to_date=date(2026, 9, 29),
            include_open_interest=True,
        )
        self.assertEqual(rows[0]["timestamp"], "2026-09-29T03:45:00+00:00")
        self.assertEqual(rows[0]["open_interest"], 350)

    def test_history_import_preserves_session_open_and_close_boundaries(self):
        smart_api = FakeSmartApi()
        smart_api.getCandleData = lambda params: (
            setattr(smart_api, "candle_params", params)
            or {"status": True, "data": [
                ["2026-09-29T09:15:00+05:30", 100, 102, 99, 101, 2000],
                ["2026-09-29T15:30:00+05:30", 101, 103, 100, 102, 3000],
            ]}
        )
        adapter = self.adapter(smart_api)

        rows = adapter.historical_candles(
            instrument_token=12345,
            symbol="NIFTY",
            exchange="NSE",
            interval="minute",
            from_date=date(2026, 9, 29),
            to_date=date(2026, 9, 29),
        )

        self.assertEqual(smart_api.candle_params["fromdate"], "2026-09-29 09:15")
        self.assertEqual(smart_api.candle_params["todate"], "2026-09-29 15:30")
        self.assertEqual(
            [row["timestamp"] for row in rows],
            ["2026-09-29T03:45:00+00:00", "2026-09-29T10:00:00+00:00"],
        )

    def test_tick_normalizes_prices_depth_and_event_schema(self):
        adapter = self.adapter()
        tick = adapter.normalize_tick({
            "token": "99926000",
            "exchange_type": 1,
            "sequence_number": 4,
            "exchange_timestamp": int(datetime.now().timestamp() * 1000),
            "last_traded_price": 2_201_000,
            "last_traded_quantity": 2,
            "volume_trade_for_the_day": 1200,
            "open_interest": 0,
            "open_price_of_the_day": 2_190_000,
            "high_price_of_the_day": 2_205_000,
            "low_price_of_the_day": 2_185_000,
            "closed_price": 2_195_000,
            "best_5_buy_data": [{"price": 2_200_900, "quantity": 10, "no of orders": 2}],
            "best_5_sell_data": [],
        }, {"exchange": "NSE", "tradingsymbol": "NIFTY 50"})
        self.assertEqual(tick.last_price, 22_010)
        self.assertEqual(tick.bids[0]["price"], 22_009)
        self.assertEqual(tick.open_interest, 0)
        self.assertEqual(tick.as_event()["source"], "angelone_smartapi")

    def test_option_quote_scenarios_preserve_oi_volume_and_spread(self):
        adapter = self.adapter()
        scenarios = (
            ("high OI liquid", 1_000_000, 100_000, 500, 500, 10_005),
            ("low OI low liquidity wide spread", 0, 1, 1, 1, 10_500),
        )
        ticks = {}
        for name, open_interest, volume, bid_quantity, ask_quantity, ask_price in scenarios:
            with self.subTest(scenario=name):
                tick = adapter.normalize_tick(
                    {
                        "token": "99926010",
                        "exchange_timestamp": int(datetime.now().timestamp() * 1000),
                        "last_traded_price": 10_200,
                        "last_traded_quantity": 1,
                        "volume_trade_for_the_day": volume,
                        "open_interest": open_interest,
                        "best_5_buy_data": [{"price": 10_000, "quantity": bid_quantity, "no of orders": 5}],
                        "best_5_sell_data": [{"price": ask_price, "quantity": ask_quantity, "no of orders": 4}],
                    },
                    {"exchange": "NFO", "tradingsymbol": "NIFTY26OCT22000CE"},
                )
                ticks[name] = tick
                self.assertEqual(tick.open_interest, open_interest)
                self.assertEqual(tick.volume, volume)
                self.assertEqual(tick.bids[0]["quantity"], bid_quantity)
                self.assertEqual(tick.asks[0]["quantity"], ask_quantity)

        liquid = ticks["high OI liquid"]
        wide = ticks["low OI low liquidity wide spread"]
        liquid_spread = liquid.asks[0]["price"] - liquid.bids[0]["price"]
        wide_spread = wide.asks[0]["price"] - wide.bids[0]["price"]
        self.assertGreater(liquid.open_interest, wide.open_interest)
        self.assertGreater(liquid.volume, wide.volume)
        self.assertGreater(wide_spread, liquid_spread)

    def test_tick_normalization_rejects_a_negative_price(self):
        adapter = self.adapter()

        with self.assertRaisesRegex(ValueError, "invalid last traded price"):
            adapter.normalize_tick({
                "token": "99926000",
                "exchange_timestamp": int(datetime.now().timestamp() * 1000),
                "last_traded_price": -10_000,
            }, {"exchange": "NSE", "tradingsymbol": "NIFTY 50"})

    def test_rest_fallback_quotes_use_the_normalized_tick_contract(self):
        adapter = self.adapter()
        ticks = []
        count = adapter._poll_rest_fallback(
            [{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}],
            on_tick=ticks.append,
        )

        self.assertEqual(count, 1)
        self.assertEqual(ticks[0].symbol, "NIFTY 50")
        self.assertEqual(ticks[0].last_price, 22_010)
        self.assertEqual(ticks[0].source, "angelone_smartapi")

    def test_websocket_starts_snap_quote_subscription_without_order_api(self):
        adapter = self.adapter()
        ticks = []
        statuses = []
        socket = adapter.start_stream(
            [{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}],
            on_tick=ticks.append,
            on_status=lambda name, data: statuses.append((name, data)),
        )
        adapter._websocket_thread.join(timeout=2)
        self.assertEqual(socket.subscriptions[0][0], "northstar1")
        self.assertEqual(socket.subscriptions[0][1], FakeWebSocket.SNAP_QUOTE)
        self.assertEqual(ticks[0].symbol, "NIFTY 50")
        self.assertEqual(statuses[0][0], "market.connected")
        self.assertFalse(hasattr(adapter, "place_order"))

    @pytest.mark.chaos
    def test_silent_feed_triggers_empty_rest_fallback_without_an_extra_tick(self):
        smart_api = FakeSmartApi()
        smart_api.getMarketData = lambda *_args: {"status": True, "data": {"fetched": []}}
        adapter = None

        class SilentAfterInitialTickWebSocket(FakeWebSocket):
            def connect(self):
                super().connect()
                adapter._stream_stop.wait(3)

        adapter = AngelOneMarketDataAdapter(
            api_key="api-key",
            smart_api=smart_api,
            websocket_factory=SilentAfterInitialTickWebSocket,
            instrument_loader=master_rows,
            stale_tick_seconds=1,
            max_reconnect_attempts=1,
        )
        adapter.fallback_interval_seconds = 0.01
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        ticks = []
        statuses = []
        empty_fallback = threading.Event()

        def record_status(event_type, payload):
            statuses.append((event_type, payload))
            if event_type == "market.rest_fallback_complete" and payload.get("quotes") == 0:
                empty_fallback.set()

        adapter.start_stream(
            [{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}],
            on_tick=ticks.append,
            on_status=record_status,
        )
        try:
            self.assertTrue(empty_fallback.wait(timeout=3), statuses)
        finally:
            adapter.logout()

        self.assertEqual(len(ticks), 1)
        self.assertIn(("market.rest_fallback_complete", {"quotes": 0}), statuses)

    def test_logout_clears_session_and_closes_feed(self):
        adapter = self.adapter()
        adapter.start_stream(
            [{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}],
            on_tick=lambda _tick: None,
        )
        adapter._websocket_thread.join(timeout=2)
        socket = adapter.websocket
        adapter.logout()
        self.assertIsNone(adapter.session)
        self.assertTrue(socket.closed)


if __name__ == "__main__":
    unittest.main()