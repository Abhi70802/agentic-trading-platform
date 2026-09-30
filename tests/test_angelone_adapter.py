import unittest
from datetime import date, datetime, timedelta

from angelone_adapter import (
    AngelOneCredentials,
    AngelOneMarketDataAdapter,
    AngelOneRestClient,
    _verified_websocket_class,
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

    def post(self, url, *, json, headers, timeout):
        self.calls.append((url, json, headers, timeout))
        if url.endswith("loginByPassword"):
            return FakeHttpResponse({"status": True, "data": {
                "jwtToken": "jwt-secret", "refreshToken": "refresh-secret", "feedToken": "feed-secret",
            }})
        return FakeHttpResponse({"status": True, "data": {"fetched": []}})


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
        socket.connect()
        self.assertNotIn("sslopt", socket.wsapp.run_options)
        self.assertEqual(socket.wsapp.run_options["ping_interval"], 10)

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