import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from core import demo_candles
from llm_gateway import LLMGateway
from angelone_adapter import AngelOneSession
from store import EventStore


class FakeAngelOneAdapter:
    def __init__(self):
        self.session = None
        self.logged_out = False
        self.feed_started = False
        self.spot_reference_stale = False

    def login(self, *, client_code, pin, totp):
        self.received = (client_code, pin, totp)
        self.session = AngelOneSession(
            client_code=client_code,
            auth_token="jwt-secret",
            refresh_token="refresh-secret",
            feed_token="feed-secret",
            login_date=datetime.now(ZoneInfo("Asia/Kolkata")).date(),
        )
        return self.session

    def logout(self):
        self.logged_out = True
        self.session = None

    def initial_option_watchlist(self, *, allow_stale_quotes=False):
        self.allow_stale_quotes = allow_stale_quotes
        return ([{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}], {"NIFTY": 22000})

    def start_stream(self, instruments, *, on_tick, on_status):
        self.feed_started = True
        on_status("market.connected", {"subscriptions": len(instruments), "mode": "snap_quote"})

    def historical_candles(self, **kwargs):
        return [{
            "symbol": kwargs["symbol"],
            "timestamp": "2026-09-01T03:45:00+00:00",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 500,
            "open_interest": 250 if kwargs["include_open_interest"] else None,
        }]


class FakeModelProvider:
    name = "offline-fake"

    def generate_structured(self, *, model, prompt, context):
        return {
            "instrument": context["instruments"][0],
            "market": "NSE",
            "direction": "LONG",
            "time_horizon": "intraday",
            "entry_range": {"low": 100, "high": 101},
            "stop_loss": 99,
            "take_profit": 103,
            "thesis": "Mock evidence-linked hypothesis.",
            "supporting_evidence": [{"source_id": context["evidence_ids"][0], "claim": "Supplied price context"}],
            "contradictory_evidence": [],
            "invalidating_conditions": ["Close below stop"],
            "confidence": 0.8,
            "data_timestamp": context["data_timestamp"],
            "strategy": "mock",
        }


class ProviderStatusApiTests(unittest.TestCase):
    def setUp(self):
        api.angel_runtime.update({"adapter": None, "instruments": []})
        api.angel_state.update({"connected": False, "last_error": None, "watchlist_count": 0, "session_active": False})
        api.angel_state["spot_reference_stale"] = False
        api.angel_state["fresh_index_ticks"] = set()

    def tearDown(self):
        adapter = api.angel_runtime.get("adapter")
        if adapter is not None:
            adapter.logout()
        api.angel_runtime.update({"adapter": None, "instruments": []})
        api.angel_state.update({"connected": False, "last_error": None, "watchlist_count": 0, "session_active": False})
        api.angel_state["spot_reference_stale"] = False
        api.angel_state["fresh_index_ticks"] = set()

    def test_angelone_status_is_disabled_without_key_or_opt_in(self):
        settings = {
            "ANGELONE_API_KEY": "",
            "ANGELONE_MARKET_DATA_ENABLED": "false",
        }
        with patch.dict(os.environ, settings, clear=False), TestClient(api.app) as client:
            response = client.get("/api/angelone/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "angelone_smartapi")
        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["credentials_configured"])
        self.assertFalse(payload["session_active"])
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["option_underlyings"], ["NIFTY", "BANKNIFTY"])

    def test_health_never_claims_orders_enabled(self):
        with TestClient(api.app) as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["broker_execution_enabled"])

    def test_llm_status_reports_selected_provider_but_stays_disabled_by_default(self):
        with (
            patch.object(api, "LLM_ENABLED", False),
            patch.object(api, "LLM_PROVIDER", "copilot"),
            patch.object(api, "llm_gateway", LLMGateway()),
            TestClient(api.app) as client,
        ):
            response = client.get("/api/llm/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "copilot")
        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["tools_enabled"])
        self.assertFalse(payload["broker_access"])

    def test_login_is_disabled_by_default(self):
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "false"}, clear=False), TestClient(api.app) as client:
            response = client.post("/api/angelone/login", json={"client_code": "AB1234", "pin": "7391", "totp": "123456"})
        self.assertEqual(response.status_code, 503)

    def test_manual_login_and_feed_start_do_not_return_tokens_or_pin(self):
        adapter = FakeAngelOneAdapter()
        env = {"ANGELONE_MARKET_DATA_ENABLED": "true", "ANGELONE_API_KEY": "test-api-key"}
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "login.db")
            with (
                patch.dict(os.environ, env, clear=False),
                patch.object(api, "store", test_store),
                patch.object(api.AngelOneMarketDataAdapter, "from_environment", return_value=adapter),
                TestClient(api.app) as client,
            ):
                login = client.post("/api/angelone/login", json={"client_code": "AB1234", "pin": "7391", "totp": "123456"})
                feed = client.post("/api/angelone/feed/start")
                status = client.get("/api/angelone/status")
                response_text = login.text + status.text + feed.text
                session_fields = set(adapter.session.__dict__)
        self.assertEqual(login.status_code, 200)
        self.assertEqual(feed.status_code, 200)
        self.assertTrue(adapter.feed_started)
        self.assertTrue(status.json()["session_active"])
        self.assertTrue(status.json()["connected"])
        for secret in ("7391", "123456", "jwt-secret", "refresh-secret", "feed-secret", "test-api-key"):
            self.assertNotIn(secret, response_text)
        self.assertNotIn("pin", session_fields)
        self.assertNotIn("totp", session_fields)

    def test_login_helper_rejects_non_loopback_requests(self):
        with self.assertRaises(HTTPException) as error:
            api._require_loopback(SimpleNamespace(client=SimpleNamespace(host="198.51.100.42")))
        self.assertEqual(error.exception.status_code, 403)

    def test_tick_history_is_read_only_for_unseen_symbol(self):
        with TestClient(api.app) as client:
            response = client.get("/api/market/NO_TICKS/ticks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_historical_import_is_disabled_by_default(self):
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "false"}, clear=False), TestClient(api.app) as client:
            response = client.post("/api/angelone/history", json={
                "symbol": "NIFTY",
                "instrument_token": 99926000,
                "exchange": "NSE",
                "from_date": "2026-09-01",
                "to_date": "2026-09-02",
                "interval": "day",
            })
        self.assertEqual(response.status_code, 503)

    def test_historical_import_uses_documented_interval_limit(self):
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False), TestClient(api.app) as client:
            response = client.post("/api/angelone/history", json={
                "symbol": "NIFTY",
                "instrument_token": 99926000,
                "exchange": "NSE",
                "from_date": "2026-01-01",
                "to_date": "2026-03-01",
                "interval": "minute",
            })
        self.assertEqual(response.status_code, 422)
        self.assertIn("limited to 30 days", response.json()["detail"])

    def test_mock_historical_import_persists_oi_and_audit_event(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "history.db")
            api.angel_runtime["adapter"] = FakeAngelOneAdapter()
            api.angel_runtime["adapter"].login(client_code="AB1234", pin="7391", totp="123456")
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/angelone/history", json={
                    "symbol": "NIFTY26OCT22000CE",
                    "instrument_token": 123456,
                    "exchange": "NFO",
                    "from_date": "2026-09-01",
                    "to_date": "2026-09-02",
                    "interval": "day",
                    "include_open_interest": True,
                })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["bars_inserted"], 1)
            stored = test_store.read_candles("NIFTY26OCT22000CE")
            self.assertEqual(stored[0].open_interest, 250)
            self.assertEqual(test_store.read_events()[0]["event_type"], "market.historical_data_imported")

    def test_feed_can_start_with_provisional_spot_and_reports_stale_reference(self):
        adapter = FakeAngelOneAdapter()
        adapter.spot_reference_stale = True
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "provisional.db")
            api.angel_runtime["adapter"] = adapter
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/angelone/feed/start")
                status = client.get("/api/angelone/status")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(adapter.allow_stale_quotes)
        self.assertTrue(status.json()["spot_reference_stale"])
        self.assertTrue(status.json()["connected"])

    def test_manual_news_api_persists_source_url_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "news.db")
            payload = {
                "source": "exchange disclosure",
                "original_url": "https://example.exchange/notices/123",
                "published_at": "2026-09-30T09:00:00+05:30",
                "instruments": ["NIFTY"],
                "sector": "Index",
                "country": "IN",
                "source_reliability": 0.95,
                "title": "NIFTY company earnings beat estimates",
                "content": "Company disclosed an earnings beat in its exchange filing.",
            }
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                first = client.post("/api/news", json=payload)
                duplicate = client.post("/api/news", json={**payload, "source": "second wire"})
            self.assertTrue(first.json()["accepted"])
            self.assertEqual(first.json()["article"]["original_url"], payload["original_url"])
            self.assertEqual(first.json()["article"]["published_at"], "2026-09-30T03:30:00+00:00")
            self.assertGreaterEqual(
                datetime.fromisoformat(first.json()["article"]["ingested_at"]),
                datetime.fromisoformat(first.json()["article"]["published_at"]),
            )
            self.assertTrue(duplicate.json()["duplicate"])
            stored = test_store.read_news()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["original_url"], payload["original_url"])

    def test_openai_hypothesis_requires_connected_feed_and_uses_fresh_tick_context(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "llm.db")
            candles = demo_candles()
            test_store.save_candles(candles)
            timestamp = datetime.now(timezone.utc).isoformat()
            tick_event = {
                "event_id": "fresh-tick-1",
                "event_type": "market.tick_received",
                "timestamp": timestamp,
                "correlation_id": "trace-1",
                "source": "angelone_smartapi",
                "version": 1,
                "schema_version": "1.0",
                "payload": {
                    "symbol": "DEMO", "instrument_token": 1, "exchange": "NSE",
                    "market_timestamp": timestamp, "last_price": candles[-1].close,
                    "volume": 1000, "open_interest": None, "day_ohlc": {}, "bids": [], "asks": [],
                },
            }
            test_store.save_market_tick(tick_event)
            provider = FakeModelProvider()
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "llm_gateway", LLMGateway(provider)),
                patch.object(api, "LLM_ENABLED", True),
                patch.object(api, "LLM_MODEL", "fake-model"),
                TestClient(api.app) as client,
            ):
                api.angel_state["connected"] = False
                disconnected = client.post("/api/llm/hypotheses/DEMO")
                api.angel_state["connected"] = True
                response = client.post("/api/llm/hypotheses/DEMO")
            self.assertEqual(disconnected.status_code, 503)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["validation"]["approved"], response.json()["validation"])
            self.assertEqual(response.json()["reproducibility"]["model"], "fake-model")
            self.assertEqual(test_store.read_events()[0]["event_type"], "llm.hypothesis_created")


if __name__ == "__main__":
    unittest.main()