import asyncio
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from core import demo_candles
from llm_gateway import LLMGateway
from angelone_adapter import AngelOneMarketDataAdapter, AngelOneSession, AngelOneTick
from india_market import AssetType, Exchange, IndiaMarketPolicy, Instrument
from marketdata import MarketDataQualityGate, MarketTickQualityGate, MinuteCandleAggregator
from store import EventStore


pytestmark = pytest.mark.integration


class FakeAngelOneAdapter:
    def __init__(self):
        self.session = None
        self.logged_out = False
        self.feed_started = False
        self.spot_reference_stale = False
        self.stream_ticks = []

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

    @property
    def feed_running(self):
        return self.feed_started

    def initial_option_watchlist(self, *, allow_stale_quotes=False):
        self.allow_stale_quotes = allow_stale_quotes
        return ([{"instrument_token": 99926000, "exchange": "NSE", "tradingsymbol": "NIFTY 50"}], {"NIFTY": 22000})

    def start_stream(self, instruments, *, on_tick, on_status):
        self.feed_started = True
        on_status("market.connected", {"subscriptions": len(instruments), "mode": "snap_quote"})
        for tick in self.stream_ticks:
            on_tick(tick)

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

    def read_portfolio_snapshot(self):
        if getattr(self, "portfolio_error", False):
            raise RuntimeError("suppressed broker response")
        return {
            "provider": "angelone_smartapi",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "positions": [{"symbol": "RELIANCE-EQ", "quantity": 5, "source": "POSITION"}],
            "holdings": [{"symbol": "TCS-EQ", "quantity": 2, "source": "HOLDING"}],
            "rejected_rows": {"positions": 0, "holdings": 0},
            "orders_enabled": False,
        }


class FakeModelProvider:
    name = "offline-fake"

    def __init__(self):
        self.contexts = []

    def generate_structured(self, *, model, prompt, context):
        self.contexts.append(context)
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


class FakeReferenceDataAdapter:
    def __init__(self):
        self.price_scale = None

    def normalized_reference_data(self, *, price_scale):
        self.price_scale = price_scale
        exchange = Exchange("NSE-TEST", "Synthetic NSE fixture", "IN", "Asia/Kolkata")
        instrument = Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index")
        policy = IndiaMarketPolicy(exchanges=(exchange,), instruments=(instrument,))
        return policy, {"bad_row": 1}


class ProviderStatusApiTests(unittest.TestCase):
    def setUp(self):
        api.angel_runtime.update({"adapter": None, "instruments": []})
        api.angel_state.update({"connected": False, "last_error": None, "watchlist_count": 0, "session_active": False})
        api.angel_state["spot_reference_stale"] = False
        api.angel_state["fresh_index_ticks"] = set()
        api.angel_state["portfolio_status"] = {"state": "unavailable", "captured_at": None, "error": None}
        api.angel_runtime["portfolio_last_read"] = None

    def tearDown(self):
        adapter = api.angel_runtime.get("adapter")
        if adapter is not None:
            adapter.logout()
        api.angel_runtime.update({"adapter": None, "instruments": []})
        api.angel_state.update({"connected": False, "last_error": None, "watchlist_count": 0, "session_active": False})
        api.angel_state["spot_reference_stale"] = False
        api.angel_state["fresh_index_ticks"] = set()
        api.angel_state["portfolio_status"] = {"state": "unavailable", "captured_at": None, "error": None}
        api.angel_runtime["portfolio_last_read"] = None

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
        self.assertFalse(payload["feed_running"])
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["option_underlyings"], ["NIFTY", "BANKNIFTY"])

    def test_health_never_claims_orders_enabled(self):
        with TestClient(api.app) as client:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["broker_execution_enabled"])

    def test_portfolio_reconciliation_returns_distinct_broker_buckets(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        api.angel_runtime["adapter"] = adapter
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False), TestClient(api.app) as client:
            response = client.get("/api/angelone/portfolio/reconcile")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["positions"][0]["source"], "POSITION")
        self.assertEqual(payload["holdings"][0]["source"], "HOLDING")
        self.assertEqual(payload["reconciliation"]["basis"], "broker_snapshot_only")
        self.assertFalse(payload["reconciliation"]["internal_fill_ledger_available"])
        self.assertFalse(payload["orders_enabled"])

    def test_portfolio_reconciliation_failure_is_not_an_empty_account(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        adapter.portfolio_error = True
        api.angel_runtime["adapter"] = adapter
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False), TestClient(api.app) as client:
            response = client.get("/api/angelone/portfolio/reconcile")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(api.angel_state["portfolio_status"]["state"], "error")
        self.assertNotIn("positions", response.json())

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

    def test_india_reference_refresh_is_disabled_by_default(self):
        with patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "false"}, clear=False), TestClient(api.app) as client:
            response = client.post("/api/india/reference-data/refresh")
        self.assertEqual(response.status_code, 503)

    def test_india_reference_refresh_persists_snapshot_without_starting_feed(self):
        adapter = FakeReferenceDataAdapter()
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "reference.db")
            with (
                patch.dict(os.environ, {
                    "ANGELONE_MARKET_DATA_ENABLED": "true",
                    "ANGELONE_API_KEY": "test-api-key",
                    "ANGELONE_REFERENCE_PRICE_SCALE": "100",
                }, clear=False),
                patch.object(api, "store", test_store),
                patch.object(api.AngelOneMarketDataAdapter, "from_environment", return_value=adapter),
                TestClient(api.app) as client,
            ):
                refresh = client.post("/api/india/reference-data/refresh")
                status = client.get("/api/india/reference-data/status")
            self.assertEqual(refresh.status_code, 200, refresh.text)
            self.assertEqual(refresh.json()["record_counts"]["instruments"], 1)
            self.assertEqual(refresh.json()["rejected_rows"], {"bad_row": 1})
            self.assertFalse(refresh.json()["feed_started"])
            self.assertFalse(refresh.json()["orders_enabled"])
            self.assertEqual(adapter.price_scale, Decimal("100"))
            self.assertEqual(status.json()["snapshot_hash"], refresh.json()["snapshot_hash"])
            self.assertEqual(test_store.read_events()[0]["event_type"], "market.reference_data_imported")

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

    def test_candle_ingest_and_read_keep_timeframes_separate(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        base = {
            "symbol": "TIMEFRAME-TEST",
            "timestamp": timestamp,
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "timeframes.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "quality_gate", MarketDataQualityGate()),
                TestClient(api.app) as client,
            ):
                minute = client.post("/api/market/ingest", json={
                    "provider": "fixture", "timeframe": "1m", "events": [base],
                })
                daily = client.post("/api/market/ingest", json={
                    "provider": "fixture", "timeframe": "1d", "events": [base],
                })
                index_daily = client.post("/api/market/ingest", json={
                    "provider": "fixture", "timeframe": "1d", "events": [{**base, "symbol": "NIFTY"}],
                })
                mismatch = client.post("/api/market/ingest", json={
                    "provider": "fixture", "timeframe": "1m",
                    "events": [{**base, "timeframe": "1d"}],
                })
                minute_bars = client.get("/api/market/TIMEFRAME-TEST/candles?timeframe=1m")
                daily_bars = client.get("/api/market/TIMEFRAME-TEST/candles?timeframe=1d")
                default_bars = client.get("/api/market/TIMEFRAME-TEST/candles")
                canonical_index_bars = client.get("/api/market/NIFTY%2050/candles?timeframe=1d")

        self.assertEqual(minute.status_code, 200)
        self.assertEqual(daily.status_code, 200)
        self.assertEqual(index_daily.status_code, 200)
        self.assertEqual(mismatch.json()["rejected"][0]["reason"], "timeframe_mismatch")
        self.assertEqual(minute.json()["accepted"], 1)
        self.assertEqual(daily.json()["accepted"], 1)
        self.assertEqual(canonical_index_bars.status_code, 200)
        self.assertEqual(canonical_index_bars.json()[0]["symbol"], "NIFTY")
        self.assertEqual(minute_bars.json()[0]["timeframe"], "1m")
        self.assertEqual(daily_bars.json()[0]["timeframe"], "1d")
        self.assertEqual(default_bars.json()[0]["timeframe"], "1d")

    def test_provider_timeframes_are_stored_separately_without_rebucketing(self):
        timestamp = "2026-09-01T03:45:00+00:00"
        candle = {
            "symbol": "MULTI-TF-TEST",
            "timestamp": timestamp,
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        timeframe_pairs = (("1m", "1m"), ("5m", "5m"), ("15m", "15m"), ("1h", "1h"), ("1d", "1d"))

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "provider-timeframes.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "quality_gate", MarketDataQualityGate()),
                TestClient(api.app) as client,
            ):
                responses = [
                    client.post("/api/market/ingest", json={
                        "provider": "fixture",
                        "timeframe": timeframe,
                        "events": [candle],
                    })
                    for timeframe, _ in timeframe_pairs
                ]
                loaded = {
                    stored_timeframe: test_store.read_candles(
                        "MULTI-TF-TEST", timeframe=stored_timeframe
                    )
                    for _, stored_timeframe in timeframe_pairs
                }

        self.assertTrue(all(response.json()["accepted"] == 1 for response in responses))
        for _, timeframe in timeframe_pairs:
            self.assertEqual(len(loaded[timeframe]), 1)
            self.assertEqual(loaded[timeframe][0].timeframe, timeframe)
            self.assertEqual(loaded[timeframe][0].timestamp, timestamp)

    def test_candle_ingest_quarantines_invalid_ohlc_and_negative_volume(self):
        timestamp = datetime(2026, 9, 1, 9, 15, tzinfo=timezone.utc)
        base = {
            "symbol": "INVALID-CANDLE-TEST",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        invalid_events = [
            {**base, "timestamp": (timestamp + timedelta(minutes=index)).isoformat(), **prices}
            for index, prices in enumerate((
                {"high": 98, "low": 99},
                {"open": 103, "high": 102},
                {"close": 103},
                {"volume": -1},
            ))
        ]

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "invalid-candles.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "quality_gate", MarketDataQualityGate()),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/market/ingest", json={
                    "provider": "fixture",
                    "timeframe": "1m",
                    "events": invalid_events,
                })
                stored = test_store.read_candles("INVALID-CANDLE-TEST", timeframe="1m")

        result = response.json()
        self.assertEqual(result["accepted"], 0)
        self.assertEqual([item["reason"] for item in result["rejected"]], ["invalid_payload"] * 4)
        self.assertEqual(result["quality"]["invalid"], 4)
        self.assertEqual(stored, [])

    def test_live_candle_ingest_rejects_future_timestamps(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        event = {
            "symbol": "FUTURE-CANDLE-TEST",
            "timestamp": future.isoformat(),
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "future-candle.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "quality_gate", MarketDataQualityGate()),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/market/ingest", json={
                    "provider": "fixture",
                    "timeframe": "1m",
                    "mode": "live",
                    "events": [event],
                })
                stored = test_store.read_candles("FUTURE-CANDLE-TEST", timeframe="1m")

        self.assertEqual(response.json()["accepted"], 0)
        self.assertEqual(response.json()["rejected"][0]["reason"], "stale_live_event")
        self.assertEqual(stored, [])

    def test_multi_timeframe_feature_endpoint_keeps_series_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "multi-timeframe.db")
            daily = demo_candles()[-60:]
            hourly_base = datetime(2026, 10, 1, tzinfo=timezone.utc)
            hourly = [
                replace(
                    candle,
                    timestamp=(hourly_base + timedelta(hours=index)).isoformat(),
                    timeframe="1h",
                )
                for index, candle in enumerate(daily)
            ]
            test_store.save_candles(daily + hourly)
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                response = client.get("/api/market/DEMO/features/multi-timeframe")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(set(payload["timeframes"]), {"1d", "1h"})
        self.assertEqual(payload["timeframes"]["1d"]["timeframe"], "1d")
        self.assertEqual(payload["timeframes"]["1h"]["timeframe"], "1h")
        self.assertEqual(payload["missing_timeframes"], ["15m", "5m", "1m"])

    def test_india_regime_endpoint_requires_both_indices_and_uses_daily_series(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "regime.db")
            test_store.save_candles(demo_candles("NIFTY"))
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                missing = client.get("/api/market/regime")
                test_store.save_candles(demo_candles("BANKNIFTY"))
                ready = client.get("/api/market/regime")

        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["regime"], "INSUFFICIENT_DATA")
        self.assertEqual(missing.json()["missing_indices"], ["NIFTY BANK"])
        self.assertEqual(ready.status_code, 200)
        self.assertIn(ready.json()["regime"], {
            "STRONG_BULL_TREND", "WEAK_BULL_TREND", "STRONG_BEAR_TREND",
            "WEAK_BEAR_TREND", "RANGE_BOUND", "HIGH_VOLATILITY", "MIXED",
        })
        self.assertEqual(set(ready.json()["index_snapshots"]), {"NIFTY 50", "NIFTY BANK"})
        self.assertEqual(ready.json()["timeframe"], "1d")
        self.assertFalse(ready.json()["execution_authority"])

    def test_historical_similarity_endpoint_uses_forward_only_analogues(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "similarity.db")
            test_store.save_candles(demo_candles("SIM-TEST"))
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                response = client.get(
                    "/api/market/SIM-TEST/similarity?timeframe=1d&horizon_bars=5&top_k=3&max_distance=2"
                )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["symbol"], "SIM-TEST")
        self.assertEqual(result["timeframe"], "1d")
        self.assertLessEqual(result["sample_count"], 3)
        self.assertTrue(all(match["reference_timestamp"] < result["as_of"] for match in result["matches"]))
        self.assertTrue(all(match["holding_period_bars"] == 5 for match in result["matches"]))
        self.assertFalse(result["is_calibrated_probability"])
        self.assertFalse(result["execution_authority"])

    def test_india_expected_value_uses_supplied_cost_schedule_and_marks_probability_unvalidated(self):
        payload = {
            "product": "EQUITY_INTRADAY",
            "win_probability": "0.6",
            "average_win": "200",
            "average_loss": "100",
            "buy_turnover": "10000",
            "sell_turnover": "11000",
            "slippage_bps_per_side": "5",
            "market_impact_bps_per_side": "2",
            "brokerage_rate": "0.001",
            "brokerage_cap_per_order": None,
            "stt_buy_rate": "0",
            "stt_sell_rate": "0.001",
            "exchange_transaction_rate": "0.0001",
            "sebi_turnover_rate": "0.00001",
            "gst_rate": "0.18",
            "stamp_duty_buy_rate": "0.00002",
            "other_turnover_rate": "0",
            "gst_base_components": ["brokerage", "exchange_charges", "sebi_charges", "other_charges"],
        }
        with TestClient(api.app) as client:
            response = client.post("/api/india/expected-value", json=payload)
            boundary_responses = [
                client.post(
                    "/api/india/expected-value",
                    json={**payload, "win_probability": value},
                )
                for value in ("0", "1")
            ]
            out_of_range_responses = [
                client.post(
                    "/api/india/expected-value",
                    json={**payload, "win_probability": value},
                )
                for value in ("-0.01", "1.01")
            ]
            missing_probability = client.post(
                "/api/india/expected-value",
                json={key: value for key, value in payload.items() if key != "win_probability"},
            )
            missing_rate = client.post(
                "/api/india/expected-value",
                json={key: value for key, value in payload.items() if key != "stt_sell_rate"},
            )
            non_finite_responses = [
                client.post(
                    "/api/india/expected-value",
                    json={**payload, "win_probability": value},
                )
                for value in ("NaN", "Infinity", "-Infinity")
            ]
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(Decimal(result["expected_value"]["expected_net_value"]), Decimal("26.5942"))
        self.assertEqual(result["probability_status"], "CALLER_SUPPLIED_UNVALIDATED")
        self.assertFalse(result["execution_authority"])
        self.assertEqual([item.status_code for item in boundary_responses], [200, 200])
        self.assertEqual([item.json()["probability_status"] for item in boundary_responses], [
            "CALLER_SUPPLIED_UNVALIDATED", "CALLER_SUPPLIED_UNVALIDATED",
        ])
        self.assertEqual([item.status_code for item in out_of_range_responses], [422, 422])
        self.assertEqual(missing_probability.status_code, 422)
        self.assertEqual(missing_rate.status_code, 422)
        self.assertEqual([item.status_code for item in non_finite_responses], [422, 422, 422])

    def test_walk_forward_backtest_scores_disjoint_out_of_sample_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "walk-forward.db")
            history = demo_candles("WF-TEST")[:45]
            test_store.save_candles(history)
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                response = client.post("/api/backtests/walk-forward", json={
                    "symbol": "WF-TEST",
                    "timeframe": "1d",
                    "train_bars": 20,
                    "test_bars": 10,
                    "step_bars": 10,
                    "fee_bps": 0,
                    "market_impact_bps": 1,
                    "max_volume_participation": 0.5,
                    "execution_delay_bars": 1,
                    "india_product": "EQUITY_INTRADAY",
                    "india_charge_rates": {
                        "brokerage_rate": "0.001",
                        "brokerage_cap_per_order": None,
                        "stt_buy_rate": "0",
                        "stt_sell_rate": "0.001",
                        "exchange_transaction_rate": "0.0001",
                        "sebi_turnover_rate": "0.00001",
                        "gst_rate": "0.18",
                        "stamp_duty_buy_rate": "0.00002",
                        "other_turnover_rate": "0",
                        "gst_base_components": ["brokerage", "exchange_charges", "sebi_charges"],
                    },
                })
                overlapping = client.post("/api/backtests/walk-forward", json={
                    "symbol": "WF-TEST",
                    "timeframe": "1d",
                    "train_bars": 20,
                    "test_bars": 10,
                    "step_bars": 5,
                })

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["fold_count"], 3)
        self.assertEqual(result["total_test_bars"], 25)
        self.assertFalse(result["compounds_fold_returns"])
        self.assertLess(result["folds"][0]["training_end"], result["folds"][0]["test_start"])
        self.assertEqual(result["folds"][0]["cost_model"], "india_configured")
        self.assertEqual(result["folds"][0]["market_impact_bps"], 1)
        self.assertEqual(result["folds"][0]["execution_delay_bars"], 1)
        self.assertEqual(overlapping.status_code, 422)

    def test_backtest_api_requires_explicit_india_schedule_without_double_fee(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "india-cost-backtest.db")
            test_store.save_candles(demo_candles("COST-TEST")[:40])
            base_request = {
                "symbol": "COST-TEST",
                "fast_window": 2,
                "slow_window": 3,
                "fee_bps": 0,
                "india_product": "EQUITY_INTRADAY",
                "india_charge_rates": {
                    "brokerage_rate": "0.001",
                    "brokerage_cap_per_order": None,
                    "stt_buy_rate": "0",
                    "stt_sell_rate": "0.001",
                    "exchange_transaction_rate": "0.0001",
                    "sebi_turnover_rate": "0.00001",
                    "gst_rate": "0.18",
                    "stamp_duty_buy_rate": "0.00002",
                    "other_turnover_rate": "0",
                    "gst_base_components": ["brokerage", "exchange_charges", "sebi_charges"],
                },
            }
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                configured = client.post("/api/backtests", json=base_request)
                duplicate_fee = client.post("/api/backtests", json={**base_request, "fee_bps": 10})

        self.assertEqual(configured.status_code, 200, configured.text)
        self.assertEqual(configured.json()["cost_model"], "india_configured")
        self.assertEqual(duplicate_fee.status_code, 422)

    def test_market_breadth_api_uses_explicit_universe_and_reports_missing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "breadth.db")
            test_store.save_candles(demo_candles("ADV-A") + demo_candles("ADV-B"))
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                response = client.post("/api/market/breadth", json={
                    "symbols": ["ADV-A", "ADV-B", "NO-HISTORY"],
                    "timeframe": "1d",
                    "sector_by_symbol": {"ADV-A": "BANKS", "ADV-B": "IT"},
                    "index_symbols": ["ADV-A"],
                })

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["universe_size"], 3)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["missing_symbols"], ["NO-HISTORY"])
        self.assertEqual(set(result["sector_breadth"]), {"BANKS", "IT"})
        self.assertIn("ADV-A", result["index_participation"])
        self.assertFalse(result["execution_authority"])

    @pytest.mark.event_driven
    def test_accepted_ticks_emit_features_only_when_minute_bar_closes(self):
        minute_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        ticks = [
            AngelOneTick(
                event_id=f"feature-tick-{index}",
                timestamp=timestamp.isoformat(),
                ingested_at=datetime.now(timezone.utc).isoformat(),
                correlation_id=f"feature-trace-{index}",
                source="fixture_feed",
                instrument_token=123,
                exchange="NSE",
                symbol="FEATURE-TEST",
                last_price=100 + index,
                last_trade_quantity=2,
                volume=10 + index,
                open_interest=None,
                day_ohlc={},
                bids=(),
                asks=(),
                sequence=index,
            )
            for index, timestamp in enumerate((minute_start - timedelta(seconds=1), minute_start))
        ]
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "live-features.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "angel_tick_quality", MarketTickQualityGate(max_age_seconds=120)),
                patch.object(api, "angel_candle_aggregator", MinuteCandleAggregator()),
                patch.object(api, "app_loop", None),
            ):
                api._handle_angelone_tick(ticks[0])
                self.assertFalse(any(event["event_type"] == "market.feature_snapshot_created" for event in test_store.read_events()))
                api._handle_angelone_tick(ticks[1])
                feature_events = [
                    event for event in test_store.read_events()
                    if event["event_type"] == "market.feature_snapshot_created"
                ]

        self.assertEqual(len(feature_events), 1)
        self.assertEqual(feature_events[0]["payload"]["symbol"], "FEATURE-TEST")
        self.assertEqual(feature_events[0]["payload"]["timeframe"], "1m")
        self.assertFalse(feature_events[0]["payload"]["is_provisional"])
        self.assertEqual(feature_events[0]["payload"]["feature_snapshot"]["bar_count"], 1)

    @pytest.mark.event_driven
    def test_tick_event_is_consumed_once_and_stale_or_duplicate_ticks_are_rejected(self):
        adapter = AngelOneMarketDataAdapter(api_key="test-key", smart_api=SimpleNamespace())
        current_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        def normalize(timestamp, sequence, price):
            return adapter.normalize_tick({
                "token": "123",
                "sequence_number": sequence,
                "exchange_timestamp": int(timestamp.timestamp() * 1000),
                "last_traded_price": int(price * 100),
                "last_traded_quantity": 2,
                "volume_trade_for_the_day": 10 + sequence,
            }, {"exchange": "NSE", "tradingsymbol": "TICK-TEST"})

        first = normalize(current_minute - timedelta(seconds=1), 1, 100)
        next_minute = normalize(current_minute, 2, 101)
        stale = normalize(current_minute - timedelta(minutes=10), 3, 99)
        self.assertEqual(first.last_price, 100)
        self.assertEqual(first.as_event()["event_type"], "market.tick_received")

        class Consumer:
            def __init__(self):
                self.messages = []

            async def send_json(self, message):
                self.messages.append(message)

        class DecisionConsumer:
            def __init__(self):
                self.event_ids = []

            def process_tick(self, *, tick, **_kwargs):
                self.event_ids.append(tick.event_id)
                return {"status": "NO TRADE", "execution": False}

        tick_consumer = Consumer()
        feature_consumer = Consumer()
        decision_consumer = DecisionConsumer()
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "tick-flow.db")
            with (
                patch.object(api, "store", test_store),
                patch.object(api, "angel_tick_quality", MarketTickQualityGate(max_age_seconds=120)),
                patch.object(api, "angel_candle_aggregator", MinuteCandleAggregator()),
                patch.dict(api.angel_runtime, {"decision_pipeline": decision_consumer}),
                patch.object(api, "tick_clients", {tick_consumer}),
                patch.object(api, "feature_clients", {feature_consumer}),
                patch.object(api, "app_loop", SimpleNamespace(is_running=lambda: True)),
                patch.object(
                    api.asyncio,
                    "run_coroutine_threadsafe",
                    side_effect=lambda coroutine, _loop: asyncio.run(coroutine),
                ),
            ):
                api._handle_angelone_tick(first)
                api._handle_angelone_tick(first)
                api._handle_angelone_tick(next_minute)
                api._handle_angelone_tick(stale)

            events = test_store.read_events()
            received = [event for event in events if event["event_type"] == "market.tick_received"]
            rejected = [event["payload"]["reason"] for event in events if event["event_type"] == "market.tick_rejected"]
            features = [event for event in events if event["event_type"] == "market.feature_snapshot_created"]
            completed = [event for event in events if event["event_type"] == "decision.pipeline_completed"]
            stored_ticks = test_store.read_market_ticks("TICK-TEST")

        self.assertEqual(len(received), 2)
        self.assertEqual(len(stored_ticks), 2)
        self.assertEqual(stored_ticks[0]["last_price"], 100)
        self.assertCountEqual(rejected, ["duplicate_tick", "stale_tick"])
        self.assertEqual(len(features), 1)
        self.assertEqual(len(completed), 2)
        self.assertEqual(decision_consumer.event_ids, [first.event_id, next_minute.event_id])
        self.assertEqual(len(tick_consumer.messages), 2)
        self.assertEqual(tick_consumer.messages[0]["last_price"], 100)
        self.assertEqual(len(feature_consumer.messages), 1)

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

    def test_previous_day_historical_import_uses_authenticated_read_only_path(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        adapter.historical_candles = lambda **kwargs: [{
            "symbol": kwargs["symbol"],
            "timestamp": f"{kwargs['from_date'].isoformat()}T03:45:00+00:00",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 500,
            "open_interest": None,
        }]
        api.angel_runtime["adapter"] = adapter
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "previous-day.db")
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/angelone/history/previous-day", json={
                    "symbol": "NIFTY",
                    "instrument_token": 99926000,
                    "exchange": "NSE",
                    "interval": "day",
                })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["date_basis"], "previous_calendar_day")
        self.assertEqual(response.json()["bars_inserted"], 1)
        self.assertIn("holiday calendar", response.json()["warning"])

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
            self.assertEqual(stored[0].timeframe, "1d")
            self.assertEqual(test_store.read_events()[0]["event_type"], "market.historical_data_imported")

    def test_historical_import_rejects_future_rows_and_negative_volume(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        adapter.historical_candles = lambda **_kwargs: [
            {
                "symbol": "NIFTY",
                "timestamp": "2026-09-01T03:45:00+00:00",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": -1,
                "open_interest": None,
            },
            {
                "symbol": "NIFTY",
                "timestamp": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": 10,
                "open_interest": None,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "invalid-history.db")
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.dict(api.angel_runtime, {"adapter": adapter}),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/angelone/history", json={
                    "symbol": "NIFTY",
                    "instrument_token": 99926000,
                    "exchange": "NSE",
                    "from_date": "2026-09-01",
                    "to_date": "2026-09-02",
                    "interval": "day",
                })
                stored = test_store.read_candles("NIFTY", timeframe="1d")

        self.assertNotEqual(response.status_code, 200, response.text)
        self.assertEqual(stored, [])

    def test_historical_import_rejects_malformed_financial_and_out_of_range_rows(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        valid_row = {
            "symbol": "NIFTY",
            "timestamp": "2026-09-01T03:45:00+00:00",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        invalid_rows = [
            {key: value for key, value in valid_row.items() if key != "close"},
            {**valid_row, "timestamp": "not-a-timestamp"},
            {**valid_row, "timestamp": "2026-09-01T09:15:00"},
            {**valid_row, "timestamp": "2026-08-31T18:29:59+00:00"},
            {**valid_row, "timestamp": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
            {**valid_row, "symbol": "BANKNIFTY"},
            {**valid_row, "high": 99},
            {**valid_row, "open": "nan"},
            {**valid_row, "volume": -1},
            {**valid_row, "open_interest": -1},
        ]

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "invalid-history-rows.db")
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.dict(api.angel_runtime, {"adapter": adapter}),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                responses = []
                for row in invalid_rows:
                    adapter.historical_candles = lambda **_kwargs: [row]
                    responses.append(client.post("/api/angelone/history", json={
                        "symbol": "NIFTY",
                        "instrument_token": 99926000,
                        "exchange": "NSE",
                        "from_date": "2026-09-01",
                        "to_date": "2026-09-02",
                        "interval": "day",
                    }))
                stored = test_store.read_candles("NIFTY", timeframe="1d")
                events = test_store.read_events()

        self.assertTrue(all(response.status_code == 502 for response in responses))
        self.assertEqual(stored, [])
        self.assertEqual(events, [])

    def test_historical_import_deduplicates_normalized_timestamps_and_rejects_conflicts(self):
        adapter = FakeAngelOneAdapter()
        adapter.login(client_code="AB1234", pin="7391", totp="123456")
        first_row = {
            "symbol": "NIFTY",
            "timestamp": "2026-09-01T03:45:00+00:00",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        }
        equivalent_row = {**first_row, "timestamp": "2026-09-01T09:15:00+05:30"}

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "deduplicated-history.db")
            with (
                patch.dict(os.environ, {"ANGELONE_MARKET_DATA_ENABLED": "true"}, clear=False),
                patch.dict(api.angel_runtime, {"adapter": adapter}),
                patch.object(api, "store", test_store),
                TestClient(api.app) as client,
            ):
                adapter.historical_candles = lambda **_kwargs: [first_row, equivalent_row]
                imported = client.post("/api/angelone/history", json={
                    "symbol": "NIFTY",
                    "instrument_token": 99926000,
                    "exchange": "NSE",
                    "from_date": "2026-09-01",
                    "to_date": "2026-09-02",
                    "interval": "day",
                })
                adapter.historical_candles = lambda **_kwargs: [equivalent_row]
                repeated = client.post("/api/angelone/history", json={
                    "symbol": "NIFTY",
                    "instrument_token": 99926000,
                    "exchange": "NSE",
                    "from_date": "2026-09-01",
                    "to_date": "2026-09-02",
                    "interval": "day",
                })
                adapter.historical_candles = lambda **_kwargs: [{**first_row, "close": 100}]
                conflict = client.post("/api/angelone/history", json={
                    "symbol": "NIFTY",
                    "instrument_token": 99926000,
                    "exchange": "NSE",
                    "from_date": "2026-09-01",
                    "to_date": "2026-09-02",
                    "interval": "day",
                })
            stored = test_store.read_candles("NIFTY", timeframe="1d")
            events = test_store.read_events()

        self.assertEqual(imported.status_code, 200, imported.text)
        self.assertEqual(imported.json()["bars_returned"], 2)
        self.assertEqual(imported.json()["bars_inserted"], 1)
        self.assertEqual(imported.json()["bars_duplicate"], 1)
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()["bars_inserted"], 0)
        self.assertEqual(repeated.json()["bars_duplicate"], 1)
        self.assertEqual(conflict.status_code, 502, conflict.text)
        self.assertEqual(len(stored), 1)
        self.assertEqual(len(events), 2)

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
                repeated = client.post("/api/angelone/feed/start")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(adapter.allow_stale_quotes)
        self.assertTrue(status.json()["spot_reference_stale"])
        self.assertTrue(status.json()["feed_running"])
        self.assertTrue(status.json()["connected"])
        self.assertEqual(repeated.status_code, 200)
        self.assertTrue(repeated.json()["already_running"])

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

    def test_news_batch_api_classifies_and_deduplicates_items(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "news-batch.db")
            item = {
                "source": "wire-a",
                "published_at": "2026-09-30T09:00:00Z",
                "instruments": ["RELIANCE"],
                "title": "Company beats estimates",
                "content": "Quarterly earnings beat estimates.",
            }
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                response = client.post("/api/news/batch", json={"items": [item, {**item, "source": "wire-b"}]})
                invalid = client.post("/api/news/batch", json={"items": [{**item, "published_at": "bad"}]})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["accepted"], 1)
            self.assertEqual(response.json()["duplicates"], 1)
            self.assertEqual(response.json()["items"][0]["intelligence"]["event_type"], "EARNINGS_BEAT")
            self.assertFalse(response.json()["orders_enabled"])
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(len(test_store.read_news()), 1)

    def test_news_api_validates_required_fields_and_article_size(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "news-validation.db")
            payload = {
                "source": "wire-a",
                "published_at": "2026-09-30T09:00:00Z",
                "title": "Large article",
                "content": "valid article content",
                "instruments": ["TEST"],
            }
            missing_source = {key: value for key, value in payload.items() if key != "source"}
            missing_timestamp = {key: value for key, value in payload.items() if key != "published_at"}
            invalid_payloads = (
                missing_source,
                missing_timestamp,
                {**payload, "content": ""},
                {**payload, "published_at": "not-a-timestamp"},
                {**payload, "content": "x" * 20_001},
            )
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                invalid_responses = [client.post("/api/news", json=item) for item in invalid_payloads]
                large_response = client.post("/api/news", json={**payload, "content": "x" * 20_000})

            self.assertTrue(all(response.status_code == 422 for response in invalid_responses))
            self.assertEqual(large_response.status_code, 200)
            self.assertTrue(large_response.json()["accepted"])
            self.assertEqual(len(test_store.read_news()[0]["content"]), 20_000)

    def test_rag_memory_api_persists_and_filters_historical_trades(self):
        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "rag.db")
            with patch.object(api, "store", test_store), TestClient(api.app) as client:
                saved = client.post("/api/rag/memory", json={
                    "document_id": "trade-1",
                    "kind": "historical_trade",
                    "text": "RELIANCE momentum bullish breakout after earnings beat",
                    "metadata": {
                        "instrument": "RELIANCE",
                        "symbol": "RELIANCE",
                        "sector": "ENERGY",
                        "strategy": "MOMENTUM",
                        "regime": "BULLISH",
                        "timeframe": "1d",
                        "source": "backtest",
                    },
                })
                duplicate = client.post("/api/rag/memory", json={
                    "document_id": "trade-1",
                    "kind": "historical_trade",
                    "text": "duplicate",
                })
                search = client.post("/api/rag/search", json={
                    "query": "RELIANCE momentum bullish breakout earnings",
                    "filters": {
                        "symbol": "RELIANCE",
                        "strategy": "MOMENTUM",
                        "regime": "BULLISH",
                        "timeframe": "1d",
                    },
                    "max_age_seconds": 60,
                })

            self.assertEqual(saved.status_code, 200)
            self.assertTrue(saved.json()["accepted"])
            self.assertTrue(duplicate.json()["duplicate"])
            self.assertEqual(search.status_code, 200)
            self.assertEqual(search.json()["results"][0]["document_id"], "trade-1")
            self.assertNotIn("embedding", search.json()["results"][0])
            self.assertFalse(search.json()["orders_enabled"])

    def test_rag_search_reports_unavailable_for_storage_timeout_and_embedding_failures(self):
        failures = (
            sqlite3.OperationalError("database unavailable"),
            TimeoutError("retrieval timed out"),
            RuntimeError("embedding failed"),
        )
        with TestClient(api.app) as client:
            for failure in failures:
                def fail_recall(*_args, error=failure, **_kwargs):
                    raise error

                with self.subTest(error=type(failure).__name__), patch.object(
                    api,
                    "HistoricalTradingMemory",
                    return_value=SimpleNamespace(recall=fail_recall),
                ):
                    response = client.post("/api/rag/search", json={"query": "RELIANCE momentum setup"})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["detail"], "Trading memory retrieval is unavailable")

    def test_llm_hypothesis_requires_connected_feed_and_uses_fresh_tick_context(self):
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

    @pytest.mark.e2e
    @pytest.mark.event_driven
    def test_mock_feed_runs_strategy_and_llm_without_touching_live_store(self):
        adapter = FakeAngelOneAdapter()
        timestamp = datetime.now(timezone.utc).isoformat()
        adapter.stream_ticks = [AngelOneTick(
            event_id="mock-feed:DEMO:1",
            timestamp=timestamp,
            ingested_at=timestamp,
            correlation_id="mock-feed-trace",
            source="mock_market_feed",
            instrument_token=1,
            exchange="NSE",
            symbol="DEMO",
            last_price=101,
            last_trade_quantity=10,
            volume=1000,
            open_interest=None,
            day_ohlc={"open": 100, "high": 101, "low": 99, "close": 100},
            bids=(),
            asks=(),
            sequence=1,
        )]
        provider = FakeModelProvider()
        live_store = api.store

        with tempfile.TemporaryDirectory() as directory:
            test_store = EventStore(Path(directory) / "mock-feed.db")
            test_store.save_candles(demo_candles())
            with (
                patch.dict(os.environ, {
                    "ANGELONE_MARKET_DATA_ENABLED": "true",
                    "ANGELONE_API_KEY": "mock-api-key",
                }, clear=False),
                patch.object(api, "store", test_store),
                patch.object(api, "angel_tick_quality", MarketTickQualityGate()),
                patch.object(api, "angel_candle_aggregator", MinuteCandleAggregator()),
                patch.object(api, "llm_gateway", LLMGateway(provider)),
                patch.object(api, "LLM_ENABLED", True),
                patch.object(api, "LLM_MODEL", "offline-fake-model"),
                patch.object(api.AngelOneMarketDataAdapter, "from_environment", return_value=adapter),
                TestClient(api.app) as client,
            ):
                login = client.post("/api/angelone/login", json={
                    "client_code": "MOCK01", "pin": "mock-pin", "totp": "123456",
                })
                feed = client.post("/api/angelone/feed/start")
                ticks = client.get("/api/market/DEMO/ticks").json()
                strategy = client.post("/api/hypotheses/DEMO")
                llm = client.post("/api/llm/hypotheses/DEMO")
                health = client.get("/api/health").json()

            self.assertEqual(login.status_code, 200)
            self.assertEqual(feed.status_code, 200)
            self.assertEqual(len(ticks), 1)
            self.assertEqual(ticks[0]["last_price"], 101)
            self.assertEqual(strategy.status_code, 200)
            self.assertEqual(llm.status_code, 200, llm.text)
            self.assertEqual(llm.json()["reproducibility"]["provider"], "offline-fake")
            self.assertEqual(provider.contexts[0]["market_snapshot"]["latest_tick"]["last_price"], 101)
            self.assertTrue(llm.json()["validation"]["approved"], llm.json()["validation"])
            self.assertFalse(health["broker_execution_enabled"])
            event_types = {event["event_type"] for event in test_store.read_events()}
            self.assertTrue({
                "market.connected",
                "market.tick_received",
                "strategy.hypothesis_validated",
                "llm.hypothesis_created",
            } <= event_types)

        self.assertIs(api.store, live_store)


if __name__ == "__main__":
    unittest.main()