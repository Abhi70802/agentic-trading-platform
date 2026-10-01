"""Local API for deterministic backtesting and paper replay; no broker adapter exists."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
import os
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core import Candle, CandleTimeframe, RiskConfig, RiskEngine, SimulationConfig, SmaCrossStrategy, run_simulation
from backtesting import run_walk_forward
from ai_agents import AdversarialAgent, StrategyAgent
from angelone_adapter import AngelOneCredentials, AngelOneMarketDataAdapter
from breadth import MarketBreadthEngine
from features import MULTI_TIMEFRAME_ANALYSIS, feature_snapshot, multi_timeframe_feature_snapshot
from india_economics import (
    IndiaExpectedValueEngine,
    IndiaProductChargeRates,
    IndiaProductType,
    IndiaTransactionCostSchedule,
)
from intelligence import build_hypothesis, ingest_news, validate_hypothesis
from india_regime import IndiaMarketRegimeEngine, IndiaRegimeConfig
from historical_similarity import historical_similarity
from llm_gateway import CopilotModelProvider, LLMGateway, OpenAIModelProvider
from marketdata import (
    JsonCandleAdapter,
    MarketDataQualityGate,
    MarketTickQualityGate,
    MinuteCandleAggregator,
    deduplicate_historical_candles,
    normalize_timestamp,
    validate_historical_candles,
)
from rag import HistoricalTradingMemory
from store import EventStore


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
store = EventStore(BASE_DIR / "data" / "platform.sqlite3")
if not store.read_candles("DEMO", timeframe="1d"):
    from core import demo_candles

    store.save_candles(demo_candles())

risk_config = RiskConfig()
quality_gate = MarketDataQualityGate()
angel_tick_quality = MarketTickQualityGate()
angel_candle_aggregator = MinuteCandleAggregator()
angel_state: dict = {
    "connected": False,
    "last_error": None,
    "watchlist_count": 0,
    "session_active": False,
    "spot_reference_stale": False,
    "fresh_index_ticks": set(),
    "portfolio_status": {"state": "unavailable", "captured_at": None, "error": None},
}
angel_state_lock = Lock()
angel_runtime: dict = {"adapter": None, "instruments": [], "portfolio_last_read": None, "decision_pipeline": None}
angel_portfolio_lock = Lock()
tick_clients: set[WebSocket] = set()
feature_clients: set[WebSocket] = set()
app_loop: asyncio.AbstractEventLoop | None = None
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").strip().lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "copilot").strip().lower()
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "auto").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
LLM_MODEL = COPILOT_MODEL if LLM_PROVIDER == "copilot" else OPENAI_MODEL
india_regime_engine = IndiaMarketRegimeEngine(IndiaRegimeConfig(
    strong_momentum_threshold_pct=float(os.getenv("INDIA_STRONG_MOMENTUM_THRESHOLD_PCT", "2.0")),
))
market_breadth_engine = MarketBreadthEngine()


def configured_llm_gateway() -> LLMGateway:
    if not LLM_ENABLED:
        return LLMGateway()
    if LLM_PROVIDER == "copilot":
        if importlib.util.find_spec("copilot") is None:
            return LLMGateway()
        return LLMGateway(CopilotModelProvider())
    if LLM_PROVIDER == "openai" and os.getenv("OPENAI_API_KEY", "").strip():
        return LLMGateway(OpenAIModelProvider.from_environment())
    return LLMGateway()


llm_gateway = configured_llm_gateway()
app = FastAPI(title="Agentic Trading Platform", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
async def configure_local_market_runtime():
    global app_loop
    app_loop = asyncio.get_running_loop()


@app.on_event("shutdown")
def stop_market_feed():
    adapter = angel_runtime.get("adapter")
    if adapter is not None:
        adapter.logout()
        angel_runtime["adapter"] = None


class SimulationRequest(BaseModel):
    symbol: str = Field(default="DEMO", pattern=r"^[A-Z0-9._-]{1,20}$")
    timeframe: CandleTimeframe = "1d"
    starting_cash: float = Field(default=100_000, gt=0, le=100_000_000)
    fast_window: int = Field(default=8, ge=2, le=100)
    slow_window: int = Field(default=21, ge=3, le=250)
    fee_bps: float = Field(default=10, ge=0, le=500)
    slippage_bps: float = Field(default=5, ge=0, le=50)
    market_impact_bps: float = Field(default=0, ge=0, le=500)
    max_volume_participation: float | None = Field(default=None, gt=0, le=1)
    execution_delay_bars: int = Field(default=0, ge=0, le=100)
    india_product: IndiaProductType | None = None
    india_charge_rates: "IndiaProductChargeRequest | None" = None


class IndiaProductChargeRequest(BaseModel):
    brokerage_rate: Decimal = Field(ge=0, le=1)
    brokerage_cap_per_order: Decimal | None = Field(gt=0)
    stt_buy_rate: Decimal = Field(ge=0, le=1)
    stt_sell_rate: Decimal = Field(ge=0, le=1)
    exchange_transaction_rate: Decimal = Field(ge=0, le=1)
    sebi_turnover_rate: Decimal = Field(ge=0, le=1)
    gst_rate: Decimal = Field(ge=0, le=1)
    stamp_duty_buy_rate: Decimal = Field(ge=0, le=1)
    other_turnover_rate: Decimal = Field(ge=0, le=1)
    gst_base_components: list[Literal["brokerage", "exchange_charges", "sebi_charges", "other_charges"]] = Field(min_length=1)


class WalkForwardBacktestRequest(BaseModel):
    symbol: str = Field(default="DEMO", pattern=r"^[A-Z0-9._-]{1,40}$")
    timeframe: CandleTimeframe = "1d"
    train_bars: int = Field(default=60, ge=2, le=4000)
    test_bars: int = Field(default=20, ge=2, le=1000)
    step_bars: int | None = Field(default=None, ge=2, le=1000)
    starting_cash: float = Field(default=100_000, gt=0, le=100_000_000)
    fast_window: int = Field(default=8, ge=2, le=100)
    slow_window: int = Field(default=21, ge=3, le=250)
    fee_bps: float = Field(default=10, ge=0, le=500)
    slippage_bps: float = Field(default=5, ge=0, le=50)
    market_impact_bps: float = Field(default=0, ge=0, le=500)
    max_volume_participation: float | None = Field(default=None, gt=0, le=1)
    execution_delay_bars: int = Field(default=0, ge=0, le=100)
    india_product: IndiaProductType | None = None
    india_charge_rates: IndiaProductChargeRequest | None = None


class KillSwitchRequest(BaseModel):
    enabled: bool


class MarketIngestRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=80)
    mode: Literal["historical", "live"] = "historical"
    timeframe: CandleTimeframe = "1d"
    events: list[dict] = Field(min_length=1, max_length=5000)


class MarketBreadthRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=500)
    timeframe: CandleTimeframe = "1d"
    sector_by_symbol: dict[str, str] = Field(default_factory=dict)
    index_symbols: list[str] = Field(default_factory=lambda: ["NIFTY 50", "NIFTY BANK"], max_length=100)


class IndiaExpectedValueRequest(BaseModel):
    product: IndiaProductType
    win_probability: Decimal = Field(ge=0, le=1)
    average_win: Decimal = Field(ge=0)
    average_loss: Decimal = Field(ge=0)
    buy_turnover: Decimal = Field(gt=0)
    sell_turnover: Decimal = Field(gt=0)
    slippage_bps_per_side: Decimal = Field(ge=0)
    market_impact_bps_per_side: Decimal = Field(ge=0)
    brokerage_rate: Decimal = Field(ge=0, le=1)
    brokerage_cap_per_order: Decimal | None = Field(gt=0)
    stt_buy_rate: Decimal = Field(ge=0, le=1)
    stt_sell_rate: Decimal = Field(ge=0, le=1)
    exchange_transaction_rate: Decimal = Field(ge=0, le=1)
    sebi_turnover_rate: Decimal = Field(ge=0, le=1)
    gst_rate: Decimal = Field(ge=0, le=1)
    stamp_duty_buy_rate: Decimal = Field(ge=0, le=1)
    other_turnover_rate: Decimal = Field(ge=0, le=1)
    gst_base_components: list[Literal["brokerage", "exchange_charges", "sebi_charges", "other_charges"]] = Field(min_length=1)


class NewsIngestRequest(BaseModel):
    article_id: str | None = Field(default=None, max_length=120)
    source: str = Field(min_length=1, max_length=120)
    original_url: str | None = Field(default=None, max_length=2000)
    published_at: str
    instruments: list[str] = Field(default_factory=list, max_length=100)
    sector: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=80)
    source_reliability: float = Field(default=0.5, ge=0, le=1)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=20_000)


class NewsBatchIngestRequest(BaseModel):
    items: list[NewsIngestRequest] = Field(min_length=1, max_length=100)


class RagMemoryRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=60)
    text: str = Field(min_length=1, max_length=20_000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=30)
    created_at: str | None = None


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    limit: int = Field(default=10, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict, max_length=10)
    max_age_seconds: int | None = Field(default=None, ge=0, le=31_536_000)

class AngelHistoricalRequest(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9._-]{1,40}$")
    instrument_token: int = Field(gt=0)
    exchange: Literal["NSE", "BSE", "NFO"]
    from_date: date
    to_date: date
    interval: Literal["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
    include_open_interest: bool = False


class AngelPreviousHistoricalRequest(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9._-]{1,40}$")
    instrument_token: int = Field(gt=0)
    exchange: Literal["NSE", "BSE", "NFO"]
    interval: Literal["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
    include_open_interest: bool = False


class AngelLoginRequest(BaseModel):
    client_code: str = Field(min_length=1, max_length=32)
    pin: str = Field(min_length=1, max_length=32)
    totp: str = Field(pattern=r"^\d{6}$")


ANGEL_INTERVAL_TIMEFRAMES = {
    "minute": "1m",
    "3minute": "3m",
    "5minute": "5m",
    "10minute": "10m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "1h",
    "day": "1d",
}


@app.get("/")
def dashboard():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "environment": "local_simulation",
        "market_data": "angelone_live_and_synthetic_demo" if angel_state["connected"] else "synthetic_demo_only",
        "broker_execution_enabled": False,
        "llm_enabled": LLM_ENABLED and llm_gateway.provider is not None,
        "kill_switch": risk_config.kill_switch,
        "data_quality": quality_gate.status(),
    }


@app.get("/api/instruments")
def instruments():
    return [
        {
            **instrument,
            "name": "Synthetic sample instrument" if instrument["symbol"] == "DEMO" else instrument["symbol"],
            "data_source": "synthetic_demo_ohlcv" if instrument["symbol"] == "DEMO" else "ingested_ohlcv",
        }
        for instrument in store.list_instruments()
    ]


@app.get("/api/angelone/status")
def angelone_status():
    credentials = AngelOneCredentials.from_environment()
    enabled = os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() == "true"
    try:
        import importlib.util

        sdk_installed = importlib.util.find_spec("SmartApi") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        sdk_installed = False
    adapter = angel_runtime.get("adapter")
    session_current = bool(adapter and adapter.session and adapter.session.is_current())
    feed_running = bool(adapter and getattr(adapter, "feed_running", False))
    if not session_current and adapter is not None and adapter.session is not None:
        adapter.session = None
        angel_state["session_active"] = False
    return {
        "provider": "angelone_smartapi",
        "enabled": enabled,
        "credentials_configured": credentials is not None,
        "sdk_installed": sdk_installed,
        "ready_to_login": enabled and credentials is not None and sdk_installed,
        "session_active": session_current,
        "feed_running": feed_running,
        "connected": angel_state["connected"],
        "last_error": angel_state["last_error"],
        "watchlist_count": angel_state["watchlist_count"],
        "watchlist": [
            {
                "symbol": item.get("tradingsymbol"),
                "exchange": item.get("exchange"),
                "instrument_type": item.get("instrument_type"),
                "underlying": item.get("name"),
                "expiry": item.get("expiry"),
                "strike": item.get("strike"),
            }
            for item in angel_runtime["instruments"]
        ],
        "spot_reference_stale": angel_state["spot_reference_stale"],
        "portfolio": dict(angel_state["portfolio_status"]),
        "quality": angel_tick_quality.status(),
        "exchanges": ["NSE", "NFO"],
        "option_underlyings": ["NIFTY", "BANKNIFTY"],
        "option_expiry": "nearest",
        "option_strikes_each_side": 5,
        "futures_enabled": False,
    }


@app.get("/api/angelone/portfolio/reconcile")
def reconcile_angelone_portfolio(request: Request):
    _require_loopback(request)
    if os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=503, detail="Angel One market data is disabled")
    adapter = angel_runtime.get("adapter")
    if adapter is None or adapter.session is None or not adapter.session.is_current():
        raise HTTPException(status_code=401, detail="Log in to Angel One first")
    with angel_portfolio_lock:
        now = time.monotonic()
        last_read = angel_runtime.get("portfolio_last_read")
        if last_read is not None and now - last_read < 1.0:
            raise HTTPException(status_code=429, detail="Portfolio refresh is limited to one request per second")
        angel_runtime["portfolio_last_read"] = now
        try:
            snapshot = adapter.read_portfolio_snapshot()
        except Exception as error:
            angel_state["portfolio_status"] = {
                "state": "error",
                "captured_at": None,
                "error": type(error).__name__,
            }
            raise HTTPException(status_code=503, detail="Angel One portfolio snapshot is unavailable") from error
    rejected = snapshot["rejected_rows"]
    state = "complete" if not any(rejected.values()) else "partial"
    summary = {
        "state": state,
        "captured_at": snapshot["captured_at"],
        "error": None,
        "positions_count": len(snapshot["positions"]),
        "holdings_count": len(snapshot["holdings"]),
        "rejected_rows": rejected,
    }
    angel_state["portfolio_status"] = summary
    store.append_events([_audit_event("portfolio.broker_snapshot_received", {
        "provider": snapshot["provider"],
        "positions_count": summary["positions_count"],
        "holdings_count": summary["holdings_count"],
        "rejected_rows": rejected,
        "complete": state == "complete",
    })])
    return {
        **snapshot,
        "reconciliation": {
            "state": state,
            "basis": "broker_snapshot_only",
            "internal_fill_ledger_available": False,
        },
    }


@app.get("/api/india/reference-data/status")
def india_reference_data_status():
    snapshot = store.read_latest_india_reference_snapshot("angelone_smartapi")
    if snapshot is None:
        return {
            "configured": False,
            "provider": "angelone_smartapi",
            "snapshot_hash": None,
            "fetched_at": None,
            "record_counts": {},
            "orders_enabled": False,
        }
    return {
        "configured": True,
        "provider": snapshot["provider"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "fetched_at": snapshot["fetched_at"],
        "record_counts": snapshot["record_counts"],
        "orders_enabled": False,
    }


@app.post("/api/india/reference-data/refresh")
def refresh_india_reference_data(request: Request):
    _require_loopback(request)
    if os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=503, detail="Angel One data access is disabled")
    if AngelOneCredentials.from_environment() is None:
        raise HTTPException(status_code=503, detail="Set ANGELONE_API_KEY in the local environment")
    try:
        adapter = AngelOneMarketDataAdapter.from_environment()
        price_scale = Decimal(os.getenv("ANGELONE_REFERENCE_PRICE_SCALE", "100"))
        policy, rejected = adapter.normalized_reference_data(price_scale=price_scale)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Reference data refresh failed: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Reference data refresh failed ({type(error).__name__})") from error
    if not policy.exchanges or not policy.instruments:
        raise HTTPException(status_code=502, detail="Reference provider returned no supported Indian instruments")
    fetched_at = datetime.now(timezone.utc).isoformat()
    snapshot = store.save_india_reference_snapshot(
        provider="angelone_smartapi",
        fetched_at=fetched_at,
        policy=policy,
    )
    store.append_events([_audit_event("market.reference_data_imported", {
        "provider": snapshot["provider"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "record_counts": {key: snapshot[key] for key in (
            "exchanges", "instruments", "contracts", "sessions", "holidays",
            "expiry_calendars", "corporate_actions",
        )},
        "rejected_rows": rejected,
    })])
    return {
        "provider": snapshot["provider"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "fetched_at": snapshot["fetched_at"],
        "record_counts": {key: snapshot[key] for key in (
            "exchanges", "instruments", "contracts", "sessions", "holidays",
            "expiry_calendars", "corporate_actions",
        )},
        "rejected_rows": rejected,
        "orders_enabled": False,
        "feed_started": False,
    }


@app.post("/api/angelone/login")
def angelone_login(request: Request, credentials: AngelLoginRequest):
    _require_loopback(request)
    if os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=503, detail="Angel One market data is disabled")
    try:
        adapter = AngelOneMarketDataAdapter.from_environment()
        session = adapter.login(
            client_code=credentials.client_code,
            pin=credentials.pin,
            totp=credentials.totp,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Angel One login failed ({type(error).__name__})") from error
    previous = angel_runtime.get("adapter")
    if previous is not None:
        previous.logout()
    angel_runtime["adapter"] = adapter
    angel_runtime["portfolio_last_read"] = None
    angel_state.update({
        "session_active": True,
        "connected": False,
        "last_error": None,
        "watchlist_count": 0,
        "spot_reference_stale": False,
        "fresh_index_ticks": set(),
        "portfolio_status": {"state": "unavailable", "captured_at": None, "error": None},
    })
    store.append_events([_audit_event("angelone.session_started", {"provider": "angelone_smartapi"})])
    expires = datetime.combine(session.login_date + timedelta(days=1), datetime.min.time(), tzinfo=ZoneInfo("Asia/Kolkata"))
    return {"session_active": True, "expires_at": expires.isoformat(), "orders_enabled": False}


@app.post("/api/angelone/logout")
def angelone_logout(request: Request):
    _require_loopback(request)
    adapter = angel_runtime.get("adapter")
    if adapter is not None:
        adapter.logout()
    angel_runtime["adapter"] = None
    angel_runtime["instruments"] = []
    angel_runtime["portfolio_last_read"] = None
    angel_state.update({
        "session_active": False,
        "connected": False,
        "last_error": None,
        "watchlist_count": 0,
        "spot_reference_stale": False,
        "fresh_index_ticks": set(),
        "portfolio_status": {"state": "unavailable", "captured_at": None, "error": None},
    })
    store.append_events([_audit_event("angelone.session_ended", {"provider": "angelone_smartapi"})])
    return {"session_active": False, "orders_enabled": False}


@app.post("/api/angelone/feed/start")
def start_angelone_feed(request: Request):
    _require_loopback(request)
    if os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=503, detail="Angel One market data is disabled")
    adapter = angel_runtime.get("adapter")
    if adapter is None or adapter.session is None or not adapter.session.is_current():
        raise HTTPException(status_code=401, detail="Log in to Angel One first")
    if getattr(adapter, "feed_running", False):
        return {
            "provider": "angelone_smartapi",
            "watchlist_count": len(angel_runtime["instruments"]) or angel_state["watchlist_count"],
            "option_underlyings": ["NIFTY", "BANKNIFTY"],
            "option_expiry": "nearest",
            "strikes_each_side": 5,
            "spot_reference_stale": angel_state["spot_reference_stale"],
            "orders_enabled": False,
            "already_running": True,
        }
    angel_state.update({
        "connected": False,
        "last_error": None,
        "watchlist_count": 0,
        "fresh_index_ticks": set(),
    })
    try:
        watchlist, _spot = adapter.initial_option_watchlist(allow_stale_quotes=True)
        angel_state["spot_reference_stale"] = adapter.spot_reference_stale
        adapter.start_stream(watchlist, on_tick=_handle_angelone_tick, on_status=_handle_angelone_status)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Angel One feed failed ({type(error).__name__})") from error
    angel_runtime["instruments"] = watchlist
    angel_state["watchlist_count"] = len(watchlist)
    return {
        "provider": "angelone_smartapi",
        "watchlist_count": len(watchlist),
        "option_underlyings": ["NIFTY", "BANKNIFTY"],
        "option_expiry": "nearest",
        "strikes_each_side": 5,
        "spot_reference_stale": angel_state["spot_reference_stale"],
        "orders_enabled": False,
    }


@app.get("/api/llm/status")
def llm_status():
    enabled = LLM_ENABLED and llm_gateway.provider is not None
    copilot_sdk_installed = importlib.util.find_spec("copilot") is not None
    return {
        "provider": LLM_PROVIDER,
        "enabled": enabled,
        "model": LLM_MODEL if enabled else None,
        "sdk_installed": copilot_sdk_installed if LLM_PROVIDER == "copilot" else None,
        "prompt_version": llm_gateway.prompt_version,
        "tools_enabled": False,
        "broker_access": False,
    }


@app.get("/api/market/{symbol}/candles")
def candles(symbol: str, limit: int = 250, timeframe: CandleTimeframe = "1d"):
    normalized_symbol = symbol.upper()
    rows = store.read_candles(normalized_symbol, min(max(limit, 2), 5000), timeframe=timeframe)
    if not rows:
        alias = {"NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY"}.get(normalized_symbol)
        if alias:
            rows = store.read_candles(alias, min(max(limit, 2), 5000), timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return [
        {"symbol": c.symbol, "timeframe": c.timeframe, "timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in rows
    ]


@app.get("/api/market/{symbol}/ticks")
def market_ticks(symbol: str, limit: int = 100):
    return store.read_market_ticks(symbol.upper(), min(max(limit, 1), 1000))

@app.post("/api/angelone/history")
def import_angelone_history(request: AngelHistoricalRequest):
    if request.include_open_interest and request.exchange != "NFO":
        raise HTTPException(status_code=422, detail="Historical open interest is only supported for NFO")
    if os.getenv("ANGELONE_MARKET_DATA_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=503, detail="Angel One market data is disabled")
    if request.from_date > request.to_date:
        raise HTTPException(status_code=422, detail="from_date must not be after to_date")
    span_days = (request.to_date - request.from_date).days
    max_span = {
        "minute": 30, "3minute": 60, "5minute": 100, "10minute": 100,
        "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
    }[request.interval]
    if span_days > max_span:
        raise HTTPException(status_code=422, detail=f"Requested interval is limited to {max_span} days per import")
    if request.to_date > datetime.now(timezone.utc).date():
        raise HTTPException(status_code=422, detail="to_date cannot be in the future")
    adapter = angel_runtime.get("adapter")
    if adapter is None or adapter.session is None or not adapter.session.is_current():
        raise HTTPException(status_code=401, detail="Log in to Angel One first")
    try:
        records = adapter.historical_candles(
            instrument_token=request.instrument_token,
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            from_date=request.from_date,
            to_date=request.to_date,
            include_open_interest=request.include_open_interest,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Angel One historical request failed ({type(error).__name__})") from error
    timeframe = ANGEL_INTERVAL_TIMEFRAMES[request.interval]
    try:
        candles_to_store, duplicate_count = validate_historical_candles(
            records,
            symbol=request.symbol,
            timeframe=timeframe,
            from_date=request.from_date,
            to_date=request.to_date,
        )
        existing = store.read_candles_for_timestamps(
            request.symbol.upper(),
            timeframe,
            [candle.timestamp for candle in candles_to_store],
        )
        candles_to_store, persisted_duplicate_count = deduplicate_historical_candles(
            candles_to_store,
            timeframe=timeframe,
            existing=existing,
        )
        duplicate_count += persisted_duplicate_count
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Angel One returned invalid historical data: {error}") from error
    inserted = store.save_candles(candles_to_store)
    store.append_events([_audit_event("market.historical_data_imported", {
        "provider": "angelone_smartapi",
        "symbol": request.symbol,
        "exchange": request.exchange,
        "interval": request.interval,
        "from_date": request.from_date.isoformat(),
        "to_date": request.to_date.isoformat(),
        "bars_returned": len(records),
        "bars_inserted": inserted,
        "bars_duplicate": duplicate_count,
        "open_interest_included": request.include_open_interest,
    })])
    return {
        "provider": "angelone_smartapi",
        "symbol": request.symbol,
        "interval": request.interval,
        "bars_returned": len(records),
        "bars_inserted": inserted,
        "bars_duplicate": duplicate_count,
        "open_interest_included": request.include_open_interest,
    }


@app.post("/api/angelone/history/previous-day")
def import_previous_angelone_history(request: AngelPreviousHistoricalRequest):
    previous_day = datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=1)
    result = import_angelone_history(AngelHistoricalRequest(
        symbol=request.symbol,
        instrument_token=request.instrument_token,
        exchange=request.exchange,
        from_date=previous_day,
        to_date=previous_day,
        interval=request.interval,
        include_open_interest=request.include_open_interest,
    ))
    return {
        **result,
        "requested_date": previous_day.isoformat(),
        "date_basis": "previous_calendar_day",
        "warning": "An authoritative NSE/BSE holiday calendar is not configured; verify this was a trading session.",
    }


@app.websocket("/ws/market/ticks")
async def market_ticks_stream(socket: WebSocket):
    await socket.accept()
    tick_clients.add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        tick_clients.discard(socket)


@app.websocket("/ws/market/features")
async def market_features_stream(socket: WebSocket):
    await socket.accept()
    feature_clients.add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        feature_clients.discard(socket)


@app.post("/api/market/ingest")
def ingest_market_data(request: MarketIngestRequest):
    adapter = JsonCandleAdapter(request.provider)
    accepted = 0
    rejected = []
    audit_events = []
    for index, raw in enumerate(request.events):
        if raw.get("timeframe") not in (None, request.timeframe):
            quality_gate.invalid += 1
            rejected.append({"index": index, "reason": "timeframe_mismatch"})
            continue
        try:
            event = adapter.normalize({**raw, "timeframe": request.timeframe})
        except (TypeError, ValueError, OverflowError) as error:
            quality_gate.invalid += 1
            rejected.append({"index": index, "reason": "invalid_payload", "detail": str(error)})
            continue
        if request.mode == "live" and not quality_gate.is_fresh(event.candle.timestamp):
            quality_gate.stale += 1
            rejected.append({"index": index, "event_id": event.event_id, "reason": "stale_live_event"})
            continue
        issue = quality_gate.review(event)
        if issue:
            rejected.append({"index": index, "event_id": event.event_id, "reason": issue})
            continue
        if store.save_candles([event.candle]) == 0:
            quality_gate.accepted -= 1
            quality_gate.duplicates += 1
            rejected.append({"index": index, "event_id": event.event_id, "reason": "duplicate_stored_candle"})
            continue
        accepted += 1
        audit_events.append(event.as_event())
    store.append_events(audit_events)
    if accepted:
        store.append_events([_audit_event("market.batch_ingested", {
            "provider": request.provider,
            "mode": request.mode,
            "timeframe": request.timeframe,
            "accepted": accepted,
            "rejected": len(rejected),
        })])
    return {"accepted": accepted, "rejected": rejected, "timeframe": request.timeframe, "quality": quality_gate.status()}


@app.get("/api/market/{symbol}/features")
def market_features(symbol: str, timeframe: CandleTimeframe = "1d"):
    rows = store.read_candles(symbol.upper(), timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return {**feature_snapshot(rows), "timeframe": timeframe}


@app.get("/api/market/{symbol}/features/multi-timeframe")
def market_multi_timeframe_features(symbol: str):
    candles_by_timeframe = {
        timeframe: store.read_candles(symbol.upper(), timeframe=timeframe)
        for timeframe in MULTI_TIMEFRAME_ANALYSIS
    }
    if not any(candles_by_timeframe.values()):
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return multi_timeframe_feature_snapshot(candles_by_timeframe)


@app.get("/api/market/regime")
def india_market_regime(timeframe: CandleTimeframe = "1d"):
    nifty_candles = store.read_candles("NIFTY 50", timeframe=timeframe) or store.read_candles(
        "NIFTY", timeframe=timeframe
    )
    banknifty_candles = store.read_candles("NIFTY BANK", timeframe=timeframe) or store.read_candles(
        "BANKNIFTY", timeframe=timeframe
    )
    candles_by_index = {
        "NIFTY 50": nifty_candles,
        "NIFTY BANK": banknifty_candles,
    }
    return india_regime_engine.classify(candles_by_index, timeframe=timeframe)


@app.post("/api/market/breadth")
def market_breadth(request: MarketBreadthRequest):
    try:
        candles_by_symbol = store.read_candles_for_symbols(
            request.symbols,
            timeframe=request.timeframe,
            limit_per_symbol=market_breadth_engine.high_low_lookback + 1,
        )
        return market_breadth_engine.calculate(
            symbols=request.symbols,
            candles_by_symbol=candles_by_symbol,
            timeframe=request.timeframe,
            sector_by_symbol=request.sector_by_symbol,
            index_symbols=tuple(request.index_symbols),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/india/expected-value")
def india_expected_value(request: IndiaExpectedValueRequest):
    try:
        rates = IndiaProductChargeRates(
            brokerage_rate=request.brokerage_rate,
            stt_buy_rate=request.stt_buy_rate,
            stt_sell_rate=request.stt_sell_rate,
            exchange_transaction_rate=request.exchange_transaction_rate,
            sebi_turnover_rate=request.sebi_turnover_rate,
            gst_rate=request.gst_rate,
            stamp_duty_buy_rate=request.stamp_duty_buy_rate,
            other_turnover_rate=request.other_turnover_rate,
            gst_base_components=tuple(request.gst_base_components),
            brokerage_cap_per_order=request.brokerage_cap_per_order,
        )
        engine = IndiaExpectedValueEngine(IndiaTransactionCostSchedule({request.product: rates}))
        costs = engine.round_trip_cost(
            product=request.product,
            buy_turnover=request.buy_turnover,
            sell_turnover=request.sell_turnover,
            slippage_bps_per_side=request.slippage_bps_per_side,
            market_impact_bps_per_side=request.market_impact_bps_per_side,
        )
        expected_value = engine.expected_value(
            win_probability=request.win_probability,
            average_win=request.average_win,
            average_loss=request.average_loss,
            round_trip_cost=costs,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "costs": costs.as_dict(),
        "expected_value": expected_value.as_dict(),
        "probability_status": "CALLER_SUPPLIED_UNVALIDATED",
        "execution_authority": False,
    }


@app.get("/api/market/{symbol}/similarity")
def market_historical_similarity(
    symbol: str,
    timeframe: CandleTimeframe = "1d",
    horizon_bars: int = Query(default=5, ge=1, le=50),
    top_k: int = Query(default=20, ge=1, le=100),
    max_distance: float = Query(default=0.35, gt=0, le=2),
):
    rows = store.read_candles(symbol.upper(), limit=5000, timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument and timeframe")
    try:
        return historical_similarity(
            rows,
            horizon_bars=horizon_bars,
            top_k=top_k,
            max_distance=max_distance,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/signals/{symbol}")
def signal(symbol: str, fast_window: int = 8, slow_window: int = 21, timeframe: CandleTimeframe = "1d"):
    try:
        strategy = SmaCrossStrategy(fast_window, slow_window)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = store.read_candles(symbol.upper(), timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "action": strategy.action(rows),
        "as_of": rows[-1].timestamp,
        "data_source": "synthetic_demo_ohlcv",
    }


@app.post("/api/news")
def create_news_item(request: NewsIngestRequest):
    try:
        item, intelligence = ingest_news(request.model_dump())
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    record = asdict(item)
    record["instruments"] = list(record["instruments"])
    record["intelligence"] = intelligence
    if not store.save_news(record):
        return {"accepted": False, "duplicate": True, "dedupe_key": item.dedupe_key}
    store.append_events([
        _audit_event("news.received", {"article_id": item.article_id, "source": item.source, "published_at": item.published_at}),
        _audit_event("news.intelligence_created", {"article_id": item.article_id, "event": intelligence}),
    ])
    return {"accepted": True, "article": record}


@app.post("/api/news/batch")
def create_news_batch(request: NewsBatchIngestRequest):
    prepared = []
    try:
        for item_request in request.items:
            item, intelligence = ingest_news(item_request.model_dump())
            record = asdict(item)
            record["instruments"] = list(record["instruments"])
            record["intelligence"] = intelligence
            prepared.append(record)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    results = []
    audit_events = []
    for record in prepared:
        accepted = store.save_news(record)
        results.append({
            "article_id": record["article_id"],
            "accepted": accepted,
            "duplicate": not accepted,
            "intelligence": record["intelligence"],
        })
        if accepted:
            audit_events.extend([
                _audit_event("news.received", {
                    "article_id": record["article_id"],
                    "source": record["source"],
                    "published_at": record["published_at"],
                }),
                _audit_event("news.intelligence_created", {
                    "article_id": record["article_id"],
                    "event": record["intelligence"],
                }),
            ])
    if audit_events:
        store.append_events(audit_events)
    accepted_count = sum(1 for result in results if result["accepted"])
    return {
        "accepted": accepted_count,
        "duplicates": len(results) - accepted_count,
        "items": results,
        "orders_enabled": False,
    }


@app.get("/api/news")
def list_news(limit: int = 50):
    return store.read_news(min(max(limit, 1), 200))


@app.post("/api/rag/memory")
def remember_trading_context(request: RagMemoryRequest):
    memory = HistoricalTradingMemory(store)
    try:
        accepted = memory.remember(
            document_id=request.document_id,
            kind=request.kind,
            text=request.text,
            metadata=request.metadata,
            created_at=request.created_at,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "accepted": accepted,
        "duplicate": not accepted,
        "document_id": request.document_id,
        "kind": request.kind.strip().upper(),
        "embedding": {"provider": "local_hashing", "dimension": memory.embedder.dimension},
        "orders_enabled": False,
    }


@app.post("/api/rag/search")
def search_trading_memory(request: RagSearchRequest):
    memory = HistoricalTradingMemory(store)
    try:
        results = memory.recall(
            request.query,
            limit=request.limit,
            filters=request.filters,
            max_age_seconds=request.max_age_seconds,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (sqlite3.Error, TimeoutError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=503, detail="Trading memory retrieval is unavailable") from error
    return {
        "query": request.query,
        "filters": request.filters,
        "results": results,
        "embedding": {"provider": "local_hashing", "dimension": memory.embedder.dimension},
        "orders_enabled": False,
    }


@app.post("/api/hypotheses/{symbol}")
def hypothesis(symbol: str, fast_window: int = 8, slow_window: int = 21, timeframe: CandleTimeframe = "1d"):
    try:
        strategy = SmaCrossStrategy(fast_window, slow_window)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = store.read_candles(symbol.upper(), timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    relevant_news = [
        item["intelligence"] for item in store.read_news(200)
        if symbol.upper() in item["instruments"]
    ]
    proposal = build_hypothesis(rows, strategy, relevant_news)
    validation = validate_hypothesis(proposal, kill_switch=risk_config.kill_switch)
    store.append_events([_audit_event("strategy.hypothesis_validated", {
        "hypothesis_id": proposal["hypothesis_id"],
        "instrument": proposal["instrument"],
        "approved": validation["approved"],
        "reasons": validation["reasons"],
    })])
    return {"hypothesis": proposal, "validation": validation, "timeframe": timeframe}


@app.post("/api/llm/hypotheses/{symbol}")
def llm_hypothesis(symbol: str, timeframe: CandleTimeframe = "1d"):
    if not LLM_ENABLED or llm_gateway.provider is None:
        raise HTTPException(status_code=503, detail="LLM hypothesis generation is disabled or not configured")
    if not angel_state["connected"]:
        raise HTTPException(status_code=503, detail="Angel One market feed is not connected")
    if angel_state["spot_reference_stale"]:
        raise HTTPException(status_code=409, detail="Option watchlist uses a stale spot reference; waiting for fresh NIFTY and BANKNIFTY index ticks")
    rows = store.read_candles(symbol.upper(), timeframe=timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    latest = rows[-1]
    recent_ticks = store.read_market_ticks(symbol.upper(), limit=1)
    if not recent_ticks or not MarketDataQualityGate.is_fresh(recent_ticks[-1]["timestamp"], max_age_seconds=30):
        raise HTTPException(status_code=409, detail="Market data is stale; no model call was made")
    snapshot = feature_snapshot(rows)
    relevant_news = [
        item for item in store.read_news(200)
        if symbol.upper() in item["instruments"] and not item["intelligence"].get("is_stale", True)
    ]
    candle_id = f"candle:{latest.symbol}:{latest.timestamp}"
    tick_id = f"tick:{recent_ticks[-1]['event_id']}"
    evidence_ids = [candle_id, tick_id, *[item["article_id"] for item in relevant_news]]
    context = {
        "instruments": [symbol.upper()],
        "data_timestamp": recent_ticks[-1]["timestamp"],
        "market_snapshot": {
            "timeframe": timeframe,
            "candle_timestamp": latest.timestamp,
            "ohlcv": {
                "open": latest.open, "high": latest.high, "low": latest.low,
                "close": latest.close, "volume": latest.volume,
            },
            "latest_tick": {
                "timestamp": recent_ticks[-1]["timestamp"],
                "last_price": recent_ticks[-1]["last_price"],
                "volume": recent_ticks[-1]["volume"],
                "open_interest": recent_ticks[-1]["open_interest"],
            },
            "features": snapshot["features"],
            "regime": snapshot["regime"],
        },
        "news_events": [item["intelligence"] for item in relevant_news],
        "evidence_ids": evidence_ids,
        "strategy_constraints": {"strategy": "sma_cross", "directional_exposure": "long_only", "quantity": "risk_engine_only"},
    }
    try:
        decision = StrategyAgent(llm_gateway).propose(model=LLM_MODEL, context=context)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    validation = validate_hypothesis(decision.output, kill_switch=risk_config.kill_switch)
    adversarial = AdversarialAgent().review(
        decision.output,
        context,
        kill_switch=risk_config.kill_switch,
    )
    audit = _audit_event("llm.hypothesis_created", {
        "model": decision.model,
        "provider": decision.provider,
        "prompt_version": decision.prompt_version,
        "context_hash": decision.context_hash,
        "timeframe": timeframe,
        "hypothesis": decision.output,
        "validation": validation,
        "adversarial_review": adversarial,
    })
    store.append_events([audit])
    return {"decision": decision.output, "validation": validation, "adversarial_review": adversarial, "reproducibility": {
        "model": decision.model,
        "provider": decision.provider,
        "prompt_version": decision.prompt_version,
        "context_hash": decision.context_hash,
        "timeframe": timeframe,
        "timestamp": decision.timestamp,
    }}


@app.post("/api/backtests")
def backtest(request: SimulationRequest):
    return _simulate(request, "backtest")


@app.post("/api/paper/replay")
def paper_replay(request: SimulationRequest):
    return _simulate(request, "paper")


def _simulation_config_from_request(request: SimulationRequest | WalkForwardBacktestRequest) -> SimulationConfig:
    if (request.india_product is None) != (request.india_charge_rates is None):
        raise ValueError("India product and its explicit charge schedule must be supplied together")
    charge_rates = None
    if request.india_charge_rates is not None:
        if request.fee_bps != 0:
            raise ValueError("Set fee_bps=0 when supplying India transaction charges")
        charge_rates = IndiaProductChargeRates(
            **{
                **request.india_charge_rates.model_dump(),
                "gst_base_components": tuple(request.india_charge_rates.gst_base_components),
            }
        )
    return SimulationConfig(
        starting_cash=request.starting_cash,
        fee_bps=request.fee_bps,
        slippage_bps=request.slippage_bps,
        market_impact_bps=request.market_impact_bps,
        max_volume_participation=request.max_volume_participation,
        execution_delay_bars=request.execution_delay_bars,
        india_product=request.india_product,
        india_charge_rates=charge_rates,
    )


@app.post("/api/backtests/walk-forward")
def walk_forward_backtest(request: WalkForwardBacktestRequest):
    if request.slow_window <= request.fast_window:
        raise HTTPException(status_code=422, detail="slow_window must be greater than fast_window")
    rows = store.read_candles(request.symbol, limit=5000, timeframe=request.timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument and timeframe")
    try:
        result = run_walk_forward(
            rows,
            lambda: SmaCrossStrategy(request.fast_window, request.slow_window),
            train_bars=request.train_bars,
            test_bars=request.test_bars,
            step_bars=request.step_bars,
            config=_simulation_config_from_request(request),
            risk_engine=RiskEngine(risk_config),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_events([_audit_event("backtest.walk_forward_completed", {
        key: result[key]
        for key in (
            "symbol", "timeframe", "method", "training_bars_initial", "test_bars_requested",
            "step_bars", "fold_count", "total_test_bars", "total_closed_trades",
            "mean_fold_return_pct", "worst_fold_drawdown_pct",
        )
    })])
    return result


def _simulate(request: SimulationRequest, mode: Literal["backtest", "paper"]):
    if request.slow_window <= request.fast_window:
        raise HTTPException(status_code=422, detail="slow_window must be greater than fast_window")
    rows = store.read_candles(request.symbol, timeframe=request.timeframe)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    try:
        result = run_simulation(
            rows,
            SmaCrossStrategy(request.fast_window, request.slow_window),
            config=_simulation_config_from_request(request),
            risk_engine=RiskEngine(risk_config),
            mode=mode,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result["timeframe"] = request.timeframe
    store.append_events(result["events"])
    return result


@app.get("/api/risk")
def risk_status():
    return {
        "kill_switch": risk_config.kill_switch,
        "max_position_fraction": risk_config.max_position_fraction,
        "max_exposure_fraction": risk_config.max_exposure_fraction,
        "new_entries_allowed": not risk_config.kill_switch,
        "execution_mode": "simulation_only",
    }


@app.post("/api/risk/kill-switch")
def set_kill_switch(request: KillSwitchRequest):
    global risk_config
    risk_config = RiskConfig(kill_switch=request.enabled)
    return risk_status()


@app.get("/api/events")
def events(limit: int = 50):
    return store.read_events(min(max(limit, 1), 200))


@app.get("/api/paper/events")
def paper_events(limit: int = 40):
    return store.read_simulation_events("paper", min(max(limit, 1), 100))


def _audit_event(event_type: str, payload: dict) -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": str(uuid4()),
        "source": "platform_api",
        "version": 1,
        "schema_version": "1.0",
        "payload": payload,
    }


def _require_loopback(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Credential actions are available only from this machine")


def _handle_angelone_status(event_type: str, payload: dict) -> None:
    safe_payload = {
        key: value for key, value in payload.items()
        if key in {"code", "attempts", "subscriptions", "mode"}
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    with angel_state_lock:
        angel_state["connected"] = event_type == "market.connected"
        if event_type in {"market.connection_error", "market.connection_closed", "market.reconnect_exhausted"}:
            angel_state["last_error"] = event_type
        elif event_type == "market.connected":
            angel_state["last_error"] = None
    store.append_events([_audit_event(event_type, safe_payload)])


def _handle_angelone_tick(tick) -> None:
    issue = angel_tick_quality.review(tick)
    if issue:
        if angel_tick_quality.should_audit_rejection(tick.symbol, issue):
            store.append_events([_audit_event("market.tick_rejected", {
                "event_id": tick.event_id,
                "symbol": tick.symbol,
                "reason": issue,
            })])
        return
    if tick.symbol in {"NIFTY 50", "NIFTY BANK"}:
        angel_state["fresh_index_ticks"].add(tick.symbol)
        if {"NIFTY 50", "NIFTY BANK"} <= angel_state["fresh_index_ticks"]:
            angel_state["spot_reference_stale"] = False
    event = tick.as_event()
    if not store.save_market_tick(event):
        return
    try:
        store.upsert_candle(angel_candle_aggregator.add(tick))
        closed_bars = angel_candle_aggregator.take_closed()
    except ValueError:
        store.append_events([_audit_event("market.tick_rejected", {
            "event_id": tick.event_id,
            "symbol": tick.symbol,
            "reason": "out_of_order_candle_aggregation",
        })])
        return
    store.append_events([event])
    for closed_bar in closed_bars:
        bars = [
            candle for candle in store.read_candles(closed_bar.symbol, limit=500, timeframe="1m")
            if candle.timestamp <= closed_bar.timestamp
        ]
        if not bars:
            continue
        snapshot = feature_snapshot(bars)
        feature_event = _audit_event("market.feature_snapshot_created", {
            "symbol": closed_bar.symbol,
            "timeframe": "1m",
            "timestamp": closed_bar.timestamp,
            "is_provisional": False,
            "feature_snapshot": snapshot,
        })
        store.append_events([feature_event])
        loop = app_loop
        if feature_clients and loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_feature_snapshot(feature_event), loop)
    if tick_clients:
        loop = app_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_tick(event["payload"]), loop)
    pipeline = angel_runtime.get("decision_pipeline")
    if pipeline is not None:
        try:
            bars = store.read_candles(tick.symbol, limit=500, timeframe="1m")
            try:
                similarity = historical_similarity(bars, horizon_bars=5, top_k=10, max_distance=0.5)
            except ValueError:
                similarity = {"status": "INSUFFICIENT_DATA"}
            memory_results = HistoricalTradingMemory(store).recall(
                f"{tick.symbol} market decision", limit=5, filters={"instrument": tick.symbol}
            )
            index_candles = {
                symbol: store.read_candles(symbol, timeframe="1d")
                for symbol in ("NIFTY 50", "NIFTY BANK")
            }
            regime = india_regime_engine.classify(index_candles, timeframe="1d")
            pipeline_result = pipeline.process_tick(
                tick=tick,
                candles=bars,
                context={
                    "model": LLM_MODEL,
                    "news_events": [
                        item["intelligence"] for item in store.read_news(200)
                        if tick.symbol in item["instruments"] and not item["intelligence"].get("is_stale", True)
                    ],
                    "portfolio": dict(angel_state.get("portfolio_status", {})),
                    "market_breadth": {"status": "UNAVAILABLE", "reason": "No explicit point-in-time universe configured"},
                    "india_regime": regime,
                    "historical_similarity": similarity,
                    "trading_memory": memory_results,
                },
            )
            store.append_events([_audit_event("decision.pipeline_completed", {
                "symbol": tick.symbol,
                "status": pipeline_result.get("status"),
                "execution": pipeline_result.get("execution", False),
            })])
        except Exception as error:
            store.append_events([_audit_event("decision.pipeline_failed", {
                "symbol": tick.symbol,
                "error_type": type(error).__name__,
            })])


async def _broadcast_tick(message: dict) -> None:
    stale = []
    for client in tuple(tick_clients):
        try:
            await client.send_json(message)
        except Exception:
            stale.append(client)
    for client in stale:
        tick_clients.discard(client)


async def _broadcast_feature_snapshot(event: dict) -> None:
    stale = []
    for client in tuple(feature_clients):
        try:
            await client.send_json(event)
        except Exception:
            stale.append(client)
    for client in stale:
        feature_clients.discard(client)