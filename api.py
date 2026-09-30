"""Local API for deterministic backtesting and paper replay; no broker adapter exists."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core import Candle, RiskConfig, RiskEngine, SimulationConfig, SmaCrossStrategy, run_simulation
from angelone_adapter import AngelOneCredentials, AngelOneMarketDataAdapter
from features import feature_snapshot
from intelligence import build_hypothesis, ingest_news, validate_hypothesis
from llm_gateway import CopilotModelProvider, LLMGateway, OpenAIModelProvider
from marketdata import (
    JsonCandleAdapter,
    MarketDataQualityGate,
    MarketTickQualityGate,
    MinuteCandleAggregator,
)
from store import EventStore


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
store = EventStore(BASE_DIR / "data" / "platform.sqlite3")
if not store.read_candles("DEMO"):
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
}
angel_state_lock = Lock()
angel_runtime: dict = {"adapter": None, "instruments": []}
tick_clients: set[WebSocket] = set()
app_loop: asyncio.AbstractEventLoop | None = None
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").strip().lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "copilot").strip().lower()
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "auto").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
LLM_MODEL = COPILOT_MODEL if LLM_PROVIDER == "copilot" else OPENAI_MODEL


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
    starting_cash: float = Field(default=100_000, gt=0, le=100_000_000)
    fast_window: int = Field(default=8, ge=2, le=100)
    slow_window: int = Field(default=21, ge=3, le=250)
    fee_bps: float = Field(default=10, ge=0, le=500)
    slippage_bps: float = Field(default=5, ge=0, le=50)


class KillSwitchRequest(BaseModel):
    enabled: bool


class MarketIngestRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=80)
    mode: Literal["historical", "live"] = "historical"
    events: list[dict] = Field(min_length=1, max_length=5000)


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

class AngelHistoricalRequest(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9._-]{1,40}$")
    instrument_token: int = Field(gt=0)
    exchange: Literal["NSE", "BSE", "NFO"]
    from_date: date
    to_date: date
    interval: Literal["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
    include_open_interest: bool = False


class AngelLoginRequest(BaseModel):
    client_code: str = Field(min_length=1, max_length=32)
    pin: str = Field(min_length=1, max_length=32)
    totp: str = Field(pattern=r"^\d{6}$")


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
        "quality": angel_tick_quality.status(),
        "exchanges": ["NSE", "NFO"],
        "option_underlyings": ["NIFTY", "BANKNIFTY"],
        "option_expiry": "nearest",
        "option_strikes_each_side": 5,
        "futures_enabled": False,
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
    angel_state.update({
        "session_active": True,
        "connected": False,
        "last_error": None,
        "watchlist_count": 0,
        "spot_reference_stale": False,
        "fresh_index_ticks": set(),
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
    angel_state.update({
        "session_active": False,
        "connected": False,
        "last_error": None,
        "watchlist_count": 0,
        "spot_reference_stale": False,
        "fresh_index_ticks": set(),
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
def candles(symbol: str, limit: int = 250):
    rows = store.read_candles(symbol.upper(), min(max(limit, 2), 5000))
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return [
        {"symbol": c.symbol, "timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
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
    candles_to_store = [
        Candle(
            row["symbol"], row["timestamp"], row["open"], row["high"],
            row["low"], row["close"], row["volume"], row["open_interest"],
        )
        for row in records
    ]
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
        "open_interest_included": request.include_open_interest,
    })])
    return {
        "provider": "angelone_smartapi",
        "symbol": request.symbol,
        "interval": request.interval,
        "bars_returned": len(records),
        "bars_inserted": inserted,
        "open_interest_included": request.include_open_interest,
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


@app.post("/api/market/ingest")
def ingest_market_data(request: MarketIngestRequest):
    adapter = JsonCandleAdapter(request.provider)
    accepted = 0
    rejected = []
    audit_events = []
    for index, raw in enumerate(request.events):
        try:
            event = adapter.normalize(raw)
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
            "accepted": accepted,
            "rejected": len(rejected),
        })])
    return {"accepted": accepted, "rejected": rejected, "quality": quality_gate.status()}


@app.get("/api/market/{symbol}/features")
def market_features(symbol: str):
    rows = store.read_candles(symbol.upper())
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return feature_snapshot(rows)


@app.get("/api/signals/{symbol}")
def signal(symbol: str, fast_window: int = 8, slow_window: int = 21):
    try:
        strategy = SmaCrossStrategy(fast_window, slow_window)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = store.read_candles(symbol.upper())
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    return {
        "symbol": symbol.upper(),
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


@app.get("/api/news")
def list_news(limit: int = 50):
    return store.read_news(min(max(limit, 1), 200))


@app.post("/api/hypotheses/{symbol}")
def hypothesis(symbol: str, fast_window: int = 8, slow_window: int = 21):
    try:
        strategy = SmaCrossStrategy(fast_window, slow_window)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = store.read_candles(symbol.upper())
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
    return {"hypothesis": proposal, "validation": validation}


@app.post("/api/llm/hypotheses/{symbol}")
def llm_hypothesis(symbol: str):
    if not LLM_ENABLED or llm_gateway.provider is None:
        raise HTTPException(status_code=503, detail="LLM hypothesis generation is disabled or not configured")
    if not angel_state["connected"]:
        raise HTTPException(status_code=503, detail="Angel One market feed is not connected")
    if angel_state["spot_reference_stale"]:
        raise HTTPException(status_code=409, detail="Option watchlist uses a stale spot reference; waiting for fresh NIFTY and BANKNIFTY index ticks")
    rows = store.read_candles(symbol.upper())
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
        decision = llm_gateway.propose(model=LLM_MODEL, context=context)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    validation = validate_hypothesis(decision.output, kill_switch=risk_config.kill_switch)
    audit = _audit_event("llm.hypothesis_created", {
        "model": decision.model,
        "provider": decision.provider,
        "prompt_version": decision.prompt_version,
        "context_hash": decision.context_hash,
        "hypothesis": decision.output,
        "validation": validation,
    })
    store.append_events([audit])
    return {"decision": decision.output, "validation": validation, "reproducibility": {
        "model": decision.model,
        "provider": decision.provider,
        "prompt_version": decision.prompt_version,
        "context_hash": decision.context_hash,
        "timestamp": decision.timestamp,
    }}


@app.post("/api/backtests")
def backtest(request: SimulationRequest):
    return _simulate(request, "backtest")


@app.post("/api/paper/replay")
def paper_replay(request: SimulationRequest):
    return _simulate(request, "paper")


def _simulate(request: SimulationRequest, mode: Literal["backtest", "paper"]):
    if request.slow_window <= request.fast_window:
        raise HTTPException(status_code=422, detail="slow_window must be greater than fast_window")
    rows = store.read_candles(request.symbol)
    if not rows:
        raise HTTPException(status_code=404, detail="No stored candles for this instrument")
    result = run_simulation(
        rows,
        SmaCrossStrategy(request.fast_window, request.slow_window),
        config=SimulationConfig(request.starting_cash, request.fee_bps, request.slippage_bps),
        risk_engine=RiskEngine(risk_config),
        mode=mode,
    )
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
    except ValueError:
        store.append_events([_audit_event("market.tick_rejected", {
            "event_id": tick.event_id,
            "symbol": tick.symbol,
            "reason": "out_of_order_candle_aggregation",
        })])
        return
    store.append_events([event])
    if tick_clients:
        loop = app_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_tick(event["payload"]), loop)


async def _broadcast_tick(message: dict) -> None:
    stale = []
    for client in tuple(tick_clients):
        try:
            await client.send_json(message)
        except Exception:
            stale.append(client)
    for client in stale:
        tick_clients.discard(client)