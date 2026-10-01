"""Read-only Angel One SmartAPI market-data integration."""

from __future__ import annotations

import os
import re
import importlib.util
import socket
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import Callable
from uuid import getnode, uuid4
from zoneinfo import ZoneInfo

from kite_adapter import DEFAULT_OPTION_UNDERLYINGS, select_nearest_option_contracts
from india_market import (
    AssetType,
    Contract,
    Exchange,
    ExpiryCalendar,
    IndiaMarketPolicy,
    Instrument,
    LotSize,
    OptionType,
    TickSize,
)


INSTRUMENT_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
INDEX_TOKENS = {"NIFTY": "99926000", "BANKNIFTY": "99926009"}
EXCHANGE_TYPES = {"NSE": 1, "NFO": 2, "BSE": 3}


@dataclass(frozen=True, repr=False)
class AngelOneCredentials:
    api_key: str

    @classmethod
    def from_environment(cls) -> AngelOneCredentials | None:
        api_key = os.getenv("ANGELONE_API_KEY", "").strip()
        return cls(api_key) if api_key else None

    def __repr__(self) -> str:
        return "AngelOneCredentials(api_key=<redacted>)"


class AngelOneRestClient:
    """Small quiet REST client; request/response bodies are never logged."""

    def __init__(self, api_key: str, *, http_session=None):
        import requests

        self.api_key = api_key
        self._http = http_session or requests.Session()
        self._auth_token: str | None = None
        self._refresh_token: str | None = None
        self._feed_token: str | None = None
        self._public_ip: str | None = os.getenv("ANGELONE_CLIENT_PUBLIC_IP")

    def _headers(self, *, authenticated: bool) -> dict[str, str]:
        if not self._public_ip:
            import requests

            try:
                self._public_ip = requests.get("https://api.ipify.org", timeout=3).text.strip()
            except requests.RequestException as error:
                raise RuntimeError("Could not determine the public IP required by SmartAPI") from error
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            local_ip = "127.0.0.1"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": local_ip,
            "X-ClientPublicIP": self._public_ip,
            "X-MACAddress": ":".join(f"{byte:02x}" for byte in getnode().to_bytes(6, "big")),
            "X-PrivateKey": self.api_key,
        }
        if authenticated:
            if not self._auth_token:
                raise RuntimeError("Angel One session is not authenticated")
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _post(self, path: str, payload: dict, *, authenticated: bool) -> dict:
        try:
            response = self._http.post(
                f"https://apiconnect.angelone.in{path}",
                json=payload,
                headers=self._headers(authenticated=authenticated),
                timeout=10,
            )
            body = response.json()
        except Exception as error:
            raise RuntimeError("Angel One request failed; response details were suppressed") from error
        if response.status_code >= 400 or not isinstance(body, dict) or body.get("status") is not True:
            error_code = body.get("errorcode", "unknown") if isinstance(body, dict) else "invalid_response"
            raise RuntimeError(f"Angel One request was rejected (HTTP {response.status_code}, code {error_code})")
        return body

    def _get(self, path: str) -> dict:
        try:
            response = self._http.get(
                f"https://apiconnect.angelone.in{path}",
                headers=self._headers(authenticated=True),
                timeout=10,
            )
            body = response.json()
        except Exception as error:
            raise RuntimeError("Angel One request failed; response details were suppressed") from error
        if response.status_code >= 400 or not isinstance(body, dict) or body.get("status") is not True:
            error_code = body.get("errorcode", "unknown") if isinstance(body, dict) else "invalid_response"
            raise RuntimeError(f"Angel One request was rejected (HTTP {response.status_code}, code {error_code})")
        return body

    def generateSession(self, client_code: str, pin: str, totp: str) -> dict:
        response = self._post(
            "/rest/auth/angelbroking/user/v1/loginByPassword",
            {"clientcode": client_code, "password": pin, "totp": totp},
            authenticated=False,
        )
        data = response.get("data", {})
        self._auth_token = data.get("jwtToken")
        self._refresh_token = data.get("refreshToken")
        self._feed_token = data.get("feedToken")
        return response

    def getMarketData(self, mode: str, exchange_tokens: dict) -> dict:
        return self._post(
            "/rest/secure/angelbroking/market/v1/quote/",
            {"mode": mode, "exchangeTokens": exchange_tokens},
            authenticated=True,
        )

    def getCandleData(self, params: dict) -> dict:
        return self._post("/rest/secure/angelbroking/historical/v1/getCandleData", params, authenticated=True)

    def getOIData(self, params: dict) -> dict:
        return self._post("/rest/secure/angelbroking/historical/v1/getOIData", params, authenticated=True)

    def getPosition(self) -> dict:
        return self._get("/rest/secure/angelbroking/order/v1/getPosition")

    def getHolding(self) -> dict:
        return self._get("/rest/secure/angelbroking/portfolio/v1/getHolding")

    def terminateSession(self, client_code: str) -> dict:
        response = self._post(
            "/rest/secure/angelbroking/user/v1/logout",
            {"clientcode": client_code},
            authenticated=True,
        )
        self._auth_token = None
        self._refresh_token = None
        self._feed_token = None
        return response


@dataclass(frozen=True, repr=False)
class AngelOneSession:
    client_code: str
    auth_token: str
    refresh_token: str
    feed_token: str
    login_date: date

    def __repr__(self) -> str:
        return f"AngelOneSession(client_code=<redacted>, login_date={self.login_date.isoformat()})"

    def is_current(self, *, now: datetime | None = None) -> bool:
        local_now = (now or datetime.now(ZoneInfo("Asia/Kolkata"))).astimezone(ZoneInfo("Asia/Kolkata"))
        return local_now.date() == self.login_date


@dataclass(frozen=True)
class AngelOneTick:
    event_id: str
    timestamp: str
    ingested_at: str
    correlation_id: str
    source: str
    instrument_token: int
    exchange: str
    symbol: str
    last_price: float
    last_trade_quantity: int | None
    volume: int | None
    open_interest: int | None
    day_ohlc: dict
    bids: tuple[dict, ...]
    asks: tuple[dict, ...]
    sequence: int | None

    def as_event(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": "market.tick_received",
            "timestamp": self.ingested_at,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "version": 1,
            "schema_version": "1.0",
            "payload": {
                "instrument_token": self.instrument_token,
                "exchange": self.exchange,
                "symbol": self.symbol,
                "market_timestamp": self.timestamp,
                "last_price": self.last_price,
                "last_trade_quantity": self.last_trade_quantity,
                "volume": self.volume,
                "open_interest": self.open_interest,
                "day_ohlc": self.day_ohlc,
                "bids": list(self.bids),
                "asks": list(self.asks),
                "sequence": self.sequence,
            },
        }


class AngelOneMarketDataAdapter:
    """SmartAPI client wrapper exposing only auth and market-data methods."""

    def __init__(
        self,
        *,
        api_key: str,
        smart_api,
        websocket_factory=None,
        instrument_loader: Callable[[], list[dict]] | None = None,
        max_quote_age_seconds: int = 30,
        fallback_interval_seconds: float = 5,
        stale_tick_seconds: float = 15,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 1,
    ):
        if not api_key:
            raise ValueError("Angel One SmartAPI key is required")
        self._api_key = api_key
        self._client = smart_api
        self._websocket_factory = websocket_factory
        self._instrument_loader = instrument_loader
        self.max_quote_age_seconds = max_quote_age_seconds
        self.session: AngelOneSession | None = None
        self.websocket = None
        self._websocket_thread: threading.Thread | None = None
        self._fallback_thread: threading.Thread | None = None
        self._stream_stop: threading.Event | None = None
        self._stream_connected = threading.Event()
        self._last_tick_monotonic: float | None = None
        self._last_pong_monotonic: float | None = None
        self.fallback_interval_seconds = max(1.0, fallback_interval_seconds)
        self.stale_tick_seconds = max(1.0, stale_tick_seconds)
        self.max_reconnect_attempts = max(1, max_reconnect_attempts)
        self.reconnect_backoff_seconds = max(0.1, reconnect_backoff_seconds)
        self.spot_reference_stale = False
        self.spot_reference_age_seconds: dict[str, float] = {}

    @property
    def feed_running(self) -> bool:
        stop_event = self._stream_stop
        return bool(
            stop_event is not None
            and not stop_event.is_set()
            and any(
                thread is not None and thread.is_alive()
                for thread in (self._websocket_thread, self._fallback_thread)
            )
        )

    @classmethod
    def from_environment(cls) -> AngelOneMarketDataAdapter:
        credentials = AngelOneCredentials.from_environment()
        if credentials is None:
            raise RuntimeError("Set ANGELONE_API_KEY in agentic-trading-platform/.env")
        try:
            websocket_factory = _load_websocket_factory()
        except (ImportError, FileNotFoundError) as error:
            raise RuntimeError("Install the optional Angel One SDK from requirements-angelone.txt") from error
        return cls(
            api_key=credentials.api_key,
            smart_api=AngelOneRestClient(credentials.api_key),
            websocket_factory=websocket_factory,
        )

    def login(self, *, client_code: str, pin: str, totp: str) -> AngelOneSession:
        if not client_code or not pin or not re.fullmatch(r"\d{6}", totp):
            raise ValueError("Client code, PIN and a six-digit TOTP are required")
        try:
            response = self._client.generateSession(client_code, pin, totp)
        except Exception as error:
            raise RuntimeError("Angel One authentication failed; check local credentials and TOTP") from error
        data = response.get("data", {}) if isinstance(response, dict) else {}
        if not response or response.get("status") is not True or not all(
            data.get(key) for key in ("jwtToken", "refreshToken", "feedToken")
        ):
            raise RuntimeError("Angel One authentication failed; check local credentials and TOTP")
        self.session = AngelOneSession(
            client_code=client_code,
            auth_token=data["jwtToken"],
            refresh_token=data["refreshToken"],
            feed_token=data["feedToken"],
            login_date=datetime.now(ZoneInfo("Asia/Kolkata")).date(),
        )
        return self.session

    def logout(self) -> None:
        if self._stream_stop is not None:
            self._stream_stop.set()
        self._stream_connected.clear()
        if self.websocket is not None:
            try:
                self.websocket.close_connection()
            except Exception:
                pass
            finally:
                self.websocket = None
        current_thread = threading.current_thread()
        for thread in (self._websocket_thread, self._fallback_thread):
            if thread is not None and thread is not current_thread and thread.is_alive():
                thread.join(timeout=3)
        if self.session is not None:
            try:
                self._client.terminateSession(self.session.client_code)
            finally:
                self.session = None

    def _require_session(self) -> AngelOneSession:
        if self.session is None or not self.session.is_current():
            self.session = None
            raise RuntimeError("Angel One session is missing or expired; log in again")
        return self.session

    def instrument_master(self) -> list[dict]:
        if self._instrument_loader is not None:
            return self._instrument_loader()
        import requests

        response = requests.get(INSTRUMENT_MASTER_URL, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Angel One returned an invalid instrument master")
        return data

    def normalized_reference_data(
        self,
        *,
        price_scale: Decimal,
        as_of: date | None = None,
    ) -> tuple[IndiaMarketPolicy, dict[str, int]]:
        return normalize_angelone_instrument_master(
            self.instrument_master(),
            price_scale=price_scale,
            as_of=as_of,
        )

    def read_portfolio_snapshot(self) -> dict:
        """Read current broker positions and holdings; this method never submits or changes orders."""
        self._require_session()
        positions_response = self._client.getPosition()
        holdings_response = self._client.getHolding()
        position_rows = _portfolio_response_rows(positions_response, "net")
        holding_rows = _portfolio_response_rows(holdings_response, "holdings")
        rejected = {"positions": 0, "holdings": 0}
        positions = []
        holdings = []
        for row in position_rows:
            normalized = _normalize_broker_position(row, source="POSITION")
            if normalized is None:
                rejected["positions"] += 1
            elif normalized["quantity"] != 0:
                positions.append(normalized)
        for row in holding_rows:
            normalized = _normalize_broker_position(row, source="HOLDING")
            if normalized is None:
                rejected["holdings"] += 1
            elif normalized["quantity"] != 0:
                holdings.append(normalized)
        return {
            "provider": "angelone_smartapi",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
            "holdings": holdings,
            "rejected_rows": rejected,
            "orders_enabled": False,
        }

    def initial_option_watchlist(
        self,
        *,
        underlyings: tuple[str, ...] = DEFAULT_OPTION_UNDERLYINGS,
        strikes_each_side: int = 5,
        as_of: date | None = None,
        allow_stale_quotes: bool = False,
    ) -> tuple[list[dict], dict[str, float]]:
        self._require_session()
        requested = {name.upper() for name in underlyings}
        if not requested or not requested <= INDEX_TOKENS.keys():
            raise ValueError("Only NIFTY and BANKNIFTY index options are configured")
        tokens = [INDEX_TOKENS[name] for name in sorted(requested)]
        response = self._client.getMarketData("FULL", {"NSE": tokens})
        fetched = (response.get("data") or {}).get("fetched", []) if response else []
        quotes = {str(item.get("symbolToken")): item for item in fetched}
        spot_prices: dict[str, float] = {}
        quote_ages: dict[str, float] = {}
        now = datetime.now(timezone.utc)
        for underlying in requested:
            token = INDEX_TOKENS[underlying]
            quote = quotes.get(token)
            if not quote:
                raise ValueError(f"Angel One returned no spot quote for {underlying}")
            raw_timestamp = quote.get("exchFeedTime")
            if raw_timestamp is None and allow_stale_quotes:
                age = float(self.max_quote_age_seconds + 1)
            else:
                try:
                    quote_timestamp = _angel_timestamp(raw_timestamp)
                    age = (now - quote_timestamp).total_seconds()
                except (TypeError, ValueError):
                    if not allow_stale_quotes:
                        raise ValueError(f"Angel One spot quote for {underlying} has an invalid timestamp")
                    age = float(self.max_quote_age_seconds + 1)
            quote_ages[underlying] = age
            if (age < 0 or age > self.max_quote_age_seconds) and not allow_stale_quotes:
                raise ValueError(f"Angel One spot quote for {underlying} is stale")
            spot_prices[underlying] = float(quote["ltp"])

        normalized_options = []
        original_by_token: dict[str, dict] = {}
        for row in self.instrument_master():
            segment = str(row.get("exch_seg", "")).lower()
            instrument_type = str(row.get("instrumenttype", "")).upper()
            symbol = str(row.get("symbol", ""))
            option_type = "CE" if symbol.endswith("CE") else "PE" if symbol.endswith("PE") else ""
            underlying = str(row.get("name", "")).upper()
            if segment not in {"nse_fo", "nfo"} or instrument_type != "OPTIDX" or option_type not in {"CE", "PE"}:
                continue
            try:
                strike = float(row["strike"]) / 100
                token = str(row["token"])
                expiry = _angel_expiry(row["expiry"])
            except (KeyError, TypeError, ValueError):
                continue
            original_by_token[token] = row
            normalized_options.append({
                "name": underlying,
                "instrument_type": option_type,
                "segment": "NFO-OPT",
                "expiry": expiry,
                "strike": strike,
                "instrument_token": token,
                "exchange": "NFO",
                "tradingsymbol": symbol,
            })
        selected = select_nearest_option_contracts(
            normalized_options,
            spot_prices=spot_prices,
            underlyings=requested,
            strikes_each_side=strikes_each_side,
            as_of=as_of,
        )
        if not selected:
            raise ValueError("Angel One instrument master has no matching listed options")
        self.spot_reference_stale = any(
            age < 0 or age > self.max_quote_age_seconds
            for age in quote_ages.values()
        )
        self.spot_reference_age_seconds = quote_ages
        watchlist = [
            {
                "instrument_token": INDEX_TOKENS[name],
                "exchange": "NSE",
                "tradingsymbol": "NIFTY 50" if name == "NIFTY" else "NIFTY BANK",
                "name": name,
                "instrument_type": "INDEX",
            }
            for name in sorted(requested)
        ]
        for contract in selected:
            raw = original_by_token[str(contract.instrument_token)]
            watchlist.append({
                "instrument_token": int(raw["token"]),
                "exchange": "NFO",
                "tradingsymbol": raw["symbol"],
                "name": contract.underlying,
                "instrument_type": contract.option_type,
                "expiry": contract.expiry.isoformat(),
                "strike": contract.strike,
            })
        return watchlist, spot_prices

    def historical_candles(
        self,
        *,
        instrument_token: int,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: date,
        to_date: date,
        include_open_interest: bool = False,
    ) -> list[dict]:
        self._require_session()
        intervals = {
            "minute": "ONE_MINUTE", "3minute": "THREE_MINUTE", "5minute": "FIVE_MINUTE",
            "10minute": "TEN_MINUTE", "15minute": "FIFTEEN_MINUTE", "30minute": "THIRTY_MINUTE",
            "60minute": "ONE_HOUR", "day": "ONE_DAY",
        }
        if interval not in intervals:
            raise ValueError("Unsupported Angel One historical interval")
        exchange = exchange.upper()
        if exchange not in {"NSE", "NFO", "BSE"}:
            raise ValueError("Unsupported exchange for Angel One historical data")
        params = {
            "exchange": exchange,
            "symboltoken": str(instrument_token),
            "interval": intervals[interval],
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_date.strftime("%Y-%m-%d 15:30"),
        }
        response = self._client.getCandleData(params)
        if not response or response.get("status") is not True:
            raise RuntimeError("Angel One historical candle request failed")
        oi_by_time = {}
        if include_open_interest and exchange == "NFO":
            oi_response = self._client.getOIData(params)
            if oi_response and oi_response.get("status") is True:
                oi_by_time = {
                    _angel_timestamp(item["time"]).isoformat(): float(item["oi"])
                    for item in oi_response.get("data", [])
                }
        rows = []
        for row in response.get("data", []):
            timestamp = _angel_timestamp(row[0]).isoformat()
            rows.append({
                "symbol": symbol.upper(),
                "timestamp": timestamp,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "open_interest": oi_by_time.get(timestamp),
                "source": "angelone_smartapi",
            })
        return rows

    def normalize_tick(self, message: dict, instrument: dict) -> AngelOneTick:
        token = int(message["token"])
        market_time = datetime.fromtimestamp(int(message["exchange_timestamp"]) / 1000, tz=timezone.utc)
        price = float(message["last_traded_price"]) / 100
        if not isfinite(price) or price <= 0:
            raise ValueError("SmartAPI tick has an invalid last traded price")
        timestamp = market_time.isoformat()
        sequence = message.get("sequence_number")
        event_id = f"angelone:{token}:{sequence}:{timestamp}:{price}"
        day_ohlc = {
            key: float(message[source]) / 100
            for key, source in {
                "open": "open_price_of_the_day",
                "high": "high_price_of_the_day",
                "low": "low_price_of_the_day",
                "close": "closed_price",
            }.items()
            if message.get(source) is not None
        }
        bids = _normalize_depth(message.get("best_5_buy_data", []))
        asks = _normalize_depth(message.get("best_5_sell_data", []))
        return AngelOneTick(
            event_id=event_id,
            timestamp=timestamp,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=str(uuid4()),
            source="angelone_smartapi",
            instrument_token=token,
            exchange=str(instrument["exchange"]),
            symbol=str(instrument["tradingsymbol"]),
            last_price=price,
            last_trade_quantity=int(message["last_traded_quantity"]) if message.get("last_traded_quantity") is not None else None,
            volume=int(message["volume_trade_for_the_day"]) if message.get("volume_trade_for_the_day") is not None else None,
            open_interest=int(message["open_interest"]) if message.get("open_interest") is not None else None,
            day_ohlc=day_ohlc,
            bids=bids,
            asks=asks,
            sequence=int(sequence) if sequence is not None else None,
        )

    def start_stream(
        self,
        instruments: list[dict],
        *,
        on_tick: Callable[[AngelOneTick], None],
        on_status: Callable[[str, dict], None] | None = None,
    ):
        session = self._require_session()
        if self._websocket_thread is not None and self._websocket_thread.is_alive():
            raise RuntimeError("Angel One market stream is already running")
        if self._websocket_factory is None:
            raise RuntimeError("Angel One WebSocket SDK is not installed")
        if not instruments:
            raise ValueError("At least one instrument must be selected")
        if len(instruments) > 1000:
            raise ValueError("SmartAPI allows at most 1000 subscriptions in one WebSocket session")
        token_map = {int(item["instrument_token"]): item for item in instruments}
        token_groups: dict[int, list[str]] = {}
        for item in instruments:
            exchange_type = 1 if item["exchange"] == "NSE" else 2 if item["exchange"] == "NFO" else 3
            token_groups.setdefault(exchange_type, []).append(str(item["instrument_token"]))
        token_list = [{"exchangeType": key, "tokens": values} for key, values in token_groups.items()]
        stop_event = threading.Event()
        self._stream_stop = stop_event
        self._last_tick_monotonic = None

        def report(event_type: str, payload: dict | None = None) -> None:
            if on_status:
                on_status(event_type, payload or {})

        def make_socket():
            socket = self._websocket_factory(
                f"Bearer {session.auth_token}", self._api_key, session.client_code, session.feed_token
            )
            mode = socket.SNAP_QUOTE

            def on_open(wsapp) -> None:
                self._stream_connected.set()
                self._last_tick_monotonic = None
                self._last_pong_monotonic = time.monotonic()
                socket.subscribe("northstar1", mode, token_list)
                report("market.connected", {"subscriptions": len(instruments), "mode": "snap_quote"})

            def on_data(_wsapp, message) -> None:
                try:
                    token = int(message["token"])
                    metadata = token_map.get(token)
                    if metadata is None:
                        report("market.unknown_instrument", {"instrument_token": token})
                        return
                    tick = self.normalize_tick(message, metadata)
                    tick_age = (datetime.now(timezone.utc) - _angel_timestamp(tick.timestamp)).total_seconds()
                    if 0 <= tick_age <= self.stale_tick_seconds:
                        self._last_tick_monotonic = time.monotonic()
                    on_tick(tick)
                except (KeyError, TypeError, ValueError, OverflowError):
                    report("market.invalid_tick")
                except Exception as error:
                    report("market.tick_handler_error", {"code": type(error).__name__})

            def on_error(*_args) -> None:
                self._stream_connected.clear()
                report("market.connection_error")

            def on_close(*_args) -> None:
                self._stream_connected.clear()
                report("market.connection_closed")

            def on_pong(*_args) -> None:
                self._last_pong_monotonic = time.monotonic()

            socket.on_open = on_open
            socket.on_data = on_data
            socket.on_error = on_error
            socket.on_close = on_close
            socket.on_pong = on_pong
            return socket

        initial_socket = make_socket()
        self.websocket = initial_socket

        def supervise_websocket() -> None:
            pending_socket = initial_socket
            failures = 0
            while not stop_event.is_set():
                try:
                    socket = pending_socket or make_socket()
                except Exception as error:
                    failures += 1
                    report("market.connection_error", {"code": type(error).__name__})
                    if failures >= self.max_reconnect_attempts:
                        report("market.reconnect_exhausted", {"attempts": failures})
                        break
                    delay = min(30.0, self.reconnect_backoff_seconds * (2 ** (failures - 1)))
                    report("market.reconnecting", {"attempts": failures, "delay_seconds": delay})
                    if stop_event.wait(delay):
                        break
                    continue
                pending_socket = None
                self.websocket = socket
                started_at = time.monotonic()
                try:
                    socket.connect()
                except Exception as error:
                    self._stream_connected.clear()
                    report("market.connection_error", {"code": type(error).__name__})
                finally:
                    self._stream_connected.clear()
                if stop_event.is_set():
                    break
                if time.monotonic() - started_at >= 30:
                    failures = 0
                failures += 1
                if failures >= self.max_reconnect_attempts:
                    report("market.reconnect_exhausted", {"attempts": failures})
                    break
                delay = min(30.0, self.reconnect_backoff_seconds * (2 ** (failures - 1)))
                report("market.reconnecting", {"attempts": failures, "delay_seconds": delay})
                if stop_event.wait(delay):
                    break

        def supervise_rest_fallback() -> None:
            while not stop_event.wait(self.fallback_interval_seconds):
                last_pong = self._last_pong_monotonic
                if (
                    self._stream_connected.is_set()
                    and last_pong is not None
                    and time.monotonic() - last_pong > 45
                ):
                    report("market.heartbeat_timeout")
                    if self.websocket is not None:
                        self.websocket.close_connection()
                    self._stream_connected.clear()
                last_tick = self._last_tick_monotonic
                stale = last_tick is None or time.monotonic() - last_tick >= self.stale_tick_seconds
                if self._stream_connected.is_set() and not stale:
                    continue
                try:
                    count = self._poll_rest_fallback(instruments, on_tick=on_tick, on_status=report)
                    report("market.rest_fallback_complete", {"quotes": count})
                except Exception as error:
                    report("market.rest_fallback_error", {"code": type(error).__name__})

        self._websocket_thread = threading.Thread(
            target=supervise_websocket, daemon=True, name="angelone-market-feed"
        )
        self._fallback_thread = threading.Thread(
            target=supervise_rest_fallback, daemon=True, name="angelone-market-rest-fallback"
        )
        self._websocket_thread.start()
        self._fallback_thread.start()
        return initial_socket

    def _poll_rest_fallback(
        self,
        instruments: list[dict],
        *,
        on_tick: Callable[[AngelOneTick], None],
        on_status: Callable[[str, dict], None] | None = None,
    ) -> int:
        self._require_session()
        batches = [instruments[index:index + 50] for index in range(0, len(instruments), 50)]
        accepted = 0
        for batch_index, batch in enumerate(batches):
            if self._stream_stop is not None and self._stream_stop.is_set():
                break
            exchange_tokens: dict[str, list[str]] = {}
            token_map = {}
            for instrument in batch:
                exchange = str(instrument["exchange"]).upper()
                token = str(instrument["instrument_token"])
                exchange_tokens.setdefault(exchange, []).append(token)
                token_map[token] = instrument
            response = self._client.getMarketData("FULL", exchange_tokens)
            data = response.get("data", {}) if isinstance(response, dict) else {}
            if (
                not isinstance(response, dict)
                or response.get("status") is not True
                or not isinstance(data, dict)
                or not isinstance(data.get("fetched"), list)
            ):
                raise RuntimeError("Angel One quote fallback response is invalid")
            for quote in data["fetched"]:
                try:
                    token = str(quote["symbolToken"])
                    instrument = token_map.get(token)
                    if instrument is None:
                        continue
                    tick = self.normalize_rest_quote(quote, instrument)
                    on_tick(tick)
                    self._last_tick_monotonic = time.monotonic()
                    accepted += 1
                except (KeyError, TypeError, ValueError, OverflowError):
                    if on_status:
                        on_status("market.invalid_rest_quote", {})
            if batch_index + 1 < len(batches):
                if self._stream_stop is not None and self._stream_stop.wait(1.0):
                    break
                if self._stream_stop is None:
                    time.sleep(1.0)
        return accepted

    def normalize_rest_quote(self, quote: dict, instrument: dict) -> AngelOneTick:
        token = int(quote["symbolToken"])
        timestamp = _angel_timestamp(quote.get("exchFeedTime") or quote.get("exchangeTimestamp"))
        quote_age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if quote_age < 0 or quote_age > self.max_quote_age_seconds:
            raise ValueError("Angel One REST quote is stale")
        last_price = _portfolio_decimal(quote, "ltp")
        if last_price is None or last_price <= 0:
            raise ValueError("Angel One REST quote has an invalid last price")
        timestamp_text = timestamp.isoformat()
        event_id = f"angelone-rest:{token}:{timestamp_text}:{last_price}"
        day_ohlc = {}
        for field in ("open", "high", "low", "close"):
            value = _portfolio_decimal(quote, field)
            if value is not None and value > 0:
                day_ohlc[field] = float(value)
        volume_value = _portfolio_decimal(quote, "tradeVolume", "volume")
        volume = int(volume_value) if volume_value is not None and volume_value >= 0 else None
        return AngelOneTick(
            event_id=event_id,
            timestamp=timestamp_text,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=str(uuid4()),
            source="angelone_smartapi",
            instrument_token=token,
            exchange=str(instrument["exchange"]),
            symbol=str(instrument["tradingsymbol"]),
            last_price=float(last_price),
            last_trade_quantity=None,
            volume=volume,
            open_interest=int(_portfolio_decimal(quote, "opnInterest", "openInterest") or 0) or None,
            day_ohlc=day_ohlc,
            bids=(),
            asks=(),
            sequence=None,
        )


def _angel_timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        stamp = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(stamp, timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.strptime(value, "%d-%b-%Y %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    else:
        raise ValueError("Unsupported Angel One timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    return parsed.astimezone(timezone.utc)


def _portfolio_response_rows(response: dict, collection: str) -> list[dict]:
    if not isinstance(response, dict) or response.get("status") is not True:
        raise RuntimeError("Angel One portfolio response is invalid")
    data = response.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = data.get(collection)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        if collection == "net" and isinstance(data.get("positions"), list):
            return [row for row in data["positions"] if isinstance(row, dict)]
    return []


def _portfolio_decimal(row: dict, *keys: str) -> Decimal | None:
    for key in keys:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = Decimal(str(raw).replace(" ", "").replace(",", ""))
        except (InvalidOperation, ValueError):
            return None
        return value if value.is_finite() else None
    return None


def _normalize_broker_position(row: dict, *, source: str) -> dict | None:
    symbol = str(row.get("tradingsymbol") or row.get("symbol") or "").strip().upper()
    exchange = str(row.get("exchange") or "").strip().upper()
    instrument_token = str(row.get("symboltoken") or row.get("instrument_token") or "").strip()
    if not symbol or exchange not in {"NSE", "BSE", "NFO", "BFO"} or not instrument_token:
        return None
    if source == "HOLDING":
        quantity_value = _portfolio_decimal(row, "quantity")
        average_price = _portfolio_decimal(row, "averageprice", "average_price")
    else:
        quantity_value = _portfolio_decimal(row, "netqty", "netQuantity", "net_quantity")
        net_quantity_fields = ("netqty", "netQuantity", "net_quantity")
        if quantity_value is None and not any(row.get(key) not in (None, "") for key in net_quantity_fields):
            buy_quantity = _portfolio_decimal(row, "buyqty") or Decimal("0")
            carry_buy = _portfolio_decimal(row, "cfbuyqty") or Decimal("0")
            sell_quantity = _portfolio_decimal(row, "sellqty") or Decimal("0")
            carry_sell = _portfolio_decimal(row, "cfsellqty") or Decimal("0")
            quantity_value = buy_quantity + carry_buy - sell_quantity - carry_sell
        average_price = _portfolio_decimal(row, "avgnetprice", "averageprice", "buyavgprice")
    if quantity_value is None or quantity_value != quantity_value.to_integral_value():
        return None
    quantity = int(quantity_value)
    if quantity != 0 and (average_price is None or average_price <= 0):
        return None
    if average_price is not None:
        average_price = abs(average_price)
    last_price = _portfolio_decimal(row, "ltp", "lastprice", "last_price")
    reported_pnl = _portfolio_decimal(row, "profitandloss", "pnl")
    return {
        "source": source,
        "symbol": symbol,
        "exchange": exchange,
        "instrument_token": instrument_token,
        "product_type": str(row.get("producttype") or row.get("product") or "UNKNOWN").upper(),
        "quantity": quantity,
        "average_entry_price": str(average_price) if average_price is not None else None,
        "last_price": str(last_price) if last_price is not None else None,
        "market_value": str(last_price * quantity) if last_price is not None else None,
        "reported_pnl": str(reported_pnl) if reported_pnl is not None else None,
    }


def normalize_angelone_instrument_master(
    rows: list[dict],
    *,
    price_scale: Decimal,
    as_of: date | None = None,
) -> tuple[IndiaMarketPolicy, dict[str, int]]:
    """Convert supported Angel One master rows into India-domain reference records."""
    if not price_scale.is_finite() or price_scale <= 0:
        raise ValueError("Angel One price scale must be a finite positive decimal")
    effective_date = as_of or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    segments = {
        "NSE": ("NSE", "NSE-CM"),
        "BSE": ("BSE", "BSE-CM"),
        "NFO": ("NSE", "NFO"),
        "NSE_FO": ("NSE", "NFO"),
        "BFO": ("BSE", "BFO"),
        "BSE_FO": ("BSE", "BFO"),
    }
    exchange_names = {"NSE": "National Stock Exchange of India", "BSE": "BSE Limited"}
    asset_types = {
        "OPTIDX": AssetType.OPTION,
        "OPTSTK": AssetType.OPTION,
        "FUTIDX": AssetType.FUTURE,
        "FUTSTK": AssetType.FUTURE,
    }
    rejected: dict[str, int] = {}
    exchanges: dict[str, Exchange] = {}
    instruments: dict[tuple[str, str], Instrument] = {}
    underlying_aliases: dict[str, Instrument] = {}
    derivative_rows: list[tuple[dict, str, str]] = []

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    def register_exchange(code: str) -> None:
        exchanges.setdefault(code, Exchange(code, exchange_names[code], "IN", "Asia/Kolkata"))

    def register_instrument(instrument: Instrument) -> None:
        key = (instrument.exchange_code, instrument.symbol)
        instruments.setdefault(key, instrument)
        for alias in (instrument.symbol, instrument.name):
            normalized_alias = alias.strip().upper()
            if normalized_alias:
                underlying_aliases.setdefault(normalized_alias, instruments[key])
        if instrument.symbol.endswith("-EQ"):
            underlying_aliases.setdefault(instrument.symbol[:-3], instruments[key])

    for row in rows:
        if not isinstance(row, dict):
            reject("invalid_master_row")
            continue
        provider_segment = str(row.get("exch_seg", "")).strip().upper()
        mapped = segments.get(provider_segment)
        if mapped is None:
            reject("unsupported_exchange_segment")
            continue
        exchange_code, canonical_segment = mapped
        raw_type = str(row.get("instrumenttype", "")).strip().upper()
        symbol = str(row.get("symbol", "")).strip().upper()
        name = str(row.get("name") or symbol).strip().upper()
        raw_token = row.get("token")
        token = str(raw_token).strip() if raw_token is not None else ""
        if not symbol or not token:
            reject("missing_symbol_or_token")
            continue

        if provider_segment in {"NSE", "BSE"}:
            if raw_type in {"", "EQUITY"}:
                asset_type = AssetType.EQUITY
            elif raw_type in {"AMXIDX", "INDEX"}:
                asset_type = AssetType.INDEX
            else:
                reject("unsupported_cash_instrument_type")
                continue
            register_exchange(exchange_code)
            register_instrument(Instrument(
                exchange_code=exchange_code,
                symbol=symbol,
                asset_type=asset_type,
                name=name,
                segment=canonical_segment,
                instrument_token=token,
            ))
            continue

        if raw_type not in asset_types:
            reject("unsupported_derivative_instrument_type")
            continue
        register_exchange(exchange_code)
        derivative_rows.append((row, exchange_code, canonical_segment))

    contracts: list[Contract] = []
    expiry_groups: dict[tuple[str, str, AssetType], set[date]] = {}
    contract_keys: set[tuple[str, str, date]] = set()
    for row, exchange_code, canonical_segment in derivative_rows:
        symbol = str(row["symbol"]).strip().upper()
        name = str(row.get("name") or "").strip().upper()
        underlying = underlying_aliases.get(name)
        if underlying is None and name in INDEX_TOKENS:
            underlying = Instrument(
                exchange_code="NSE",
                symbol=name,
                asset_type=AssetType.INDEX,
                name=name,
                segment="INDEX",
                instrument_token=INDEX_TOKENS[name],
            )
            register_exchange("NSE")
            register_instrument(underlying)
            underlying = underlying_aliases[name]
        if underlying is None:
            reject("unresolved_derivative_underlying")
            continue
        try:
            expiry = _angel_expiry(row["expiry"])
            raw_lot_size = row["lotsize"]
            if isinstance(raw_lot_size, bool):
                raise ValueError("Lot size must be a positive integer")
            lot_size_value = Decimal(str(raw_lot_size))
            if (
                not lot_size_value.is_finite()
                or lot_size_value <= 0
                or lot_size_value != lot_size_value.to_integral_value()
            ):
                raise ValueError("Lot size must be a positive integer")
            lot_size = int(lot_size_value)
            raw_tick = row.get("tick_size", row.get("ticksize"))
            tick_size = Decimal(str(raw_tick)) / price_scale
            asset_type = asset_types[str(row["instrumenttype"]).strip().upper()]
            strike = None
            option_type = None
            if asset_type == AssetType.OPTION:
                strike = Decimal(str(row["strike"])) / price_scale
                option_type = OptionType.CALL if symbol.endswith("CE") else OptionType.PUT if symbol.endswith("PE") else None
                if option_type is None:
                    raise ValueError("Option side is missing")
            contract_instrument = Instrument(
                exchange_code=exchange_code,
                symbol=symbol,
                asset_type=asset_type,
                name=name,
                segment=canonical_segment,
                instrument_token=str(row["token"]),
            )
            contract = Contract(
                instrument=contract_instrument,
                underlying=underlying,
                expiry=expiry,
                lot_size=LotSize(lot_size, effective_date),
                tick_size=TickSize(tick_size, effective_date),
                strike=strike,
                option_type=option_type,
                product_type=str(row["instrumenttype"]).strip().upper(),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
            reject("invalid_derivative_reference_fields")
            continue
        key = (exchange_code, symbol, expiry)
        if key in contract_keys:
            reject("duplicate_derivative_contract")
            continue
        contract_keys.add(key)
        contracts.append(contract)
        expiry_groups.setdefault((exchange_code, underlying.symbol, asset_type), set()).add(expiry)

    expiry_calendars = tuple(
        ExpiryCalendar(exchange_code, underlying_symbol, asset_type, tuple(sorted(expiries)))
        for (exchange_code, underlying_symbol, asset_type), expiries in sorted(
            expiry_groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
        )
    )
    policy = IndiaMarketPolicy(
        exchanges=tuple(exchanges[code] for code in sorted(exchanges)),
        instruments=tuple(instruments[key] for key in sorted(instruments)),
        contracts=tuple(contracts),
        expiry_calendars=expiry_calendars,
    )
    return policy, rejected


def _angel_expiry(value) -> date:
    if isinstance(value, date):
        return value
    for pattern in ("%d%b%Y", "%d%b%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value).upper(), pattern).date()
        except ValueError:
            continue
    raise ValueError("Invalid Angel One option expiry")


def _normalize_depth(rows: list[dict]) -> tuple[dict, ...]:
    levels = []
    for row in rows[:5]:
        raw_price = row.get("price", 0)
        raw_quantity = row.get("quantity", 0)
        price = float(raw_price) / 100
        quantity = int(raw_quantity)
        orders = int(row.get("no of orders", row.get("num_of_orders", 0)))
        if isfinite(price) and price > 0 and quantity >= 0 and orders >= 0:
            levels.append({"price": price, "quantity": quantity, "orders": orders})
    return tuple(levels)


def _load_websocket_factory():
    package_spec = importlib.util.find_spec("SmartApi")
    if package_spec is None or package_spec.origin is None:
        raise ImportError("SmartAPI SDK is not installed")
    module_path = Path(package_spec.origin).with_name("smartWebSocketV2.py")
    spec = importlib.util.spec_from_file_location("northstar_smartwebsocketv2", module_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError("SmartAPI WebSocket V2 module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _verified_websocket_class(module.SmartWebSocketV2, module.websocket)


def _verified_websocket_class(base_class, websocket_module):
    class VerifiedSmartWebSocketV2(base_class):
        def _on_pong(self, wsapp, data):
            super()._on_pong(wsapp, data)
            callback = getattr(self, "on_pong", None)
            if callback is not None:
                callback(wsapp, data)

        def connect(self):
            headers = {
                "Authorization": self.auth_token,
                "x-api-key": self.api_key,
                "x-client-code": self.client_code,
                "x-feed-token": self.feed_token,
            }
            self.wsapp = websocket_module.WebSocketApp(
                self.ROOT_URI,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_data=self._on_data,
                on_ping=self._on_ping,
                on_pong=self._on_pong,
            )
            self.wsapp.run_forever(ping_interval=30, ping_timeout=10)

    return VerifiedSmartWebSocketV2