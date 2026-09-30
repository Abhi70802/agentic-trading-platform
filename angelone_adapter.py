"""Read-only Angel One SmartAPI market-data integration."""

from __future__ import annotations

import os
import re
import importlib.util
import socket
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Callable
from uuid import getnode, uuid4
from zoneinfo import ZoneInfo

from kite_adapter import DEFAULT_OPTION_UNDERLYINGS, select_nearest_option_contracts


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
        self.spot_reference_stale = False
        self.spot_reference_age_seconds: dict[str, float] = {}

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
        if self.websocket is not None:
            self.websocket.close_connection()
            self.websocket = None
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
        socket = self._websocket_factory(f"Bearer {session.auth_token}", self._api_key, session.client_code, session.feed_token)
        mode = socket.SNAP_QUOTE

        def on_open(wsapp) -> None:
            socket.subscribe("northstar1", mode, token_list)
            if on_status:
                on_status("market.connected", {"subscriptions": len(instruments), "mode": "snap_quote"})

        def on_data(_wsapp, message) -> None:
            try:
                token = int(message["token"])
                metadata = token_map.get(token)
                if metadata is None:
                    if on_status:
                        on_status("market.unknown_instrument", {"instrument_token": token})
                    return
                on_tick(self.normalize_tick(message, metadata))
            except (KeyError, TypeError, ValueError, OverflowError):
                if on_status:
                    on_status("market.invalid_tick", {})

        def on_error(*_args) -> None:
            if on_status:
                on_status("market.connection_error", {})

        def on_close(*_args) -> None:
            if on_status:
                on_status("market.connection_closed", {})

        socket.on_open = on_open
        socket.on_data = on_data
        socket.on_error = on_error
        socket.on_close = on_close
        self.websocket = socket
        self._websocket_thread = threading.Thread(target=socket.connect, daemon=True, name="angelone-market-feed")
        self._websocket_thread.start()
        return socket


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
            self.wsapp.run_forever(ping_interval=self.HEART_BEAT_INTERVAL)

    return VerifiedSmartWebSocketV2