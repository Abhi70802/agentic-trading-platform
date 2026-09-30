"""Read-only Kite Connect market-data adapter; order methods are deliberately absent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from typing import Callable, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo


MAX_KITE_SUBSCRIPTIONS = 3000
SUPPORTED_EXCHANGES = ("NSE", "BSE", "NFO")
DEFAULT_OPTION_UNDERLYINGS = ("NIFTY", "BANKNIFTY")


@dataclass(frozen=True, repr=False)
class KiteCredentials:
    api_key: str
    access_token: str

    @classmethod
    def from_environment(cls) -> KiteCredentials | None:
        api_key = os.getenv("KITE_API_KEY", "").strip()
        access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
        return cls(api_key, access_token) if api_key and access_token else None

    def __repr__(self) -> str:
        return "KiteCredentials(api_key=<redacted>, access_token=<redacted>)"


@dataclass(frozen=True)
class OptionContract:
    instrument_token: int
    exchange: str
    trading_symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: str


@dataclass(frozen=True)
class NormalizedTick:
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
    sequence: None = None

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
            },
        }


def select_nearest_option_contracts(
    instruments: Iterable[dict],
    *,
    spot_prices: dict[str, float],
    underlyings: Iterable[str] = DEFAULT_OPTION_UNDERLYINGS,
    strikes_each_side: int = 5,
    as_of: date | None = None,
) -> list[OptionContract]:
    """Select CE/PE contracts at the nearest expiry and ATM ± N listed strikes."""
    if strikes_each_side < 0:
        raise ValueError("strikes_each_side cannot be negative")
    today = as_of or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    requested = {name.strip().upper() for name in underlyings}
    prices = {name.strip().upper(): float(value) for name, value in spot_prices.items()}
    if not requested or not requested <= prices.keys():
        raise ValueError("A positive spot price is required for every option underlying")
    if any(not isfinite(value) or value <= 0 for value in prices.values()):
        raise ValueError("Spot prices must be finite and positive")

    candidates: dict[str, list[dict]] = {name: [] for name in requested}
    for item in instruments:
        underlying = str(item.get("name", "")).upper()
        option_type = str(item.get("instrument_type", "")).upper()
        segment = str(item.get("segment", "")).upper()
        expiry = _as_date(item.get("expiry"))
        try:
            strike = float(item.get("strike", 0))
        except (TypeError, ValueError):
            continue
        if (
            underlying in candidates
            and option_type in {"CE", "PE"}
            and segment.endswith("OPT")
            and expiry is not None
            and expiry >= today
            and isfinite(strike)
            and strike > 0
        ):
            candidates[underlying].append({**item, "_expiry": expiry, "_strike": strike})

    selected: list[OptionContract] = []
    for underlying in sorted(requested):
        rows = candidates[underlying]
        if not rows:
            continue
        expiry = min(row["_expiry"] for row in rows)
        expiry_rows = [row for row in rows if row["_expiry"] == expiry]
        strikes = sorted({row["_strike"] for row in expiry_rows})
        atm_index = min(range(len(strikes)), key=lambda index: abs(strikes[index] - prices[underlying]))
        first = max(0, atm_index - strikes_each_side)
        last = min(len(strikes), atm_index + strikes_each_side + 1)
        chosen_strikes = set(strikes[first:last])
        for row in expiry_rows:
            if row["_strike"] not in chosen_strikes:
                continue
            try:
                selected.append(OptionContract(
                    instrument_token=int(row["instrument_token"]),
                    exchange=str(row.get("exchange", "NFO")),
                    trading_symbol=str(row["tradingsymbol"]),
                    underlying=underlying,
                    expiry=expiry,
                    strike=row["_strike"],
                    option_type=str(row["instrument_type"]).upper(),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(selected, key=lambda item: (item.underlying, item.strike, item.option_type))


class KiteMarketDataAdapter:
    """Read-only REST/WebSocket market data; use a separate gated adapter for OMS."""

    def __init__(
        self,
        *,
        api_key: str,
        access_token: str,
        kite_client,
        ticker_factory=None,
        max_subscriptions: int = MAX_KITE_SUBSCRIPTIONS,
    ):
        if not api_key or not access_token:
            raise ValueError("Kite API key and access token are required")
        self._api_key = api_key
        self._access_token = access_token
        self._kite = kite_client
        self._ticker_factory = ticker_factory
        self.max_subscriptions = min(max_subscriptions, MAX_KITE_SUBSCRIPTIONS)

    @classmethod
    def from_environment(cls) -> KiteMarketDataAdapter:
        credentials = KiteCredentials.from_environment()
        if credentials is None:
            raise RuntimeError("Set KITE_API_KEY and KITE_ACCESS_TOKEN in the local environment")
        try:
            from kiteconnect import KiteConnect, KiteTicker
        except ImportError as error:
            raise RuntimeError("Install the optional Kite SDK from requirements-kite.txt") from error
        client = KiteConnect(api_key=credentials.api_key)
        client.set_access_token(credentials.access_token)
        return cls(
            api_key=credentials.api_key,
            access_token=credentials.access_token,
            kite_client=client,
            ticker_factory=KiteTicker,
        )

    def instrument_master(self, exchanges: Iterable[str] = SUPPORTED_EXCHANGES) -> list[dict]:
        rows = []
        for exchange in exchanges:
            normalized = exchange.upper()
            if normalized not in SUPPORTED_EXCHANGES:
                raise ValueError(f"Unsupported Kite exchange: {normalized}")
            rows.extend(self._kite.instruments(normalized))
        return rows

    def initial_option_watchlist(
        self,
        *,
        underlyings: Iterable[str] = DEFAULT_OPTION_UNDERLYINGS,
        strikes_each_side: int = 5,
        max_quote_age_seconds: int = 30,
        as_of: date | None = None,
    ) -> tuple[list[dict], dict[str, float]]:
        requested = {name.upper() for name in underlyings}
        index_symbols = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}
        if not requested or not requested <= index_symbols.keys():
            raise ValueError("Only the configured NIFTY and BANKNIFTY underlyings are supported")
        index_rows = [
            row for row in self._kite.instruments("NSE")
            if str(row.get("segment", "")).upper() == "INDICES"
            and str(row.get("tradingsymbol", "")).upper() in {index_symbols[name] for name in requested}
        ]
        by_symbol = {str(row.get("tradingsymbol", "")).upper(): row for row in index_rows}
        if any(index_symbols[name] not in by_symbol for name in requested):
            raise ValueError("Kite NSE instrument master is missing a required index")
        quote_keys = {name: f"NSE:{index_symbols[name]}" for name in requested}
        quotes = self._kite.quote(*quote_keys.values())
        now = datetime.now(timezone.utc)
        spot_prices: dict[str, float] = {}
        for underlying, key in quote_keys.items():
            quote = quotes.get(key)
            if not quote:
                raise ValueError(f"Kite returned no spot quote for {underlying}")
            timestamp_value = quote.get("timestamp") or quote.get("exchange_timestamp") or quote.get("last_trade_time")
            if timestamp_value is None:
                raise ValueError(f"Kite spot quote for {underlying} has no market timestamp")
            quote_time = _as_utc(timestamp_value)
            age = (now - quote_time).total_seconds()
            if age < 0 or age > max_quote_age_seconds:
                raise ValueError(f"Kite spot quote for {underlying} is stale")
            spot_prices[underlying] = float(quote["last_price"])

        option_master = self._kite.instruments("NFO")
        contracts = select_nearest_option_contracts(
            option_master,
            spot_prices=spot_prices,
            underlyings=requested,
            strikes_each_side=strikes_each_side,
            as_of=as_of,
        )
        by_token = {int(row["instrument_token"]): row for row in option_master}
        selected = [by_symbol[index_symbols[name]] for name in sorted(requested)]
        selected.extend(by_token[contract.instrument_token] for contract in contracts)
        if not contracts:
            raise ValueError("Kite returned no listed options for the configured underlyings")
        if len(selected) > self.max_subscriptions:
            raise ValueError("Selected option watchlist exceeds the Kite connection limit")
        return selected, spot_prices

    def historical_candles(
        self,
        *,
        instrument_token: int,
        from_date,
        to_date,
        interval: str,
        symbol: str,
        continuous: bool = False,
        include_open_interest: bool = False,
    ) -> list[dict]:
        rows = self._kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            interval,
            continuous=continuous,
            oi=include_open_interest,
        )
        normalized = []
        for row in rows:
            stamp = _as_utc(row["date"])
            normalized.append({
                "symbol": symbol.upper(),
                "timestamp": stamp.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "open_interest": int(row["oi"]) if row.get("oi") is not None else None,
                "source": "kite_connect",
            })
        return normalized

    def normalize_tick(self, tick: dict, instrument: dict, *, correlation_id: str | None = None) -> NormalizedTick:
        token = int(tick["instrument_token"])
        price = float(tick["last_price"])
        if not isfinite(price) or price <= 0:
            raise ValueError("Kite tick must contain a finite positive last price")
        market_time = _as_utc(tick.get("exchange_timestamp") or tick.get("last_trade_time") or datetime.now(timezone.utc))
        ingested_at = datetime.now(timezone.utc).isoformat()
        ohlc = tick.get("ohlc") or {}
        day_ohlc = {
            key: float(ohlc[key])
            for key in ("open", "high", "low", "close")
            if ohlc.get(key) is not None
        }
        depth = tick.get("depth") or {}
        bids = _normalize_depth(depth.get("buy", []))
        asks = _normalize_depth(depth.get("sell", []))
        volume = tick.get("volume_traded")
        open_interest = tick.get("oi")
        event_seed = f"{token}:{market_time.isoformat()}:{price}:{volume}:{open_interest}"
        return NormalizedTick(
            event_id=f"kite:{event_seed}",
            timestamp=market_time.isoformat(),
            ingested_at=ingested_at,
            correlation_id=correlation_id or str(uuid4()),
            source="kite_connect",
            instrument_token=token,
            exchange=str(instrument.get("exchange", "")),
            symbol=str(instrument.get("tradingsymbol", "")),
            last_price=price,
            last_trade_quantity=(
                int(tick["last_traded_quantity"])
                if tick.get("last_traded_quantity") is not None
                else None
            ),
            volume=int(volume) if volume is not None else None,
            open_interest=int(open_interest) if open_interest is not None else None,
            day_ohlc=day_ohlc,
            bids=bids,
            asks=asks,
        )

    def start_stream(
        self,
        instruments: list[dict],
        *,
        on_tick: Callable[[NormalizedTick], None],
        on_status: Callable[[str, dict], None] | None = None,
        full_mode: bool = True,
    ):
        if self._ticker_factory is None:
            raise RuntimeError("KiteTicker is unavailable; construct the adapter with a ticker factory")
        if not instruments:
            raise ValueError("At least one instrument must be selected")
        if len(instruments) > self.max_subscriptions:
            raise ValueError(f"Kite subscription limit is {self.max_subscriptions} instruments per connection")
        token_map = {int(item["instrument_token"]): item for item in instruments}
        ticker = self._ticker_factory(self._api_key, self._access_token, reconnect=True)

        def status(event_type: str, **payload) -> None:
            if on_status:
                on_status(event_type, payload)

        def connected(socket, _response) -> None:
            tokens = list(token_map)
            socket.subscribe(tokens)
            mode = socket.MODE_FULL if full_mode else socket.MODE_QUOTE
            socket.set_mode(mode, tokens)
            status("market.connected", subscriptions=len(tokens), mode="full" if full_mode else "quote")

        def ticks(_socket, tick_rows) -> None:
            for row in tick_rows:
                token = int(row.get("instrument_token", 0))
                metadata = token_map.get(token)
                if metadata is None:
                    status("market.unknown_instrument", instrument_token=token)
                    continue
                try:
                    on_tick(self.normalize_tick(row, metadata))
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    status("market.invalid_tick", instrument_token=token, error=str(error))

        ticker.on_connect = connected
        ticker.on_ticks = ticks
        ticker.on_reconnect = lambda _socket, attempts: status("market.reconnecting", attempts=attempts)
        ticker.on_noreconnect = lambda *_args: status("market.reconnect_exhausted")
        ticker.on_error = lambda _socket, code, message: status("market.connection_error", code=code, message=message)
        ticker.on_close = lambda _socket, code, reason: status("market.connection_closed", code=code, reason=reason)
        ticker.connect(threaded=True)
        return ticker


def _normalize_depth(rows: list[dict]) -> tuple[dict, ...]:
    normalized = []
    for row in rows[:5]:
        price = float(row.get("price", 0))
        quantity = int(row.get("quantity", 0))
        orders = int(row.get("orders", 0))
        if isfinite(price) and price > 0 and quantity >= 0 and orders >= 0:
            normalized.append({"price": price, "quantity": quantity, "orders": orders})
    return tuple(normalized)


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _as_utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Kite timestamps must be datetime or ISO-8601 values")
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    return value.astimezone(timezone.utc)