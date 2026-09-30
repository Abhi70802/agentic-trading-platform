"""Provider-neutral market event normalization and in-process quality gates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Protocol
from uuid import uuid4

from core import Candle


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    timestamp: str
    correlation_id: str
    source: str
    version: int
    schema_version: str
    sequence: int | None
    candle: Candle

    def as_event(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": "market.candle_received",
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "version": self.version,
            "schema_version": self.schema_version,
            "payload": {"sequence": self.sequence, "candle": self.candle.__dict__},
        }


class MarketDataAdapter(Protocol):
    """A broker/provider adapter converts its messages to normalized candle events."""

    name: str

    def normalize(self, message: dict) -> MarketEvent: ...


class JsonCandleAdapter:
    """Adapter for normalized JSON or common provider key aliases."""

    def __init__(self, name: str):
        self.name = name

    def normalize(self, message: dict) -> MarketEvent:
        symbol = str(message.get("symbol", message.get("ticker", ""))).strip().upper()
        timestamp = normalize_timestamp(message.get("timestamp", message.get("t")))
        price_fields = {
            "open": message.get("open", message.get("o")),
            "high": message.get("high", message.get("h")),
            "low": message.get("low", message.get("l")),
            "close": message.get("close", message.get("c")),
            "volume": message.get("volume", message.get("v", 0)),
        }
        prices = {key: float(value) for key, value in price_fields.items()}
        if not all(isfinite(value) for value in prices.values()) or prices["volume"] < 0:
            raise ValueError("Candle prices and volume must be finite; volume cannot be negative")
        candle = Candle(symbol, timestamp, **prices)
        raw_sequence = message.get("sequence", message.get("seq"))
        sequence = int(raw_sequence) if raw_sequence is not None else None
        event_id = str(message.get("event_id") or f"{self.name}:{symbol}:{timestamp}:{sequence or 0}")
        return MarketEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=str(message.get("correlation_id") or uuid4()),
            source=self.name,
            version=1,
            schema_version="1.0",
            sequence=sequence,
            candle=candle,
        )


def normalize_timestamp(value: object) -> str:
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
    else:
        raise ValueError("A provider timestamp is required")
    return parsed.isoformat()


class MarketDataQualityGate:
    """Reject duplicate and out-of-order events; tracks bounded IDs per process."""

    def __init__(self, max_seen: int = 50_000):
        self.max_seen = max_seen
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._latest_time: dict[tuple[str, str], str] = {}
        self._latest_sequence: dict[tuple[str, str], int] = {}
        self.accepted = 0
        self.duplicates = 0
        self.out_of_order = 0
        self.invalid = 0
        self.stale = 0

    def review(self, event: MarketEvent) -> str | None:
        key = (event.source, event.candle.symbol)
        event_key = f"{event.source}:{event.event_id}"
        if event_key in self._seen:
            self.duplicates += 1
            return "duplicate_event"
        latest_time = self._latest_time.get(key)
        if latest_time and event.candle.timestamp <= latest_time:
            self.out_of_order += 1
            return "out_of_order_timestamp"
        if event.sequence is not None:
            latest_sequence = self._latest_sequence.get(key)
            if latest_sequence is not None and event.sequence <= latest_sequence:
                self.out_of_order += 1
                return "out_of_order_sequence"
        self._seen[event_key] = None
        if len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)
        self._latest_time[key] = event.candle.timestamp
        if event.sequence is not None:
            self._latest_sequence[key] = event.sequence
        self.accepted += 1
        return None

    @staticmethod
    def is_fresh(timestamp: str, *, now: datetime | None = None, max_age_seconds: int = 30) -> bool:
        observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        age = ((now or datetime.now(timezone.utc)) - observed.astimezone(timezone.utc)).total_seconds()
        return 0 <= age <= max_age_seconds

    def status(self) -> dict:
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "invalid": self.invalid,
            "stale": self.stale,
        }


class MarketTickQualityGate:
    """Tick-specific checks; equal exchange timestamps are allowed when IDs differ."""

    def __init__(self, *, max_seen: int = 100_000, max_age_seconds: int = 30):
        self.max_seen = max_seen
        self.max_age_seconds = max_age_seconds
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._latest_timestamp: dict[tuple[str, str], str] = {}
        self._last_rejection_audit: dict[tuple[str, str], datetime] = {}
        self.accepted = 0
        self.duplicates = 0
        self.out_of_order = 0
        self.stale = 0

    def review(self, tick, *, now: datetime | None = None) -> str | None:
        if tick.event_id in self._seen:
            self.duplicates += 1
            return "duplicate_tick"
        if not MarketDataQualityGate.is_fresh(tick.timestamp, now=now, max_age_seconds=self.max_age_seconds):
            self.stale += 1
            return "stale_tick"
        key = (tick.source, tick.symbol)
        previous = self._latest_timestamp.get(key)
        if previous and tick.timestamp < previous:
            self.out_of_order += 1
            return "out_of_order_tick"
        self._seen[tick.event_id] = None
        if len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)
        self._latest_timestamp[key] = max(previous or tick.timestamp, tick.timestamp)
        self.accepted += 1
        return None

    def status(self) -> dict:
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "stale": self.stale,
        }

    def should_audit_rejection(
        self,
        symbol: str,
        reason: str,
        *,
        now: datetime | None = None,
        interval_seconds: int = 60,
    ) -> bool:
        current = now or datetime.now(timezone.utc)
        key = (symbol, reason)
        previous = self._last_rejection_audit.get(key)
        if previous is not None and (current - previous).total_seconds() < interval_seconds:
            return False
        self._last_rejection_audit[key] = current
        return True


class MinuteCandleAggregator:
    """Builds provisional one-minute OHLCV bars from ordered accepted Kite ticks."""

    def __init__(self):
        self._bars: dict[str, Candle] = {}

    def add(self, tick) -> Candle:
        tick_time = datetime.fromisoformat(tick.timestamp.replace("Z", "+00:00"))
        minute = tick_time.replace(second=0, microsecond=0)
        timestamp = minute.isoformat()
        previous = self._bars.get(tick.symbol)
        if previous is not None and timestamp < previous.timestamp:
            raise ValueError("Cannot aggregate an out-of-order tick")
        price = float(tick.last_price)
        trade_quantity = max(0, int(tick.last_trade_quantity or 0))
        if previous is None or timestamp > previous.timestamp:
            candle = Candle(tick.symbol, timestamp, price, price, price, price, float(trade_quantity))
        else:
            candle = Candle(
                tick.symbol,
                timestamp,
                previous.open,
                max(previous.high, price),
                min(previous.low, price),
                price,
                previous.volume + trade_quantity,
            )
        self._bars[tick.symbol] = candle
        return candle