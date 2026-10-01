"""Provider-neutral market event normalization and in-process quality gates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from core import Candle
from india_market import IndiaMarketPolicy


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
        timeframe = str(message.get("timeframe", "legacy"))
        candle = Candle(symbol, timestamp, **prices, timeframe=timeframe)
        raw_sequence = message.get("sequence", message.get("seq"))
        sequence = int(raw_sequence) if raw_sequence is not None else None
        event_id = str(message.get("event_id") or f"{self.name}:{symbol}:{timeframe}:{timestamp}:{sequence or 0}")
        return MarketEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=str(message.get("correlation_id") or uuid4()),
            source=self.name,
            version=1,
            schema_version="1.1",
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


def deduplicate_historical_candles(
    candles: list[Candle],
    *,
    timeframe: str,
    existing: list[Candle] | None = None,
) -> tuple[list[Candle], int]:
    def canonical(candle: Candle) -> Candle:
        parsed = datetime.fromisoformat(candle.timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return Candle(
            candle.symbol.strip().upper(), parsed.astimezone(timezone.utc).isoformat(),
            candle.open, candle.high, candle.low, candle.close,
            candle.volume, candle.open_interest, timeframe,
        )

    seen: dict[tuple[str, str, str], Candle] = {}
    for candle in existing or []:
        normalized = canonical(candle)
        seen[(normalized.symbol, normalized.timeframe, normalized.timestamp)] = normalized

    unique: list[Candle] = []
    duplicate_count = 0
    for candle in candles:
        normalized = canonical(candle)
        key = (normalized.symbol, normalized.timeframe, normalized.timestamp)
        prior = seen.get(key)
        if prior is not None:
            if prior != normalized:
                raise ValueError(f"Historical batch contains conflicting duplicate timestamp {normalized.timestamp}")
            duplicate_count += 1
            continue
        seen[key] = normalized
        unique.append(normalized)
    return unique, duplicate_count


def validate_historical_candles(
    rows: list[dict],
    *,
    symbol: str,
    timeframe: str,
    from_date: date,
    to_date: date,
    existing: list[Candle] | None = None,
    now: datetime | None = None,
) -> tuple[list[Candle], int]:
    """Validate a complete provider batch and return only new, deduplicated candles."""
    if not isinstance(rows, list):
        raise ValueError("Historical provider response must be a list of rows")
    if from_date > to_date:
        raise ValueError("Historical date range is invalid")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Historical validation time must include a timezone")
    current = current.astimezone(timezone.utc)
    india_timezone = ZoneInfo("Asia/Kolkata")
    required_fields = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}

    accepted: list[Candle] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required_fields <= row.keys():
            raise ValueError(f"Historical row {index} is missing required schema fields")
        row_symbol = row["symbol"]
        if not isinstance(row_symbol, str) or row_symbol.strip().upper() != symbol.upper():
            raise ValueError(f"Historical row {index} symbol does not match the requested instrument")
        timestamp_value = row["timestamp"]
        if not isinstance(timestamp_value, str):
            raise ValueError(f"Historical row {index} timestamp must be an ISO-8601 string")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Historical row {index} timestamp is invalid") from error
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            raise ValueError(f"Historical row {index} timestamp must include a timezone")
        parsed_timestamp = parsed_timestamp.astimezone(timezone.utc)
        local_date = parsed_timestamp.astimezone(india_timezone).date()
        if parsed_timestamp > current:
            raise ValueError(f"Historical row {index} timestamp is in the future")
        if not from_date <= local_date <= to_date:
            raise ValueError(f"Historical row {index} timestamp is outside the requested date range")

        numeric_fields = ("open", "high", "low", "close", "volume")
        values = {}
        for field in numeric_fields:
            raw_value = row[field]
            if isinstance(raw_value, bool):
                raise ValueError(f"Historical row {index} {field} must be a finite number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Historical row {index} {field} must be a finite number") from error
            if not isfinite(value):
                raise ValueError(f"Historical row {index} {field} must be a finite number")
            values[field] = value
        raw_open_interest = row.get("open_interest")
        if raw_open_interest is None:
            open_interest = None
        else:
            if isinstance(raw_open_interest, bool):
                raise ValueError(f"Historical row {index} open_interest must be finite and non-negative")
            try:
                open_interest = float(raw_open_interest)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Historical row {index} open_interest must be finite and non-negative") from error
            if not isfinite(open_interest) or open_interest < 0:
                raise ValueError(f"Historical row {index} open_interest must be finite and non-negative")

        candle = Candle(
            row_symbol.strip().upper(), parsed_timestamp.isoformat(),
            values["open"], values["high"], values["low"], values["close"],
            values["volume"], open_interest, timeframe,
        )
        accepted.append(candle)

    return deduplicate_historical_candles(accepted, timeframe=timeframe, existing=existing)


class MarketDataQualityGate:
    """Reject duplicate and out-of-order events; tracks bounded IDs per process."""

    def __init__(self, max_seen: int = 50_000):
        self.max_seen = max_seen
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._latest_time: dict[tuple[str, str, str], str] = {}
        self._latest_sequence: dict[tuple[str, str, str], int] = {}
        self.accepted = 0
        self.duplicates = 0
        self.out_of_order = 0
        self.invalid = 0
        self.stale = 0

    def review(self, event: MarketEvent) -> str | None:
        key = (event.source, event.candle.timeframe, event.candle.symbol)
        event_key = f"{event.source}:{event.candle.timeframe}:{event.event_id}"
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
        self._closed: list[Candle] = []

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
            if previous is not None:
                self._closed.append(previous)
            candle = Candle(
                tick.symbol, timestamp, price, price, price, price,
                float(trade_quantity), timeframe="1m",
            )
        else:
            candle = Candle(
                tick.symbol,
                timestamp,
                previous.open,
                max(previous.high, price),
                min(previous.low, price),
                price,
                previous.volume + trade_quantity,
                timeframe="1m",
            )
        self._bars[tick.symbol] = candle
        return candle

    def take_closed(self) -> list[Candle]:
        closed, self._closed = self._closed, []
        return closed


@dataclass(frozen=True)
class IncompleteCandleBucket:
    symbol: str
    timeframe: str
    timestamp: str
    reasons: tuple[str, ...]
    missing_minutes: tuple[str, ...]


@dataclass(frozen=True)
class CandleResampleResult:
    candles: tuple[Candle, ...]
    incomplete_buckets: tuple[IncompleteCandleBucket, ...]
    missing_sessions: tuple[date, ...] = ()


class SessionCandleResampler:
    """Derive complete bars from ordered 1m candles using explicit sessions and holidays.

    Incomplete buckets are reported but never emitted. Late corrections require rerunning
    the batch with the corrected candles in timestamp order.
    """

    _TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "1d": None}

    def __init__(self, policy: IndiaMarketPolicy, *, exchange_code: str, segment: str):
        exchange = policy.exchange(exchange_code)
        sessions = [
            session for session in policy.sessions
            if session.exchange_code == exchange.code and session.segment == segment
        ]
        if len(sessions) != 1:
            raise ValueError(f"Exactly one trading session is required for {exchange.code}:{segment}")
        self._session = sessions[0]
        self._timezone = ZoneInfo(exchange.timezone_name)
        self._holiday_dates = {
            holiday.session_date
            for holiday in policy.holidays
            if holiday.exchange_code == exchange.code and holiday.segment in (None, segment)
        }
        open_seconds = self._session.opens_at.hour * 3600 + self._session.opens_at.minute * 60 + self._session.opens_at.second
        close_seconds = self._session.closes_at.hour * 3600 + self._session.closes_at.minute * 60 + self._session.closes_at.second
        if self._session.opens_at.second or self._session.closes_at.second or (close_seconds - open_seconds) % 60:
            raise ValueError("Trading session boundaries must align to whole minutes")
        self._duration_minutes = (close_seconds - open_seconds) // 60

    def resample(
        self,
        candles: list[Candle],
        timeframe: str,
        *,
        as_of: datetime | None = None,
    ) -> CandleResampleResult:
        if timeframe not in self._TIMEFRAME_MINUTES:
            raise ValueError("Supported resample timeframes are 5m, 15m, 1h, and 1d")
        cutoff = as_of or datetime.now(timezone.utc)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("Resampling cutoff must include a timezone")
        cutoff = cutoff.astimezone(timezone.utc)
        if not candles:
            return CandleResampleResult((), ())

        symbol = candles[0].symbol
        last_timestamp: datetime | None = None
        candles_by_date: dict[date, dict[str, Candle]] = {}
        for index, candle in enumerate(candles):
            if candle.timeframe != "1m":
                raise ValueError(f"Resample input row {index} must be a 1m candle")
            if candle.symbol != symbol:
                raise ValueError("Resample input must contain exactly one instrument")
            try:
                timestamp = datetime.fromisoformat(candle.timestamp.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Resample input row {index} has an invalid timestamp") from error
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"Resample input row {index} timestamp must include a timezone")
            timestamp = timestamp.astimezone(timezone.utc)
            if last_timestamp is not None and timestamp <= last_timestamp:
                reason = "duplicate" if timestamp == last_timestamp else "late or out-of-order"
                raise ValueError(f"Resample input contains a {reason} timestamp")
            last_timestamp = timestamp
            if timestamp.second or timestamp.microsecond:
                raise ValueError(f"Resample input row {index} timestamp must align to a minute")

            local_timestamp = timestamp.astimezone(self._timezone)
            session_date = local_timestamp.date()
            if session_date in self._holiday_dates:
                raise ValueError(f"Resample input contains a candle on configured holiday {session_date}")
            if session_date.weekday() not in self._session.weekdays:
                raise ValueError(f"Resample input contains a candle outside configured trading weekdays: {session_date}")
            local_time = local_timestamp.time().replace(tzinfo=None)
            if not self._session.opens_at <= local_time < self._session.closes_at:
                raise ValueError(f"Resample input candle is outside the configured session: {candle.timestamp}")
            candles_by_date.setdefault(session_date, {})[timestamp.isoformat()] = candle

        duration_minutes = self._duration_minutes
        timeframe_minutes = self._TIMEFRAME_MINUTES[timeframe] or duration_minutes

        resampled: list[Candle] = []
        incomplete: list[IncompleteCandleBucket] = []
        for session_date in sorted(candles_by_date):
            session_open = datetime.combine(session_date, self._session.opens_at, tzinfo=self._timezone)
            session_candles = candles_by_date[session_date]
            for bucket_offset in range(0, duration_minutes, timeframe_minutes):
                expected_count = min(timeframe_minutes, duration_minutes - bucket_offset)
                bucket_start = session_open + timedelta(minutes=bucket_offset)
                expected = [
                    bucket_start + timedelta(minutes=minute_offset)
                    for minute_offset in range(expected_count)
                ]
                elapsed = [
                    minute for minute in expected
                    if minute.astimezone(timezone.utc) + timedelta(minutes=1) <= cutoff
                ]
                if not elapsed and bucket_start.astimezone(timezone.utc) > cutoff:
                    continue
                present = [
                    session_candles.get(minute.astimezone(timezone.utc).isoformat())
                    for minute in elapsed
                ]
                missing = tuple(
                    minute.astimezone(timezone.utc).isoformat()
                    for minute, candle in zip(elapsed, present)
                    if candle is None
                )
                reasons = []
                if timeframe != "1d" and expected_count < timeframe_minutes:
                    reasons.append("partial_session_bucket")
                if missing:
                    reasons.append("missing_minutes")
                if len(elapsed) < expected_count:
                    reasons.append("in_progress")
                bucket_timestamp = bucket_start.astimezone(timezone.utc).isoformat()
                if reasons:
                    incomplete.append(IncompleteCandleBucket(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=bucket_timestamp,
                        reasons=tuple(reasons),
                        missing_minutes=missing,
                    ))
                    continue

                complete = [candle for candle in present if candle is not None]
                resampled.append(Candle(
                    symbol,
                    bucket_timestamp,
                    complete[0].open,
                    max(candle.high for candle in complete),
                    min(candle.low for candle in complete),
                    complete[-1].close,
                    sum(candle.volume for candle in complete),
                    complete[-1].open_interest,
                    timeframe,
                ))

        observed_dates = set(candles_by_date)
        missing_sessions = []
        current_date = min(observed_dates)
        last_date = max(observed_dates)
        while current_date <= last_date:
            is_session_day = current_date.weekday() in self._session.weekdays
            if is_session_day and current_date not in self._holiday_dates and current_date not in observed_dates:
                missing_sessions.append(current_date)
            current_date += timedelta(days=1)

        return CandleResampleResult(tuple(resampled), tuple(incomplete), tuple(missing_sessions))

    def resample_all(
        self,
        candles: list[Candle],
        *,
        as_of: datetime | None = None,
    ) -> dict[str, CandleResampleResult]:
        cutoff = as_of or datetime.now(timezone.utc)
        return {
            timeframe: self.resample(candles, timeframe, as_of=cutoff)
            for timeframe in self._TIMEFRAME_MINUTES
        }