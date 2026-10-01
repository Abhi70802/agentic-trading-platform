"""India-only market reference-data contracts; no exchange rules are implicit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class AssetType(StrEnum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class OptionType(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True)
class Exchange:
    code: str
    name: str
    country_code: str
    timezone_name: str

    def __post_init__(self) -> None:
        if not self.code or self.code != self.code.upper():
            raise ValueError("Exchange code must be non-empty uppercase text")
        if self.country_code != "IN":
            raise ValueError("Initial market policy accepts Indian exchanges only")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Exchange timezone must be a valid IANA timezone") from error


@dataclass(frozen=True)
class Instrument:
    exchange_code: str
    symbol: str
    asset_type: AssetType
    name: str
    segment: str = ""
    instrument_token: str | None = None
    currency: str = "INR"

    def __post_init__(self) -> None:
        if not self.exchange_code or self.exchange_code != self.exchange_code.upper():
            raise ValueError("Instrument exchange code must be non-empty uppercase text")
        if not self.symbol or self.symbol != self.symbol.upper():
            raise ValueError("Instrument symbol must be non-empty uppercase text")
        if self.segment and self.segment != self.segment.upper():
            raise ValueError("Instrument segment must be uppercase text")
        if self.currency != "INR":
            raise ValueError("Initial market policy accepts INR instruments only")


@dataclass(frozen=True)
class LotSize:
    quantity: int
    effective_from: date
    effective_to: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool) or self.quantity < 1:
            raise ValueError("Lot size must be positive")
        if not isinstance(self.effective_from, date) or isinstance(self.effective_from, datetime):
            raise ValueError("Lot-size effective date must be a date")
        if self.effective_to is not None and (
            not isinstance(self.effective_to, date)
            or isinstance(self.effective_to, datetime)
            or self.effective_to < self.effective_from
        ):
            raise ValueError("Lot-size effective range is invalid")

    def accepts(self, quantity: int) -> bool:
        return (
            isinstance(quantity, int)
            and not isinstance(quantity, bool)
            and quantity > 0
            and quantity % self.quantity == 0
        )


@dataclass(frozen=True)
class TickSize:
    value: Decimal
    effective_from: date
    effective_to: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal) or not self.value.is_finite() or self.value <= 0:
            raise ValueError("Tick size must be a finite positive decimal")
        if not isinstance(self.effective_from, date) or isinstance(self.effective_from, datetime):
            raise ValueError("Tick-size effective date must be a date")
        if self.effective_to is not None and (
            not isinstance(self.effective_to, date)
            or isinstance(self.effective_to, datetime)
            or self.effective_to < self.effective_from
        ):
            raise ValueError("Tick-size effective range is invalid")

    def accepts(self, price: Decimal) -> bool:
        if not price.is_finite() or price <= 0:
            return False
        ticks = price / self.value
        return ticks == ticks.to_integral_value()


@dataclass(frozen=True)
class Contract:
    instrument: Instrument
    underlying: Instrument
    expiry: date
    lot_size: LotSize
    tick_size: TickSize
    strike: Decimal | None = None
    option_type: OptionType | None = None
    product_type: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, Instrument) or not isinstance(self.underlying, Instrument):
            raise ValueError("Contract instrument and underlying references are required")
        if not isinstance(self.expiry, date) or isinstance(self.expiry, datetime):
            raise ValueError("Contract expiry must be a date")
        if not isinstance(self.lot_size, LotSize) or not isinstance(self.tick_size, TickSize):
            raise ValueError("Contract lot size and tick size references are required")
        if self.instrument.asset_type not in {AssetType.FUTURE, AssetType.OPTION}:
            raise ValueError("Contracts must be futures or options")
        if self.underlying.asset_type not in {AssetType.EQUITY, AssetType.INDEX}:
            raise ValueError("Contract underlying must be an equity or index")
        if self.instrument.asset_type == AssetType.OPTION:
            if (
                not isinstance(self.strike, Decimal)
                or not self.strike.is_finite()
                or self.strike <= 0
            ):
                raise ValueError("Options require a finite positive strike")
            if not isinstance(self.option_type, OptionType):
                raise ValueError("Options require a call or put type")
        elif self.strike is not None or self.option_type is not None:
            raise ValueError("Futures cannot declare option strike or type")


@dataclass(frozen=True)
class TradingSession:
    exchange_code: str
    segment: str
    opens_at: time
    closes_at: time
    weekdays: frozenset[int]

    def __post_init__(self) -> None:
        if not self.segment:
            raise ValueError("Trading session segment is required")
        if self.opens_at.tzinfo is not None or self.closes_at.tzinfo is not None:
            raise ValueError("Trading session times must be local wall-clock times")
        if self.opens_at >= self.closes_at:
            raise ValueError("Trading session close must follow its open")
        if not self.weekdays or any(day not in range(7) for day in self.weekdays):
            raise ValueError("Trading session weekdays must be explicitly configured")


@dataclass(frozen=True)
class TradingHoliday:
    exchange_code: str
    session_date: date
    reason: str
    segment: str | None = None


@dataclass(frozen=True)
class ExpiryCalendar:
    exchange_code: str
    underlying_symbol: str
    asset_type: AssetType
    expiry_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if self.asset_type not in {AssetType.FUTURE, AssetType.OPTION}:
            raise ValueError("Expiry calendars apply only to futures or options")
        if tuple(sorted(set(self.expiry_dates))) != self.expiry_dates:
            raise ValueError("Expiry dates must be unique and sorted")


@dataclass(frozen=True)
class CorporateAction:
    exchange_code: str
    symbol: str
    action_type: str
    announced_at: datetime
    ex_date: date
    terms: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.announced_at.tzinfo is None or self.announced_at.utcoffset() is None:
            raise ValueError("Corporate-action announcement time must include a timezone")
        if not self.action_type:
            raise ValueError("Corporate-action type is required")


@dataclass(frozen=True)
class IndiaMarketPolicy:
    """Explicit India market reference data; empty configuration enables no market rules."""

    exchanges: tuple[Exchange, ...] = ()
    instruments: tuple[Instrument, ...] = ()
    contracts: tuple[Contract, ...] = ()
    sessions: tuple[TradingSession, ...] = ()
    holidays: tuple[TradingHoliday, ...] = ()
    expiry_calendars: tuple[ExpiryCalendar, ...] = ()
    corporate_actions: tuple[CorporateAction, ...] = ()

    def __post_init__(self) -> None:
        exchange_codes = [exchange.code for exchange in self.exchanges]
        if len(exchange_codes) != len(set(exchange_codes)):
            raise ValueError("Exchange codes must be unique")
        known_exchanges = set(exchange_codes)

        instrument_keys = [(item.exchange_code, item.symbol) for item in self.instruments]
        if len(instrument_keys) != len(set(instrument_keys)):
            raise ValueError("Instrument exchange and symbol pairs must be unique")
        known_instruments = set(instrument_keys)
        for instrument in self.instruments:
            if instrument.exchange_code not in known_exchanges:
                raise ValueError(f"Exchange reference is missing for {instrument.symbol}")

        contract_keys: list[tuple[str, str, date]] = []
        for contract in self.contracts:
            if contract.instrument.exchange_code not in known_exchanges:
                raise ValueError(f"Exchange reference is missing for {contract.instrument.symbol}")
            if contract.underlying.exchange_code not in known_exchanges:
                raise ValueError(f"Underlying exchange reference is missing for {contract.underlying.symbol}")
            underlying_key = (contract.underlying.exchange_code, contract.underlying.symbol)
            if underlying_key not in known_instruments:
                raise ValueError(f"Instrument reference is missing for underlying {contract.underlying.symbol}")
            contract_keys.append((contract.instrument.exchange_code, contract.instrument.symbol, contract.expiry))
        if len(contract_keys) != len(set(contract_keys)):
            raise ValueError("Contract exchange, symbol, and expiry triples must be unique")

        session_keys = [(session.exchange_code, session.segment) for session in self.sessions]
        if len(session_keys) != len(set(session_keys)):
            raise ValueError("Trading sessions must be unique per exchange and segment")
        for record in (*self.sessions, *self.holidays, *self.expiry_calendars):
            if record.exchange_code not in known_exchanges:
                raise ValueError(f"Exchange reference is missing for {record.exchange_code}")

        for action in self.corporate_actions:
            if (action.exchange_code, action.symbol) not in known_instruments:
                raise ValueError(f"Instrument reference is missing for corporate action {action.symbol}")

    def exchange(self, code: str) -> Exchange:
        for exchange in self.exchanges:
            if exchange.code == code.upper():
                return exchange
        raise ValueError(f"No Indian exchange reference configured for {code}")

    def instrument(self, exchange_code: str, symbol: str) -> Instrument:
        for instrument in self.instruments:
            if (instrument.exchange_code, instrument.symbol) == (exchange_code.upper(), symbol.upper()):
                return instrument
        raise ValueError(f"No Indian instrument reference configured for {exchange_code}:{symbol}")