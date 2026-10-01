"""Deterministic synthetic Indian-market fixtures for offline tests only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from math import exp, isfinite, log
from random import Random
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

from core import Candle
from india_market import (
    AssetType,
    Contract,
    Exchange,
    IndiaMarketPolicy,
    Instrument,
    LotSize,
    OptionType,
    TickSize,
)


class MarketScenario(str, Enum):
    NORMAL_BULL_TREND = "Normal Bull Trend"
    NORMAL_BEAR_TREND = "Normal Bear Trend"
    RANGE_BOUND = "Range Bound"
    SHARP_INTRADAY_REVERSAL = "Sharp Intraday Reversal"
    GAP_UP = "Gap Up"
    GAP_DOWN = "Gap Down"
    FALSE_BREAKOUT = "False Breakout"
    BREAKOUT_WITH_STRONG_VOLUME = "Breakout With Strong Volume"
    BREAKOUT_WITHOUT_VOLUME = "Breakout Without Volume"
    LOW_VOLATILITY = "Low Volatility"
    HIGH_VOLATILITY = "High Volatility"
    EXTREME_VOLATILITY = "Extreme Volatility"
    FLASH_MOVE = "Flash-Move Scenario"
    NEWS_SHOCK = "News Shock"
    EXPIRY_DAY_VOLATILITY = "Expiry-Day Volatility"
    LIQUIDITY_COLLAPSE = "Liquidity Collapse"
    STALE_MARKET_DATA = "Stale Market Data"
    MISSING_CANDLES = "Missing Candles"
    DUPLICATE_CANDLES = "Duplicate Candles"
    OUT_OF_ORDER_CANDLES = "Out-of-Order Candles"


DEFAULT_START = datetime(2026, 10, 1, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
_BASE_VOLUME = 1_000.0
_BAR_INTERVAL = timedelta(minutes=1)


@dataclass(frozen=True)
class SyntheticMarketUniverse:
    """Test-only Indian-market reference data and arbitrary simulation price anchors."""

    policy: IndiaMarketPolicy
    reference_prices: Mapping[str, float]


def build_synthetic_indian_market() -> SyntheticMarketUniverse:
    """Build test references; prices and contract terms are not real market data."""
    exchanges = (
        Exchange("NSE", "Synthetic NSE test exchange", "IN", "Asia/Kolkata"),
        Exchange("BSE", "Synthetic BSE test exchange", "IN", "Asia/Kolkata"),
    )
    equities = (
        Instrument("NSE", "TESTEQNSE", AssetType.EQUITY, "Synthetic NSE equity", "CM-TEST"),
        Instrument("BSE", "TESTEQBSE", AssetType.EQUITY, "Synthetic BSE equity", "CM-TEST"),
    )
    indices = (
        Instrument("NSE", "NIFTY", AssetType.INDEX, "Synthetic NIFTY-like index", "INDEX-TEST"),
        Instrument("NSE", "BANKNIFTY", AssetType.INDEX, "Synthetic BANKNIFTY-like index", "INDEX-TEST"),
        Instrument("NSE", "SYNTH_SECTOR_IT", AssetType.INDEX, "Synthetic IT sector index", "INDEX-TEST"),
        Instrument("NSE", "SYNTH_SECTOR_AUTO", AssetType.INDEX, "Synthetic auto sector index", "INDEX-TEST"),
        Instrument("NSE", "SYNTH_INDIA_VIX_LIKE", AssetType.INDEX, "Synthetic volatility-index proxy", "INDEX-TEST"),
    )
    reference_by_symbol = {
        "TESTEQNSE": 100.0,
        "TESTEQBSE": 100.0,
        "NIFTY": 1_000.0,
        "BANKNIFTY": 1_500.0,
        "SYNTH_SECTOR_IT": 750.0,
        "SYNTH_SECTOR_AUTO": 600.0,
        "SYNTH_INDIA_VIX_LIKE": 20.0,
    }
    contracts: list[Contract] = []
    for underlying in (indices[0], indices[1]):
        underlying_price = Decimal(str(reference_by_symbol[underlying.symbol]))
        future = Instrument(
            "NSE",
            f"{underlying.symbol}-TEST-FUT",
            AssetType.FUTURE,
            f"Synthetic {underlying.symbol}-like future",
            "NFO-TEST",
        )
        contracts.append(Contract(
            instrument=future,
            underlying=underlying,
            expiry=date(2099, 1, 1),
            lot_size=LotSize(1, date(2026, 1, 1)),
            tick_size=TickSize(Decimal("0.01"), date(2026, 1, 1)),
            product_type="SYNTHETIC",
        ))
        reference_by_symbol[future.symbol] = float(underlying_price)
        for option_type, suffix in ((OptionType.CALL, "CALL"), (OptionType.PUT, "PUT")):
            option = Instrument(
                "NSE",
                f"{underlying.symbol}-TEST-{suffix}",
                AssetType.OPTION,
                f"Synthetic {underlying.symbol}-like {suffix.lower()} option",
                "NFO-TEST",
            )
            contracts.append(Contract(
                instrument=option,
                underlying=underlying,
                expiry=date(2099, 1, 1),
                lot_size=LotSize(1, date(2026, 1, 1)),
                tick_size=TickSize(Decimal("0.01"), date(2026, 1, 1)),
                strike=underlying_price,
                option_type=option_type,
                product_type="SYNTHETIC",
            ))
            reference_by_symbol[option.symbol] = 25.0

    policy = IndiaMarketPolicy(
        exchanges=exchanges,
        instruments=(*equities, *indices),
        contracts=tuple(contracts),
    )
    return SyntheticMarketUniverse(policy, MappingProxyType(reference_by_symbol))


def generate_market_scenario(
    scenario: MarketScenario | str,
    *,
    symbol: str | Instrument = "NIFTY",
    start_price: float = 100.0,
    bars: int = 60,
    seed: int = 0,
    start_time: datetime = DEFAULT_START,
    interval: timedelta = _BAR_INTERVAL,
) -> list[Candle]:
    """Generate reproducible synthetic OHLCV bars; anomaly scenarios alter the sequence."""
    scenario = MarketScenario(scenario)
    symbol = symbol.symbol if isinstance(symbol, Instrument) else symbol.upper()
    if not symbol:
        raise ValueError("A market-data symbol is required")
    if not isfinite(start_price) or start_price <= 0:
        raise ValueError("start_price must be finite and positive")
    if bars < 8:
        raise ValueError("At least 8 bars are required for scenario events")
    if start_time.tzinfo is None or start_time.utcoffset() is None:
        raise ValueError("start_time must include a timezone")
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")

    rng = Random(seed)
    shock_index = bars // 2
    gap_index = bars // 3
    collapse_index = (bars * 2) // 3
    volatility = {
        MarketScenario.NORMAL_BULL_TREND: 0.0015,
        MarketScenario.NORMAL_BEAR_TREND: 0.0015,
        MarketScenario.RANGE_BOUND: 0.0008,
        MarketScenario.SHARP_INTRADAY_REVERSAL: 0.0015,
        MarketScenario.GAP_UP: 0.0015,
        MarketScenario.GAP_DOWN: 0.0015,
        MarketScenario.FALSE_BREAKOUT: 0.0007,
        MarketScenario.BREAKOUT_WITH_STRONG_VOLUME: 0.0007,
        MarketScenario.BREAKOUT_WITHOUT_VOLUME: 0.0007,
        MarketScenario.LOW_VOLATILITY: 0.00025,
        MarketScenario.HIGH_VOLATILITY: 0.006,
        MarketScenario.EXTREME_VOLATILITY: 0.018,
        MarketScenario.FLASH_MOVE: 0.0015,
        MarketScenario.NEWS_SHOCK: 0.002,
        MarketScenario.EXPIRY_DAY_VOLATILITY: 0.009,
        MarketScenario.LIQUIDITY_COLLAPSE: 0.002,
        MarketScenario.STALE_MARKET_DATA: 0.0015,
        MarketScenario.MISSING_CANDLES: 0.0015,
        MarketScenario.DUPLICATE_CANDLES: 0.0015,
        MarketScenario.OUT_OF_ORDER_CANDLES: 0.0015,
    }[scenario]

    candles: list[Candle] = []
    previous_close = start_price
    range_center = start_price
    for index in range(bars):
        drift = 0.0005 if scenario == MarketScenario.NORMAL_BULL_TREND else 0.0
        if scenario == MarketScenario.NORMAL_BEAR_TREND:
            drift = -0.0005
        elif scenario == MarketScenario.SHARP_INTRADAY_REVERSAL:
            drift = 0.0018 if index < shock_index else -0.0035
        elif scenario == MarketScenario.RANGE_BOUND:
            drift = max(-0.002, min(0.002, -0.35 * log(previous_close / range_center)))
        elif scenario == MarketScenario.BREAKOUT_WITH_STRONG_VOLUME and index > shock_index:
            drift = 0.001

        bar_open = previous_close
        if scenario in {MarketScenario.GAP_UP, MarketScenario.GAP_DOWN} and index == gap_index:
            gap = 0.02 if scenario == MarketScenario.GAP_UP else -0.02
            bar_open = previous_close * (1 + gap)

        log_return = rng.gauss(drift, volatility)
        close = bar_open * exp(log_return)
        if scenario in {
            MarketScenario.BREAKOUT_WITH_STRONG_VOLUME,
            MarketScenario.BREAKOUT_WITHOUT_VOLUME,
        } and index == shock_index:
            close = bar_open * 1.015
        elif scenario == MarketScenario.FALSE_BREAKOUT and index == shock_index:
            close = bar_open * 1.015
        elif scenario == MarketScenario.FALSE_BREAKOUT and index == shock_index + 1:
            close = bar_open / 1.015 * 0.995
        elif scenario == MarketScenario.FLASH_MOVE and index == shock_index:
            close = bar_open * 0.945
        elif scenario == MarketScenario.FLASH_MOVE and index == shock_index + 1:
            close = bar_open * 1.04
        elif scenario == MarketScenario.NEWS_SHOCK and index == shock_index:
            close = bar_open * (1.05 if rng.getrandbits(1) else 0.95)

        wick_scale = 3.0 if scenario == MarketScenario.LIQUIDITY_COLLAPSE and index >= collapse_index else 1.0
        wick = abs(rng.gauss(volatility * 0.35, volatility * 0.15)) * wick_scale
        high = max(bar_open, close) * (1 + wick)
        low = min(bar_open, close) * (1 - wick)

        volume_multiplier = rng.lognormvariate(0.0, 0.15)
        if scenario == MarketScenario.EXPIRY_DAY_VOLATILITY:
            volume_multiplier *= 2.0
        if scenario == MarketScenario.BREAKOUT_WITH_STRONG_VOLUME and index == shock_index:
            volume_multiplier = 4.0
        elif scenario == MarketScenario.BREAKOUT_WITHOUT_VOLUME and index == shock_index:
            volume_multiplier = 0.25
        elif scenario == MarketScenario.NEWS_SHOCK and index == shock_index:
            volume_multiplier *= 4.0
        elif scenario == MarketScenario.LIQUIDITY_COLLAPSE and index >= collapse_index:
            volume_multiplier *= 0.03

        timestamp_start = start_time
        if scenario == MarketScenario.STALE_MARKET_DATA:
            timestamp_start -= timedelta(hours=1)
        timestamp = timestamp_start + index * interval
        candles.append(Candle(
            symbol=symbol,
            timestamp=timestamp.isoformat(),
            open=round(bar_open, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close, 4),
            volume=round(max(1.0, _BASE_VOLUME * volume_multiplier), 2),
            timeframe="1m",
        ))
        previous_close = close

    if scenario == MarketScenario.MISSING_CANDLES:
        del candles[shock_index]
    elif scenario == MarketScenario.DUPLICATE_CANDLES:
        candles.insert(shock_index + 1, candles[shock_index])
    elif scenario == MarketScenario.OUT_OF_ORDER_CANDLES:
        candles[shock_index], candles[shock_index + 1] = candles[shock_index + 1], candles[shock_index]
    return candles