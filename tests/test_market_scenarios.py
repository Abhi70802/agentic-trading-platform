from datetime import datetime, timedelta
from math import log
from statistics import pstdev

import pytest

from india_market import AssetType, OptionType
from market_scenarios import (
    DEFAULT_START,
    MarketScenario,
    build_synthetic_indian_market,
    generate_market_scenario,
)
from marketdata import MarketDataQualityGate


def test_synthetic_universe_covers_supported_indian_market_segments():
    universe = build_synthetic_indian_market()
    instruments = {(item.exchange_code, item.symbol): item for item in universe.policy.instruments}

    assert {exchange.code for exchange in universe.policy.exchanges} == {"NSE", "BSE"}
    assert instruments[("NSE", "TESTEQNSE")].asset_type == AssetType.EQUITY
    assert instruments[("BSE", "TESTEQBSE")].asset_type == AssetType.EQUITY
    assert {"NIFTY", "BANKNIFTY"} <= {
        item.symbol for item in universe.policy.instruments if item.asset_type == AssetType.INDEX
    }
    assert {"SYNTH_SECTOR_IT", "SYNTH_SECTOR_AUTO", "SYNTH_INDIA_VIX_LIKE"} <= set(universe.reference_prices)
    assert {contract.instrument.asset_type for contract in universe.policy.contracts} == {
        AssetType.FUTURE,
        AssetType.OPTION,
    }
    assert {contract.option_type for contract in universe.policy.contracts if contract.option_type} == {
        OptionType.CALL,
        OptionType.PUT,
    }
    assert all(contract.product_type == "SYNTHETIC" for contract in universe.policy.contracts)


def test_generation_is_reproducible_for_the_same_seed():
    first = generate_market_scenario(MarketScenario.NEWS_SHOCK, seed=314, start_price=1_000)
    repeated = generate_market_scenario(MarketScenario.NEWS_SHOCK, seed=314, start_price=1_000)
    different = generate_market_scenario(MarketScenario.NEWS_SHOCK, seed=315, start_price=1_000)

    assert first == repeated
    assert first != different


def test_scenarios_generate_bars_for_each_synthetic_instrument_and_contract():
    universe = build_synthetic_indian_market()

    for symbol, reference_price in universe.reference_prices.items():
        candles = generate_market_scenario(
            MarketScenario.NORMAL_BULL_TREND,
            symbol=symbol,
            start_price=reference_price,
            bars=8,
            seed=17,
        )
        assert len(candles) == 8
        assert all(bar.symbol == symbol for bar in candles)


@pytest.mark.parametrize("scenario", list(MarketScenario))
def test_every_scenario_emits_valid_positive_ohlcv_bars(scenario):
    candles = generate_market_scenario(scenario, seed=71, start_price=1_000, bars=24)

    assert candles
    for candle in candles:
        assert candle.open > 0
        assert candle.close > 0
        assert candle.low <= min(candle.open, candle.close)
        assert candle.high >= max(candle.open, candle.close)
        assert candle.volume > 0
        assert candle.timeframe == "1m"


def test_trend_reversal_and_range_profiles_have_distinct_shapes():
    bullish = generate_market_scenario(MarketScenario.NORMAL_BULL_TREND, seed=12)
    bearish = generate_market_scenario(MarketScenario.NORMAL_BEAR_TREND, seed=12)
    range_bound = generate_market_scenario(MarketScenario.RANGE_BOUND, seed=12)
    reversal = generate_market_scenario(MarketScenario.SHARP_INTRADAY_REVERSAL, seed=12)

    assert bullish[-1].close > bullish[0].close
    assert bearish[-1].close < bearish[0].close
    assert max(bar.close for bar in range_bound) - min(bar.close for bar in range_bound) < 0.08 * range_bound[0].close
    midpoint = len(reversal) // 2
    assert reversal[midpoint].close > reversal[0].close
    assert reversal[-1].close < reversal[midpoint].close


@pytest.mark.parametrize(
    ("scenario", "direction"),
    [(MarketScenario.GAP_UP, 1), (MarketScenario.GAP_DOWN, -1)],
)
def test_gap_scenarios_jump_at_the_open(scenario, direction):
    candles = generate_market_scenario(scenario, seed=5, bars=30)
    gap_index = len(candles) // 3
    gap_return = candles[gap_index].open / candles[gap_index - 1].close - 1

    assert direction * gap_return >= 0.019


def test_breakout_volume_profiles_are_distinguishable():
    strong = generate_market_scenario(MarketScenario.BREAKOUT_WITH_STRONG_VOLUME, seed=8)
    quiet = generate_market_scenario(MarketScenario.BREAKOUT_WITHOUT_VOLUME, seed=8)
    breakout_index = len(strong) // 2

    assert strong[breakout_index].close > strong[breakout_index].open
    assert quiet[breakout_index].close > quiet[breakout_index].open
    assert strong[breakout_index].volume > quiet[breakout_index].volume * 10


def test_false_breakout_returns_below_the_pre_breakout_close():
    candles = generate_market_scenario(MarketScenario.FALSE_BREAKOUT, seed=8)
    breakout_index = len(candles) // 2

    assert candles[breakout_index].close > candles[breakout_index - 1].close
    assert candles[breakout_index + 1].close < candles[breakout_index - 1].close


def test_volatility_shock_and_liquidity_scenarios_are_distinguishable():
    low = generate_market_scenario(MarketScenario.LOW_VOLATILITY, seed=23)
    high = generate_market_scenario(MarketScenario.HIGH_VOLATILITY, seed=23)
    extreme = generate_market_scenario(MarketScenario.EXTREME_VOLATILITY, seed=23)
    expiry = generate_market_scenario(MarketScenario.EXPIRY_DAY_VOLATILITY, seed=23)
    flash = generate_market_scenario(MarketScenario.FLASH_MOVE, seed=23)
    news = generate_market_scenario(MarketScenario.NEWS_SHOCK, seed=23)
    liquidity = generate_market_scenario(MarketScenario.LIQUIDITY_COLLAPSE, seed=23)

    def return_volatility(candles):
        returns = [log(current.close / previous.close) for previous, current in zip(candles, candles[1:])]
        return pstdev(returns)

    assert return_volatility(low) < return_volatility(high) < return_volatility(extreme)
    assert return_volatility(expiry) > return_volatility(low)
    shock_index = len(flash) // 2
    assert flash[shock_index].close < flash[shock_index].open
    assert flash[shock_index + 1].close > flash[shock_index + 1].open
    assert max(abs(log(bar.close / bar.open)) for bar in news) >= 0.04
    collapse_index = (len(liquidity) * 2) // 3
    assert max(bar.volume for bar in liquidity[collapse_index:]) < min(
        bar.volume for bar in liquidity[:collapse_index]
    )


def test_market_data_quality_scenarios_include_stale_missing_duplicate_and_reordered_bars():
    stale = generate_market_scenario(MarketScenario.STALE_MARKET_DATA, seed=9)
    missing = generate_market_scenario(MarketScenario.MISSING_CANDLES, seed=9)
    duplicate = generate_market_scenario(MarketScenario.DUPLICATE_CANDLES, seed=9)
    out_of_order = generate_market_scenario(MarketScenario.OUT_OF_ORDER_CANDLES, seed=9)

    assert not MarketDataQualityGate.is_fresh(
        stale[-1].timestamp,
        now=DEFAULT_START + timedelta(hours=1),
    )
    missing_times = [bar.timestamp for bar in missing]
    assert any(
        right - left > timedelta(minutes=1)
        for left, right in zip(
            (datetime.fromisoformat(value) for value in missing_times),
            (datetime.fromisoformat(value) for value in missing_times[1:]),
        )
    )
    assert any(left == right for left, right in zip(duplicate, duplicate[1:]))
    assert any(
        datetime.fromisoformat(right.timestamp) < datetime.fromisoformat(left.timestamp)
        for left, right in zip(out_of_order, out_of_order[1:])
    )


def test_scenario_arguments_reject_ambiguous_or_invalid_inputs():
    with pytest.raises(ValueError, match="At least 8 bars"):
        generate_market_scenario(MarketScenario.GAP_UP, bars=7)
    with pytest.raises(ValueError, match="finite and positive"):
        generate_market_scenario(MarketScenario.GAP_UP, start_price=0)
    with pytest.raises(ValueError, match="timezone"):
        generate_market_scenario(
            MarketScenario.GAP_UP,
            start_time=DEFAULT_START.replace(tzinfo=None),
        )