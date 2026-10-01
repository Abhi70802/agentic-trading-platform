from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import exp, isfinite, nan, inf, sqrt
from random import Random
from statistics import mean, pstdev

import pytest

from breadth import MarketBreadthEngine
from core import Candle
from features import FeatureValidator, feature_snapshot


def make_bars(closes, volumes=None, *, highs=None, lows=None):
    start = datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
    volumes = volumes or [10 + index for index in range(len(closes))]
    highs = highs or [close + 2 for close in closes]
    lows = lows or [close - 1 for close in closes]
    return [
        Candle(
            "FEATURE-TEST",
            (start + timedelta(minutes=index)).isoformat(),
            close,
            highs[index],
            lows[index],
            close,
            volumes[index],
            timeframe="1m",
        )
        for index, close in enumerate(closes)
    ]


def reference_ema(values, period):
    alpha = 2 / (period + 1)
    result = mean(values[:period])
    for value in values[period:]:
        result = alpha * value + (1 - alpha) * result
    return result


def numeric_features(snapshot):
    return [value for value in snapshot["features"].values() if isinstance(value, (int, float))]


def test_feature_snapshot_matches_independent_hand_calculations():
    closes = [float(value) for value in range(100, 160)]
    volumes = [float(value) for value in range(10, 70)]
    snapshot = feature_snapshot(make_bars(closes, volumes))
    features = snapshot["features"]

    expected_ema_12 = reference_ema(closes, 12)
    expected_ema_26 = reference_ema(closes, 26)
    returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
    last_returns = returns[-20:]
    last_volumes = volumes[-20:]
    expected_volume_zscore = (last_volumes[-1] - mean(last_volumes)) / pstdev(last_volumes)
    expected_vwap = sum((close + 1 / 3) * volume for close, volume in zip(closes[-20:], last_volumes)) / sum(last_volumes)

    assert features["sma_20"] == 149.5
    assert features["sma_50"] == 134.5
    assert features["ema_12"] == round(expected_ema_12, 6)
    assert features["ema_26"] == round(expected_ema_26, 6)
    assert features["rsi_14"] == 100
    assert features["macd"] == round(expected_ema_12 - expected_ema_26, 6)
    assert features["atr_14"] == 3
    assert features["atr_pct"] == round(3 / closes[-1] * 100, 6)
    assert features["bollinger_middle"] == 149.5
    assert features["bollinger_upper"] == pytest.approx(149.5 + 2 * sqrt(33.25), abs=1e-6)
    assert features["bollinger_lower"] == pytest.approx(149.5 - 2 * sqrt(33.25), abs=1e-6)
    assert features["momentum_20_pct"] == round((closes[-1] / closes[-21] - 1) * 100, 6)
    assert features["annualized_volatility_pct"] == round(pstdev(last_returns) * sqrt(252) * 100, 6)
    assert features["volume_zscore_20"] == round(expected_volume_zscore, 6)
    assert features["vwap_20"] == round(expected_vwap, 6)
    assert snapshot["regime"]["trend"] == "UPTREND"


def test_constant_prices_and_volumes_have_finite_neutral_indicators():
    snapshot = feature_snapshot(make_bars(
        [100.0] * 60,
        [500.0] * 60,
        highs=[100.0] * 60,
        lows=[100.0] * 60,
    ))
    features = snapshot["features"]

    assert features["sma_20"] == 100
    assert features["ema_12"] == 100
    assert features["ema_26"] == 100
    assert features["rsi_14"] == 50
    assert features["macd"] == 0
    assert features["atr_14"] == 0
    assert features["annualized_volatility_pct"] == 0
    assert features["vwap_20"] == 100
    assert features["momentum_20_pct"] == 0
    assert features["volume_zscore_20"] == 0
    assert snapshot["regime"]["trend"] == "RANGE_BOUND"


def test_short_history_returns_insufficient_values_without_fabricating_indicators():
    snapshot = feature_snapshot(make_bars([100.0 + index for index in range(10)]))
    features = snapshot["features"]

    for name in (
        "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "macd", "atr_14",
        "atr_pct", "bollinger_upper", "bollinger_middle", "bollinger_lower", "momentum_20_pct",
    ):
        assert features[name] is None, name
    assert features["vwap_20"] is not None
    assert features["volume_zscore_20"] is not None
    assert snapshot["regime"]["trend"] == "INSUFFICIENT_DATA"
    assert snapshot["regime"]["risk_posture"] == "UNKNOWN"


def test_extreme_but_finite_prices_keep_all_calculated_features_finite():
    closes = [1e150 * (1 + index * 1e-4) for index in range(60)]
    volumes = [1e100 * (1 + index * 1e-5) for index in range(60)]
    highs = [close * 1.01 for close in closes]
    lows = [close * 0.99 for close in closes]
    snapshot = feature_snapshot(make_bars(closes, volumes, highs=highs, lows=lows))

    assert numeric_features(snapshot)
    assert all(isfinite(value) for value in numeric_features(snapshot))


def test_seeded_randomized_ohlcv_produces_only_finite_numeric_features():
    rng = Random(7319)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    previous_close = 1_000.0
    for index in range(500):
        opening = previous_close
        close = opening * exp(rng.gauss(0, 0.003))
        high = max(opening, close) * (1 + rng.uniform(0, 0.002))
        low = min(opening, close) * (1 - rng.uniform(0, 0.002))
        candles.append(Candle(
            "RANDOM-TEST",
            (start + timedelta(minutes=index)).isoformat(),
            opening,
            high,
            low,
            close,
            float(rng.randint(1, 100_000)),
            timeframe="1m",
        ))
        previous_close = close

    snapshot = feature_snapshot(candles)

    assert snapshot["bar_count"] == 500
    assert all(isfinite(value) for value in numeric_features(snapshot))


def test_feature_snapshot_rejects_missing_values_at_candle_construction():
    with pytest.raises((TypeError, ValueError)):
        Candle("FEATURE-TEST", "2026-10-01T09:15:00+00:00", 100, 102, 99, None, 10, timeframe="1m")


def test_feature_snapshot_rejects_non_finite_values_even_if_a_candle_was_mutated():
    candles = make_bars([100.0 + index for index in range(60)])
    object.__setattr__(candles[-1], "high", nan)

    with pytest.raises(ValueError, match="finite"):
        feature_snapshot(candles)


def test_feature_validator_rejects_non_finite_output_values():
    snapshot = feature_snapshot(make_bars([100.0 + index for index in range(60)]))
    snapshot["features"]["macd"] = inf

    with pytest.raises(ValueError, match="macd.*finite"):
        FeatureValidator.validate(snapshot)


def test_feature_snapshot_validates_derived_output(monkeypatch):
    monkeypatch.setattr("features._round", lambda value: inf if value is not None else None)

    with pytest.raises(ValueError, match="finite"):
        feature_snapshot(make_bars([100.0 + index for index in range(60)]))


def test_feature_snapshot_rejects_overflowed_returns_before_statistics():
    candles = [
        Candle("FEATURE-TEST", "2026-10-01T09:15:00+00:00", 1e-300, 1.1e-300, 0.9e-300, 1e-300, 1, timeframe="1m"),
        Candle("FEATURE-TEST", "2026-10-01T09:16:00+00:00", 1e300, 1.1e300, 0.9e300, 1e300, 1, timeframe="1m"),
        Candle("FEATURE-TEST", "2026-10-01T09:17:00+00:00", 1e-300, 1.1e-300, 0.9e-300, 1e-300, 1, timeframe="1m"),
    ]

    with pytest.raises(ValueError, match="returns must be finite"):
        feature_snapshot(candles)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("open", "high", "low", "close", "volume", "open_interest")
        for value in (nan, inf, -inf)
    ],
)
def test_non_finite_candle_input_is_rejected_at_construction(field, value):
    candles = make_bars([100.0 + index for index in range(60)])
    with pytest.raises(ValueError, match="finite"):
        replace(candles[-1], **{field: value})


def test_feature_snapshot_rejects_duplicate_timestamps():
    candles = make_bars([100.0 + index for index in range(60)])
    candles[-1] = replace(candles[-1], timestamp=candles[-2].timestamp)

    with pytest.raises(ValueError, match="timestamp"):
        feature_snapshot(candles)


def test_breadth_detects_new_high_and_relative_volume_on_known_series():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    closes = [100, 101, 99, 102, 100, 101, 110]
    volumes = [10, 10, 10, 10, 10, 10, 30]
    candles = [
        Candle(
            "BREAKOUT-TEST",
            (start + timedelta(days=index)).isoformat(),
            close,
            close + 1,
            close - 1,
            close,
            volume,
            timeframe="1d",
        )
        for index, (close, volume) in enumerate(zip(closes, volumes))
    ]
    result = MarketBreadthEngine(
        moving_average_windows=(3,),
        high_low_lookback=5,
        volume_lookback=3,
        high_relative_volume_threshold=1.5,
    ).calculate(
        symbols=["BREAKOUT-TEST"],
        candles_by_symbol={"BREAKOUT-TEST": candles},
        timeframe="1d",
    )

    assert result["new_highs_lookback"] == {"count": 1, "evaluated_count": 1, "lookback": 5}
    assert result["volume_breadth"]["high_relative_volume_count"] == 1
    assert result["volume_breadth"]["relative_volume_evaluated_count"] == 1