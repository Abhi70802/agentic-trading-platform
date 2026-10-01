"""Deterministic cross-sectional breadth over an explicitly supplied universe."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite
from statistics import mean

from core import Candle


class MarketBreadthEngine:
    def __init__(
        self,
        *,
        moving_average_windows: tuple[int, ...] = (20, 50),
        high_low_lookback: int = 252,
        volume_lookback: int = 20,
        high_relative_volume_threshold: float = 1.5,
    ):
        if not moving_average_windows or any(window < 1 for window in moving_average_windows):
            raise ValueError("Moving-average windows must be positive")
        if high_low_lookback < 1 or volume_lookback < 1:
            raise ValueError("Breadth history windows must be positive")
        if not isfinite(high_relative_volume_threshold) or high_relative_volume_threshold <= 0:
            raise ValueError("Relative-volume threshold must be finite and positive")
        self.moving_average_windows = tuple(sorted(set(moving_average_windows)))
        self.high_low_lookback = high_low_lookback
        self.volume_lookback = volume_lookback
        self.high_relative_volume_threshold = high_relative_volume_threshold

    def calculate(
        self,
        *,
        symbols: list[str] | tuple[str, ...],
        candles_by_symbol: dict[str, list[Candle]],
        timeframe: str,
        sector_by_symbol: dict[str, str] | None = None,
        index_symbols: tuple[str, ...] = ("NIFTY 50", "NIFTY BANK"),
    ) -> dict:
        universe = [symbol.strip().upper() for symbol in symbols]
        if not universe or any(not symbol for symbol in universe):
            raise ValueError("Breadth universe must contain non-empty symbols")
        if len(universe) != len(set(universe)):
            raise ValueError("Breadth universe symbols must be unique")
        if not timeframe:
            raise ValueError("Breadth timeframe is required")

        sector_map = {key.strip().upper(): value.strip() for key, value in (sector_by_symbol or {}).items()}
        prepared: dict[str, list[Candle]] = {}
        excluded: dict[str, str] = {}
        missing: list[str] = []
        last_timestamps: Counter[str] = Counter()

        for symbol in universe:
            candles = candles_by_symbol.get(symbol)
            if not candles:
                missing.append(symbol)
                continue
            ordered = sorted(candles, key=lambda candle: _timestamp_key(candle.timestamp))
            if any(candle.symbol.upper() != symbol for candle in ordered):
                excluded[symbol] = "symbol_mismatch"
            elif any(candle.timeframe != timeframe for candle in ordered):
                excluded[symbol] = "timeframe_mismatch"
            elif len({_timestamp_key(candle.timestamp) for candle in ordered}) != len(ordered):
                excluded[symbol] = "duplicate_timestamps"
            elif len(ordered) < 2:
                excluded[symbol] = "insufficient_bars_for_direction"
            else:
                prepared[symbol] = ordered
                last_timestamps[_timestamp_key(ordered[-1].timestamp)] += 1

        as_of_key = max(last_timestamps, key=lambda stamp: (last_timestamps[stamp], stamp)) if last_timestamps else None
        aligned: dict[str, list[Candle]] = {}
        for symbol, ordered in prepared.items():
            if _timestamp_key(ordered[-1].timestamp) != as_of_key:
                excluded[symbol] = "latest_bar_not_aligned"
            else:
                aligned[symbol] = ordered

        advances = declines = unchanged = 0
        advancing_volume = declining_volume = unchanged_volume = 0.0
        above_ma_counts = {window: 0 for window in self.moving_average_windows}
        above_ma_evaluated = {window: 0 for window in self.moving_average_windows}
        new_highs = new_lows = new_high_low_evaluated = 0
        high_relative_volume = relative_volume_evaluated = 0
        sector_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"advances": 0, "declines": 0, "unchanged": 0, "sample_count": 0})
        changes: dict[str, str] = {}

        for symbol, ordered in aligned.items():
            previous, current = ordered[-2], ordered[-1]
            if current.close > previous.close:
                direction = "ADVANCING"
                advances += 1
                advancing_volume += current.volume
            elif current.close < previous.close:
                direction = "DECLINING"
                declines += 1
                declining_volume += current.volume
            else:
                direction = "UNCHANGED"
                unchanged += 1
                unchanged_volume += current.volume
            changes[symbol] = direction

            sector = sector_map.get(symbol)
            if sector:
                sector_counts[sector]["sample_count"] += 1
                sector_key = {"ADVANCING": "advances", "DECLINING": "declines", "UNCHANGED": "unchanged"}[direction]
                sector_counts[sector][sector_key] += 1

            closes = [candle.close for candle in ordered]
            for window in self.moving_average_windows:
                if len(closes) >= window:
                    average = mean(closes[-window:])
                    above_ma_evaluated[window] += 1
                    if current.close > average:
                        above_ma_counts[window] += 1

            if len(closes) >= self.high_low_lookback + 1:
                previous_window = closes[-self.high_low_lookback - 1:-1]
                new_high_low_evaluated += 1
                new_highs += current.close > max(previous_window)
                new_lows += current.close < min(previous_window)

            volumes = [candle.volume for candle in ordered]
            if len(volumes) >= self.volume_lookback + 1:
                average_volume = mean(volumes[-self.volume_lookback - 1:-1])
                relative_volume_evaluated += 1
                if average_volume > 0 and current.volume / average_volume >= self.high_relative_volume_threshold:
                    high_relative_volume += 1

        sample_count = len(aligned)
        above_moving_averages = {
            f"sma_{window}": {
                "above_count": above_ma_counts[window],
                "evaluated_count": above_ma_evaluated[window],
                "percent_above": _percentage(above_ma_counts[window], above_ma_evaluated[window]),
            }
            for window in self.moving_average_windows
        }
        sector_breadth = {}
        for sector, counts in sorted(sector_counts.items()):
            sector_breadth[sector] = {
                **counts,
                "advance_decline_ratio": _ratio(counts["advances"], counts["declines"]),
            }
        normalized_indices = {symbol.upper() for symbol in index_symbols}
        index_participation = {
            symbol: changes.get(symbol)
            for symbol in sorted(normalized_indices)
        }

        return {
            "timeframe": timeframe,
            "as_of": as_of_key,
            "universe_size": len(universe),
            "sample_count": sample_count,
            "missing_symbols": sorted(missing),
            "excluded_symbols": dict(sorted(excluded.items())),
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "percent_advancing": _percentage(advances, sample_count),
            "percent_declining": _percentage(declines, sample_count),
            "advance_decline_ratio": _ratio(advances, declines),
            "above_moving_averages": above_moving_averages,
            "new_highs_lookback": {"count": new_highs, "evaluated_count": new_high_low_evaluated, "lookback": self.high_low_lookback},
            "new_lows_lookback": {"count": new_lows, "evaluated_count": new_high_low_evaluated, "lookback": self.high_low_lookback},
            "volume_breadth": {
                "advancing_volume": advancing_volume,
                "declining_volume": declining_volume,
                "unchanged_volume": unchanged_volume,
                "net_advancing_volume": advancing_volume - declining_volume,
                "high_relative_volume_count": high_relative_volume,
                "relative_volume_evaluated_count": relative_volume_evaluated,
                "relative_volume_threshold": self.high_relative_volume_threshold,
            },
            "sector_breadth": sector_breadth,
            "index_participation": index_participation,
            "execution_authority": False,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _percentage(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 4) if denominator else None


def _timestamp_key(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()