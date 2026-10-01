from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest

from core import Candle
from marketdata import JsonCandleAdapter, MarketDataQualityGate
from store import EventStore


@pytest.mark.performance
def test_provider_normalization_and_quality_gate_handle_five_thousand_events():
    adapter = JsonCandleAdapter("load-fixture")
    gate = MarketDataQualityGate(max_seen=6_000)
    start = datetime(2026, 10, 1, 3, 45, tzinfo=timezone.utc)
    event_count = 5_000

    started = perf_counter()
    for sequence in range(event_count):
        event = adapter.normalize({
            "symbol": "NIFTY 50",
            "timestamp": (start + timedelta(seconds=sequence)).isoformat(),
            "open": 22_000,
            "high": 22_002,
            "low": 21_998,
            "close": 22_001,
            "volume": 1_000 + sequence,
            "timeframe": "1m",
            "sequence": sequence,
        })
        assert gate.review(event) is None
    elapsed = perf_counter() - started

    assert gate.status()["accepted"] == event_count
    assert gate.status()["duplicates"] == 0
    assert elapsed < 15, f"Normalized {event_count} events in {elapsed:.2f}s"


@pytest.mark.performance
def test_sqlite_bulk_candle_load_round_trips_five_thousand_bars():
    bar_count = 5_000
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(
            "NIFTY 50",
            (start + timedelta(minutes=index)).isoformat(),
            22_000,
            22_002,
            21_998,
            22_001,
            1_000 + index,
            timeframe="1m",
        )
        for index in range(bar_count)
    ]

    with TemporaryDirectory() as directory:
        store = EventStore(Path(directory) / "load.db")
        started = perf_counter()
        saved = store.save_candles(candles)
        loaded = store.read_candles("NIFTY 50", limit=bar_count, timeframe="1m")
        elapsed = perf_counter() - started

    assert saved == bar_count
    assert loaded == candles
    assert elapsed < 15, f"Persisted and read {bar_count} bars in {elapsed:.2f}s"