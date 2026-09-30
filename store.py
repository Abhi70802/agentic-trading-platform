"""Small local SQLite store for numerical bars and auditable platform events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from core import Candle


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    open_interest REAL,
                    PRIMARY KEY (symbol, timestamp)
                );
                CREATE INDEX IF NOT EXISTS candles_symbol_time
                    ON candles(symbol, timestamp);
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    schema_version TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_time ON events(timestamp DESC);
                CREATE TABLE IF NOT EXISTS news_items (
                    article_id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    instruments TEXT NOT NULL,
                    sector TEXT,
                    country TEXT,
                    source_reliability REAL NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    original_url TEXT,
                    intelligence TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS news_published_time
                    ON news_items(published_at DESC);
                CREATE TABLE IF NOT EXISTS market_ticks (
                    event_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    instrument_token INTEGER NOT NULL,
                    exchange TEXT NOT NULL,
                    market_timestamp TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    last_price REAL NOT NULL,
                    volume INTEGER,
                    open_interest INTEGER,
                    day_ohlc TEXT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS market_ticks_symbol_time
                    ON market_ticks(symbol, market_timestamp DESC);
                """
            )
            candle_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(candles)")
            }
            if "open_interest" not in candle_columns:
                connection.execute("ALTER TABLE candles ADD COLUMN open_interest REAL")
            news_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(news_items)")
            }
            if "original_url" not in news_columns:
                connection.execute("ALTER TABLE news_items ADD COLUMN original_url TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_candles(self, candles: list[Candle]) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.executemany(
                """INSERT OR IGNORE INTO candles
                   (symbol, timestamp, open, high, low, close, volume, open_interest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.symbol, c.timestamp, c.open, c.high, c.low, c.close, c.volume, c.open_interest)
                    for c in candles
                ],
            )
            return cursor.rowcount

    def upsert_candle(self, candle: Candle) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO candles (symbol, timestamp, open, high, low, close, volume, open_interest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, timestamp) DO UPDATE SET
                       high=MAX(candles.high, excluded.high),
                       low=MIN(candles.low, excluded.low),
                       close=excluded.close,
                       volume=excluded.volume,
                       open_interest=COALESCE(excluded.open_interest, candles.open_interest)""",
                (candle.symbol, candle.timestamp, candle.open, candle.high, candle.low,
                 candle.close, candle.volume, candle.open_interest),
            )

    def read_candles(self, symbol: str, limit: int = 5000) -> list[Candle]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [
            Candle(
                row["symbol"], row["timestamp"], row["open"], row["high"],
                row["low"], row["close"], row["volume"], row["open_interest"],
            )
            for row in reversed(rows)
        ]

    def list_instruments(self) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """SELECT symbol, COUNT(*) AS bars, MAX(timestamp) AS latest_timestamp
                   FROM candles GROUP BY symbol ORDER BY symbol"""
            ).fetchall()
        return [
            {"symbol": row["symbol"], "bars": row["bars"], "latest_timestamp": row["latest_timestamp"]}
            for row in rows
        ]

    def append_events(self, events: list[dict]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executemany(
                """INSERT OR IGNORE INTO events
                   (event_id, event_type, timestamp, correlation_id, source, version,
                    schema_version, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        event["event_id"], event["event_type"], event["timestamp"],
                        event["correlation_id"], event["source"], event["version"],
                        event["schema_version"], json.dumps(event["payload"]),
                    )
                    for event in events
                ],
            )

    def save_news(self, item: dict) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO news_items
                   (article_id, dedupe_key, source, published_at, ingested_at,
                    instruments, sector, country, source_reliability, title, content, original_url, intelligence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item["article_id"], item["dedupe_key"], item["source"],
                    item["published_at"], item["ingested_at"],
                    json.dumps(item["instruments"]), item.get("sector"), item.get("country"),
                    item["source_reliability"], item["title"], item["content"], item.get("original_url"),
                    json.dumps(item["intelligence"]),
                ),
            )
            return cursor.rowcount == 1

    def read_news(self, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM news_items ORDER BY published_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "article_id": row["article_id"],
                "dedupe_key": row["dedupe_key"],
                "source": row["source"],
                "published_at": row["published_at"],
                "ingested_at": row["ingested_at"],
                "instruments": json.loads(row["instruments"]),
                "sector": row["sector"],
                "country": row["country"],
                "source_reliability": row["source_reliability"],
                "title": row["title"],
                "content": row["content"],
                "original_url": row["original_url"],
                "intelligence": json.loads(row["intelligence"]),
            }
            for row in rows
        ]

    def save_market_tick(self, event: dict) -> bool:
        payload = event["payload"]
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO market_ticks
                   (event_id, symbol, instrument_token, exchange, market_timestamp,
                    ingested_at, last_price, volume, open_interest, day_ohlc, bids,
                    asks, correlation_id, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"], payload["symbol"], payload["instrument_token"],
                    payload["exchange"], payload["market_timestamp"], event["timestamp"],
                    payload["last_price"], payload.get("volume"), payload.get("open_interest"),
                    json.dumps(payload["day_ohlc"]), json.dumps(payload["bids"]),
                    json.dumps(payload["asks"]), event["correlation_id"], event["source"],
                ),
            )
            return cursor.rowcount == 1

    def read_market_ticks(self, symbol: str, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """SELECT * FROM market_ticks WHERE symbol = ?
                   ORDER BY market_timestamp DESC, rowid DESC LIMIT ?""",
                (symbol.upper(), limit),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "symbol": row["symbol"],
                "instrument_token": row["instrument_token"],
                "exchange": row["exchange"],
                "timestamp": row["market_timestamp"],
                "ingested_at": row["ingested_at"],
                "last_price": row["last_price"],
                "volume": row["volume"],
                "open_interest": row["open_interest"],
                "day_ohlc": json.loads(row["day_ohlc"]),
                "bids": json.loads(row["bids"]),
                "asks": json.loads(row["asks"]),
                "correlation_id": row["correlation_id"],
                "source": row["source"],
            }
            for row in reversed(rows)
        ]

    def read_events(self, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "correlation_id": row["correlation_id"],
                "source": row["source"],
                "version": row["version"],
                "schema_version": row["schema_version"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]