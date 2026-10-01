"""Small local SQLite store for numerical bars and auditable platform events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Iterable

from core import Candle
from india_market import IndiaMarketPolicy
from rag import MemoryDocument, cosine_similarity


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _reference_json_value(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported reference-data value: {type(value).__name__}")


def _reference_snapshot_record(row: sqlite3.Row) -> dict:
    return {
        "provider": row["provider"],
        "snapshot_hash": row["snapshot_hash"],
        "fetched_at": row["fetched_at"],
        "record_counts": json.loads(row["record_counts"]),
        "payload": json.loads(row["payload"]),
    }


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            existing_tables = {
                row["name"] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "candles" in existing_tables:
                existing_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(candles)")
                }
                if "timeframe" not in existing_columns:
                    connection.execute("DROP INDEX IF EXISTS candles_symbol_time")
                    connection.execute("ALTER TABLE candles RENAME TO candles_legacy_timeframe")
                    connection.execute(
                        """CREATE TABLE candles (
                            symbol TEXT NOT NULL,
                            timeframe TEXT NOT NULL DEFAULT 'legacy',
                            timestamp TEXT NOT NULL,
                            open REAL NOT NULL,
                            high REAL NOT NULL,
                            low REAL NOT NULL,
                            close REAL NOT NULL,
                            volume REAL NOT NULL,
                            open_interest REAL,
                            PRIMARY KEY (symbol, timeframe, timestamp)
                        )"""
                    )
                    open_interest_column = "open_interest" if "open_interest" in existing_columns else "NULL"
                    connection.execute(
                        f"""INSERT INTO candles
                            (symbol, timeframe, timestamp, open, high, low, close, volume, open_interest)
                            SELECT symbol, 'legacy', timestamp, open, high, low, close, volume, {open_interest_column}
                            FROM candles_legacy_timeframe"""
                    )
                    connection.execute("DROP TABLE candles_legacy_timeframe")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT 'legacy',
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    open_interest REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                );
                CREATE INDEX IF NOT EXISTS candles_symbol_timeframe_time
                    ON candles(symbol, timeframe, timestamp);
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
                CREATE INDEX IF NOT EXISTS events_type ON events(event_type);
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
                CREATE TABLE IF NOT EXISTS rag_documents (
                    document_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS rag_documents_kind_time
                    ON rag_documents(kind, created_at DESC);
                CREATE TABLE IF NOT EXISTS prediction_records (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    outcome TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS prediction_records_symbol_time
                    ON prediction_records(symbol, created_at DESC);
                CREATE TABLE IF NOT EXISTS trade_journal (
                    trade_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    realized_pnl REAL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS trade_journal_symbol_time
                    ON trade_journal(symbol, opened_at DESC);
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
                CREATE TABLE IF NOT EXISTS india_reference_snapshots (
                    provider TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    record_counts TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (provider, snapshot_hash)
                );
                CREATE INDEX IF NOT EXISTS india_reference_snapshots_latest
                    ON india_reference_snapshots(provider, fetched_at DESC);
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

    def save_india_reference_snapshot(
        self,
        *,
        provider: str,
        fetched_at: str,
        policy: IndiaMarketPolicy,
    ) -> dict:
        if not provider.strip():
            raise ValueError("Reference-data provider is required")
        payload = json.dumps(
            asdict(policy),
            default=_reference_json_value,
            sort_keys=True,
            separators=(",", ":"),
        )
        snapshot_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        record_counts = {
            "exchanges": len(policy.exchanges),
            "instruments": len(policy.instruments),
            "contracts": len(policy.contracts),
            "sessions": len(policy.sessions),
            "holidays": len(policy.holidays),
            "expiry_calendars": len(policy.expiry_calendars),
            "corporate_actions": len(policy.corporate_actions),
        }
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO india_reference_snapshots
                   (provider, snapshot_hash, fetched_at, record_counts, payload)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(provider, snapshot_hash) DO UPDATE SET
                       fetched_at=excluded.fetched_at""",
                (provider, snapshot_hash, fetched_at, json.dumps(record_counts, sort_keys=True), payload),
            )
        return {"provider": provider, "snapshot_hash": snapshot_hash, "fetched_at": fetched_at, **record_counts}

    def read_latest_india_reference_snapshot(self, provider: str) -> dict | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """SELECT * FROM india_reference_snapshots
                   WHERE provider=? ORDER BY fetched_at DESC LIMIT 1""",
                (provider,),
            ).fetchone()
        return _reference_snapshot_record(row) if row else None

    def read_india_reference_snapshot(self, provider: str, snapshot_hash: str) -> dict | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM india_reference_snapshots WHERE provider=? AND snapshot_hash=?",
                (provider, snapshot_hash),
            ).fetchone()
        return _reference_snapshot_record(row) if row else None

    def save_candles(self, candles: list[Candle]) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.executemany(
                """INSERT OR IGNORE INTO candles
                   (symbol, timeframe, timestamp, open, high, low, close, volume, open_interest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.symbol, c.timeframe, c.timestamp, c.open, c.high, c.low, c.close, c.volume, c.open_interest)
                    for c in candles
                ],
            )
            return cursor.rowcount

    def upsert_candle(self, candle: Candle) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume, open_interest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
                       high=MAX(candles.high, excluded.high),
                       low=MIN(candles.low, excluded.low),
                       close=excluded.close,
                       volume=excluded.volume,
                       open_interest=COALESCE(excluded.open_interest, candles.open_interest)""",
                (candle.symbol, candle.timeframe, candle.timestamp, candle.open, candle.high, candle.low,
                 candle.close, candle.volume, candle.open_interest),
            )

    def read_candles(self, symbol: str, limit: int = 5000, *, timeframe: str | None = None) -> list[Candle]:
        with closing(self._connect()) as connection, connection:
            if timeframe is None:
                rows = connection.execute(
                    "SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC, timeframe LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, timeframe, limit),
                ).fetchall()
        return [
            Candle(
                row["symbol"], row["timestamp"], row["open"], row["high"],
                row["low"], row["close"], row["volume"], row["open_interest"], row["timeframe"],
            )
            for row in reversed(rows)
        ]

    def read_candles_for_timestamps(
        self,
        symbol: str,
        timeframe: str,
        timestamps: list[str],
    ) -> list[Candle]:
        unique_timestamps = list(dict.fromkeys(timestamps))
        rows = []
        with closing(self._connect()) as connection, connection:
            for offset in range(0, len(unique_timestamps), 900):
                batch = unique_timestamps[offset:offset + 900]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(connection.execute(
                    f"SELECT * FROM candles WHERE symbol = ? AND timeframe = ? AND timestamp IN ({placeholders})",
                    (symbol, timeframe, *batch),
                ).fetchall())
        return [
            Candle(
                row["symbol"], row["timestamp"], row["open"], row["high"],
                row["low"], row["close"], row["volume"], row["open_interest"], row["timeframe"],
            )
            for row in rows
        ]

    def read_candles_for_symbols(
        self,
        symbols: list[str],
        *,
        timeframe: str,
        limit_per_symbol: int = 253,
    ) -> dict[str, list[Candle]]:
        if not symbols or limit_per_symbol < 1:
            raise ValueError("Symbols and a positive per-symbol candle limit are required")
        normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
        if not normalized_symbols:
            raise ValueError("At least one non-empty symbol is required")
        placeholders = ",".join("?" for _ in normalized_symbols)
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                f"""SELECT symbol, timeframe, timestamp, open, high, low, close, volume, open_interest
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS candle_rank
                        FROM candles WHERE timeframe=? AND symbol IN ({placeholders})
                    )
                    WHERE candle_rank <= ? ORDER BY symbol, timestamp""",
                (timeframe, *normalized_symbols, limit_per_symbol),
            ).fetchall()
        result: dict[str, list[Candle]] = {symbol: [] for symbol in normalized_symbols}
        for row in rows:
            result[row["symbol"]].append(Candle(
                row["symbol"], row["timestamp"], row["open"], row["high"],
                row["low"], row["close"], row["volume"], row["open_interest"], row["timeframe"],
            ))
        return result

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

    def save_memory_document(self, document: MemoryDocument) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO rag_documents
                   (document_id, kind, text, metadata, embedding, embedding_dimension, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.document_id,
                    document.kind,
                    document.text,
                    json.dumps(document.metadata, sort_keys=True),
                    json.dumps(list(document.embedding)),
                    len(document.embedding),
                    document.created_at,
                ),
            )
            return cursor.rowcount == 1

    def save_prediction(self, record: dict) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO prediction_records
                   (prediction_id, symbol, predicted_probability, outcome, created_at, resolved_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["prediction_id"], record["symbol"], record["predicted_probability"],
                    record.get("outcome"), record["created_at"], record.get("resolved_at"),
                    json.dumps(record.get("metadata", {}), sort_keys=True),
                ),
            )
            return cursor.rowcount == 1

    def save_trade_journal(self, record: dict) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO trade_journal
                   (trade_id, client_order_id, symbol, market, side, quantity, entry_price,
                    exit_price, realized_pnl, status, opened_at, closed_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["trade_id"], record["client_order_id"], record["symbol"], record["market"],
                    record["side"], record["quantity"], record.get("entry_price"), record.get("exit_price"),
                    record.get("realized_pnl"), record["status"], record["opened_at"], record.get("closed_at"),
                    json.dumps(record.get("metadata", {}), sort_keys=True),
                ),
            )
            return cursor.rowcount == 1

    def read_trade_journal(self, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM trade_journal ORDER BY opened_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "trade_id": row["trade_id"], "client_order_id": row["client_order_id"],
                "symbol": row["symbol"], "market": row["market"], "side": row["side"],
                "quantity": row["quantity"], "entry_price": row["entry_price"], "exit_price": row["exit_price"],
                "realized_pnl": row["realized_pnl"], "status": row["status"], "opened_at": row["opened_at"],
                "closed_at": row["closed_at"], "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def search_memory_documents(
        self,
        query_embedding: Iterable[float],
        *,
        limit: int = 10,
        filters: dict | None = None,
        created_after: str | None = None,
    ) -> list[dict]:
        if limit < 1 or limit > 100:
            raise ValueError("Memory search limit must be between one and 100")
        filters = dict(filters or {})
        allowed_filters = {
            "kind", "instrument", "symbol", "sector", "timeframe", "source",
            "event_type", "strategy", "regime",
        }
        unknown = set(filters) - allowed_filters
        if unknown:
            raise ValueError(f"Unsupported memory filters: {', '.join(sorted(unknown))}")
        cutoff = _utc_datetime(created_after) if created_after is not None else None
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT * FROM rag_documents").fetchall()
        matches = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            if cutoff is not None and _utc_datetime(row["created_at"]) < cutoff:
                continue
            if row["kind"] != str(filters.get("kind", row["kind"])).upper():
                continue
            if any(
                str(metadata.get(key, metadata.get("instrument" if key == "symbol" else "symbol" if key == "instrument" else key, ""))).upper()
                != str(value).upper()
                for key, value in filters.items()
                if key != "kind"
            ):
                continue
            score = cosine_similarity(query_embedding, json.loads(row["embedding"]))
            matches.append({
                "document_id": row["document_id"],
                "kind": row["kind"],
                "text": row["text"],
                "metadata": metadata,
                "score": round(score, 6),
                "created_at": row["created_at"],
            })
        return sorted(matches, key=lambda item: (-item["score"], item["created_at"], item["document_id"]))[:limit]

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

    @staticmethod
    def _event_record(row) -> dict:
        return {
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "timestamp": row["timestamp"],
            "correlation_id": row["correlation_id"],
            "source": row["source"],
            "version": row["version"],
            "schema_version": row["schema_version"],
            "payload": json.loads(row["payload"]),
        }

    def read_events(self, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._event_record(row) for row in rows]

    def read_simulation_events(self, mode: str, limit: int = 40) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            run_rows = connection.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY rowid DESC",
                ("simulation.completed",),
            ).fetchall()
            runs = [
                event for event in map(self._event_record, run_rows)
                if event["payload"].get("mode") == mode
            ][:limit]
            correlation_ids = list(dict.fromkeys(event["correlation_id"] for event in runs))
            if not correlation_ids:
                return []
            placeholders = ", ".join("?" for _ in correlation_ids)
            rows = connection.execute(
                f"SELECT * FROM events WHERE correlation_id IN ({placeholders}) ORDER BY rowid DESC LIMIT ?",
                (*correlation_ids, max(limit * 20, limit)),
            ).fetchall()
        return [self._event_record(row) for row in rows]