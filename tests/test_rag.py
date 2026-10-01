import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pytest

from rag import HashingEmbedder, HistoricalTradingMemory, cosine_similarity
from store import EventStore


@pytest.mark.component
class RagTests(unittest.TestCase):
    def test_hashing_embeddings_are_normalized_and_reproducible(self):
        embedder = HashingEmbedder(64)
        first = embedder.embed("NIFTY earnings beat estimates")
        second = embedder.embed("NIFTY earnings beat estimates")
        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine_similarity(first, second), 1.0, places=6)
        self.assertEqual(len(first), 64)

    def test_memory_persists_and_filters_by_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = HistoricalTradingMemory(EventStore(Path(directory) / "rag.db"))
            self.assertTrue(memory.remember(
                document_id="trade-1",
                kind="HISTORICAL_TRADE",
                text="NIFTY breakout after earnings beat with controlled risk",
                metadata={"instrument": "NIFTY", "timeframe": "1d", "source": "backtest"},
            ))
            self.assertTrue(memory.remember(
                document_id="trade-2",
                kind="HISTORICAL_TRADE",
                text="BANKNIFTY reversal after negative news",
                metadata={"instrument": "BANKNIFTY", "timeframe": "1d", "source": "backtest"},
            ))
            self.assertFalse(memory.remember(
                document_id="trade-1",
                kind="HISTORICAL_TRADE",
                text="duplicate document",
            ))
            results = memory.recall(
                "NIFTY earnings breakout",
                filters={"instrument": "NIFTY", "timeframe": "1d"},
            )
            self.assertEqual([item["document_id"] for item in results], ["trade-1"])
            self.assertGreater(results[0]["score"], 0)
            self.assertNotIn("embedding", results[0])

    def test_unknown_filters_and_empty_text_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = HistoricalTradingMemory(EventStore(Path(directory) / "rag.db"))
            with self.assertRaisesRegex(ValueError, "searchable content"):
                memory.recall("   ")
            with self.assertRaisesRegex(ValueError, "Unsupported memory filters"):
                memory.recall("query", filters={"account": "x"})

    def test_recall_filters_symbol_sector_strategy_and_regime_before_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = HistoricalTradingMemory(EventStore(Path(directory) / "filtered-rag.db"))
            documents = (
                ("reliance-momentum-bull", "HISTORICAL_TRADE", {"symbol": "RELIANCE", "sector": "ENERGY", "strategy": "MOMENTUM", "regime": "BULLISH"}),
                ("hdfc-momentum-bull", "HISTORICAL_TRADE", {"symbol": "HDFCBANK", "sector": "FINANCIALS", "strategy": "MOMENTUM", "regime": "BULLISH"}),
                ("reliance-momentum-bear", "HISTORICAL_TRADE", {"instrument": "RELIANCE", "sector": "ENERGY", "strategy": "MOMENTUM", "regime": "BEARISH"}),
                ("reliance-news", "NEWS_EVENT", {"symbol": "RELIANCE", "sector": "ENERGY", "strategy": "MOMENTUM", "regime": "BULLISH", "event_type": "EARNINGS_BEAT"}),
                ("infy-mean-reversion", "HISTORICAL_TRADE", {"symbol": "INFY", "sector": "IT", "strategy": "MEAN_REVERSION", "regime": "BULLISH"}),
            )
            for document_id, kind, metadata in documents:
                self.assertTrue(memory.remember(
                    document_id=document_id,
                    kind=kind,
                    text="RELIANCE momentum bullish breakout earnings setup",
                    metadata=metadata,
                    created_at="2026-10-01T09:00:00Z",
                ))

            query = "RELIANCE momentum bullish breakout earnings setup"
            target = memory.recall(query, filters={
                "symbol": "RELIANCE",
                "strategy": "MOMENTUM",
                "regime": "BULLISH",
                "kind": "HISTORICAL_TRADE",
            })

            self.assertEqual([item["document_id"] for item in target], ["reliance-momentum-bull"])
            self.assertEqual(
                {item["document_id"] for item in memory.recall(query, filters={"sector": "ENERGY"})},
                {"reliance-momentum-bull", "reliance-momentum-bear", "reliance-news"},
            )
            self.assertEqual(
                {item["document_id"] for item in memory.recall(query, filters={"strategy": "MEAN_REVERSION"})},
                {"infy-mean-reversion"},
            )
            self.assertEqual(
                {item["document_id"] for item in memory.recall(query, filters={"event_type": "EARNINGS_BEAT"})},
                {"reliance-news"},
            )
            self.assertEqual(memory.recall(query, filters={"symbol": "UNKNOWN"}), [])

    def test_empty_and_stale_memory_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = HistoricalTradingMemory(EventStore(Path(directory) / "stale-rag.db"))
            query = "RELIANCE momentum trade setup"
            self.assertEqual(memory.recall(query), [])

            memory.remember(
                document_id="old-trade",
                kind="HISTORICAL_TRADE",
                text=query,
                metadata={"symbol": "RELIANCE"},
                created_at="2026-09-01T09:00:00Z",
            )
            memory.remember(
                document_id="recent-trade",
                kind="HISTORICAL_TRADE",
                text=query,
                metadata={"symbol": "RELIANCE"},
                created_at="2026-10-01T09:59:30Z",
            )

            results = memory.recall(
                query,
                filters={"symbol": "RELIANCE"},
                max_age_seconds=60,
                now=datetime.fromisoformat("2026-10-01T10:00:00+00:00"),
            )
            self.assertEqual([item["document_id"] for item in results], ["recent-trade"])


if __name__ == "__main__":
    unittest.main()
