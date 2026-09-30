import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import Candle
from features import feature_snapshot
from intelligence import build_hypothesis, ingest_news, validate_hypothesis
from marketdata import JsonCandleAdapter, MarketDataQualityGate
from store import EventStore


def bars(count=60):
    result = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        close = 100 + index
        result.append(Candle(
            "TEST", (base + timedelta(days=index)).isoformat(), close - 0.2,
            close + 0.5, close - 0.5, close, 1000 + index * 10,
        ))
    return result


class AlwaysBuy:
    def action(self, history):
        return "BUY"


class ResearchServiceTests(unittest.TestCase):
    def test_provider_adapter_normalizes_aliases_and_millisecond_timestamps(self):
        event = JsonCandleAdapter("provider-a").normalize({
            "ticker": "test",
            "t": 1735689600000,
            "o": 10,
            "h": 12,
            "l": 9,
            "c": 11,
            "v": 100,
            "seq": 7,
        })
        self.assertEqual(event.candle.symbol, "TEST")
        self.assertEqual(event.candle.timestamp, "2025-01-01T00:00:00+00:00")
        self.assertEqual(event.sequence, 7)
        self.assertEqual(event.as_event()["schema_version"], "1.0")

    def test_adapter_rejects_non_finite_prices(self):
        with self.assertRaises(ValueError):
            JsonCandleAdapter("provider-a").normalize({
                "symbol": "TEST", "timestamp": "2025-01-01T00:00:00Z",
                "open": float("nan"), "high": 12, "low": 9, "close": 11,
            })

    def test_quality_gate_rejects_duplicates_and_out_of_order_messages(self):
        adapter = JsonCandleAdapter("provider-a")
        gate = MarketDataQualityGate()
        first = adapter.normalize({
            "symbol": "TEST", "timestamp": "2025-01-02T00:00:00Z",
            "open": 10, "high": 12, "low": 9, "close": 11, "sequence": 2,
        })
        self.assertIsNone(gate.review(first))
        self.assertEqual(gate.review(first), "duplicate_event")
        older = adapter.normalize({
            "symbol": "TEST", "timestamp": "2025-01-01T00:00:00Z",
            "open": 10, "high": 12, "low": 9, "close": 11, "sequence": 3,
        })
        self.assertEqual(gate.review(older), "out_of_order_timestamp")
        self.assertEqual(gate.status()["duplicates"], 1)
        self.assertEqual(gate.status()["out_of_order"], 1)

    def test_market_freshness_uses_explicit_timestamp_window(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        fresh = (now - timedelta(seconds=10)).isoformat()
        stale = (now - timedelta(seconds=60)).isoformat()
        self.assertTrue(MarketDataQualityGate.is_fresh(fresh, now=now, max_age_seconds=30))
        self.assertFalse(MarketDataQualityGate.is_fresh(stale, now=now, max_age_seconds=30))

    def test_feature_snapshot_returns_indicators_and_regime(self):
        snapshot = feature_snapshot(bars())
        self.assertEqual(snapshot["bar_count"], 60)
        self.assertAlmostEqual(snapshot["features"]["sma_20"], 149.5)
        self.assertEqual(snapshot["features"]["rsi_14"], 100.0)
        self.assertEqual(snapshot["regime"]["trend"], "UPTREND")
        self.assertEqual(snapshot["regime"]["risk_posture"], "RISK_ON")
        self.assertIsNotNone(snapshot["features"]["atr_14"])

    def test_insufficient_history_is_explicit_not_fabricated(self):
        snapshot = feature_snapshot(bars(5))
        self.assertIsNone(snapshot["features"]["sma_20"])
        self.assertEqual(snapshot["regime"]["trend"], "INSUFFICIENT_DATA")

    def test_news_classification_keeps_publication_and_ingestion_times(self):
        item, event = ingest_news({
            "article_id": "news-1",
            "source": "licensed-feed",
            "published_at": "2025-02-01T09:00:00+05:30",
            "instruments": ["test"],
            "source_reliability": 0.9,
            "title": "Company beats estimates",
            "content": "Quarterly earnings beat analyst estimates.",
        }, ingested_at="2025-02-01T04:00:00+00:00")
        self.assertEqual(item.published_at, "2025-02-01T03:30:00+00:00")
        self.assertEqual(item.ingested_at, "2025-02-01T04:00:00+00:00")
        self.assertEqual(event["event_type"], "EARNINGS_BEAT")
        self.assertGreater(event["confidence"], 0.7)
        self.assertEqual(event["evidence"][0]["article_id"], "news-1")
        self.assertFalse(event["is_stale"])

    def test_stale_news_is_flagged_and_not_used_as_current_hypothesis_evidence(self):
        _, event = ingest_news({
            "article_id": "stale-news",
            "source": "licensed-feed",
            "published_at": "2025-01-01T00:00:00Z",
            "instruments": ["TEST"],
            "source_reliability": 1,
            "title": "Company beats estimates",
            "content": "Quarterly earnings beat analyst estimates.",
        }, ingested_at="2025-01-03T00:00:00Z")
        self.assertTrue(event["is_stale"])
        proposal = build_hypothesis(bars(), AlwaysBuy(), [event])
        self.assertNotIn("stale-news", proposal["data_source_ids"])

    def test_news_deduplicates_across_sources_and_persists_intelligence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "news.db")
            common = {
                "published_at": "2025-02-01T09:00:00Z",
                "title": "Company beats estimates",
                "content": "Quarterly earnings beat analyst estimates.",
                "source_reliability": 0.8,
            }
            first, first_event = ingest_news({**common, "source": "wire-a"})
            second, _ = ingest_news({**common, "source": "wire-b"})
            first_record = {**first.__dict__, "instruments": list(first.instruments), "intelligence": first_event}
            self.assertTrue(store.save_news(first_record))
            second_record = {**second.__dict__, "instruments": list(second.instruments), "intelligence": first_event}
            self.assertFalse(store.save_news(second_record))
            self.assertEqual(store.read_news()[0]["intelligence"]["event_type"], "EARNINGS_BEAT")

    def test_news_source_url_is_validated_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "source-url.db")
            item, intelligence = ingest_news({
                "source": "exchange",
                "published_at": "2025-02-01T09:00:00Z",
                "title": "NSE company filing",
                "content": "Company disclosed an exchange filing.",
                "original_url": "https://example.exchange/filing/123",
            })
            record = {
                **item.__dict__,
                "instruments": list(item.instruments),
                "intelligence": intelligence,
            }
            self.assertTrue(store.save_news(record))
            self.assertEqual(store.read_news()[0]["original_url"], "https://example.exchange/filing/123")
            with self.assertRaisesRegex(ValueError, "HTTP or HTTPS"):
                ingest_news({
                    "source": "exchange",
                    "published_at": "2025-02-01T09:00:00Z",
                    "title": "Bad link",
                    "content": "Testing invalid source link.",
                    "original_url": "javascript:alert(1)",
                })

    def test_hypothesis_has_evidence_and_guard_rejects_stale_data(self):
        candles = bars()
        proposal = build_hypothesis(candles, AlwaysBuy())
        self.assertEqual(proposal["direction"], "LONG")
        self.assertTrue(proposal["supporting_evidence"])
        result = validate_hypothesis(proposal, now=datetime.now(timezone.utc))
        self.assertFalse(result["approved"])
        self.assertIn("Market data is stale or timestamp is in the future", result["reasons"])
        self.assertFalse(result["execution_enabled"])

    def test_hypothesis_validation_approves_fresh_context_but_never_execution(self):
        candles = bars()
        proposal = build_hypothesis(candles, AlwaysBuy())
        current_time = datetime.fromisoformat(proposal["data_timestamp"])
        result = validate_hypothesis(proposal, now=current_time)
        self.assertTrue(result["approved"])
        self.assertEqual(result["quantity"], 0)
        self.assertFalse(result["execution_enabled"])


if __name__ == "__main__":
    unittest.main()