import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_agents import AdversarialAgent
from core import Candle
from features import MULTI_TIMEFRAME_ANALYSIS, feature_snapshot, multi_timeframe_feature_snapshot
from intelligence import build_hypothesis, ingest_news, validate_hypothesis
from marketdata import JsonCandleAdapter, MarketDataQualityGate
from store import EventStore


def bars(count=60, timeframe="legacy"):
    result = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        close = 100 + index
        result.append(Candle(
            "TEST", (base + timedelta(days=index)).isoformat(), close - 0.2,
            close + 0.5, close - 0.5, close, 1000 + index * 10, timeframe=timeframe,
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
        self.assertEqual(event.as_event()["schema_version"], "1.1")

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
        self.assertEqual(snapshot["timeframe"], "legacy")
        self.assertAlmostEqual(snapshot["features"]["sma_20"], 149.5)
        self.assertEqual(snapshot["features"]["rsi_14"], 100.0)
        self.assertEqual(snapshot["regime"]["trend"], "UPTREND")
        self.assertEqual(snapshot["regime"]["risk_posture"], "RISK_ON")
        self.assertIsNotNone(snapshot["features"]["atr_14"])

    def test_multi_timeframe_features_keep_frames_separate_and_report_missing(self):
        daily = bars(timeframe="1d")
        hourly = bars(30, timeframe="1h")
        snapshot = multi_timeframe_feature_snapshot({"1d": daily, "1h": hourly})
        self.assertEqual(snapshot["symbol"], "TEST")
        self.assertEqual(set(snapshot["timeframes"]), {"1d", "1h"})
        self.assertEqual(snapshot["timeframes"]["1d"]["timeframe"], "1d")
        self.assertEqual(snapshot["timeframes"]["1h"]["timeframe"], "1h")
        self.assertEqual(snapshot["missing_timeframes"], ["15m", "5m", "1m"])
        self.assertEqual(MULTI_TIMEFRAME_ANALYSIS, ("1d", "1h", "15m", "5m", "1m"))

    def test_feature_snapshot_rejects_mixed_timeframes(self):
        with self.assertRaisesRegex(ValueError, "exactly one timeframe"):
            feature_snapshot([bars(1)[0], bars(1, timeframe="1d")[0]])

    def test_insufficient_history_is_explicit_not_fabricated(self):
        snapshot = feature_snapshot(bars(5))
        self.assertIsNone(snapshot["features"]["sma_20"])
        self.assertEqual(snapshot["regime"]["trend"], "INSUFFICIENT_DATA")

    def test_news_classification_keeps_publication_and_ingestion_times(self):
        item, event = ingest_news({
            "article_id": "news-1",
            "source": "licensed-feed",
            "published_at": "2025-02-01T09:00:00+05:30",
            "instruments": ["test", "TEST", " "],
            "sector": "IT",
            "country": "IN",
            "source_reliability": 0.9,
            "title": "Company beats estimates",
            "content": "Quarterly earnings beat analyst estimates.",
        }, ingested_at="2025-02-01T04:00:00+00:00")
        self.assertEqual(item.published_at, "2025-02-01T03:30:00+00:00")
        self.assertEqual(item.ingested_at, "2025-02-01T04:00:00+00:00")
        self.assertEqual(item.source, "licensed-feed")
        self.assertEqual(item.instruments, ("TEST",))
        self.assertEqual(event["event_type"], "EARNINGS_BEAT")
        self.assertEqual(event["instruments"], ["TEST"])
        self.assertEqual(event["sector"], "IT")
        self.assertEqual(event["country"], "IN")
        self.assertGreater(event["confidence"], 0.7)
        self.assertEqual(event["evidence"][0]["article_id"], "news-1")
        self.assertEqual(event["evidence"][0]["source"], "licensed-feed")
        self.assertFalse(event["is_stale"])

        _, future_event = ingest_news({
            "source": "licensed-feed",
            "published_at": "2025-02-01T04:01:00Z",
            "title": "Company beats estimates",
            "content": "A future-dated article for freshness classification.",
        }, ingested_at="2025-02-01T04:00:00Z")
        self.assertTrue(future_event["is_stale"])

    def test_news_extraction_preserves_multiple_event_types(self):
        _, event = ingest_news({
            "source": "licensed-feed",
            "published_at": "2025-02-01T09:00:00Z",
            "title": "Company beats estimates despite regulatory investigation and guidance cut",
            "content": "Profit rose, but the regulator opened an investigation and management cut guidance.",
            "source_reliability": 1,
        })
        self.assertEqual(event["event_type"], "GUIDANCE_REDUCTION")
        self.assertEqual(
            {item["event_type"] for item in event["extracted_events"]},
            {"EARNINGS_BEAT", "REGULATORY_ACTION", "GUIDANCE_REDUCTION"},
        )
        event_sentiments = {item["event_type"]: item["sentiment"] for item in event["extracted_events"]}
        self.assertEqual(event_sentiments["EARNINGS_BEAT"], "POSITIVE")
        self.assertEqual(event_sentiments["GUIDANCE_REDUCTION"], "NEGATIVE")

    def test_synthetic_news_examples_extract_entities_and_expected_event_metadata(self):
        examples = (
            ("positive earnings surprise", "Reliance beats estimates", "Quarterly earnings beat analyst estimates.", "RELIANCE", "Energy", "EARNINGS_BEAT", "POSITIVE", "MEDIUM", 0.82),
            ("negative earnings guidance", "Reliance cuts guidance", "Management lowered the outlook for the year.", "RELIANCE", "Energy", "GUIDANCE_REDUCTION", "NEGATIVE", "HIGH", 0.88),
            ("regulatory action", "SEBI opens investigation into Reliance", "The regulator announced a formal investigation.", "RELIANCE", "Energy", "REGULATORY_ACTION", "NEGATIVE", "HIGH", 0.84),
            ("management change", "TCS management change announced", "A new chief executive will take office.", "TCS", "IT", "MANAGEMENT_CHANGE", "NEUTRAL", "MEDIUM", 0.74),
            ("acquisition announcement", "Reliance announces acquisition", "The company agreed to acquire a competitor.", "RELIANCE", "Energy", "MERGER_ACQUISITION", "POSITIVE", "HIGH", 0.80),
            ("product launch", "TCS product launch", "The company launched a new product.", "TCS", "IT", "PRODUCT_LAUNCH", "POSITIVE", "MEDIUM", 0.72),
            ("large order announcement", "Larsen wins large order", "The company secured a major order.", "LT", "Industrials", "LARGE_ORDER", "POSITIVE", "HIGH", 0.80),
            ("credit downgrade", "Agency announces credit rating cut", "The issuer received a credit downgrade.", "RELIANCE", "Energy", "CREDIT_DOWNGRADE", "NEGATIVE", "HIGH", 0.86),
            ("dividend announcement", "Reliance dividend announcement", "The board declared a dividend.", "RELIANCE", "Energy", "DIVIDEND_ANNOUNCEMENT", "POSITIVE", "MEDIUM", 0.76),
            ("macro policy announcement", "RBI policy rate decision announced", "The central bank announced its policy rate.", "NIFTY", "Index", "RATE_DECISION", "NEUTRAL", "HIGH", 0.75),
        )

        for name, title, content, symbol, sector, event_type, sentiment, severity, base_confidence in examples:
            with self.subTest(example=name):
                item, event = ingest_news({
                    "source": "licensed-feed",
                    "published_at": "2025-02-01T09:00:00+05:30",
                    "instruments": [symbol.lower(), f" {symbol.lower()} ", ""],
                    "sector": sector,
                    "source_reliability": 0.8,
                    "title": title,
                    "content": content,
                }, ingested_at="2025-02-01T04:00:00Z")

                self.assertEqual(item.instruments, (symbol,))
                self.assertEqual(event["instruments"], [symbol])
                self.assertEqual(event["sector"], sector)
                self.assertEqual(event["event_type"], event_type)
                self.assertEqual(event["sentiment"], sentiment)
                self.assertEqual(event["severity"], severity)
                self.assertEqual(event["confidence"], round(base_confidence * 0.9, 3))
                self.assertEqual(item.published_at, "2025-02-01T03:30:00+00:00")
                self.assertEqual(event["evidence"][0]["published_at"], item.published_at)

    def test_positive_earnings_and_negative_guidance_both_remain_in_hypothesis(self):
        _, positive_event = ingest_news({
            "article_id": "positive-earnings",
            "source": "wire-a",
            "published_at": "2025-02-27T08:00:00Z",
            "instruments": ["TEST"],
            "title": "Company beats estimates",
            "content": "Quarterly earnings beat analyst estimates.",
            "source_reliability": 1,
        }, ingested_at="2025-02-27T08:05:00Z")
        _, negative_event = ingest_news({
            "article_id": "negative-guidance",
            "source": "wire-b",
            "published_at": "2025-02-28T08:00:00Z",
            "instruments": ["TEST"],
            "title": "Company cuts guidance",
            "content": "Management lowered the full-year outlook.",
            "source_reliability": 1,
        }, ingested_at="2025-02-28T08:05:00Z")

        baseline = build_hypothesis(bars(), AlwaysBuy())
        proposal = build_hypothesis(bars(), AlwaysBuy(), [positive_event, negative_event])

        self.assertIn("positive-earnings", proposal["data_source_ids"])
        self.assertIn("negative-guidance", proposal["data_source_ids"])
        self.assertEqual(
            {item["source_id"] for item in proposal["contradictory_evidence"]},
            {"negative-guidance"},
        )
        self.assertEqual(proposal["confidence"], baseline["confidence"] - 0.15)
        review = AdversarialAgent().review(
            proposal,
            {"evidence_ids": proposal["data_source_ids"]},
            now=datetime.fromisoformat(proposal["data_timestamp"]),
        )
        self.assertEqual(proposal["confidence"], 0.5)
        self.assertTrue(review["approved"], review)
        self.assertFalse(review["execution_authority"])

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

    def test_news_published_after_historical_decision_cannot_influence_hypothesis(self):
        _, before_event = ingest_news({
            "article_id": "before-decision",
            "source": "wire-a",
            "published_at": "2025-02-28T23:59:00Z",
            "instruments": ["TEST"],
            "title": "Company beats estimates",
            "content": "Quarterly earnings beat estimates.",
        }, ingested_at="2025-03-01T00:30:00Z")
        _, after_event = ingest_news({
            "article_id": "after-decision",
            "source": "wire-b",
            "published_at": "2025-03-01T00:01:00Z",
            "instruments": ["TEST"],
            "title": "Company guidance cut",
            "content": "The company cuts guidance for the coming year.",
        }, ingested_at="2025-03-01T00:30:00Z")

        self.assertFalse(after_event["is_stale"])
        proposal = build_hypothesis(bars(), AlwaysBuy(), [
            before_event,
            after_event,
        ])

        self.assertIn("before-decision", proposal["data_source_ids"])
        self.assertNotIn("after-decision", proposal["data_source_ids"])
        self.assertNotIn("after-decision", [
            item["source_id"] for item in proposal["contradictory_evidence"]
        ])
        self.assertEqual(proposal["confidence"], build_hypothesis(bars(), AlwaysBuy())["confidence"])

        later_source = {
            **before_event["evidence"][0],
            "article_id": "later-source-copy",
            "source": "wire-b",
            "published_at": "2025-03-01T00:01:00+00:00",
        }
        combined_event = {**before_event, "evidence": [before_event["evidence"][0], later_source]}
        combined_proposal = build_hypothesis(bars(), AlwaysBuy(), [combined_event])
        self.assertIn("before-decision", combined_proposal["data_source_ids"])
        self.assertNotIn("later-source-copy", combined_proposal["data_source_ids"])

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
            stored = store.read_news()[0]
            self.assertEqual(stored["intelligence"]["event_type"], "EARNINGS_BEAT")
            self.assertEqual(stored["source"], "wire-a")
            self.assertEqual(first_event["evidence"][0]["source"], "wire-a")

    def test_news_storage_orders_out_of_order_ingestion_by_publication_time(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "out-of-order-news.db")
            for article_id, published_at in (
                ("newer-article", "2025-02-03T09:00:00Z"),
                ("older-article", "2025-02-01T09:00:00Z"),
            ):
                item, intelligence = ingest_news({
                    "article_id": article_id,
                    "source": "licensed-feed",
                    "published_at": published_at,
                    "title": f"Article {article_id}",
                    "content": f"Distinct content for {article_id}.",
                }, ingested_at="2025-02-03T10:00:00Z")
                record = {**item.__dict__, "instruments": list(item.instruments), "intelligence": intelligence}
                self.assertTrue(store.save_news(record))

            self.assertEqual(
                [item["article_id"] for item in store.read_news()],
                ["newer-article", "older-article"],
            )

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