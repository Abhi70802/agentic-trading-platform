"""Structured news intelligence and non-LLM trade hypothesis guardrails."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from core import Candle, Strategy
from features import feature_snapshot


@dataclass(frozen=True)
class NewsItem:
    article_id: str
    dedupe_key: str
    source: str
    published_at: str
    ingested_at: str
    instruments: tuple[str, ...]
    sector: str | None
    country: str | None
    source_reliability: float
    title: str
    content: str
    original_url: str | None = None


EVENT_RULES = (
    ("GUIDANCE_REDUCTION", ("guidance cut", "cuts guidance", "lowered guidance", "outlook reduced"), "NEGATIVE", "HIGH", "days_to_weeks", 0.88),
    ("EARNINGS_BEAT", ("earnings beat", "beats estimates", "profit rose", "record revenue"), "POSITIVE", "MEDIUM", "days_to_weeks", 0.82),
    ("REGULATORY_ACTION", ("regulatory action", "investigation", "antitrust", "fined", "license revoked"), "NEGATIVE", "HIGH", "weeks_to_months", 0.84),
    ("MANAGEMENT_CHANGE", ("management change", "new ceo", "ceo resignation", "chief executive appointed"), "NEUTRAL", "MEDIUM", "weeks_to_months", 0.74),
    ("MERGER_ACQUISITION", ("acquisition", "merger", "to acquire", "takeover offer"), "POSITIVE", "HIGH", "weeks_to_months", 0.80),
    ("PRODUCT_LAUNCH", ("product launch", "launches new product", "launched a new product"), "POSITIVE", "MEDIUM", "days_to_weeks", 0.72),
    ("LARGE_ORDER", ("large order", "major order", "order win", "wins contract"), "POSITIVE", "HIGH", "days_to_weeks", 0.80),
    ("CREDIT_DOWNGRADE", ("downgrade", "credit rating cut", "default risk"), "NEGATIVE", "HIGH", "weeks_to_months", 0.86),
    ("DIVIDEND_ANNOUNCEMENT", ("dividend announcement", "announces dividend", "dividend declared"), "POSITIVE", "MEDIUM", "days_to_weeks", 0.76),
    ("RATE_DECISION", ("interest rate", "rate decision", "policy rate", "central bank", "monetary policy"), "NEUTRAL", "HIGH", "days_to_weeks", 0.75),
)
NEWS_FRESHNESS_SECONDS = 86_400


def ingest_news(payload: dict, *, ingested_at: str | None = None) -> tuple[NewsItem, dict]:
    source = str(payload.get("source", "")).strip()
    title = str(payload.get("title", "")).strip()
    content = str(payload.get("content", "")).strip()
    if not source or not title or not content:
        raise ValueError("News source, title and content are required")
    reliability = float(payload.get("source_reliability", 0.5))
    if not 0 <= reliability <= 1:
        raise ValueError("source_reliability must be between zero and one")
    original_url = payload.get("original_url")
    if original_url:
        parsed_url = urlparse(str(original_url))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("original_url must be an absolute HTTP or HTTPS URL")
    published = _iso(payload.get("published_at"))
    received = _iso(ingested_at or datetime.now(timezone.utc).isoformat())
    normalized = re.sub(r"\W+", " ", f"{title} {content}".casefold()).strip()
    dedupe_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    symbols = tuple(sorted({str(item).strip().upper() for item in payload.get("instruments", []) if str(item).strip()}))
    item = NewsItem(
        article_id=str(payload.get("article_id") or uuid4()),
        dedupe_key=dedupe_key,
        source=source,
        published_at=published,
        ingested_at=received,
        instruments=symbols,
        sector=payload.get("sector"),
        country=payload.get("country"),
        source_reliability=reliability,
        title=title,
        content=content,
        original_url=str(original_url) if original_url else None,
    )
    return item, classify_news(item)


def classify_news(item: NewsItem) -> dict:
    text = f"{item.title} {item.content}".casefold()
    matched_rules = [rule for rule in EVENT_RULES if any(phrase in text for phrase in rule[1])]
    extracted_events = []
    for event_rule in matched_rules:
        event_type, _, sentiment, severity, horizon, base_confidence = event_rule
        extracted_events.append({
            "event_type": event_type,
            "sentiment": sentiment,
            "severity": severity,
            "expected_horizon": horizon,
            "confidence": round(base_confidence * (0.5 + item.source_reliability / 2), 3),
        })
    if extracted_events:
        primary = extracted_events[0]
        event_type = primary["event_type"]
        sentiment = primary["sentiment"]
        severity = primary["severity"]
        horizon = primary["expected_horizon"]
        confidence = primary["confidence"]
    else:
        event_type, sentiment, severity, horizon, confidence = "UNCLASSIFIED", "NEUTRAL", "LOW", "unknown", 0.0
    publication_lag = (
        datetime.fromisoformat(item.ingested_at) - datetime.fromisoformat(item.published_at)
    ).total_seconds()
    return {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "severity": severity,
        "sentiment": sentiment,
        "expected_horizon": horizon,
        "instruments": list(item.instruments),
        "sector": item.sector,
        "country": item.country,
        "confidence": confidence,
        "extracted_events": extracted_events,
        "source_reliability": item.source_reliability,
        "publication_lag_seconds": round(publication_lag, 3),
        "is_stale": publication_lag < 0 or publication_lag > NEWS_FRESHNESS_SECONDS,
        "evidence": [{"article_id": item.article_id, "source": item.source, "published_at": item.published_at}],
    }


def build_hypothesis(candles: list[Candle], strategy: Strategy, news_events: list[dict] | None = None) -> dict:
    if not candles:
        raise ValueError("No market data is available")
    ordered = sorted(candles, key=lambda item: _utc_datetime(item.timestamp))
    snapshot = feature_snapshot(ordered)
    action = strategy.action(ordered)
    latest = ordered[-1]
    decision_time = _utc_datetime(latest.timestamp)
    features = snapshot["features"]
    atr = features["atr_14"]
    usable_news = []
    for event in news_events or []:
        if latest.symbol not in event.get("instruments", []) or event.get("is_stale", True):
            continue
        eligible_evidence = []
        for evidence in event.get("evidence", []):
            if not isinstance(evidence, dict) or not isinstance(evidence.get("published_at"), str):
                continue
            try:
                published_at = _utc_datetime(evidence["published_at"])
            except ValueError:
                continue
            if published_at <= decision_time:
                eligible_evidence.append(evidence)
        if eligible_evidence:
            usable_news.append({**event, "evidence": eligible_evidence})
    supporting = [{"source_id": f"candle:{latest.symbol}:{latest.timestamp}", "claim": "Latest normalized OHLCV bar used for the signal"}]
    supporting.extend(
        {"source_id": evidence["article_id"], "claim": f"{event['event_type']} · {event['sentiment']}"}
        for event in usable_news
        for evidence in event.get("evidence", [])
    )
    contradictory = [
        {"source_id": evidence["article_id"], "claim": f"{event['event_type']} has negative classified impact"}
        for event in usable_news if event.get("sentiment") == "NEGATIVE"
        for evidence in event.get("evidence", [])
    ]
    if action != "BUY" or atr is None:
        return {
            "hypothesis_id": str(uuid4()),
            "instrument": latest.symbol,
            "market": "UNCONFIGURED",
            "direction": "NONE",
            "time_horizon": "signal_dependent",
            "entry_range": None,
            "stop_loss": None,
            "take_profit": None,
            "thesis": "No actionable long entry: the deterministic SMA crossover has not produced a BUY signal.",
            "supporting_evidence": supporting,
            "contradictory_evidence": contradictory,
            "invalidating_conditions": ["Signal is no longer BUY", "Market data is stale or fails quality checks"],
            "confidence": 0.0,
            "expected_risk": None,
            "expected_reward": None,
            "data_timestamp": latest.timestamp,
            "data_source_ids": [f"candle:{latest.symbol}:{latest.timestamp}"],
            "strategy": "sma_cross",
            "regime": snapshot["regime"],
            "llm_used": False,
        }
    stop = max(0.01, latest.close - 2 * atr)
    target = latest.close + 4 * atr
    confidence = 0.65 if snapshot["regime"]["trend"] == "UPTREND" else 0.55
    if contradictory:
        confidence = max(0.35, confidence - 0.15)
    return {
        "hypothesis_id": str(uuid4()),
        "instrument": latest.symbol,
        "market": "UNCONFIGURED",
        "direction": "LONG",
        "time_horizon": "days_to_weeks",
        "entry_range": {"low": round(latest.close * 0.995, 4), "high": round(latest.close * 1.005, 4)},
        "stop_loss": round(stop, 4),
        "take_profit": round(target, 4),
        "thesis": "A deterministic moving-average crossover generated a long candidate; this is a hypothesis, not an execution instruction.",
        "supporting_evidence": supporting,
        "contradictory_evidence": contradictory,
        "invalidating_conditions": ["Close falls below the proposed stop", "Crossover reverses", "Market data is stale or fails quality checks"],
        "confidence": confidence,
        "expected_risk": round(latest.close - stop, 4),
        "expected_reward": round(target - latest.close, 4),
        "data_timestamp": latest.timestamp,
        "data_source_ids": [f"candle:{latest.symbol}:{latest.timestamp}", *[e["source_id"] for e in supporting[1:]]],
        "strategy": "sma_cross",
        "regime": snapshot["regime"],
        "llm_used": False,
    }


def validate_hypothesis(
    hypothesis: dict,
    *,
    now: datetime | None = None,
    max_data_age_seconds: int = 300,
    minimum_confidence: float = 0.5,
    kill_switch: bool = False,
) -> dict:
    reasons: list[str] = []
    timestamp = _iso(hypothesis.get("data_timestamp"))
    age_seconds = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(timestamp)).total_seconds()
    source_ids = hypothesis.get("data_source_ids") or [
        item.get("source_id")
        for item in hypothesis.get("supporting_evidence", [])
        if isinstance(item, dict) and item.get("source_id")
    ]
    if hypothesis.get("direction") != "LONG":
        reasons.append("No actionable LONG hypothesis")
    if not hypothesis.get("instrument") or not source_ids:
        reasons.append("Instrument or source evidence is missing")
    if age_seconds < 0 or age_seconds > max_data_age_seconds:
        reasons.append("Market data is stale or timestamp is in the future")
    if float(hypothesis.get("confidence", 0)) < minimum_confidence:
        reasons.append("Confidence is below the configured minimum")
    if not hypothesis.get("supporting_evidence"):
        reasons.append("Supporting evidence is required")
    stop = hypothesis.get("stop_loss")
    target = hypothesis.get("take_profit")
    entry = hypothesis.get("entry_range")
    if not stop or not target or not isinstance(entry, dict):
        reasons.append("Entry range, stop-loss and take-profit are required")
    elif entry.get("low", 0) <= stop or target <= entry.get("high", 0):
        reasons.append("Risk levels do not define a valid long reward/risk range")
    if kill_switch:
        reasons.append("Emergency kill switch is active")
    return {
        "approved": not reasons,
        "reasons": reasons,
        "age_seconds": round(age_seconds, 3),
        "quantity": 0,
        "execution_enabled": False,
        "note": "Validation does not authorize or submit an order.",
    }


def _iso(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("An ISO-8601 timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)