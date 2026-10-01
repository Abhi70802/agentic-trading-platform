import unittest
from datetime import datetime, time, timezone

import pytest

from compliance import MarketCompliancePolicy
from core import RiskConfig, RiskEngine
from execution import ExecutionService, PaperBrokerAdapter
from paper_trading import LivePaperTradingSession, PaperTradeDecision
from portfolio import PortfolioSnapshot, PositionSizer, SizingConfig


class Tick:
    timestamp = "2026-10-01T10:00:00+00:00"


def policy():
    return MarketCompliancePolicy(
        market="NSE", timezone_name="Asia/Kolkata", session_open=time(9, 15), session_close=time(15, 30),
        require_exchange_calendar=True,
    )


def service():
    return ExecutionService(
        risk_engine=RiskEngine(RiskConfig(max_position_fraction=0.2, max_exposure_fraction=0.5)),
        position_sizer=PositionSizer(SizingConfig(max_loss_fraction=0.01)),
        paper_broker=PaperBrokerAdapter(0),
    )


@pytest.mark.component
class PaperTradingTests(unittest.TestCase):
    def decision(self, approved=True):
        return PaperTradeDecision("paper-1", "TEST", "NSE", 100, 90, approved, "decision-1")

    @pytest.mark.chaos
    def test_stale_live_data_never_reaches_decision_or_execution(self):
        session = LivePaperTradingSession(
            execution=service(), portfolio=PortfolioSnapshot(100_000, ()), policy=policy(), paper_mode_enabled=True,
        )
        called = []
        result = session.on_tick(
            type("StaleTick", (), {"timestamp": "2026-09-30T10:00:00+00:00"})(),
            decision_factory=lambda *_: called.append(True),
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(called, [])

    def test_live_decision_requires_explicit_paper_mode_and_human_approval(self):
        session = LivePaperTradingSession(
            execution=service(), portfolio=PortfolioSnapshot(100_000, ()), policy=policy(), paper_mode_enabled=False,
        )
        disabled = session.on_tick(Tick(), decision_factory=lambda *_: self.decision(), now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))
        self.assertEqual(disabled["status"], "PAPER_MODE_DISABLED")

        session.paper_mode_enabled = True
        session.news_provider = lambda _symbol: []
        pending = session.on_tick(Tick(), decision_factory=lambda *_: self.decision(), now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc), calendar_session=True)
        self.assertEqual(pending["status"], "PENDING_APPROVAL")
        self.assertFalse(pending["execution"])

    def test_approved_paper_mode_runs_simulated_fill_only(self):
        session = LivePaperTradingSession(
            execution=service(), portfolio=PortfolioSnapshot(100_000, ()), policy=policy(), paper_mode_enabled=True, human_approval=True,
            news_provider=lambda _symbol: [],
        )
        result = session.on_tick(Tick(), decision_factory=lambda *_: self.decision(), now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc), calendar_session=True)
        self.assertEqual(result["status"], "FILLED")
        self.assertEqual(result["broker"], "paper_only")
        self.assertGreater(result["filled_quantity"], 0)
        self.assertEqual(session.last_order.average_fill_price, 100)

    def test_pending_execution_cancels_when_breaking_news_arrives_after_signal(self):
        t1 = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)
        t2 = t1.replace(second=5)
        t3 = t1.replace(second=10)
        news = []
        session = LivePaperTradingSession(
            execution=service(),
            portfolio=PortfolioSnapshot(100_000, ()),
            policy=policy(),
            paper_mode_enabled=True,
            news_provider=lambda _symbol: news,
        )
        tick = type("SignalTick", (), {"timestamp": t1.isoformat()})()

        pending = session.on_tick(
            tick,
            decision_factory=lambda *_: self.decision(),
            now=t1,
            calendar_session=True,
        )
        self.assertEqual(pending["status"], "PENDING_APPROVAL")

        news.append({
            "article_id": "breaking-news-1",
            "ingested_at": t2.isoformat(),
            "intelligence": {
                "event_type": "REGULATORY_ACTION",
                "severity": "HIGH",
                "sentiment": "NEGATIVE",
                "instruments": ["TEST"],
                "is_stale": False,
                "evidence": [{"article_id": "breaking-news-1", "published_at": t2.isoformat()}],
            },
        })

        result = session.approve_pending(now=t3, calendar_session=True)

        self.assertEqual(result["status"], "CANCELLED")
        self.assertIn("high-severity news", result["reason"])
        self.assertFalse(result["execution"])
        self.assertEqual(session.last_order.status, "REJECTED")
        self.assertIsNone(session.execution.paper_broker.get("paper-1"))

    def test_configured_continue_policy_allows_pending_order_after_news(self):
        t1 = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)
        t2 = t1.replace(second=5)
        t3 = t1.replace(second=10)
        news = []
        session = LivePaperTradingSession(
            execution=service(),
            portfolio=PortfolioSnapshot(100_000, ()),
            policy=policy(),
            paper_mode_enabled=True,
            news_provider=lambda _symbol: news,
            news_race_policy="CONTINUE",
        )
        tick = type("SignalTick", (), {"timestamp": t1.isoformat()})()
        pending = session.on_tick(
            tick,
            decision_factory=lambda *_: self.decision(),
            now=t1,
            calendar_session=True,
        )
        news.append({
            "ingested_at": t2.isoformat(),
            "intelligence": {
                "severity": "HIGH",
                "instruments": ["TEST"],
                "is_stale": False,
                "evidence": [{"published_at": t2.isoformat()}],
            },
        })

        result = session.approve_pending(now=t3, calendar_session=True)

        self.assertEqual(pending["status"], "PENDING_APPROVAL")
        self.assertEqual(result["status"], "FILLED")
        self.assertTrue(result["execution"])

    def test_approved_order_does_not_fill_before_open_or_after_close(self):
        outside_session_times = (
            datetime(2026, 10, 1, 3, 44, 59, tzinfo=timezone.utc),
            datetime(2026, 10, 1, 10, 0, 1, tzinfo=timezone.utc),
        )
        for now in outside_session_times:
            with self.subTest(now=now):
                session = LivePaperTradingSession(
                    execution=service(),
                    portfolio=PortfolioSnapshot(100_000, ()),
                    policy=policy(),
                    paper_mode_enabled=True,
                    human_approval=True,
                    news_provider=lambda _symbol: [],
                )
                tick = type("FreshTick", (), {"timestamp": now.isoformat()})()
                result = session.on_tick(
                    tick,
                    decision_factory=lambda *_: self.decision(),
                    now=now,
                    calendar_session=True,
                )

                self.assertEqual(result["status"], "REJECTED")
                self.assertIsNotNone(session.last_order)
                self.assertEqual(session.last_order.status, "REJECTED")
                self.assertEqual(session.last_order.filled_quantity, 0)

    def test_cancel_policy_fails_closed_without_a_news_provider(self):
        t1 = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)
        session = LivePaperTradingSession(
            execution=service(),
            portfolio=PortfolioSnapshot(100_000, ()),
            policy=policy(),
            paper_mode_enabled=True,
            human_approval=True,
        )

        result = session.on_tick(
            type("SignalTick", (), {"timestamp": t1.isoformat()})(),
            decision_factory=lambda *_: self.decision(),
            now=t1,
            calendar_session=True,
        )

        self.assertEqual(result["status"], "CANCELLED")
        self.assertIn("could not be revalidated", result["reason"])
        self.assertIsNone(session.execution.paper_broker.get("paper-1"))


if __name__ == "__main__":
    unittest.main()
