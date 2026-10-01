import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_agents import AdversarialAgent, StrategyAgent
from alpha_strategies import AlphaAggregator, MomentumAlpha, SmaAlpha
from core import Candle
from llm_gateway import LLMGateway
from rag import HistoricalTradingMemory
from runtime_pipeline import RuntimeDecisionPipeline
from store import EventStore
from execution import Order
from trade_governance import TradeTrace


def eligible_evaluation():
    return {
        "eligible_for_risk_review": True,
        "execution_authority": False,
        "expected_value": {"expected_net_value": "12.5"},
        "round_trip_cost": {"total_cost": "2.5"},
        "position_sizing": {"quantity": 5},
        "portfolio_optimization": {"eligible_for_risk_review": True},
        "compliance": {"allowed": True},
    }


def candidate_pipeline(
    *, calibration, evaluation, executor_calls, evaluator_calls=None, order_status="FILLED", model_confidence=0.5,
):
    paper_order = SimpleNamespace(
        symbol="TEST",
        client_order_id="paper-probe",
        market="NSE",
        side="BUY",
        status=order_status,
        filled_quantity=1,
        average_fill_price=100.0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    def evaluate(*_):
        if evaluator_calls is not None:
            evaluator_calls.append(True)
        return evaluation

    return RuntimeDecisionPipeline(
        strategy_agent=SimpleNamespace(propose=lambda **_: SimpleNamespace(
            output={
                "instrument": "TEST",
                "direction": "LONG",
                "confidence": model_confidence,
                "thesis": "Evidence-backed test candidate.",
            },
            context_hash="a" * 64,
        )),
        adversarial_agent=SimpleNamespace(review=lambda *args, **kwargs: {"approved": True}),
        alpha_strategies=[SmaAlpha(), MomentumAlpha()],
        historical_validator=lambda *_: calibration,
        decision_evaluator=evaluate,
        paper_executor=lambda *_: (executor_calls.append(True) or paper_order),
        journal_store=SimpleNamespace(save_prediction=lambda _: True, save_trade_journal=lambda _: True),
        memory=SimpleNamespace(remember=lambda **_: True),
    )


@pytest.mark.component
class RuntimePipelineTests(unittest.TestCase):
    def candles(self):
        base = datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
        return [Candle("TEST", (base + timedelta(minutes=index)).isoformat(), 100 + index, 101 + index, 99 + index, 100 + index, 1000, timeframe="1m") for index in range(30)]

    @pytest.mark.chaos
    def test_trade_trace_rejects_non_finite_risk_measurements(self):
        valid_trace = {
            "trade_id": "trade-1",
            "symbol": "TEST",
            "decision_timestamp": "2026-10-01T10:00:00+00:00",
            "context_hash": "a" * 64,
            "strategy_names": ("sma_cross",),
            "alpha_score": 0.7,
            "calibrated_probability": "0.7",
            "expected_net_value": "12.5",
            "evidence_ids": ("candle-1",),
            "explanation": "Validated test trace",
            "historical_validation": {},
            "portfolio_impact": {},
            "risk_controls": {},
            "compliance": {},
            "audit_event_ids": ("event-1",),
            "backtest_reference": "test",
        }
        invalid_metrics = (
            ("alpha_score", float("nan")),
            ("alpha_score", float("inf")),
            ("calibrated_probability", "NaN"),
            ("calibrated_probability", "Infinity"),
            ("calibrated_probability", "-Infinity"),
            ("expected_net_value", "NaN"),
            ("expected_net_value", "Infinity"),
            ("expected_net_value", "-Infinity"),
        )
        for field, value in invalid_metrics:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "measurements"):
                TradeTrace(**{**valid_trace, field: value})

    def test_pipeline_connects_alpha_ai_validation_paper_journal_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "pipeline.db")
            pipeline = RuntimeDecisionPipeline(
                strategy_agent=SimpleNamespace(propose=lambda **_: SimpleNamespace(
                    output={"instrument": "TEST", "direction": "LONG", "thesis": "Two independent alpha signals agree."}, context_hash="a" * 64,
                )),
                adversarial_agent=SimpleNamespace(review=lambda *args, **kwargs: {"approved": True}),
                alpha_strategies=[SmaAlpha(), MomentumAlpha()],
                historical_validator=lambda *_: {"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
                decision_evaluator=lambda *_: eligible_evaluation(),
                paper_executor=lambda *_: Order(
                    "order-1", "paper-1", "TEST", "NSE", "BUY", "LIMIT", 5, "FILLED",
                    datetime.now(timezone.utc).isoformat(), requested_price=100, filled_quantity=5, average_fill_price=100,
                ),
                journal_store=store,
                memory=HistoricalTradingMemory(store),
            )
            result = pipeline.process_tick(
                tick=SimpleNamespace(event_id="tick-1", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
                candles=self.candles(),
                context={"model": "fake", "evidence_ids": []},
                now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
            )
            self.assertEqual(result["status"], "FILLED")
            self.assertEqual(len(store.read_trade_journal()), 1)
            trace = store.read_trade_journal()[0]["metadata"]["trade_trace"]
            self.assertIn("measurements", trace)
            self.assertIn("explainability", trace)
            self.assertTrue(trace["reproducibility_hash"])
            self.assertIn("risk_controls", trace)
            self.assertEqual(HistoricalTradingMemory(store).recall("TEST trade", filters={"instrument": "TEST"})[0]["document_id"], "trade:paper-1")

    def test_pipeline_keeps_completed_trade_when_memory_write_is_unavailable(self):
        executor_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
        )

        def unavailable_memory(**_kwargs):
            raise RuntimeError("memory storage unavailable")

        pipeline.memory = SimpleNamespace(remember=unavailable_memory)
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="tick-rag-down", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "fake", "evidence_ids": []},
            now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "FILLED")
        self.assertEqual(executor_calls, [True])
        self.assertEqual(result["stages"]["learn"], "UNAVAILABLE")

    def test_valid_llm_long_hypothesis_requires_adversarial_review_before_order(self):
        class ContextualProvider:
            name = "mock-llm"

            def generate_structured(self, *, model, prompt, context):
                return {
                    "instrument": "TEST",
                    "market": "NSE",
                    "direction": "LONG",
                    "time_horizon": "intraday",
                    "entry_range": {"low": 125, "high": 130},
                    "stop_loss": 120,
                    "take_profit": 140,
                    "thesis": "Mocked model hypothesis; approval remains downstream.",
                    "supporting_evidence": [{
                        "source_id": context["evidence_ids"][0],
                        "claim": "Deterministic alpha signal",
                    }],
                    "contradictory_evidence": [],
                    "invalidating_conditions": ["Close below stop"],
                    "confidence": 0.8,
                    "data_timestamp": context["data_timestamp"],
                    "strategy": "sma_cross",
                }

        executor_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
        )
        pipeline.strategy_agent = StrategyAgent(LLMGateway(ContextualProvider()))
        pipeline.adversarial_agent = AdversarialAgent()
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="llm-long", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "mock", "evidence_ids": []},
            now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "ADVERSARIAL_REJECTED")
        self.assertFalse(result["adversarial_review"]["approved"])
        self.assertEqual(executor_calls, [])

    def test_long_strategy_with_mixed_evidence_below_confidence_floor_is_blocked(self):
        class ContextualProvider:
            name = "mock-llm"

            def generate_structured(self, *, model, prompt, context):
                return {
                    "instrument": "TEST",
                    "market": "NSE",
                    "direction": "LONG",
                    "time_horizon": "intraday",
                    "entry_range": {"low": 125, "high": 130},
                    "stop_loss": 120,
                    "take_profit": 140,
                    "thesis": "Strong positive evidence, offset by a verified guidance cut.",
                    "supporting_evidence": [{
                        "source_id": "strong-positive-report",
                        "claim": "Audited earnings materially exceeded estimates",
                    }],
                    "contradictory_evidence": [{
                        "source_id": "strong-negative-report",
                        "claim": "The company issued a verified material guidance reduction",
                    }],
                    "invalidating_conditions": ["Close below stop"],
                    "confidence": 0.4,
                    "data_timestamp": context["data_timestamp"],
                    "strategy": "sma_cross",
                }

        executor_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
        )
        pipeline.strategy_agent = StrategyAgent(LLMGateway(ContextualProvider()))
        pipeline.adversarial_agent = AdversarialAgent()
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="mixed-evidence", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={
                "model": "mock",
                "evidence_ids": ["strong-positive-report", "strong-negative-report"],
            },
            now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "ADVERSARIAL_REJECTED")
        self.assertIn("Confidence is below the configured minimum", result["adversarial_review"]["risk_flags"])
        self.assertEqual(executor_calls, [])

    def test_adversarial_approval_cannot_rescue_unsupported_short_strategy(self):
        executor_calls = []
        review_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
        )
        pipeline.strategy_agent = SimpleNamespace(propose=lambda **_: SimpleNamespace(
            output={"instrument": "TEST", "direction": "SHORT", "thesis": "Strong bearish signal."},
            context_hash="c" * 64,
        ))
        pipeline.adversarial_agent = SimpleNamespace(review=lambda *args, **kwargs: (
            review_calls.append(True),
            {"approved": True, "risk_flags": [], "execution_authority": False},
        )[1])
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="unsupported-short", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "mock", "evidence_ids": []},
            now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "DECISION_REJECTED")
        self.assertEqual(result["decision"]["action"], "NO TRADE")
        self.assertEqual(review_calls, [])
        self.assertEqual(executor_calls, [])

    def test_pipeline_blocks_execution_authority_from_decision_output(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "authority.db")
            executed = []
            pipeline = RuntimeDecisionPipeline(
                strategy_agent=SimpleNamespace(propose=lambda **_: SimpleNamespace(output={"instrument": "TEST", "direction": "LONG"}, context_hash="a" * 64)),
                adversarial_agent=SimpleNamespace(review=lambda *args, **kwargs: {"approved": True}),
                alpha_strategies=[SmaAlpha(), MomentumAlpha()],
                historical_validator=lambda *_: {"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
                decision_evaluator=lambda *_: {"eligible_for_risk_review": True, "execution_authority": True},
                paper_executor=lambda *_: executed.append(True),
                journal_store=store,
                memory=HistoricalTradingMemory(store),
            )
            result = pipeline.process_tick(
                tick=SimpleNamespace(event_id="tick-2", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
                candles=self.candles(), context={"model": "fake"},
                now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
            )
            self.assertEqual(result["status"], "EXECUTION_AUTHORITY_VIOLATION")
            self.assertEqual(executed, [])

    def test_pipeline_preserves_market_metadata_through_journal_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "metadata.db")
            captured_context = {}
            captured_regimes = []

            def aggregate_with_regime(signals, *, regime=None):
                captured_regimes.append(regime)
                return AlphaAggregator().aggregate(signals, regime=regime)

            pipeline = RuntimeDecisionPipeline(
                strategy_agent=SimpleNamespace(propose=lambda **kwargs: (
                    captured_context.update(kwargs["context"]),
                    SimpleNamespace(output={"instrument": "TEST", "direction": "LONG", "thesis": "Metadata-linked decision."}, context_hash="b" * 64),
                )[1]),
                adversarial_agent=SimpleNamespace(review=lambda *args, **kwargs: {"approved": True}),
                alpha_strategies=[SmaAlpha(), MomentumAlpha()],
                historical_validator=lambda *_: {"status": "CALIBRATED", "calibrated_probability": Decimal("0.7"), "source": "walk_forward_v1"},
                decision_evaluator=lambda *_: {
                    "eligible_for_risk_review": True,
                    "execution_authority": False,
                    "expected_value": {"expected_net_value": "12.5"},
                    "round_trip_cost": {"total_cost": "2.5"},
                    "portfolio_optimization": {"eligible_for_risk_review": True, "symbol": "TEST"},
                    "position_sizing": {"quantity": 5},
                    "compliance": {"allowed": True},
                },
                paper_executor=lambda *_: Order(
                    "order-meta", "paper-meta", "TEST", "NSE", "BUY", "LIMIT", 5, "FILLED",
                    datetime.now(timezone.utc).isoformat(), requested_price=100, filled_quantity=5, average_fill_price=100,
                ),
                alpha_aggregator=SimpleNamespace(aggregate=aggregate_with_regime),
                journal_store=store,
                memory=HistoricalTradingMemory(store),
            )
            result = pipeline.process_tick(
                tick=SimpleNamespace(event_id="tick-meta", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
                candles=self.candles(),
                context={
                    "model": "fake-meta",
                    "evidence_ids": ["candle:TEST:1m:1", "news:42"],
                    "timeframe": "1m",
                    "data_source": "fixture_nse",
                    "market_regime": "RISK_ON",
                },
                now=datetime(2026, 10, 1, 9, 44, tzinfo=timezone.utc),
            )
            self.assertEqual(result["status"], "FILLED")
            self.assertEqual(captured_context["timeframe"], "1m")
            self.assertEqual(captured_context["data_source"], "fixture_nse")
            self.assertEqual(captured_context["market_regime"], "RISK_ON")
            self.assertEqual(captured_regimes, ["RISK_ON"])
            journal = store.read_trade_journal()[0]
            trace = journal["metadata"]["trade_trace"]
            self.assertIn("news:42", trace["explainability"]["evidence_ids"])
            self.assertEqual(trace["backtest_reference"], "historical_validation")
            memory = HistoricalTradingMemory(store).recall("TEST trade", filters={"instrument": "TEST"})
            self.assertEqual(memory[0]["metadata"]["status"], "FILLED")

    def test_neutral_alpha_is_hold_only_for_an_existing_long_position(self):
        pipeline = RuntimeDecisionPipeline(
            strategy_agent=None,
            adversarial_agent=None,
            alpha_strategies=[SmaAlpha(), MomentumAlpha()],
            historical_validator=None,
            decision_evaluator=None,
            paper_executor=None,
            journal_store=None,
            memory=None,
        )
        tick = SimpleNamespace(event_id="neutral", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00")
        held = pipeline.process_tick(
            tick=tick,
            candles=[],
            context={"model": "unused", "portfolio": {"positions": [{"symbol": "TEST", "quantity": 2}]}},
        )
        flat = pipeline.process_tick(tick=tick, candles=[], context={"model": "unused", "portfolio": {"positions": []}})
        self.assertEqual(held["decision"]["action"], "HOLD")
        self.assertEqual(held["status"], "HOLD")
        self.assertEqual(flat["decision"]["action"], "NO TRADE")
        self.assertEqual(flat["status"], "NO TRADE")
        self.assertFalse(flat["execution"])
        self.assertEqual(flat["stages"]["validate_against_history"], "NOT_RUN")

    def test_sell_alpha_is_explicit_but_never_opens_a_short_or_submits_an_order(self):
        pipeline = RuntimeDecisionPipeline(
            strategy_agent=None,
            adversarial_agent=None,
            alpha_strategies=[SmaAlpha(), MomentumAlpha()],
            historical_validator=None,
            decision_evaluator=None,
            paper_executor=None,
            journal_store=None,
            memory=None,
        )
        base = datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
        candles = []
        for index in range(30):
            close = 130 - index
            candles.append(Candle(
                "TEST", (base + timedelta(minutes=index)).isoformat(),
                close + 0.5, close + 1, close - 1, close, 1000, timeframe="1m",
            ))
        tick = SimpleNamespace(event_id="sell", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00")
        no_position = pipeline.process_tick(tick=tick, candles=candles, context={"model": "unused"})
        held_position = pipeline.process_tick(
            tick=tick,
            candles=candles,
            context={"model": "unused", "portfolio": {"positions": [{"symbol": "TEST", "quantity": 2}]}},
        )
        self.assertEqual(no_position["decision"]["signal_action"], "SELL")
        self.assertEqual(no_position["decision"]["action"], "NO TRADE")
        self.assertEqual(held_position["decision"]["action"], "SELL")
        self.assertEqual(held_position["status"], "SELL_EXIT_UNAVAILABLE")
        self.assertFalse(held_position["execution"])
        self.assertEqual(held_position["stages"]["execute"], "BLOCKED_EXIT_PATH_UNAVAILABLE")

    @pytest.mark.chaos
    def test_maximum_model_confidence_cannot_bypass_historical_calibration(self):
        evaluator_calls = []
        executor_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "INSUFFICIENT_SAMPLES", "calibrated_probability": None},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
            evaluator_calls=evaluator_calls,
            model_confidence=1.0,
        )
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="uncalibrated", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "test"},
        )
        self.assertEqual(result["status"], "CALIBRATION_REJECTED")
        self.assertEqual(result["decision"]["action"], "NO TRADE")
        self.assertEqual(evaluator_calls, [])
        self.assertEqual(executor_calls, [])

    @pytest.mark.chaos
    def test_missing_authority_or_gate_outputs_fail_closed_before_paper_execution(self):
        calibration = {"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")}
        executor_calls = []
        missing_authority = eligible_evaluation()
        del missing_authority["execution_authority"]
        first = candidate_pipeline(calibration=calibration, evaluation=missing_authority, executor_calls=executor_calls)
        authority_result = first.process_tick(
            tick=SimpleNamespace(event_id="missing-authority", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "test"},
        )
        self.assertEqual(authority_result["status"], "EXECUTION_AUTHORITY_VIOLATION")

        second = candidate_pipeline(
            calibration=calibration,
            evaluation={"eligible_for_risk_review": True, "execution_authority": False},
            executor_calls=executor_calls,
        )
        incomplete_result = second.process_tick(
            tick=SimpleNamespace(event_id="missing-gates", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "test"},
        )
        self.assertEqual(incomplete_result["status"], "DECISION_REJECTED")
        self.assertIn("expected_value", incomplete_result["decision"]["reason"])
        self.assertEqual(executor_calls, [])

    def test_rejected_paper_order_is_not_reported_as_an_execution(self):
        executor_calls = []
        pipeline = candidate_pipeline(
            calibration={"status": "CALIBRATED", "calibrated_probability": Decimal("0.7")},
            evaluation=eligible_evaluation(),
            executor_calls=executor_calls,
            order_status="REJECTED",
        )
        result = pipeline.process_tick(
            tick=SimpleNamespace(event_id="rejected-order", symbol="TEST", timestamp="2026-10-01T09:44:00+00:00"),
            candles=self.candles(),
            context={"model": "test"},
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["execution"])
        self.assertEqual(executor_calls, [True])


if __name__ == "__main__":
    unittest.main()
