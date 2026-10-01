"""Runtime orchestration for the research-to-paper decision path."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

from alpha_strategies import AlphaAggregator
from trade_governance import TradeTrace


_PIPELINE_STAGES = (
    "observe",
    "analyze",
    "generate_trade_hypotheses",
    "challenge",
    "validate_against_history",
    "estimate_probability",
    "calculate_expected_net_value",
    "optimize_portfolio_risk",
    "determine_position_size",
    "check_compliance",
    "execute",
    "analyze_result",
    "learn",
)


def _new_stage_trace() -> dict[str, str]:
    return {stage: "NOT_RUN" for stage in _PIPELINE_STAGES}


def _decision_result(
    *,
    status: str,
    action: str,
    signal_action: str,
    reason: str,
    alpha: dict[str, Any],
    stages: dict[str, str],
    **details: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "decision": {"action": action, "signal_action": signal_action, "reason": reason},
        "alpha": alpha,
        "stages": dict(stages),
        "execution": False,
        **details,
    }


def _has_open_long_position(context: dict, symbol: str) -> bool:
    portfolio = context.get("portfolio", {})
    if isinstance(portfolio, dict):
        positions = [*portfolio.get("positions", []), *portfolio.get("holdings", [])]
    else:
        positions = [*getattr(portfolio, "positions", ()), *getattr(portfolio, "holdings", ())]
    for position in positions:
        held_symbol = position.get("symbol", "") if isinstance(position, dict) else getattr(position, "symbol", "")
        quantity = position.get("quantity", 0) if isinstance(position, dict) else getattr(position, "quantity", 0)
        if str(held_symbol).upper() == symbol.upper() and isinstance(quantity, (int, float)) and quantity > 0:
            return True
    return False


class RuntimeDecisionPipeline:
    """Connects context, alpha, AI review, calibrated decision, paper OMS, and memory."""

    def __init__(
        self,
        *,
        strategy_agent,
        adversarial_agent,
        alpha_strategies,
        historical_validator,
        decision_evaluator,
        paper_executor,
        journal_store,
        memory,
        alpha_aggregator=None,
    ):
        self.strategy_agent = strategy_agent
        self.adversarial_agent = adversarial_agent
        self.alpha_strategies = tuple(alpha_strategies)
        self.historical_validator = historical_validator
        self.decision_evaluator = decision_evaluator
        self.paper_executor = paper_executor
        self.journal_store = journal_store
        self.memory = memory
        self.alpha_aggregator = alpha_aggregator or AlphaAggregator()

    def process_tick(self, *, tick, candles: list, context: dict, now: datetime | None = None) -> dict:
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        stages = _new_stage_trace()
        stages["observe"] = "COMPLETED"
        evidence_id = f"tick:{tick.event_id}"
        signals = [
            strategy.generate(tick.symbol, candles, evidence_id=evidence_id)
            for strategy in self.alpha_strategies
        ]
        aggregate = self.alpha_aggregator.aggregate(signals, regime=context.get("market_regime"))
        enriched_context = {
            **context,
            "instruments": [tick.symbol],
            "data_timestamp": tick.timestamp,
            "alpha_aggregate": aggregate,
            "evidence_ids": sorted(set(context.get("evidence_ids", [])) | {evidence_id, *aggregate["evidence_ids"]}),
        }
        stages["analyze"] = "COMPLETED"
        signal_action = aggregate["direction"]
        if signal_action != "BUY":
            has_position = _has_open_long_position(context, tick.symbol)
            if signal_action == "SELL" and has_position:
                stages["generate_trade_hypotheses"] = "SELL_SIGNAL"
                stages["execute"] = "BLOCKED_EXIT_PATH_UNAVAILABLE"
                return _decision_result(
                    status="SELL_EXIT_UNAVAILABLE",
                    action="SELL",
                    signal_action=signal_action,
                    reason="A held long position has a sell signal, but the validated sell-to-close execution path is not configured; no order was sent.",
                    alpha=aggregate,
                    stages=stages,
                )
            if signal_action == "SELL":
                action = "NO TRADE"
                reason = "A sell signal cannot open a short position, and no long position is available to close."
                stages["generate_trade_hypotheses"] = "BLOCKED_SHORT_ENTRY"
                stages["execute"] = "BLOCKED_SHORT_ENTRY"
            elif signal_action == "HOLD" and has_position:
                action = "HOLD"
                reason = "No new entry signal; retain the existing long position."
                stages["generate_trade_hypotheses"] = "SKIPPED_NO_ENTRY_SIGNAL"
            else:
                action = "NO TRADE"
                reason = "No deterministic entry signal and no existing long position to hold."
                stages["generate_trade_hypotheses"] = "SKIPPED_NO_ENTRY_SIGNAL"
            return _decision_result(
                status=action,
                action=action,
                signal_action=signal_action,
                reason=reason,
                alpha=aggregate,
                stages=stages,
            )

        decision = self.strategy_agent.propose(model=context["model"], context=enriched_context)
        stages["generate_trade_hypotheses"] = "COMPLETED"
        candidate = decision.output
        if (
            candidate.get("direction") != "LONG"
            or str(candidate.get("instrument", "")).upper() != tick.symbol.upper()
        ):
            stages["generate_trade_hypotheses"] = "REJECTED_UNSUPPORTED_DIRECTION_OR_SYMBOL"
            return _decision_result(
                status="DECISION_REJECTED",
                action="NO TRADE",
                signal_action=signal_action,
                reason="Only a LONG hypothesis for the observed instrument is supported by this decision path.",
                alpha=aggregate,
                stages=stages,
                hypothesis=candidate,
            )
        adversarial = self.adversarial_agent.review(decision.output, enriched_context, now=now)
        stages["challenge"] = "COMPLETED" if adversarial["approved"] else "REJECTED"
        if not adversarial["approved"]:
            stages["execute"] = "BLOCKED_ADVERSARIAL_REVIEW"
            return _decision_result(
                status="ADVERSARIAL_REJECTED",
                action="NO TRADE",
                signal_action=signal_action,
                reason="The adversarial review rejected the candidate.",
                alpha=aggregate,
                stages=stages,
                adversarial_review=adversarial,
            )
        calibration = self.historical_validator(decision.output, enriched_context)
        stages["validate_against_history"] = "COMPLETED" if calibration else "UNAVAILABLE"
        calibrated_probability = calibration.get("calibrated_probability")
        if (
            calibration.get("status") != "CALIBRATED"
            or not isinstance(calibrated_probability, Decimal)
            or not calibrated_probability.is_finite()
            or not Decimal("0") <= calibrated_probability <= Decimal("1")
        ):
            stages["estimate_probability"] = "REJECTED_UNCALIBRATED"
            stages["execute"] = "BLOCKED_UNCALIBRATED_PROBABILITY"
            return _decision_result(
                status="CALIBRATION_REJECTED",
                action="NO TRADE",
                signal_action=signal_action,
                reason="A finite calibrated probability from completed historical outcomes is required.",
                alpha=aggregate,
                stages=stages,
                adversarial_review=adversarial,
                historical_validation=calibration,
            )
        stages["estimate_probability"] = "CALIBRATED"
        evaluated = self.decision_evaluator(decision.output, calibration, enriched_context)
        stage_fields = {
            "calculate_expected_net_value": "expected_value",
            "optimize_portfolio_risk": "portfolio_optimization",
            "determine_position_size": "position_sizing",
            "check_compliance": "compliance",
        }
        for stage, field in stage_fields.items():
            has_stage_output = isinstance(evaluated.get(field), dict)
            has_cost_output = stage != "calculate_expected_net_value" or isinstance(evaluated.get("round_trip_cost"), dict)
            stages[stage] = "COMPLETED" if has_stage_output and has_cost_output else "MISSING"
        if evaluated.get("execution_authority") is not False:
            stages["execute"] = "BLOCKED_INVALID_AUTHORITY"
            return _decision_result(
                status="EXECUTION_AUTHORITY_VIOLATION",
                action="NO TRADE",
                signal_action=signal_action,
                reason="The deterministic evaluator must explicitly declare execution_authority=False.",
                alpha=aggregate,
                stages=stages,
                adversarial_review=adversarial,
                decision_evaluation=evaluated,
            )

        required_fields = (*stage_fields.values(), "round_trip_cost")
        missing_fields = [field for field in required_fields if not isinstance(evaluated.get(field), dict)]
        if missing_fields:
            stages["execute"] = "BLOCKED_MISSING_GATE_OUTPUTS"
            return _decision_result(
                status="DECISION_REJECTED",
                action="NO TRADE",
                signal_action=signal_action,
                reason=f"Deterministic evaluator omitted required gate outputs: {', '.join(missing_fields)}.",
                alpha=aggregate,
                stages=stages,
                adversarial_review=adversarial,
                decision_evaluation=evaluated,
            )

        gate_reasons = list(evaluated.get("reasons", []))
        try:
            expected_net_value = Decimal(str(evaluated["expected_value"].get("expected_net_value")))
        except (InvalidOperation, TypeError, ValueError):
            expected_net_value = Decimal("NaN")
        if not expected_net_value.is_finite() or expected_net_value <= 0:
            stages["calculate_expected_net_value"] = "REJECTED_NON_POSITIVE_OR_MISSING"
            gate_reasons.append("Expected net value after costs must be positive")
        try:
            round_trip_cost = Decimal(str(evaluated["round_trip_cost"].get("total_cost")))
        except (InvalidOperation, TypeError, ValueError):
            round_trip_cost = Decimal("NaN")
        if not round_trip_cost.is_finite() or round_trip_cost < 0:
            stages["calculate_expected_net_value"] = "REJECTED_MISSING_COSTS"
            gate_reasons.append("A finite, non-negative round-trip cost result is required")
        quantity = evaluated["position_sizing"].get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            stages["determine_position_size"] = "REJECTED_ZERO_OR_INVALID"
            gate_reasons.append("Deterministic position sizing did not produce a positive quantity")
        if evaluated["portfolio_optimization"].get("eligible_for_risk_review") is not True:
            stages["optimize_portfolio_risk"] = "REJECTED"
            gate_reasons.append("Portfolio optimization did not approve the candidate")
        if evaluated["compliance"].get("allowed") is not True:
            stages["check_compliance"] = "REJECTED"
            gate_reasons.append("Compliance did not allow the candidate")
        if evaluated.get("eligible_for_risk_review") is not True:
            gate_reasons.extend(gate_reasons or ["Deterministic evaluation did not approve the candidate"])
        if gate_reasons:
            stages["execute"] = "BLOCKED_DETERMINISTIC_GATES"
            return _decision_result(
                status="DECISION_REJECTED",
                action="NO TRADE",
                signal_action=signal_action,
                reason="; ".join(dict.fromkeys(gate_reasons)),
                alpha=aggregate,
                stages=stages,
                adversarial_review=adversarial,
                historical_validation=calibration,
                decision_evaluation=evaluated,
            )

        prediction_id = str(uuid4())
        self.journal_store.save_prediction({
            "prediction_id": prediction_id,
            "symbol": tick.symbol,
            "predicted_probability": float(calibrated_probability),
            "created_at": timestamp,
            "metadata": {"context_hash": decision.context_hash, "alpha": aggregate},
        })
        stages["execute"] = "PAPER_ORDER_REQUESTED"
        order = self.paper_executor(decision.output, evaluated, tick, enriched_context)
        stages["execute"] = order.status
        trade_id = str(uuid4())
        trace = TradeTrace(
            trade_id=trade_id,
            symbol=order.symbol,
            decision_timestamp=timestamp,
            context_hash=decision.context_hash,
            strategy_names=tuple(signal.strategy for signal in signals),
            alpha_score=aggregate["score"],
            calibrated_probability=str(calibration.get("calibrated_probability")),
            expected_net_value=str(evaluated.get("expected_value", {}).get("expected_net_value", "0")),
            evidence_ids=tuple(enriched_context["evidence_ids"]),
            explanation=str(decision.output.get("thesis", "Evidence-linked strategy decision")),
            historical_validation=calibration,
            portfolio_impact=evaluated.get("portfolio_optimization", {}),
            risk_controls=evaluated.get("position_sizing", {}),
            compliance=evaluated.get("compliance", {}),
            audit_event_ids=(prediction_id, trade_id),
            backtest_reference=str(enriched_context.get("backtest_reference", "historical_validation")),
        )
        self.journal_store.save_trade_journal({
            "trade_id": trade_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "market": order.market,
            "side": order.side,
            "quantity": order.filled_quantity,
            "entry_price": order.average_fill_price,
            "status": order.status,
            "opened_at": order.created_at,
            "metadata": {"prediction_id": prediction_id, "decision": evaluated, "trade_trace": trace.as_dict()},
        })
        try:
            self.memory.remember(
                document_id=f"trade:{order.client_order_id}",
                kind="PAPER_TRADE",
                text=f"{order.symbol} {order.status} {order.filled_quantity} at {order.average_fill_price}",
                metadata={"instrument": order.symbol, "status": order.status, "prediction_id": prediction_id},
                created_at=timestamp,
            )
        except Exception:
            stages["learn"] = "UNAVAILABLE"
        else:
            stages["learn"] = "MEMORY_RECORDED_NO_RETRAIN"
        stages["analyze_result"] = "RECORDED_ONLY"
        return {
            "status": order.status,
            "decision": {"action": "BUY", "signal_action": signal_action, "reason": "Candidate passed deterministic gates; see paper order result."},
            "alpha": aggregate,
            "stages": stages,
            "adversarial_review": adversarial,
            "decision_evaluation": evaluated,
            "order": order.__dict__,
            "execution": order.status in {"FILLED", "PARTIALLY_FILLED"},
        }
