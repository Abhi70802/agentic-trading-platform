"""Safe strategy and adversarial agents over the structured LLM gateway."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

from intelligence import validate_hypothesis
from llm_gateway import LLMGateway, ModelDecisionRecord


class StrategyAgent:
    """Requests an evidence-linked hypothesis; it cannot determine quantity or execute."""

    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway

    def propose(self, *, model: str, context: dict) -> ModelDecisionRecord:
        return self.gateway.propose(model=model, context=context)


class AdversarialAgent:
    """Independently stress-tests a strategy output without calling tools or brokers."""

    def review(
        self,
        hypothesis: dict,
        context: dict,
        *,
        now: datetime | None = None,
        kill_switch: bool = False,
    ) -> dict:
        flags: list[str] = []
        revisions: list[str] = []
        evidence_gaps: list[str] = []
        try:
            validation = validate_hypothesis(
                hypothesis,
                now=now,
                max_data_age_seconds=300,
                kill_switch=kill_switch,
            )
        except (TypeError, ValueError) as error:
            validation = {"approved": False, "reasons": [str(error)]}
        for reason in validation.get("reasons", []):
            flags.append(reason)

        supporting = hypothesis.get("supporting_evidence", [])
        contradictory = hypothesis.get("contradictory_evidence", [])
        allowed_evidence = set(context.get("evidence_ids", []))
        referenced = {
            item.get("source_id") for item in [*supporting, *contradictory]
            if isinstance(item, dict) and item.get("source_id")
        }
        missing = sorted(allowed_evidence - referenced)
        if missing:
            evidence_gaps.extend(missing)
            revisions.append("Review all supplied evidence before accepting the hypothesis")
        if not contradictory:
            flags.append("No contradictory evidence was supplied")
            revisions.append("Seek an independent disconfirming source")
        entry = hypothesis.get("entry_range")
        stop = hypothesis.get("stop_loss")
        target = hypothesis.get("take_profit")
        if hypothesis.get("direction") == "LONG" and isinstance(entry, dict):
            low, high = entry.get("low"), entry.get("high")
            numbers = (low, high, stop, target)
            if not all(isinstance(value, (int, float)) and isfinite(value) for value in numbers):
                flags.append("Long risk levels are not finite numbers")
            elif low <= stop or target <= high:
                flags.append("Long reward and risk levels are invalid")
        if hypothesis.get("direction") == "SHORT":
            flags.append("Short direction conflicts with the long-only strategy constraint")
        return {
            "agent": "adversarial",
            "approved": not flags,
            "risk_flags": flags,
            "required_revisions": revisions,
            "evidence_gaps": evidence_gaps,
            "validation": validation,
            "execution_authority": False,
            "reviewed_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        }
