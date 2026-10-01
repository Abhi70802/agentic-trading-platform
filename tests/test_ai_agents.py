import unittest
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ai_agents import AdversarialAgent, StrategyAgent
from llm_gateway import LLMGateway, OpenAIModelProvider


class FakeProvider:
    name = "fake"

    def __init__(self, output):
        self.output = output

    def generate_structured(self, *, model, prompt, context):
        return self.output


class FakeOpenAIClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )))


def hypothesis(direction="LONG"):
    return {
        "instrument": "TEST",
        "market": "NSE",
        "direction": direction,
        "time_horizon": "days",
        "entry_range": {"low": 99, "high": 101},
        "stop_loss": 95,
        "take_profit": 110,
        "thesis": "Evidence-backed candidate.",
        "supporting_evidence": [{"source_id": "bar-1", "claim": "Signal"}],
        "contradictory_evidence": [{"source_id": "news-1", "claim": "Risk"}],
        "invalidating_conditions": ["Close below stop"],
        "confidence": 0.7,
        "data_timestamp": "2026-10-01T10:00:00+00:00",
        "strategy": "sma_cross",
    }


def model_context():
    return {
        "instruments": ["TEST"],
        "evidence_ids": ["bar-1", "news-1"],
        "data_timestamp": "2026-10-01T10:00:00+00:00",
    }


@pytest.mark.component
class AiAgentTests(unittest.TestCase):
    def test_strategy_agent_delegates_to_structured_gateway(self):
        decision = StrategyAgent(LLMGateway(FakeProvider(hypothesis()))).propose(
            model="fake-v1",
            context={
                "instruments": ["TEST"],
                "evidence_ids": ["bar-1", "news-1"],
                "data_timestamp": "2026-10-01T10:00:00+00:00",
            },
        )
        self.assertEqual(decision.output["instrument"], "TEST")
        self.assertEqual(decision.provider, "fake")

    def test_valid_trade_hypothesis_passes_gateway_validation(self):
        decision = LLMGateway(FakeProvider(hypothesis())).propose(model="fake-v1", context=model_context())

        self.assertEqual(decision.output, hypothesis())

    def test_openai_provider_rejects_malformed_json_without_live_requests(self):
        provider = OpenAIModelProvider(api_key="test-key", client=FakeOpenAIClient('{"direction":'))

        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            provider.generate_structured(model="fake-v1", prompt="test", context=model_context())

    def test_gateway_rejects_non_object_provider_output(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            LLMGateway(FakeProvider("{malformed json")).propose(model="fake-v1", context=model_context())

    def test_gateway_rejects_unsafe_or_incomplete_hypotheses(self):
        invalid_outputs = (
            ("missing fields", lambda output: output.pop("market")),
            ("invalid symbol", lambda output: output.update(instrument="OTHER")),
            ("invalid direction", lambda output: output.update(direction="BUY")),
            ("missing stop loss", lambda output: output.pop("stop_loss")),
            ("missing evidence", lambda output: output.update(supporting_evidence=[])),
            ("extremely high confidence", lambda output: output.update(confidence=1_000_000)),
            ("confidence below zero", lambda output: output.update(confidence=-0.01)),
            ("negative price", lambda output: output.update(stop_loss=-1)),
            ("negative quantity", lambda output: output.update(quantity=-1)),
            (
                "contradictory output with unknown source",
                lambda output: output.update(contradictory_evidence=[{
                    "source_id": "unknown-news",
                    "claim": "Unverified negative news",
                }]),
            ),
            (
                "hallucinated news",
                lambda output: output.update(supporting_evidence=[{
                    "source_id": "fabricated-news",
                    "claim": "Company announced a major order",
                }]),
            ),
            ("stale-context decision", lambda output: output.update(data_timestamp="2026-10-01T09:59:00+00:00")),
        )

        for name, mutate in invalid_outputs:
            with self.subTest(case=name):
                output = deepcopy(hypothesis())
                mutate(output)
                with self.assertRaises(ValueError):
                    LLMGateway(FakeProvider(output)).propose(model="fake-v1", context=model_context())

    def test_gateway_rejects_hallucinated_claims_without_supplied_sources(self):
        hallucinations = (
            (
                "unsupported news",
                "fabricated-news",
                "NSE announced XYZ company acquired ABC",
            ),
            (
                "unsupported price",
                "fabricated-price",
                "XYZ shares closed at 999.99 on the NSE",
            ),
            (
                "unsupported financial metric",
                "fabricated-financials",
                "XYZ revenue grew 47 percent year over year",
            ),
            (
                "unsupported historical statistic",
                "fabricated-backtest",
                "XYZ delivered an 89 percent win rate over the last ten years",
            ),
        )

        for name, source_id, claim in hallucinations:
            with self.subTest(claim_type=name):
                output = deepcopy(hypothesis())
                output["thesis"] = claim
                output["supporting_evidence"] = [{"source_id": source_id, "claim": claim}]

                with self.assertRaisesRegex(ValueError, "supplied evidence IDs"):
                    LLMGateway(FakeProvider(output)).propose(model="fake-v1", context=model_context())

    def test_valid_contradictory_evidence_is_preserved_for_adversarial_review(self):
        output = hypothesis()
        output["contradictory_evidence"] = [{"source_id": "news-1", "claim": "Regulatory investigation creates downside risk"}]
        decision = LLMGateway(FakeProvider(output)).propose(model="fake-v1", context=model_context())

        self.assertEqual(decision.output["supporting_evidence"], hypothesis()["supporting_evidence"])
        self.assertEqual(decision.output["contradictory_evidence"], output["contradictory_evidence"])

    def test_adversarial_agent_approves_complete_long_hypothesis(self):
        result = AdversarialAgent().review(
            hypothesis(),
            {"evidence_ids": ["bar-1", "news-1"]},
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(result["approved"], result)
        self.assertFalse(result["execution_authority"])
        self.assertEqual(result["risk_flags"], [])

    def test_adversarial_agent_rejects_short_and_missing_contradiction(self):
        result = AdversarialAgent().review(
            {**hypothesis("SHORT"), "contradictory_evidence": []},
            {"evidence_ids": ["bar-1"]},
            now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(result["approved"])
        self.assertIn("Short direction conflicts with the long-only strategy constraint", result["risk_flags"])
        self.assertIn("No contradictory evidence was supplied", result["risk_flags"])


if __name__ == "__main__":
    unittest.main()
