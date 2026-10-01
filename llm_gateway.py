"""LLM provider boundary and strict evidence-linked structured-output validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


REQUIRED_HYPOTHESIS_FIELDS = {
    "instrument", "market", "direction", "time_horizon", "entry_range", "stop_loss",
    "take_profit", "thesis", "supporting_evidence", "contradictory_evidence",
    "invalidating_conditions", "confidence", "data_timestamp", "strategy",
}

COPILOT_HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "instrument": {"type": "string"},
        "market": {"type": "string"},
        "direction": {"type": "string", "enum": ["LONG", "SHORT", "NONE"]},
        "time_horizon": {"type": "string"},
        "entry_range": {
            "type": "object",
            "properties": {"low": {"type": "number"}, "high": {"type": "number"}},
            "required": ["low", "high"],
            "additionalProperties": False,
        },
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "thesis": {"type": "string"},
        "supporting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "claim": {"type": "string"}},
                "required": ["source_id", "claim"],
                "additionalProperties": False,
            },
        },
        "contradictory_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "claim": {"type": "string"}},
                "required": ["source_id", "claim"],
                "additionalProperties": False,
            },
        },
        "invalidating_conditions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "data_timestamp": {"type": "string"},
        "strategy": {"type": "string"},
    },
    "required": sorted(REQUIRED_HYPOTHESIS_FIELDS),
    "additionalProperties": False,
}


class ModelProvider(Protocol):
    name: str

    def generate_structured(self, *, model: str, prompt: str, context: dict) -> dict: ...


class OpenAIModelProvider:
    name = "openai"

    def __init__(self, *, api_key: str, client=None, max_output_tokens: int = 1200, max_context_chars: int = 30_000):
        if not api_key and client is None:
            raise ValueError("An OpenAI API key or injected client is required")
        if max_output_tokens < 1 or max_context_chars < 1:
            raise ValueError("LLM token and context limits must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self.max_output_tokens = max_output_tokens
        self.max_context_chars = max_context_chars

    @classmethod
    def from_environment(cls) -> OpenAIModelProvider:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the local environment")
        return cls(api_key=api_key)

    def generate_structured(self, *, model: str, prompt: str, context: dict) -> dict:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        if len(serialized) > self.max_context_chars:
            raise ValueError("LLM context exceeds the configured character budget")
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{prompt} Return one JSON object only."},
                {"role": "user", "content": serialized},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty structured response")
        try:
            output = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("OpenAI returned malformed JSON") from error
        if not isinstance(output, dict):
            raise ValueError("OpenAI response must be a JSON object")
        return output


class CopilotModelProvider:
    name = "copilot"

    def __init__(
        self,
        *,
        client_factory=None,
        permission_denied_factory=None,
        max_context_chars: int = 30_000,
        max_output_chars: int = 20_000,
        timeout_seconds: float = 90,
    ):
        if min(max_context_chars, max_output_chars) < 1 or timeout_seconds <= 0:
            raise ValueError("Copilot context, output, and timeout limits must be positive")
        self._client_factory = client_factory
        self._permission_denied_factory = permission_denied_factory
        self.max_context_chars = max_context_chars
        self.max_output_chars = max_output_chars
        self.timeout_seconds = timeout_seconds

    def generate_structured(self, *, model: str, prompt: str, context: dict) -> dict:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        if len(serialized) > self.max_context_chars:
            raise ValueError("LLM context exceeds the configured character budget")
        request_prompt = (
            f"{prompt} Return one JSON object only, with no markdown. "
            f"Do not use tools. Context: {serialized}"
        )
        try:
            return asyncio.run(self._generate(model=model, prompt=request_prompt))
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(f"Copilot request failed ({type(error).__name__})") from error

    async def _generate(self, *, model: str, prompt: str) -> dict:
        client_factory = self._client_factory
        permission_denied_factory = self._permission_denied_factory
        if client_factory is None or permission_denied_factory is None:
            from copilot import CopilotClient
            from copilot.rpc import PermissionDecisionUserNotAvailable

            client_factory = client_factory or CopilotClient
            permission_denied_factory = permission_denied_factory or PermissionDecisionUserNotAvailable
        with tempfile.TemporaryDirectory(prefix="northstar-copilot-") as workspace:
            async with client_factory(
                working_directory=workspace,
                log_level="error",
            ) as client:
                session = await client.create_session(
                    model=model,
                    working_directory=workspace,
                    available_tools=[],
                    excluded_tools=[],
                    tools=[],
                    mcp_servers={},
                    on_permission_request=lambda *_: permission_denied_factory(),
                    enable_config_discovery=False,
                    skip_custom_instructions=True,
                    enable_session_store=False,
                    enable_skills=False,
                    enable_host_git_operations=False,
                    enable_file_hooks=False,
                    enable_on_demand_instruction_discovery=False,
                    enable_session_telemetry=False,
                    memory={"enabled": False},
                    infinite_sessions={"enabled": False},
                )
                try:
                    response = await session.send_and_wait(
                        prompt,
                        timeout=self.timeout_seconds,
                        response_schema=COPILOT_HYPOTHESIS_SCHEMA,
                    )
                finally:
                    await client.delete_session(session.session_id)
        if response is None or getattr(response, "data", None) is None:
            raise ValueError("Copilot returned no assistant response")
        data = response.data
        if getattr(data, "tool_requests", None):
            raise ValueError("Copilot returned a tool request; tool use is disabled")
        content = getattr(data, "content", None)
        if not isinstance(content, str) or not content:
            raise ValueError("Copilot returned an empty structured response")
        if len(content) > self.max_output_chars:
            raise ValueError("Copilot response exceeds the configured character budget")
        try:
            output = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Copilot returned malformed JSON") from error
        if not isinstance(output, dict):
            raise ValueError("Copilot response must be a JSON object")
        return output


@dataclass(frozen=True)
class ModelDecisionRecord:
    model: str
    provider: str
    prompt_version: str
    context_hash: str
    timestamp: str
    output: dict


class LLMGateway:
    def __init__(self, provider: ModelProvider | None = None, *, prompt_version: str = "trade-hypothesis-v1"):
        self.provider = provider
        self.prompt_version = prompt_version

    def propose(self, *, model: str, context: dict) -> ModelDecisionRecord:
        if self.provider is None:
            raise RuntimeError("No LLM provider configured; model inference is disabled")
        context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
        prompt = (
            "Return a structured trade hypothesis using only supplied context. "
            "Cite evidence IDs for every factual claim. Never determine order quantity."
        )
        output = self.provider.generate_structured(model=model, prompt=prompt, context=context)
        validate_hypothesis_output(output, context)
        return ModelDecisionRecord(
            model=model,
            provider=self.provider.name,
            prompt_version=self.prompt_version,
            context_hash=hashlib.sha256(context_json.encode("utf-8")).hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            output=output,
        )


def validate_hypothesis_output(output: dict, context: dict) -> None:
    if not isinstance(output, dict):
        raise ValueError("Model output must be a JSON object")
    missing = REQUIRED_HYPOTHESIS_FIELDS - output.keys()
    if missing:
        raise ValueError(f"Model output is missing required fields: {', '.join(sorted(missing))}")
    if "quantity" in output:
        raise ValueError("The model must not determine order quantity")
    unexpected = output.keys() - REQUIRED_HYPOTHESIS_FIELDS
    if unexpected:
        raise ValueError(f"Model output contains unsupported fields: {', '.join(sorted(unexpected))}")
    if output["direction"] not in {"LONG", "SHORT", "NONE"}:
        raise ValueError("Model direction is invalid")
    for field in ("instrument", "market", "time_horizon", "thesis", "strategy"):
        if not isinstance(output[field], str) or not output[field].strip():
            raise ValueError(f"Model field {field} must be a non-empty string")
    allowed_symbols = set(context.get("instruments", []))
    if output["instrument"] not in allowed_symbols:
        raise ValueError("Model returned an instrument absent from supplied context")
    allowed_evidence = set(context.get("evidence_ids", []))
    evidence_ids = set()
    for field in ("supporting_evidence", "contradictory_evidence"):
        if not isinstance(output[field], list):
            raise ValueError(f"{field} must be a list")
        for item in output[field]:
            if (
                not isinstance(item, dict)
                or set(item) != {"source_id", "claim"}
                or not isinstance(item["source_id"], str)
                or not item["source_id"].strip()
                or not isinstance(item["claim"], str)
                or not item["claim"].strip()
            ):
                raise ValueError(f"{field} entries must contain a source_id and claim")
            evidence_ids.add(item["source_id"])
    if not output.get("supporting_evidence") or not evidence_ids <= allowed_evidence:
        raise ValueError("Supporting evidence must reference supplied evidence IDs")
    if not isinstance(output["invalidating_conditions"], list) or any(
        not isinstance(condition, str) for condition in output["invalidating_conditions"]
    ):
        raise ValueError("Invalidating conditions must be a list of strings")
    confidence = output.get("confidence")
    if not _finite_number(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Model confidence must be between zero and one")
    low, high = _validated_entry_range(output["entry_range"])
    stop = output["stop_loss"]
    target = output["take_profit"]
    if not _finite_number(stop) or not _finite_number(target) or stop <= 0 or target <= 0:
        raise ValueError("Stop loss and take profit must be finite positive prices")
    if output["direction"] == "LONG" and not (stop < low <= high < target):
        raise ValueError("Long stop loss and take profit must bracket the entry range")
    if output["direction"] == "SHORT" and not (target < low <= high < stop):
        raise ValueError("Short stop loss and take profit must bracket the entry range")
    model_timestamp = _parse_timestamp(output["data_timestamp"])
    context_timestamp = context.get("data_timestamp")
    if context_timestamp and model_timestamp != _parse_timestamp(context_timestamp):
        raise ValueError("Model data timestamp does not match the supplied context")


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated_entry_range(entry_range: dict) -> tuple[float, float]:
    if not isinstance(entry_range, dict) or set(entry_range) != {"low", "high"}:
        raise ValueError("Entry range must contain low and high prices")
    low, high = entry_range["low"], entry_range["high"]
    if not _finite_number(low) or not _finite_number(high) or low <= 0 or high <= 0 or low > high:
        raise ValueError("Entry range must contain finite positive ordered prices")
    return low, high


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("Model data timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Model data timestamp must be ISO-8601") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)