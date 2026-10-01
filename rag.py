"""Local deterministic embeddings and historical trading memory."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbedder:
    """Reproducible local embeddings; suitable for retrieval, not semantic proof."""

    def __init__(self, dimension: int = 128):
        if dimension < 16:
            raise ValueError("Embedding dimension must be at least 16")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(str(text).casefold())
        if not tokens:
            raise ValueError("Embedding text must contain searchable content")
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 + min(len(token), 12) / 12
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("Could not create a non-zero embedding")
        return [round(value / norm, 8) for value in vector]


@dataclass(frozen=True)
class MemoryDocument:
    document_id: str
    kind: str
    text: str
    metadata: dict
    embedding: tuple[float, ...]
    created_at: str

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "kind": self.kind,
            "text": self.text,
            "metadata": self.metadata,
            "embedding_dimension": len(self.embedding),
            "created_at": self.created_at,
        }


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("Vectors must have the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Vectors must be non-zero")
    return sum(a * b for a, b in zip(left_values, right_values)) / (left_norm * right_norm)


def build_memory_document(
    *,
    document_id: str,
    kind: str,
    text: str,
    metadata: dict | None = None,
    embedder: HashingEmbedder | None = None,
    created_at: str | None = None,
) -> MemoryDocument:
    if not document_id.strip() or not kind.strip():
        raise ValueError("Memory document id and kind are required")
    normalized_text = " ".join(str(text).split())
    embedder = embedder or HashingEmbedder()
    return MemoryDocument(
        document_id=document_id.strip(),
        kind=kind.strip().upper(),
        text=normalized_text,
        metadata=dict(metadata or {}),
        embedding=tuple(embedder.embed(normalized_text)),
        created_at=_timestamp(created_at),
    )


def _timestamp(value: str | None) -> str:
    parsed = datetime.fromisoformat((value or datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


class HistoricalTradingMemory:
    def __init__(self, store, *, embedder: HashingEmbedder | None = None):
        self.store = store
        self.embedder = embedder or HashingEmbedder()

    def remember(
        self,
        *,
        document_id: str,
        kind: str,
        text: str,
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> bool:
        document = build_memory_document(
            document_id=document_id,
            kind=kind,
            text=text,
            metadata=metadata,
            embedder=self.embedder,
            created_at=created_at,
        )
        return self.store.save_memory_document(document)

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict | None = None,
        max_age_seconds: int | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        if max_age_seconds is not None and max_age_seconds < 0:
            raise ValueError("Memory maximum age must be non-negative")
        created_after = None
        if max_age_seconds is not None:
            current = now or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            created_after = (current.astimezone(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        embedding = self.embedder.embed(query)
        return self.store.search_memory_documents(
            embedding,
            limit=limit,
            filters=filters,
            created_after=created_after,
        )
