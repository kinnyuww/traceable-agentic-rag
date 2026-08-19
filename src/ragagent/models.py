from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import numpy as np

from ragagent.config import Settings


def lexical_terms(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", lowered)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk_terms: list[str] = []
    for run in cjk_runs:
        cjk_terms.extend(run)
        cjk_terms.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return latin + cjk_terms


def fts_text(text: str) -> str:
    return " ".join(lexical_terms(text))


def query_coverage(query: str, document: str) -> float:
    query_terms = set(lexical_terms(query))
    if not query_terms:
        return 0.0
    document_terms = set(lexical_terms(document))
    return len(query_terms & document_terms) / len(query_terms)


class EmbeddingClient(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RerankClient(Protocol):
    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retry_count: int = 0


class ChatClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> ChatResult: ...


class ModelServiceError(RuntimeError):
    """Sanitized model-provider error safe to persist in traces and API responses."""


def _model_error(provider: str, exc: Exception) -> ModelServiceError:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTP {exc.response.status_code}"
    elif isinstance(exc, httpx.TimeoutException):
        detail = "timeout"
    elif isinstance(exc, httpx.RequestError):
        detail = type(exc).__name__
    else:
        detail = type(exc).__name__
    return ModelServiceError(f"{provider} model request failed: {detail}")


def trust_environment_proxy(url: str) -> bool:
    """Never route local model traffic through macOS/system HTTP proxies."""

    hostname = (urlparse(url).hostname or "").lower()
    return hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
        "model-runner.docker.internal",
    }


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json_payload: dict[str, Any],
    max_attempts: int = 5,
) -> tuple[httpx.Response, int]:
    """Retry only transient transport/5xx failures with a bounded backoff."""

    for attempt in range(max_attempts):
        try:
            response = await client.post(url, headers=headers, json=json_payload)
            if response.status_code < 500:
                response.raise_for_status()
                return response, attempt
            if attempt == max_attempts - 1:
                response.raise_for_status()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == max_attempts - 1:
                raise
        await asyncio.sleep(0.5 * (2**attempt))
    raise RuntimeError("unreachable model retry state")


class DeterministicEmbeddingClient:
    """Network-free feature hashing used for tests and graceful local fallback."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for term in lexical_terms(text):
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


class OpenAIEmbeddingClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", dimensions: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions
        self.last_retry_count = 0
        self.trust_env = trust_environment_proxy(base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=self.trust_env) as client:
                response, self.last_retry_count = await _post_with_retry(
                    client,
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json_payload={"model": self.model, "input": texts},
                )
                payload = response.json()
            ordered = sorted(payload["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
        except Exception as exc:
            raise _model_error("embedding", exc) from exc
        if vectors:
            self.dimensions = len(vectors[0])
        return vectors


class LexicalRerankClient:
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_terms = set(lexical_terms(query))
        scores: list[float] = []
        for document in documents:
            document_terms = set(lexical_terms(document))
            overlap = len(query_terms & document_terms)
            coverage = overlap / max(1, len(query_terms))
            precision = overlap / max(1, len(document_terms))
            scores.append(min(1.0, 0.82 * coverage + 0.18 * math.sqrt(precision)))
        return scores


class HttpRerankClient:
    """Cohere/vLLM-style rerank endpoint adapter."""

    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.last_latency_ms: float | None = None
        self.last_retry_count = 0
        self.trust_env = trust_environment_proxy(endpoint)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {"query": query, "documents": documents}
        if self.model:
            payload["model"] = self.model
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=self.trust_env) as client:
                response, self.last_retry_count = await _post_with_retry(
                    client,
                    self.endpoint,
                    headers=headers,
                    json_payload=payload,
                )
                body = response.json()
        except Exception as exc:
            raise _model_error("rerank", exc) from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
        results = body.get("results", body.get("data", []))
        scores = [0.0] * len(documents)
        for fallback_index, item in enumerate(results):
            index = int(item.get("index", fallback_index))
            score = item.get("relevance_score", item.get("score", 0.0))
            if 0 <= index < len(scores):
                scores[index] = float(score)
        return scores


class OpenAIChatClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_seconds: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.last_retry_count = 0
        self.trust_env = trust_environment_proxy(base_url)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> ChatResult:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, trust_env=self.trust_env
            ) as client:
                response, self.last_retry_count = await _post_with_retry(
                    client,
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json_payload=payload,
                )
                body = response.json()
        except Exception as exc:
            raise _model_error("chat", exc) from exc
        usage = body.get("usage", {})
        return ChatResult(
            content=body["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            retry_count=self.last_retry_count,
        )


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingClient(
            settings.embedding_base_url,
            settings.embedding_model,
            settings.embedding_api_key,
            settings.embedding_dimensions,
        )
    return DeterministicEmbeddingClient(settings.embedding_dimensions)


def build_rerank_client(settings: Settings) -> RerankClient:
    if settings.rerank_provider == "http":
        return HttpRerankClient(
            settings.rerank_endpoint,
            settings.rerank_model,
            settings.rerank_api_key,
        )
    return LexicalRerankClient()


def build_chat_client(settings: Settings) -> ChatClient | None:
    if not settings.llm_enabled:
        return None
    return OpenAIChatClient(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_timeout_seconds,
    )
