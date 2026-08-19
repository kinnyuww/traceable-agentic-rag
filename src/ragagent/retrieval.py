from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragagent.config import Settings
from ragagent.models import EmbeddingClient, LexicalRerankClient, RerankClient, query_coverage
from ragagent.repositories import Repository
from ragagent.schemas import RetrievalHit, SourceLocation


@dataclass(frozen=True)
class RetrievalResult:
    index_version_id: str
    query: str
    hits: list[RetrievalHit]
    latency_ms: float
    diagnostics: dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        repository: Repository,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
        settings: Settings,
    ):
        self.repository = repository
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.settings = settings

    async def retrieve(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        index_version_id: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        index_id = self.repository.resolve_index_id(knowledge_base_id, index_version_id)
        dense_rows = self.repository.load_dense_chunks(index_id)
        if not dense_rows:
            return RetrievalResult(
                index_version_id=index_id,
                query=query,
                hits=[],
                latency_ms=(time.perf_counter() - started) * 1000,
                diagnostics={
                    "dense_candidates": [],
                    "sparse_candidates": [],
                    "fused_candidates": [],
                },
            )

        embedding_started = time.perf_counter()
        query_vector = np.asarray((await self.embedding_client.embed([query]))[0], dtype=np.float32)
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm:
            query_vector /= query_norm

        dense_ranked: list[tuple[str, float]] = []
        for row in dense_rows:
            vector = row["embedding"]
            norm = float(np.linalg.norm(vector))
            score = float(np.dot(query_vector, vector / norm)) if norm else 0.0
            dense_ranked.append((row["id"], score))
        dense_ranked.sort(key=lambda item: item[1], reverse=True)
        dense_ranked = dense_ranked[: self.settings.retrieve_dense_k]

        sparse_rows = self.repository.sparse_search(
            index_id, query, self.settings.retrieve_sparse_k
        )
        sparse_ranked = [(row["id"], float(row["bm25_score"])) for row in sparse_rows]

        row_by_id = {row["id"]: row for row in dense_rows}
        dense_score_by_id = dict(dense_ranked)
        sparse_score_by_id = dict(sparse_ranked)
        fused_scores: dict[str, float] = {}
        rrf_constant = 60.0
        for rank, (chunk_id, _) in enumerate(dense_ranked, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
        for rank, (chunk_id, _) in enumerate(sparse_ranked, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
        fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[
            : self.settings.retrieve_fused_k
        ]
        fused_rows = [row_by_id[chunk_id] for chunk_id, _ in fused if chunk_id in row_by_id]
        documents = [f"{row['contextual_text']}\n{row['text']}" for row in fused_rows]
        rerank_started = time.perf_counter()
        rerank_error: str | None = None
        rerank_mode = type(self.rerank_client).__name__
        if documents:
            try:
                rerank_scores = await self.rerank_client.rerank(query, documents)
            except Exception as exc:
                rerank_error = f"{type(exc).__name__}: {exc}"[:1000]
                rerank_mode = "LexicalRerankClient(fallback)"
                rerank_scores = await LexicalRerankClient().rerank(query, documents)
        else:
            rerank_scores = []
        rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000

        hits: list[RetrievalHit] = []
        for row, rerank_score in zip(fused_rows, rerank_scores, strict=True):
            source = SourceLocation(**row["source"])
            hits.append(
                RetrievalHit(
                    chunk_id=row["id"],
                    text=row["text"],
                    contextual_text=row["contextual_text"],
                    source=source,
                    dense_score=dense_score_by_id.get(row["id"]),
                    sparse_score=sparse_score_by_id.get(row["id"]),
                    rrf_score=fused_scores[row["id"]],
                    rerank_score=float(rerank_score),
                    query_coverage=query_coverage(query, documents[len(hits)]),
                )
            )
        hits.sort(key=lambda hit: (hit.rerank_score, hit.rrf_score), reverse=True)
        limit = top_k or self.settings.rerank_k
        hits = hits[:limit]
        latency_ms = (time.perf_counter() - started) * 1000

        def candidate(chunk_id: str, score_name: str, score: float) -> dict[str, Any]:
            row = row_by_id.get(chunk_id, {})
            source = row.get("source", {})
            return {
                "chunk_id": chunk_id,
                "document_id": row.get("document_id"),
                "filename": source.get("filename"),
                "page": source.get("page"),
                "section": source.get("section"),
                score_name: round(score, 6),
            }

        diagnostics = {
            "dense_candidates": [
                candidate(chunk_id, "score", score) for chunk_id, score in dense_ranked
            ],
            "sparse_candidates": [
                candidate(chunk_id, "bm25", score) for chunk_id, score in sparse_ranked
            ],
            "fused_candidates": [
                candidate(chunk_id, "rrf", score) for chunk_id, score in fused
            ],
            "reranked_candidates": [
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.source.document_id,
                    "filename": hit.source.filename,
                    "page": hit.source.page,
                    "section": hit.source.section,
                    "rerank": round(hit.rerank_score, 6),
                    "coverage": round(hit.query_coverage, 6),
                }
                for hit in hits
            ],
            "providers": {
                "embedding": type(self.embedding_client).__name__,
                "rerank": rerank_mode,
            },
            "stage_latency_ms": {
                "embedding": round(embedding_latency_ms, 3),
                "rerank": round(rerank_latency_ms, 3),
            },
            "model_retry_counts": {
                "embedding": int(getattr(self.embedding_client, "last_retry_count", 0)),
                "rerank": int(getattr(self.rerank_client, "last_retry_count", 0)),
            },
            "rerank_error": rerank_error,
            "candidate_counts": {
                "dense": len(dense_ranked),
                "sparse": len(sparse_ranked),
                "fused": len(fused),
                "final": len(hits),
            },
            "latency_ms": round(latency_ms, 3),
        }
        return RetrievalResult(index_id, query, hits, latency_ms, diagnostics)
