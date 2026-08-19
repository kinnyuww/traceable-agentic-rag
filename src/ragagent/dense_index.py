from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from usearch.index import Index

from ragagent.config import Settings
from ragagent.repositories import Repository

DenseBackend = Literal["exact", "hnsw"]


@dataclass(frozen=True)
class DenseSearchResult:
    ranked: list[tuple[str, float]]
    rows: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class DenseIndexManager:
    """Build and query immutable exact/HNSW dense indexes.

    Exact indexes are cached as a normalized matrix after their first query.
    HNSW indexes are persisted beside SQLite and memory-mapped on first use.
    """

    def __init__(self, repository: Repository, settings: Settings):
        self.repository = repository
        self.settings = settings
        self._exact_cache: dict[str, tuple[list[dict[str, Any]], np.ndarray]] = {}
        self._hnsw_cache: dict[str, tuple[Index, list[str]]] = {}

    def initialize(self) -> None:
        self.settings.vector_indexes_dir.mkdir(parents=True, exist_ok=True)

    def resolve_backend(self, requested: str, chunk_count: int) -> DenseBackend:
        if requested == "auto":
            return (
                "hnsw"
                if chunk_count >= self.settings.dense_auto_hnsw_min_chunks
                else "exact"
            )
        if requested not in {"exact", "hnsw"}:
            raise ValueError(f"Unsupported dense backend: {requested}")
        return requested  # type: ignore[return-value]

    def build_hnsw(self, index_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            raise ValueError("Cannot build an HNSW index without chunks")
        self.initialize()
        vectors = np.asarray([item["embedding"] for item in chunks], dtype=np.float32)
        if vectors.ndim != 2 or not vectors.shape[1]:
            raise ValueError("HNSW vectors must be a non-empty 2D matrix")
        keys = np.arange(len(chunks), dtype=np.uint64)
        index = Index(
            ndim=vectors.shape[1],
            metric="cos",
            dtype="f32",
            connectivity=self.settings.hnsw_connectivity,
            expansion_add=self.settings.hnsw_expansion_add,
            expansion_search=self.settings.hnsw_expansion_search,
        )
        index.add(keys, vectors)

        native_path, metadata_path = self._paths(index_id)
        native_temp = native_path.with_suffix(".usearch.tmp")
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        index.save(native_temp)
        metadata_temp.write_text(
            json.dumps(
                {
                    "index_version_id": index_id,
                    "chunk_ids": [item["id"] for item in chunks],
                    "dimensions": int(vectors.shape[1]),
                    "metric": "cos",
                    "connectivity": self.settings.hnsw_connectivity,
                    "expansion_add": self.settings.hnsw_expansion_add,
                    "expansion_search": self.settings.hnsw_expansion_search,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        native_temp.replace(native_path)
        metadata_temp.replace(metadata_path)
        self._hnsw_cache.pop(index_id, None)
        return {
            "artifact": str(native_path),
            "chunks": len(chunks),
            "dimensions": int(vectors.shape[1]),
            "connectivity": self.settings.hnsw_connectivity,
            "expansion_add": self.settings.hnsw_expansion_add,
            "expansion_search": self.settings.hnsw_expansion_search,
        }

    def search(
        self,
        *,
        index_id: str,
        backend: DenseBackend,
        query_vector: np.ndarray,
        limit: int,
    ) -> DenseSearchResult:
        if backend == "hnsw":
            return self._search_hnsw(index_id, query_vector, limit)
        return self._search_exact(index_id, query_vector, limit)

    def _search_exact(
        self, index_id: str, query_vector: np.ndarray, limit: int
    ) -> DenseSearchResult:
        cached = self._exact_cache.get(index_id)
        cache_hit = cached is not None
        if cached is None:
            rows = self.repository.load_dense_chunks(index_id)
            if rows:
                matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
            else:
                matrix = np.empty((0, query_vector.size), dtype=np.float32)
            cached = (rows, matrix)
            self._exact_cache[index_id] = cached
        rows, matrix = cached
        if not rows:
            return DenseSearchResult([], [], self._diagnostics("exact", 0, cache_hit))
        scores = matrix @ query_vector
        count = min(limit, len(rows))
        if count == len(rows):
            positions = np.argsort(-scores)
        else:
            unsorted = np.argpartition(-scores, count - 1)[:count]
            positions = unsorted[np.argsort(-scores[unsorted])]
        ranked = [(rows[int(position)]["id"], float(scores[position])) for position in positions]
        return DenseSearchResult(
            ranked,
            rows,
            self._diagnostics("exact", len(rows), cache_hit),
        )

    def _search_hnsw(
        self, index_id: str, query_vector: np.ndarray, limit: int
    ) -> DenseSearchResult:
        cached = self._hnsw_cache.get(index_id)
        cache_hit = cached is not None
        if cached is None:
            native_path, metadata_path = self._paths(index_id)
            if not native_path.exists() or not metadata_path.exists():
                raise RuntimeError(f"HNSW artifact is missing for index {index_id}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            chunk_ids = metadata["chunk_ids"]
            index = Index.restore(
                native_path,
                view=True,
                expansion_search=self.settings.hnsw_expansion_search,
            )
            if index is None or len(index) != len(chunk_ids):
                raise RuntimeError(f"HNSW artifact metadata is inconsistent for index {index_id}")
            cached = (index, chunk_ids)
            self._hnsw_cache[index_id] = cached
        index, chunk_ids = cached
        count = min(limit, len(chunk_ids))
        matches = index.search(query_vector.astype(np.float32), count=count)
        ranked = [
            (chunk_ids[int(key)], float(1.0 - distance))
            for key, distance in zip(matches.keys, matches.distances, strict=True)
        ]
        rows = self.repository.load_chunks_by_ids(index_id, [item[0] for item in ranked])
        return DenseSearchResult(
            ranked,
            rows,
            self._diagnostics("hnsw", len(chunk_ids), cache_hit),
        )

    def _paths(self, index_id: str) -> tuple[Path, Path]:
        return (
            self.settings.vector_indexes_dir / f"{index_id}.usearch",
            self.settings.vector_indexes_dir / f"{index_id}.json",
        )

    def _diagnostics(
        self, backend: DenseBackend, indexed_chunks: int, cache_hit: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": backend,
            "indexed_chunks": indexed_chunks,
            "cache_hit": cache_hit,
            "exact": backend == "exact",
        }
        if backend == "hnsw":
            payload["parameters"] = {
                "connectivity": self.settings.hnsw_connectivity,
                "expansion_search": self.settings.hnsw_expansion_search,
            }
        return payload
