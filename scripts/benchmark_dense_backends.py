#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from usearch.index import Index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact cosine and USearch HNSW")
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260819)
    vectors = normalize(rng.normal(size=(args.vectors, args.dimensions)).astype(np.float32))
    queries = normalize(rng.normal(size=(args.queries, args.dimensions)).astype(np.float32))

    build_started = time.perf_counter()
    hnsw = Index(
        ndim=args.dimensions,
        metric="cos",
        dtype="f32",
        connectivity=32,
        expansion_add=512,
        expansion_search=512,
    )
    hnsw.add(np.arange(args.vectors, dtype=np.uint64), vectors)
    build_ms = (time.perf_counter() - build_started) * 1000

    exact_latencies: list[float] = []
    hnsw_latencies: list[float] = []
    recalls: list[float] = []
    for query in queries:
        started = time.perf_counter()
        scores = vectors @ query
        exact_ids = np.argpartition(-scores, args.top_k - 1)[: args.top_k]
        exact_latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        matches = hnsw.search(query, count=args.top_k)
        hnsw_latencies.append((time.perf_counter() - started) * 1000)
        recalls.append(len(set(map(int, matches.keys)) & set(map(int, exact_ids))) / args.top_k)

    payload = {
        "kind": "synthetic_dense_microbenchmark",
        "warning": "Random vectors measure the local implementation, not domain retrieval quality.",
        "seed": 20260819,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "dataset": {
            "vectors": args.vectors,
            "dimensions": args.dimensions,
            "queries": args.queries,
            "top_k": args.top_k,
        },
        "hnsw": {
            "connectivity": 32,
            "expansion_add": 512,
            "expansion_search": 512,
            "build_ms": round(build_ms, 3),
            "mean_recall_at_k": round(float(np.mean(recalls)), 6),
            "latency_ms": latency_summary(hnsw_latencies),
        },
        "exact": {"latency_ms": latency_summary(exact_latencies)},
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
    }


if __name__ == "__main__":
    main()
