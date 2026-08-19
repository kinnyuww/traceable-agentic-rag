#!/usr/bin/env python3
"""Run a fixed MIRACL-zh shard diagnostic with official human qrels."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import platform
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_benchmarks import (
    BenchmarkDocument,
    BenchmarkExample,
    BenchmarkSuite,
    _sha256,
    run_suite,
)

from ragagent.config import Settings


def load_suite(
    topics_path: Path, qrels_path: Path, corpus_path: Path, *, query_limit: int = 20
) -> BenchmarkSuite:
    topics: list[tuple[str, str]] = []
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        query_id, query = line.split("\t", 1)
        topics.append((query_id, query))
    qrels: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        query_id, _, document_id, relevance = line.split("\t")
        qrels[query_id].append((document_id, int(relevance)))
    judged_ids = {document_id for rows in qrels.values() for document_id, _ in rows}
    matched: dict[str, dict[str, Any]] = {}
    distractors: list[dict[str, Any]] = []
    with gzip.open(corpus_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            document_id = str(row["docid"])
            if document_id in judged_ids:
                matched[document_id] = row
            elif len(distractors) < 100:
                distractors.append(row)
    selected: list[tuple[str, str]] = []
    for query_id, query in topics:
        positives = [
            document_id
            for document_id, relevance in qrels[query_id]
            if relevance > 0 and document_id in matched
        ]
        if positives:
            selected.append((query_id, query))
        if len(selected) >= query_limit:
            break
    if len(selected) < query_limit:
        raise RuntimeError(
            f"Only {len(selected)} dev queries had a positive document in this corpus shard"
        )
    selected_judged_ids = {
        document_id
        for query_id, _ in selected
        for document_id, _ in qrels[query_id]
        if document_id in matched
    }
    documents: list[BenchmarkDocument] = []
    for document_id in sorted(selected_judged_ids):
        row = matched[document_id]
        documents.append(
            BenchmarkDocument(
                key=f"miracl_{document_id}",
                filename=f"miracl-{document_id.replace('#', '-')}.md",
                content=f"# {row.get('title', '')}\n\n{row.get('text', '')}",
            )
        )
    selected_judged_ids_set = set(selected_judged_ids)
    for row in distractors:
        document_id = str(row["docid"])
        if document_id in selected_judged_ids_set:
            continue
        documents.append(
            BenchmarkDocument(
                key=f"miracl_{document_id}",
                filename=f"miracl-{document_id.replace('#', '-')}-distractor.md",
                content=f"# {row.get('title', '')}\n\n{row.get('text', '')}",
            )
        )
    examples = [
        BenchmarkExample(
            id=query_id,
            question=query,
            expected_answer="",
            expected_document_keys=[
                f"miracl_{document_id}"
                for document_id, relevance in qrels[query_id]
                if relevance > 0 and document_id in matched
            ],
            answerable=True,
            category="miracl_zh_dev",
        )
        for query_id, query in selected
    ]
    return BenchmarkSuite(
        name="miracl-zh-dev-shard0-fixed-slice",
        source="miracl/miracl zh dev topics/qrels + miracl-corpus zh docs-0",
        sampling_policy=(
            f"first {query_limit} dev topic IDs with at least one positive in docs-0; "
            "all judged docs found in shard plus first 100 unjudged shard passages as distractors"
        ),
        documents=documents,
        examples=examples,
    )


async def main_async(args: argparse.Namespace) -> None:
    suite = load_suite(args.topics, args.qrels, args.corpus, query_limit=args.query_limit)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="traceable-rag-miracl-") as directory:
        settings = Settings(
            _env_file=None,
            data_dir=Path(directory) / "data",
            inline_jobs=True,
            embedding_provider="openai",
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_dimensions=1024,
            rerank_provider="http",
            rerank_endpoint=args.rerank_endpoint,
            rerank_model=args.rerank_model,
            llm_enabled=False,
            chunk_default_strategy=args.chunk_strategy,
        )
        result = await run_suite(suite, settings)
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "MIRACL-zh official-qrels fixed-shard diagnostic; not official leaderboard",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "embedding_model": args.embedding_model,
            "rerank_model": args.rerank_model,
            "generation_model": "disabled",
            "chunk_strategy": args.chunk_strategy,
        },
        "inputs": {
            "topics": {"path": str(args.topics), "sha256": _sha256(args.topics)},
            "qrels": {"path": str(args.qrels), "sha256": _sha256(args.qrels)},
            "corpus_shard": {"path": str(args.corpus), "sha256": _sha256(args.corpus)},
        },
        "suite": result,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/results/miracl-zh-real-model.json")
    )
    parser.add_argument(
        "--embedding-base-url",
        default="http://localhost:12434/engines/llama.cpp/v1",
    )
    parser.add_argument("--embedding-model", default="ai/qwen3-embedding:0.6B-F16")
    parser.add_argument("--rerank-endpoint", default="http://localhost:12434/rerank")
    parser.add_argument("--rerank-model", default="ai/qwen3-reranker:0.6B")
    parser.add_argument(
        "--chunk-strategy",
        choices=("auto", "structure", "semantic"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
