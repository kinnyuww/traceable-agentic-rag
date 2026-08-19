#!/usr/bin/env python3
"""Evaluate representative live-service answers with Ragas and deterministic checks.

This adapter intentionally keeps Ragas outside the application runtime.  It
calls the public REST service, then uses the configured judge only for metrics
that genuinely require an LLM.  Gold-document checks remain deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import ragas
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)


def normalize(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value.lower())


def metric_payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "value", None)
    return {
        "value": float(value) if value is not None else None,
        "reason": getattr(result, "reason", None),
    }


async def score_metric(name: str, awaitable: Any) -> tuple[str, dict[str, Any]]:
    try:
        return name, metric_payload(await awaitable)
    except Exception as exc:
        return name, {
            "value": None,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
        }


def select_examples(manifest: dict[str, Any], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index in manifest["suggested_example_indices"]:
        example = manifest["examples"][index]
        if example.get("expected_answer") and example.get("expected_document_ids"):
            selected.append(example)
        if len(selected) >= count:
            break
    return selected


async def evaluate_example(
    client: httpx.AsyncClient,
    base_url: str,
    knowledge_base_id: str,
    dataset: str,
    example: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    query_response = await client.post(
        f"{base_url}/v1/query",
        json={"knowledge_base_id": knowledge_base_id, "question": example["question"]},
    )
    query_response.raise_for_status()
    answer = query_response.json()
    retrieval_response = await client.post(
        f"{base_url}/v1/retrieve",
        json={
            "knowledge_base_id": knowledge_base_id,
            "query": example["question"],
            "top_k": 6,
        },
    )
    retrieval_response.raise_for_status()
    retrieval = retrieval_response.json()
    run_response = await client.get(f"{base_url}{answer['trace_url']}")
    run_response.raise_for_status()
    run = run_response.json()

    retrieved_contexts = [
        f"{hit['contextual_text']}\n{hit['text']}" for hit in retrieval["hits"]
    ]
    selected_contexts = [citation["quote"] for citation in answer["citations"]]
    if not selected_contexts:
        selected_contexts = retrieved_contexts[:1]
    expected_documents = set(example["expected_document_ids"])
    cited_documents = {
        citation["source"]["document_id"] for citation in answer["citations"]
    }
    gold = example["expected_answer"]
    string_gold_found = bool(gold) and normalize(gold) in normalize(
        answer["answer"] + " " + " ".join(selected_contexts)
    )

    ragas_scores = dict(
        [
            await score_metric(
                "faithfulness",
                metrics["faithfulness"].ascore(
                    user_input=example["question"],
                    response=answer["answer"],
                    retrieved_contexts=selected_contexts,
                ),
            ),
            await score_metric(
                "context_precision_with_reference",
                metrics["context_precision"].ascore(
                    user_input=example["question"],
                    reference=gold,
                    retrieved_contexts=retrieved_contexts,
                ),
            ),
            await score_metric(
                "context_recall",
                metrics["context_recall"].ascore(
                    user_input=example["question"],
                    retrieved_contexts=retrieved_contexts,
                    reference=gold,
                ),
            ),
        ]
    )
    return {
        "dataset": dataset,
        "example_id": example["id"],
        "question": example["question"],
        "reference_answer": gold,
        "response": answer["answer"],
        "route": answer["route"],
        "rounds": answer["rounds"],
        "latency_ms": answer["latency_ms"],
        "run_id": answer["run_id"],
        "trace_stages": [event["stage"] for event in run["trace"]],
        "citations": answer["citations"],
        "deterministic": {
            "expected_document_count": len(expected_documents),
            "cited_expected_document_count": len(expected_documents & cited_documents),
            "evidence_hit": bool(expected_documents & cited_documents),
            "all_expected_evidence_cited": expected_documents.issubset(cited_documents),
            "answer_or_citation_contains_gold": string_gold_found,
        },
        "retrieval": {
            "top_k": len(retrieval["hits"]),
            "hits": [
                {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["source"]["document_id"],
                    "filename": hit["source"]["filename"],
                    "rerank_score": hit["rerank_score"],
                    "preview": hit["text"][:500],
                }
                for hit in retrieval["hits"]
            ],
        },
        "ragas": ragas_scores,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({name for row in rows for name in row["ragas"]})
    ragas_means: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            row["ragas"][name].get("value")
            for row in rows
            if row["ragas"][name].get("value") is not None
        ]
        ragas_means[name] = round(statistics.fmean(values), 6) if values else None
    return {
        "examples": len(rows),
        "evidence_hit_rate": round(
            sum(row["deterministic"]["evidence_hit"] for row in rows) / len(rows), 6
        ),
        "all_expected_evidence_cited_rate": round(
            sum(row["deterministic"]["all_expected_evidence_cited"] for row in rows)
            / len(rows),
            6,
        ),
        "answer_or_citation_contains_gold_rate": round(
            sum(row["deterministic"]["answer_or_citation_contains_gold"] for row in rows)
            / len(rows),
            6,
        ),
        "mean_latency_ms": round(statistics.fmean(row["latency_ms"] for row in rows), 3),
        "ragas_mean": ragas_means,
        "ragas_errors": sum(
            "error" in metric for row in rows for metric in row["ragas"].values()
        ),
    }


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get("RAG_LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RAG_LLM_API_KEY is required for the Ragas judge")
    judge_client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.judge_base_url,
        timeout=args.judge_timeout,
        max_retries=2,
    )
    judge = llm_factory(
        args.judge_model,
        provider="openai",
        client=judge_client,
        max_tokens=args.judge_max_tokens,
    )
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.manifest_dir.glob("*.json"))
        if path.stem in set(args.datasets)
    ]
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as service_client:
        health = await service_client.get(f"{args.base_url.rstrip('/')}/v1/health")
        health.raise_for_status()
        for manifest in manifests:
            dataset = manifest["dataset"]["slug"]
            for example in select_examples(manifest, args.examples_per_dataset):
                print(
                    json.dumps(
                        {"stage": "ragas", "dataset": dataset, "question": example["question"]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                rows.append(
                    await evaluate_example(
                        service_client,
                        args.base_url.rstrip("/"),
                        manifest["knowledge_base"]["id"],
                        dataset,
                        example,
                        metrics,
                    )
                )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Representative live REST generation + Ragas diagnostic; not a leaderboard",
        "service": {
            "base_url": args.base_url.rstrip("/"),
            "health": health.json(),
        },
        "judge": {
            "framework": f"ragas {ragas.__version__}",
            "provider": "DeepSeek official OpenAI-compatible API",
            "model": args.judge_model,
            "same_model_family_as_answer_generator": True,
        },
        "metrics": {
            "deterministic": [
                "gold document evidence hit",
                "all gold documents cited",
                "normalized reference string presence",
            ],
            "ragas": [
                "faithfulness",
                "context_precision_with_reference",
                "context_recall",
            ],
        },
        "limitations": [
            "DeepSeek judges answers generated by DeepSeek, so judge scores are not independent.",
            "Ragas is run on a small guided sample to demonstrate the evaluation path and control cost.",
            "MIRACL has relevance qrels but no reference answers, so it remains a deterministic retrieval evaluation and is not included in these answer metrics.",
            "Faithfulness uses returned citation excerpts; retrieval metrics use the separate /v1/retrieve top-6 contexts.",
        ],
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["aggregate"]}, ensure_ascii=False, indent=2))
    await judge_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/demo-manifests"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["qasper", "multihoprag"],
        choices=["qasper", "multihoprag", "miracl_zh"],
    )
    parser.add_argument("--examples-per-dataset", type=int, default=1)
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/live-ragas-deepseek-v4-flash.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
