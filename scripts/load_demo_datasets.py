#!/usr/bin/env python3
"""Load the fixed public benchmark slices into a running RAG REST service.

The script deliberately uses only public REST routes.  A successful run proves
that the same upload, worker, index and query boundaries used by the Web UI can
also ingest the reproducible benchmark slices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from run_benchmarks import BenchmarkExample, BenchmarkSuite, load_multihop, load_qasper
from run_miracl_benchmark import load_suite as load_miracl


@dataclass(frozen=True)
class DemoDefinition:
    slug: str
    display_name: str
    suite: BenchmarkSuite
    suggested_indices: tuple[int, ...]


def demo_definitions(args: argparse.Namespace) -> list[DemoDefinition]:
    qasper = load_qasper(args.qasper)
    multihop = load_multihop(args.multihop)
    miracl = load_miracl(args.miracl_topics, args.miracl_qrels, args.miracl_corpus)
    return [
        DemoDefinition(
            "qasper",
            "示例 · QASPER 论文问答",
            qasper,
            (2, 4, 6, 8, 9),
        ),
        DemoDefinition(
            "multihoprag",
            "示例 · MultiHop-RAG 多跳推理",
            multihop,
            (0, 1, 8),
        ),
        DemoDefinition(
            "miracl_zh",
            "示例 · MIRACL 中文检索",
            miracl,
            (0, 1, 15, 16, 18),
        ),
    ]


def description_for(demo: DemoDefinition) -> str:
    questions = [demo.suite.examples[index].question for index in demo.suggested_indices]
    lines = [
        f"真实公开数据固定切片：{demo.suite.source}",
        f"{len(demo.suite.documents)} 个文档，{len(demo.suite.examples)} 个金标问题。",
        "示例问题：",
        *[f"- {question}" for question in questions],
    ]
    return "\n".join(lines)[:1000]


async def wait_job(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    *,
    deadline_seconds: float = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        response = await client.get(f"{base_url}/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise RuntimeError(f"Job {job_id} failed: {job.get('error')}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


async def wait_jobs(
    client: httpx.AsyncClient, base_url: str, job_ids: list[str]
) -> list[dict[str, Any]]:
    return await asyncio.gather(*(wait_job(client, base_url, job_id) for job_id in job_ids))


def example_payload(
    example: BenchmarkExample, key_to_document_id: dict[str, str]
) -> dict[str, Any]:
    return {
        "id": example.id,
        "category": example.category,
        "question": example.question,
        "expected_answer": example.expected_answer,
        "answerable": example.answerable,
        "expected_document_keys": example.expected_document_keys,
        "expected_document_ids": [
            key_to_document_id[key]
            for key in example.expected_document_keys
            if key in key_to_document_id
        ],
    }


async def create_demo(
    client: httpx.AsyncClient,
    base_url: str,
    demo: DemoDefinition,
    output_dir: Path,
    *,
    upload_batch_size: int,
    run_native_eval: bool,
) -> dict[str, Any]:
    knowledge_bases = (await client.get(f"{base_url}/v1/knowledge-bases")).json()
    ready = next(
        (
            item
            for item in knowledge_bases
            if item["name"] == demo.display_name
            and item.get("active_index_version_id")
            and item.get("document_count") == len(demo.suite.documents)
        ),
        None,
    )
    if ready:
        manifest_path = output_dir / f"{demo.slug}.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"{demo.display_name} already exists, but its local manifest is missing: {manifest_path}"
        )

    existing_names = {item["name"] for item in knowledge_bases}
    display_name = demo.display_name
    if display_name in existing_names:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        display_name = f"{display_name} · {stamp}"
    response = await client.post(
        f"{base_url}/v1/knowledge-bases",
        json={"name": display_name, "description": description_for(demo)},
    )
    response.raise_for_status()
    knowledge_base = response.json()

    key_to_document_id: dict[str, str] = {}
    parse_jobs: list[str] = []
    for start in range(0, len(demo.suite.documents), upload_batch_size):
        batch = demo.suite.documents[start : start + upload_batch_size]
        files = [
            ("files", (document.filename, document.content.encode("utf-8"), "text/markdown"))
            for document in batch
        ]
        upload = await client.post(
            f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}/documents",
            files=files,
            data={"auto_index": "false"},
        )
        upload.raise_for_status()
        items = upload.json()["items"]
        for document, item in zip(batch, items, strict=True):
            key_to_document_id[document.key] = item["document"]["id"]
            parse_jobs.append(item["job_id"])
    await wait_jobs(client, base_url, parse_jobs)

    build = await client.post(
        f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}/index-builds",
        json={"contextualize": False, "activate": True},
    )
    build.raise_for_status()
    build_payload = build.json()
    index_job = await wait_job(client, base_url, build_payload["job_id"])

    examples = [example_payload(example, key_to_document_id) for example in demo.suite.examples]
    native_evaluation: dict[str, Any] | None = None
    if run_native_eval:
        selected = [examples[index] for index in demo.suggested_indices[:3]]
        evaluation = await client.post(
            f"{base_url}/v1/evaluations",
            json={
                "knowledge_base_id": knowledge_base["id"],
                "name": f"{demo.slug}-guided-live-eval",
                "examples": [
                    {
                        "question": item["question"],
                        "expected_answer": item["expected_answer"] or None,
                        "expected_document_ids": item["expected_document_ids"],
                        "answerable": item["answerable"],
                    }
                    for item in selected
                ],
            },
        )
        evaluation.raise_for_status()
        evaluation_payload = evaluation.json()
        await wait_job(client, base_url, evaluation_payload["job_id"])
        evaluation_result = await client.get(
            f"{base_url}/v1/evaluations/{evaluation_payload['id']}"
        )
        evaluation_result.raise_for_status()
        native_evaluation = evaluation_result.json()

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "slug": demo.slug,
            "name": demo.suite.name,
            "source": demo.suite.source,
            "sampling_policy": demo.suite.sampling_policy,
        },
        "knowledge_base": knowledge_base,
        "index_version_id": build_payload["index_version_id"],
        "index_job_result": index_job.get("result"),
        "document_key_to_id": key_to_document_id,
        "suggested_example_indices": list(demo.suggested_indices),
        "examples": examples,
        "native_evaluation": native_evaluation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{demo.slug}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


async def main_async(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    selected = set(args.only or ["qasper", "multihoprag", "miracl_zh"])
    demos = [demo for demo in demo_definitions(args) if demo.slug in selected]
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        health = await client.get(f"{base_url}/v1/health")
        health.raise_for_status()
        manifests: list[dict[str, Any]] = []
        for demo in demos:
            print(
                json.dumps(
                    {
                        "stage": "loading",
                        "dataset": demo.slug,
                        "documents": len(demo.suite.documents),
                        "examples": len(demo.suite.examples),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            manifests.append(
                await create_demo(
                    client,
                    base_url,
                    demo,
                    args.output_dir,
                    upload_batch_size=args.upload_batch_size,
                    run_native_eval=args.run_native_eval,
                )
            )
    summary = {
        "status": "ready",
        "base_url": base_url,
        "knowledge_bases": [
            {
                "dataset": manifest["dataset"]["slug"],
                "id": manifest["knowledge_base"]["id"],
                "name": manifest["knowledge_base"]["name"],
                "documents": len(manifest["document_key_to_id"]),
                "index_version_id": manifest["index_version_id"],
                "native_evaluation": (
                    manifest["native_evaluation"].get("metrics")
                    if manifest.get("native_evaluation")
                    else None
                ),
            }
            for manifest in manifests
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--qasper",
        type=Path,
        default=Path("data/benchmark-sources/qasper-validation-first-rows.json"),
    )
    parser.add_argument(
        "--multihop",
        type=Path,
        default=Path("data/benchmark-sources/multihoprag-train-first-rows.json"),
    )
    parser.add_argument(
        "--miracl-topics",
        type=Path,
        default=Path("data/benchmark-sources/miracl-zh-dev-topics.tsv"),
    )
    parser.add_argument(
        "--miracl-qrels",
        type=Path,
        default=Path("data/benchmark-sources/miracl-zh-dev-qrels.tsv"),
    )
    parser.add_argument(
        "--miracl-corpus",
        type=Path,
        default=Path("data/benchmark-sources/miracl-zh-docs-0.jsonl.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo-manifests"))
    parser.add_argument("--upload-batch-size", type=int, default=20)
    parser.add_argument(
        "--only",
        action="append",
        choices=["qasper", "multihoprag", "miracl_zh"],
        help="May be passed more than once; defaults to all three demos.",
    )
    parser.add_argument("--run-native-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
