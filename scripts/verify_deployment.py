#!/usr/bin/env python3
"""Black-box REST verification for a running Docker deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


async def wait_job(client: httpx.AsyncClient, base_url: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = await client.get(f"{base_url}/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise RuntimeError(f"Job {job_id} failed: {job['error']}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


async def post_query(
    client: httpx.AsyncClient, base_url: str, knowledge_base_id: str, question: str
) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/v1/query",
        json={"knowledge_base_id": knowledge_base_id, "question": question},
    )
    response.raise_for_status()
    return response.json()


async def main_async(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        health_response = await client.get(f"{base_url}/v1/health")
        health_response.raise_for_status()
        health = health_response.json()
        openapi_response = await client.get(f"{base_url}/openapi.json")
        openapi_response.raise_for_status()
        openapi = openapi_response.json()
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        kb_response = await client.post(
            f"{base_url}/v1/knowledge-bases",
            json={"name": f"Docker smoke {stamp}", "description": "automated black-box check"},
        )
        kb_response.raise_for_status()
        kb = kb_response.json()
        fixture_bytes = args.fixture.read_bytes()
        upload_response = await client.post(
            f"{base_url}/v1/knowledge-bases/{kb['id']}/documents",
            files={"files": (args.fixture.name, fixture_bytes, "text/markdown")},
            data={"auto_index": "false"},
        )
        upload_response.raise_for_status()
        uploaded = upload_response.json()["items"][0]
        parse_job = await wait_job(client, base_url, uploaded["job_id"])
        build_response = await client.post(
            f"{base_url}/v1/knowledge-bases/{kb['id']}/index-builds",
            json={"contextualize": False, "activate": True},
        )
        build_response.raise_for_status()
        build = build_response.json()
        index_job = await wait_job(client, base_url, build["job_id"])
        single = await post_query(client, base_url, kb["id"], "员工每年有多少天带薪年假？")
        if not single["citations"] or "十二天" not in single["citations"][0]["quote"]:
            raise AssertionError("Single query did not return the expected cited evidence")
        run_response = await client.get(f"{base_url}{single['trace_url']}")
        run_response.raise_for_status()
        run = run_response.json()
        trace_stages = [event["stage"] for event in run["trace"]]
        required = {
            "query_received",
            "query_understanding",
            "retrieval_round",
            "retrieval_merge_rerank",
            "evidence_gate",
            "context_selection",
            "answer_generation",
            "run_completed",
        }
        if not required.issubset(trace_stages):
            raise AssertionError(f"Trace is incomplete: {trace_stages}")
        concurrent = await asyncio.gather(
            post_query(client, base_url, kb["id"], "年假需要提前几个工作日申请？"),
            post_query(client, base_url, kb["id"], "公司的主要办公地点在哪里？"),
        )
        evaluation_response = await client.post(
            f"{base_url}/v1/evaluations",
            json={
                "knowledge_base_id": kb["id"],
                "name": "docker-smoke-evaluation",
                "examples": [
                    {
                        "question": "员工年假有多少天？",
                        "expected_answer": "十二天",
                        "expected_document_ids": [uploaded["document"]["id"]],
                        "answerable": True,
                    },
                    {
                        "question": "文档中火星基地的负责人是谁？",
                        "expected_answer": None,
                        "expected_document_ids": [],
                        "answerable": False,
                    },
                ],
            },
        )
        evaluation_response.raise_for_status()
        evaluation_job = await wait_job(client, base_url, evaluation_response.json()["job_id"])
        evaluation_result_response = await client.get(
            f"{base_url}/v1/evaluations/{evaluation_response.json()['id']}"
        )
        evaluation_result_response.raise_for_status()
        evaluation = evaluation_result_response.json()
    concurrent_latencies = [item["latency_ms"] for item in concurrent]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "status": "passed",
        "health": health,
        "openapi_route_count": len(openapi["paths"]),
        "knowledge_base_id": kb["id"],
        "document_id": uploaded["document"]["id"],
        "index_version_id": build["index_version_id"],
        "parse_job": {
            "status": parse_job["status"],
            "trace_stages": [event["stage"] for event in parse_job["trace"]],
            "result": parse_job["result"],
        },
        "index_job": {
            "status": index_job["status"],
            "trace_stages": [event["stage"] for event in index_job["trace"]],
            "result": index_job["result"],
        },
        "single_query": {
            "route": single["route"],
            "rounds": single["rounds"],
            "latency_ms": single["latency_ms"],
            "citation_count": len(single["citations"]),
            "trace_stages": trace_stages,
        },
        "two_concurrent_queries": {
            "routes": [item["route"] for item in concurrent],
            "latencies_ms": concurrent_latencies,
            "mean_latency_ms": statistics.fmean(concurrent_latencies),
            "max_latency_ms": max(concurrent_latencies),
            "under_5_seconds": max(concurrent_latencies) <= 5000,
        },
        "evaluation_job": {
            "status": evaluation_job["status"],
            "trace_stages": [event["stage"] for event in evaluation_job["trace"]],
        },
        "evaluation": evaluation,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/docker-smoke.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/docker-e2e.json"))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
