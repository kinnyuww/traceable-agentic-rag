#!/usr/bin/env python3
"""Use an existing knowledge base or build one entirely through the REST API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


async def wait_job(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
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


async def resolve_knowledge_base(
    client: httpx.AsyncClient,
    base_url: str,
    knowledge_base_id: str | None,
    knowledge_base_name: str | None,
) -> dict[str, Any]:
    if knowledge_base_id:
        response = await client.get(f"{base_url}/v1/knowledge-bases/{knowledge_base_id}")
        response.raise_for_status()
        return response.json()
    response = await client.get(f"{base_url}/v1/knowledge-bases")
    response.raise_for_status()
    matches = [item for item in response.json() if item["name"] == knowledge_base_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one knowledge base named {knowledge_base_name!r}, found {len(matches)}"
        )
    return matches[0]


async def ask(
    client: httpx.AsyncClient,
    base_url: str,
    knowledge_base_id: str,
    question: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/v1/query",
        json={"knowledge_base_id": knowledge_base_id, "question": question},
    )
    response.raise_for_status()
    return response.json()


async def query_existing(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        knowledge_base = await resolve_knowledge_base(
            client,
            base_url,
            args.knowledge_base_id,
            args.knowledge_base_name,
        )
        result = await ask(client, base_url, knowledge_base["id"], args.question)
    return {"knowledge_base": knowledge_base, "query": result}


async def build_and_query(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    document_paths = [path.resolve() for path in args.documents]
    for path in document_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in CONTENT_TYPES:
            raise ValueError(f"Unsupported document type: {path}")

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        create = await client.post(
            f"{base_url}/v1/knowledge-bases",
            json={"name": args.name, "description": args.description},
        )
        create.raise_for_status()
        knowledge_base = create.json()
        files = [
            (
                "files",
                (
                    path.name,
                    path.read_bytes(),
                    CONTENT_TYPES[path.suffix.lower()],
                ),
            )
            for path in document_paths
        ]
        upload = await client.post(
            f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}/documents",
            files=files,
            data={"auto_index": "false"},
        )
        upload.raise_for_status()
        uploaded = upload.json()["items"]
        parse_jobs = [
            await wait_job(client, base_url, item["job_id"], args.timeout)
            for item in uploaded
        ]

        build = await client.post(
            f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}/index-builds",
            json={
                "contextualize": False,
                "activate": True,
                "chunk_strategy": args.chunk_strategy,
                "dense_backend": args.dense_backend,
            },
        )
        build.raise_for_status()
        build_request = build.json()
        index_job = await wait_job(
            client,
            base_url,
            build_request["job_id"],
            args.timeout,
        )
        current = await client.get(
            f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}"
        )
        current.raise_for_status()
        current_documents = await client.get(
            f"{base_url}/v1/knowledge-bases/{knowledge_base['id']}/documents"
        )
        current_documents.raise_for_status()
        result: dict[str, Any] = {
            "knowledge_base": current.json(),
            "documents": current_documents.json(),
            "parse_jobs": [
                {"id": job["id"], "status": job["status"], "result": job["result"]}
                for job in parse_jobs
            ],
            "index": {
                "id": build_request["index_version_id"],
                "job_id": index_job["id"],
                "status": index_job["status"],
            },
        }
        if args.question:
            result["query"] = await ask(
                client,
                base_url,
                knowledge_base["id"],
                args.question,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=600)
    commands = parser.add_subparsers(dest="command", required=True)

    query_parser = commands.add_parser("query", help="Query a knowledge base built in Web UI")
    identity = query_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--knowledge-base-id")
    identity.add_argument("--knowledge-base-name")
    query_parser.add_argument("--question", required=True)

    build_parser = commands.add_parser("build", help="Build and optionally query through REST")
    build_parser.add_argument("--name", required=True)
    build_parser.add_argument("--description", default="Created through REST API")
    build_parser.add_argument("--question")
    build_parser.add_argument(
        "--chunk-strategy", choices=("auto", "structure", "semantic"), default="auto"
    )
    build_parser.add_argument(
        "--dense-backend", choices=("auto", "exact", "hnsw"), default="auto"
    )
    build_parser.add_argument("documents", nargs="+", type=Path)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    if args.command == "query":
        result = await query_existing(args)
    else:
        result = await build_and_query(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
