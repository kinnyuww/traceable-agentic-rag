from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from ragagent.chunking import chunk_document
from ragagent.config import Settings
from ragagent.documents import (
    DocumentError,
    ParsedDocument,
    content_sha256,
    parse_document,
    safe_filename,
    validate_extension,
)
from ragagent.models import EmbeddingClient
from ragagent.repositories import ConflictError, Repository
from ragagent.schemas import (
    DocumentRead,
    DocumentStatus,
    IndexStatus,
    JobRead,
    JobStatus,
)


class ObjectStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def initialize(self) -> None:
        self.settings.objects_dir.mkdir(parents=True, exist_ok=True)
        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def put(self, digest: str, content: bytes) -> Path:
        self.initialize()
        target = self.settings.objects_dir / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return target

    def artifact_path(self, document_id: str) -> Path:
        self.initialize()
        return self.settings.artifacts_dir / f"{document_id}.json"


class IngestionService:
    def __init__(self, repository: Repository, object_store: ObjectStore, settings: Settings):
        self.repository = repository
        self.object_store = object_store
        self.settings = settings

    def accept_upload(
        self,
        *,
        knowledge_base_id: str,
        filename: str,
        content_type: str,
        content: bytes,
        auto_index: bool = False,
    ) -> tuple[DocumentRead, JobRead]:
        filename = safe_filename(filename)
        validate_extension(filename)
        max_bytes = self.settings.upload_max_mb * 1024 * 1024
        if not content:
            raise DocumentError("The uploaded file is empty")
        if len(content) > max_bytes:
            raise DocumentError(f"File exceeds the {self.settings.upload_max_mb} MB limit")
        digest = content_sha256(content)
        object_path = self.object_store.put(digest, content)
        document = self.repository.create_document(
            kb_id=knowledge_base_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(content),
            sha256=digest,
            object_path=object_path,
        )
        job = self.repository.create_job(
            "parse_document",
            {"document_id": document.id, "auto_index": auto_index},
        )
        return document, job

    def request_index_build(
        self,
        *,
        knowledge_base_id: str,
        contextualize: bool = False,
        activate: bool = True,
    ) -> tuple[str, JobRead]:
        config = {
            "chunker": "structure_v1",
            "target_chars": 1100,
            "overlap_chars": 160,
            "contextualize": contextualize,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model or "deterministic-feature-hash-v1",
        }
        index_id = self.repository.create_index_version(knowledge_base_id, config)
        job = self.repository.create_job(
            "build_index",
            {
                "knowledge_base_id": knowledge_base_id,
                "index_version_id": index_id,
                "activate": activate,
            },
        )
        return index_id, job


class JobProcessor:
    def __init__(
        self,
        repository: Repository,
        ingestion: IngestionService,
        embedding_client: EmbeddingClient,
    ):
        self.repository = repository
        self.ingestion = ingestion
        self.embedding_client = embedding_client
        self.agent_service: Any | None = None

    async def process(self, job: JobRead) -> None:
        try:
            self.repository.update_job(
                job.id, status=JobStatus.PROCESSING, stage="starting", progress=0.01
            )
            if job.type == "parse_document":
                result = await self._parse_document(job)
            elif job.type == "build_index":
                result = await self._build_index(job)
            elif job.type == "evaluation":
                result = await self._evaluate(job)
            else:
                raise ValueError(f"Unsupported job type: {job.type}")
            self.repository.update_job(
                job.id,
                status=JobStatus.SUCCEEDED,
                stage="completed",
                progress=1.0,
                result=result,
            )
        except Exception as exc:
            await self._mark_domain_failure(job, exc)
            self.repository.update_job(
                job.id,
                status=JobStatus.FAILED,
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _mark_domain_failure(self, job: JobRead, exc: Exception) -> None:
        if job.type == "parse_document":
            self.repository.update_document_status(
                job.payload["document_id"], DocumentStatus.FAILED, error=str(exc)
            )
        elif job.type == "build_index":
            self.repository.update_index_status(
                job.payload["index_version_id"], IndexStatus.FAILED, error=str(exc)
            )
        elif job.type == "evaluation":
            self.repository.update_evaluation(job.payload["evaluation_id"], status="failed")

    async def _parse_document(self, job: JobRead) -> dict[str, Any]:
        started = time.perf_counter()
        document_id = job.payload["document_id"]
        record = self.repository.get_document_record(document_id)
        self.repository.update_document_status(document_id, DocumentStatus.PARSING)
        self.repository.update_job(job.id, stage="parsing", progress=0.2)
        parsed = await asyncio.to_thread(
            parse_document,
            document_id,
            record["filename"],
            Path(record["object_path"]),
        )
        artifact_path = self.ingestion.object_store.artifact_path(document_id)
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(parsed.to_json(), encoding="utf-8")
        temporary.replace(artifact_path)
        self.repository.update_document_status(
            document_id, DocumentStatus.READY, artifact_path=artifact_path
        )
        result: dict[str, Any] = {
            "document_id": document_id,
            "sections": len(parsed.sections),
            "characters": len(parsed.text),
            "parser": Path(record["filename"]).suffix.lower().lstrip("."),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if job.payload.get("auto_index"):
            index_id, index_job = self.ingestion.request_index_build(
                knowledge_base_id=record["knowledge_base_id"]
            )
            result.update({"index_version_id": index_id, "index_job_id": index_job.id})
        return result

    async def _build_index(self, job: JobRead) -> dict[str, Any]:
        started = time.perf_counter()
        kb_id = job.payload["knowledge_base_id"]
        index_id = job.payload["index_version_id"]
        activate = bool(job.payload.get("activate", True))
        index_config = self.repository.get_index_record(index_id)["config"]
        if index_config.get("contextualize"):
            raise DocumentError(
                "LLM contextual summaries are intentionally not implemented in v0.1; "
                "use structure-aware context metadata or set contextualize=false"
            )
        self.repository.update_index_status(index_id, IndexStatus.BUILDING)
        records = self.repository.list_ready_document_records(kb_id)
        if not records:
            raise ConflictError("No parsed documents are ready for indexing")
        drafts: list[dict[str, Any]] = []
        manifest: list[dict[str, Any]] = []
        for position, record in enumerate(records, start=1):
            artifact_path = Path(record["artifact_path"])
            parsed = ParsedDocument.from_path(artifact_path)
            manifest.append(
                {
                    "document_id": record["id"],
                    "sha256": record["sha256"],
                    "version": record["version"],
                }
            )
            for draft in chunk_document(parsed):
                stable = hashlib.sha256(
                    f"{record['id']}:{record['version']}:{draft.ordinal}:{draft.text}".encode()
                ).hexdigest()[:20]
                drafts.append(
                    {
                        "id": f"chk_{stable}",
                        "knowledge_base_id": kb_id,
                        "document_id": record["id"],
                        "ordinal": draft.ordinal,
                        "text": draft.text,
                        "contextual_text": draft.contextual_text,
                        "source": draft.source.model_dump(mode="json"),
                    }
                )
            self.repository.update_job(
                job.id,
                stage="chunking",
                progress=0.1 + 0.25 * (position / len(records)),
            )
        if not drafts:
            raise DocumentError("No chunks were produced from the ready documents")

        batch_size = 32
        for start in range(0, len(drafts), batch_size):
            batch = drafts[start : start + batch_size]
            texts = [f"{item['contextual_text']}\n{item['text']}" for item in batch]
            vectors = await self.embedding_client.embed(texts)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding endpoint returned a mismatched batch size")
            for item, vector in zip(batch, vectors, strict=True):
                item["embedding"] = vector
            self.repository.update_job(
                job.id,
                stage="embedding",
                progress=0.35 + 0.5 * (min(start + batch_size, len(drafts)) / len(drafts)),
            )

        self.repository.update_job(job.id, stage="persisting", progress=0.9)
        self.repository.replace_index_chunks(index_id, drafts)
        self.repository.update_index_status(
            index_id,
            IndexStatus.ACTIVE,
            manifest=manifest,
            chunk_count=len(drafts),
            activate=activate,
        )
        return {
            "index_version_id": index_id,
            "documents": len(records),
            "chunks": len(drafts),
            "active": activate,
            "chunker": index_config["chunker"],
            "embedding_provider": index_config["embedding_provider"],
            "embedding_model": index_config["embedding_model"],
            "embedding_dimensions": len(drafts[0]["embedding"]),
            "last_embedding_retry_count": int(
                getattr(self.embedding_client, "last_retry_count", 0)
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    async def _evaluate(self, job: JobRead) -> dict[str, Any]:
        if self.agent_service is None:
            raise RuntimeError("Evaluation agent service is not configured")
        evaluation_id = job.payload["evaluation_id"]
        evaluation = self.repository.get_evaluation(evaluation_id)
        self.repository.update_evaluation(evaluation_id, status="processing")
        rows: list[dict[str, Any]] = []
        retrieval_hits = 0
        answerable_correct = 0
        for index, example in enumerate(evaluation.examples, start=1):
            response = await self.agent_service.query(
                knowledge_base_id=evaluation.knowledge_base_id,
                question=example["question"],
            )
            run = self.repository.get_run(response.run_id)
            cited_documents = {citation.source.document_id for citation in response.citations}
            expected_documents = set(example.get("expected_document_ids", []))
            evidence_hit = not expected_documents or bool(cited_documents & expected_documents)
            if evidence_hit:
                retrieval_hits += 1
            expected_answerable = bool(example.get("answerable", True))
            predicted_answerable = response.route.value not in {
                "insufficient_evidence",
                "clarify",
            }
            if predicted_answerable == expected_answerable:
                answerable_correct += 1
            rows.append(
                {
                    **example,
                    "run_id": run.id,
                    "route": response.route.value,
                    "evidence_hit": evidence_hit,
                    "predicted_answerable": predicted_answerable,
                    "latency_ms": response.latency_ms,
                }
            )
            self.repository.update_job(
                job.id,
                stage="evaluating",
                progress=index / len(evaluation.examples),
            )
        count = len(rows)
        metrics = {
            "examples": count,
            "evidence_hit_rate": retrieval_hits / count,
            "answerability_accuracy": answerable_correct / count,
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / count,
        }
        self.repository.update_evaluation(
            evaluation_id, status="succeeded", examples=rows, metrics=metrics
        )
        return {"evaluation_id": evaluation_id, "metrics": metrics}
