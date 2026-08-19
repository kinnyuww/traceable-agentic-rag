from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ragagent.db import Database, decode_json, encode_json
from ragagent.ids import new_id, utc_now
from ragagent.models import fts_text, lexical_terms
from ragagent.schemas import (
    Citation,
    DocumentRead,
    DocumentStatus,
    EvaluationRead,
    IndexStatus,
    JobEventRead,
    JobRead,
    JobStatus,
    KnowledgeBaseRead,
    Route,
    RunRead,
    TraceEventRead,
)


class NotFoundError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


class Repository:
    def __init__(self, database: Database):
        self.db = database

    def create_knowledge_base(self, name: str, description: str) -> KnowledgeBaseRead:
        now = utc_now()
        kb_id = new_id("kb")
        self.db.execute(
            "INSERT INTO knowledge_bases(id,name,description,created_at,updated_at) VALUES(?,?,?,?,?)",
            (kb_id, name, description, now, now),
        )
        return self.get_knowledge_base(kb_id)

    def list_knowledge_bases(self) -> list[KnowledgeBaseRead]:
        rows = self.db.fetch_all(
            """
            SELECT kb.*, COUNT(d.id) AS document_count
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.knowledge_base_id = kb.id
            GROUP BY kb.id
            ORDER BY kb.created_at DESC
            """
        )
        return [KnowledgeBaseRead(**dict(row)) for row in rows]

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBaseRead:
        row = self.db.fetch_one(
            """
            SELECT kb.*, COUNT(d.id) AS document_count
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.knowledge_base_id = kb.id
            WHERE kb.id = ? GROUP BY kb.id
            """,
            (kb_id,),
        )
        if not row:
            raise NotFoundError(f"Knowledge base {kb_id!r} was not found")
        return KnowledgeBaseRead(**dict(row))

    def create_document(
        self,
        *,
        kb_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        object_path: Path,
    ) -> DocumentRead:
        self.get_knowledge_base(kb_id)
        existing = self.db.fetch_one(
            "SELECT id FROM documents WHERE knowledge_base_id=? AND sha256=? ORDER BY version DESC",
            (kb_id, sha256),
        )
        if existing:
            raise ConflictError(f"This exact file already exists as {existing['id']}")
        now = utc_now()
        document_id = new_id("doc")
        self.db.execute(
            """
            INSERT INTO documents(
                id,knowledge_base_id,filename,content_type,size_bytes,sha256,object_path,
                status,version,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                document_id,
                kb_id,
                filename,
                content_type,
                size_bytes,
                sha256,
                str(object_path),
                DocumentStatus.QUEUED.value,
                1,
                now,
                now,
            ),
        )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> DocumentRead:
        row = self.db.fetch_one("SELECT * FROM documents WHERE id=?", (document_id,))
        if not row:
            raise NotFoundError(f"Document {document_id!r} was not found")
        payload = dict(row)
        payload.pop("object_path", None)
        payload.pop("artifact_path", None)
        return DocumentRead(**payload)

    def get_document_record(self, document_id: str) -> dict[str, Any]:
        row = self.db.fetch_one("SELECT * FROM documents WHERE id=?", (document_id,))
        if not row:
            raise NotFoundError(f"Document {document_id!r} was not found")
        return dict(row)

    def list_documents(self, kb_id: str) -> list[DocumentRead]:
        self.get_knowledge_base(kb_id)
        rows = self.db.fetch_all(
            "SELECT * FROM documents WHERE knowledge_base_id=? ORDER BY created_at DESC", (kb_id,)
        )
        documents: list[DocumentRead] = []
        for row in rows:
            payload = dict(row)
            payload.pop("object_path", None)
            payload.pop("artifact_path", None)
            documents.append(DocumentRead(**payload))
        return documents

    def list_ready_document_records(self, kb_id: str) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM documents WHERE knowledge_base_id=? AND status=? ORDER BY id",
            (kb_id, DocumentStatus.READY.value),
        )
        return [dict(row) for row in rows]

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        artifact_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE documents SET status=?, artifact_path=COALESCE(?,artifact_path), error=?, updated_at=?
            WHERE id=?
            """,
            (
                status.value,
                str(artifact_path) if artifact_path else None,
                error,
                utc_now(),
                document_id,
            ),
        )

    def create_index_version(self, kb_id: str, config: dict[str, Any]) -> str:
        self.get_knowledge_base(kb_id)
        index_id = new_id("idx")
        self.db.execute(
            """
            INSERT INTO index_versions(id,knowledge_base_id,status,config_json,created_at)
            VALUES(?,?,?,?,?)
            """,
            (index_id, kb_id, IndexStatus.QUEUED.value, encode_json(config), utc_now()),
        )
        return index_id

    def get_index_record(self, index_id: str) -> dict[str, Any]:
        row = self.db.fetch_one("SELECT * FROM index_versions WHERE id=?", (index_id,))
        if not row:
            raise NotFoundError(f"Index version {index_id!r} was not found")
        payload = dict(row)
        payload["config"] = decode_json(payload.pop("config_json"), {})
        payload["document_manifest"] = decode_json(payload.pop("document_manifest_json"), [])
        return payload

    def resolve_index_id(self, kb_id: str, requested: str | None = None) -> str:
        kb = self.get_knowledge_base(kb_id)
        index_id = requested or kb.active_index_version_id
        if not index_id:
            raise ConflictError("The knowledge base has no active index; build an index first")
        record = self.get_index_record(index_id)
        if record["knowledge_base_id"] != kb_id:
            raise ConflictError("The requested index does not belong to this knowledge base")
        if record["status"] != IndexStatus.ACTIVE.value:
            raise ConflictError(f"Index {index_id} is not active")
        return index_id

    def update_index_status(
        self,
        index_id: str,
        status: IndexStatus,
        *,
        manifest: list[dict[str, Any]] | None = None,
        chunk_count: int | None = None,
        error: str | None = None,
        activate: bool = False,
    ) -> None:
        now = utc_now()
        connection = self.db.transaction()
        try:
            connection.execute(
                """
                UPDATE index_versions SET status=?,
                  document_manifest_json=COALESCE(?,document_manifest_json),
                  chunk_count=COALESCE(?,chunk_count), error=?,
                  activated_at=CASE WHEN ? THEN ? ELSE activated_at END
                WHERE id=?
                """,
                (
                    status.value,
                    encode_json(manifest) if manifest is not None else None,
                    chunk_count,
                    error,
                    activate,
                    now,
                    index_id,
                ),
            )
            if activate:
                row = connection.execute(
                    "SELECT knowledge_base_id FROM index_versions WHERE id=?", (index_id,)
                ).fetchone()
                if not row:
                    raise NotFoundError(f"Index version {index_id!r} was not found")
                connection.execute(
                    "UPDATE knowledge_bases SET active_index_version_id=?, updated_at=? WHERE id=?",
                    (index_id, now, row["knowledge_base_id"]),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def replace_index_chunks(self, index_id: str, chunks: list[dict[str, Any]]) -> None:
        connection = self.db.transaction()
        try:
            connection.execute("DELETE FROM chunks_fts WHERE index_version_id=?", (index_id,))
            connection.execute("DELETE FROM chunks WHERE index_version_id=?", (index_id,))
            for chunk in chunks:
                vector = np.asarray(chunk["embedding"], dtype=np.float32)
                connection.execute(
                    """
                    INSERT INTO chunks(
                      id,index_version_id,knowledge_base_id,document_id,ordinal,text,
                      contextual_text,source_json,embedding,embedding_dim,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        chunk["id"],
                        index_id,
                        chunk["knowledge_base_id"],
                        chunk["document_id"],
                        chunk["ordinal"],
                        chunk["text"],
                        chunk["contextual_text"],
                        encode_json(chunk["source"]),
                        vector.tobytes(),
                        vector.size,
                        utc_now(),
                    ),
                )
                searchable = f"{chunk['contextual_text']} {chunk['text']}"
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id,index_version_id,text) VALUES(?,?,?)",
                    (chunk["id"], index_id, fts_text(searchable)),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load_dense_chunks(self, index_id: str) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM chunks WHERE index_version_id=? ORDER BY id", (index_id,)
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["source"] = decode_json(payload.pop("source_json"), {})
            payload["embedding"] = np.frombuffer(
                payload["embedding"], dtype=np.float32, count=payload["embedding_dim"]
            )
            result.append(payload)
        return result

    def sparse_search(self, index_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        terms = lexical_terms(query)
        if not terms:
            return []
        escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:80]]
        match_query = " OR ".join(escaped)
        rows = self.db.fetch_all(
            """
            SELECT c.*, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ? AND chunks_fts.index_version_id = ?
            ORDER BY bm25_score ASC LIMIT ?
            """,
            (match_query, index_id, limit),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["source"] = decode_json(payload.pop("source_json"), {})
            payload.pop("embedding", None)
            result.append(payload)
        return result

    def create_job(self, job_type: str, payload: dict[str, Any]) -> JobRead:
        job_id = new_id("job")
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO jobs(id,type,status,stage,progress,payload_json,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                job_type,
                JobStatus.QUEUED.value,
                "queued",
                0.0,
                encode_json(payload),
                now,
                now,
            ),
        )
        self.append_job_event(
            job_id,
            status=JobStatus.QUEUED,
            stage="queued",
            progress=0.0,
            payload={"job_type": job_type},
        )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRead:
        row = self.db.fetch_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            raise NotFoundError(f"Job {job_id!r} was not found")
        payload = dict(row)
        payload["payload"] = decode_json(payload.pop("payload_json"), {})
        payload["result"] = decode_json(payload.pop("result_json"), None)
        payload.pop("locked_at", None)
        event_rows = self.db.fetch_all(
            """
            SELECT sequence,status,stage,progress,payload_json,created_at
            FROM job_events WHERE job_id=? ORDER BY sequence
            """,
            (job_id,),
        )
        payload["trace"] = [
            JobEventRead(
                sequence=event["sequence"],
                status=event["status"],
                stage=event["stage"],
                progress=event["progress"],
                payload=decode_json(event["payload_json"], {}),
                created_at=event["created_at"],
            )
            for event in event_rows
        ]
        return JobRead(**payload)

    def append_job_event(
        self,
        job_id: str,
        *,
        status: JobStatus,
        stage: str,
        progress: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        connection = self.db.transaction()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS next FROM job_events WHERE job_id=?",
                (job_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO job_events(job_id,sequence,status,stage,progress,payload_json,created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    row["next"],
                    status.value,
                    stage,
                    progress,
                    encode_json(payload or {}),
                    utc_now(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_next_job(self) -> JobRead | None:
        connection = self.db.transaction()
        try:
            row = connection.execute(
                "SELECT id FROM jobs WHERE status=? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            now = utc_now()
            connection.execute(
                "UPDATE jobs SET status=?,stage=?,locked_at=?,updated_at=? WHERE id=?",
                (JobStatus.PROCESSING.value, "starting", now, now, row["id"]),
            )
            connection.commit()
            return self.get_job(row["id"])
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: str | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        current = self.get_job(job_id)
        selected_status = status or current.status
        selected_stage = stage or current.stage
        selected_progress = current.progress if progress is None else max(0.0, min(1.0, progress))
        self.db.execute(
            """
            UPDATE jobs SET status=?,stage=?,progress=?,result_json=?,error=?,updated_at=? WHERE id=?
            """,
            (
                selected_status.value,
                selected_stage,
                selected_progress,
                encode_json(result)
                if result is not None
                else (encode_json(current.result) if current.result is not None else None),
                error,
                utc_now(),
                job_id,
            ),
        )
        event_payload: dict[str, Any] = {}
        if error:
            event_payload["error"] = error[:2000]
        if result is not None:
            event_payload["result"] = result
        self.append_job_event(
            job_id,
            status=selected_status,
            stage=selected_stage,
            progress=selected_progress,
            payload=event_payload,
        )

    def create_run(self, kb_id: str, index_id: str | None, question: str) -> str:
        run_id = new_id("run")
        self.db.execute(
            """
            INSERT INTO runs(id,knowledge_base_id,index_version_id,question,status,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (run_id, kb_id, index_id, question, "running", utc_now()),
        )
        return run_id

    def append_trace(self, run_id: str, stage: str, payload: dict[str, Any]) -> None:
        connection = self.db.transaction()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS next FROM trace_events WHERE run_id=?",
                (run_id,),
            ).fetchone()
            connection.execute(
                "INSERT INTO trace_events(run_id,sequence,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
                (run_id, row["next"], stage, encode_json(payload), utc_now()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete_run(
        self,
        run_id: str,
        *,
        route: Route,
        answer: str,
        citations: list[Citation],
        rounds: int,
        metrics: dict[str, Any],
        status: str = "succeeded",
    ) -> None:
        self.db.execute(
            """
            UPDATE runs SET route=?,status=?,answer=?,citations_json=?,rounds=?,metrics_json=?,
              completed_at=? WHERE id=?
            """,
            (
                route.value,
                status,
                answer,
                encode_json([citation.model_dump(mode="json") for citation in citations]),
                rounds,
                encode_json(metrics),
                utc_now(),
                run_id,
            ),
        )

    def fail_run(self, run_id: str, *, error: str, latency_ms: float) -> None:
        self.db.execute(
            """
            UPDATE runs SET status='failed',metrics_json=?,completed_at=? WHERE id=?
            """,
            (encode_json({"latency_ms": latency_ms, "error": error[:2000]}), utc_now(), run_id),
        )

    def get_run(self, run_id: str) -> RunRead:
        row = self.db.fetch_one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not row:
            raise NotFoundError(f"Run {run_id!r} was not found")
        trace_rows = self.db.fetch_all(
            "SELECT sequence,stage,payload_json,created_at FROM trace_events WHERE run_id=? ORDER BY sequence",
            (run_id,),
        )
        payload = dict(row)
        payload["citations"] = [
            Citation(**citation) for citation in decode_json(payload.pop("citations_json"), [])
        ]
        payload["metrics"] = decode_json(payload.pop("metrics_json"), {})
        payload["trace"] = [
            TraceEventRead(
                sequence=item["sequence"],
                stage=item["stage"],
                payload=decode_json(item["payload_json"], {}),
                created_at=item["created_at"],
            )
            for item in trace_rows
        ]
        return RunRead(**payload)

    def create_evaluation(self, kb_id: str, name: str, examples: list[dict[str, Any]]) -> str:
        self.get_knowledge_base(kb_id)
        evaluation_id = new_id("eval")
        self.db.execute(
            """
            INSERT INTO evaluations(id,knowledge_base_id,name,status,examples_json,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (evaluation_id, kb_id, name, "queued", encode_json(examples), utc_now()),
        )
        return evaluation_id

    def update_evaluation(
        self,
        evaluation_id: str,
        *,
        status: str,
        examples: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE evaluations SET status=?,examples_json=COALESCE(?,examples_json),
              metrics_json=COALESCE(?,metrics_json),completed_at=CASE WHEN ? IN ('succeeded','failed')
              THEN ? ELSE completed_at END WHERE id=?
            """,
            (
                status,
                encode_json(examples) if examples is not None else None,
                encode_json(metrics) if metrics is not None else None,
                status,
                utc_now(),
                evaluation_id,
            ),
        )

    def get_evaluation(self, evaluation_id: str) -> EvaluationRead:
        row = self.db.fetch_one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
        if not row:
            raise NotFoundError(f"Evaluation {evaluation_id!r} was not found")
        payload = dict(row)
        payload["examples"] = decode_json(payload.pop("examples_json"), [])
        payload["metrics"] = decode_json(payload.pop("metrics_json"), {})
        return EvaluationRead(**payload)
