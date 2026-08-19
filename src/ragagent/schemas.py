from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    READY = "ready"
    FAILED = "failed"


class IndexStatus(StrEnum):
    QUEUED = "queued"
    BUILDING = "building"
    ACTIVE = "active"
    FAILED = "failed"


class Route(StrEnum):
    NO_RAG_DIRECT = "no_rag_direct"
    SINGLE_PASS_RAG = "single_pass_rag"
    ITERATIVE_RAG = "iterative_rag"
    CLARIFY = "clarify"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SourceLocation(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class KnowledgeBaseRead(BaseModel):
    id: str
    name: str
    description: str
    active_index_version_id: str | None
    document_count: int = 0
    created_at: str
    updated_at: str


class DocumentRead(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    version: int
    error: str | None = None
    created_at: str
    updated_at: str


class UploadResult(BaseModel):
    document: DocumentRead
    job_id: str


class UploadResponse(BaseModel):
    items: list[UploadResult]


class JobEventRead(BaseModel):
    sequence: int
    status: JobStatus
    stage: str
    progress: float
    payload: dict[str, Any]
    created_at: str


class JobRead(BaseModel):
    id: str
    type: str
    status: JobStatus
    stage: str
    progress: float
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    trace: list[JobEventRead] = Field(default_factory=list)


class IndexBuildRequest(BaseModel):
    contextualize: bool = False
    activate: bool = True
    chunk_strategy: Literal["auto", "structure", "semantic"] | None = None
    dense_backend: Literal["auto", "exact", "hnsw"] | None = None


class IndexBuildResponse(BaseModel):
    index_version_id: str
    job_id: str
    status: IndexStatus


class RetrieveRequest(BaseModel):
    knowledge_base_id: str
    query: str = Field(min_length=1, max_length=8000)
    index_version_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=30)


class RetrievalHit(BaseModel):
    chunk_id: str
    text: str
    contextual_text: str
    source: SourceLocation
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float
    rerank_score: float
    query_coverage: float


class RetrieveResponse(BaseModel):
    knowledge_base_id: str
    index_version_id: str
    query: str
    hits: list[RetrievalHit]
    latency_ms: float


class QueryRequest(BaseModel):
    knowledge_base_id: str
    question: str = Field(min_length=1, max_length=8000)
    index_version_id: str | None = None
    include_trace: bool = True


class Citation(BaseModel):
    id: str
    chunk_id: str
    quote: str
    source: SourceLocation


class QueryResponse(BaseModel):
    run_id: str
    route: Route
    answer: str
    citations: list[Citation]
    rounds: int
    latency_ms: float
    trace_url: str


class TraceEventRead(BaseModel):
    sequence: int
    stage: str
    payload: dict[str, Any]
    created_at: str


class RunRead(BaseModel):
    id: str
    knowledge_base_id: str
    index_version_id: str | None
    question: str
    route: Route | None
    status: str
    answer: str | None
    citations: list[Citation]
    rounds: int
    metrics: dict[str, Any]
    created_at: str
    completed_at: str | None
    trace: list[TraceEventRead]


class EvaluationExample(BaseModel):
    question: str
    expected_answer: str | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    answerable: bool = True


class EvaluationRequest(BaseModel):
    knowledge_base_id: str
    name: str = Field(default="evaluation", min_length=1, max_length=120)
    examples: list[EvaluationExample] = Field(min_length=1, max_length=500)


class EvaluationResponse(BaseModel):
    id: str
    job_id: str
    status: Literal["queued", "processing", "succeeded", "failed"]


class EvaluationRead(BaseModel):
    id: str
    knowledge_base_id: str
    name: str
    status: str
    metrics: dict[str, Any]
    examples: list[dict[str, Any]]
    created_at: str
    completed_at: str | None
