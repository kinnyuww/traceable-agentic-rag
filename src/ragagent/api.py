from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from ragagent.documents import DocumentError
from ragagent.models import ModelServiceError
from ragagent.repositories import ConflictError, NotFoundError
from ragagent.schemas import (
    EvaluationRead,
    EvaluationRequest,
    EvaluationResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    IndexStatus,
    JobRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RunRead,
    UploadResponse,
    UploadResult,
)

router = APIRouter(prefix="/v1")


def _container(request: Request):
    return request.app.state.container


def _schedule(request: Request, background: BackgroundTasks, job: JobRead) -> None:
    container = _container(request)
    if container.settings.inline_jobs:
        background.add_task(container.job_processor.process, job)


@router.get("/health")
def health(request: Request) -> dict:
    container = _container(request)
    container.database.initialize()
    return {
        "status": "ok",
        "version": "0.1.0",
        "providers": {
            "embedding": container.settings.embedding_provider,
            "rerank": container.settings.rerank_provider,
            "llm": container.settings.llm_model if container.settings.llm_enabled else "disabled",
        },
    }


@router.post("/knowledge-bases", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(payload: KnowledgeBaseCreate, request: Request) -> KnowledgeBaseRead:
    return _container(request).repository.create_knowledge_base(payload.name, payload.description)


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(request: Request) -> list[KnowledgeBaseRead]:
    return _container(request).repository.list_knowledge_bases()


@router.get("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(knowledge_base_id: str, request: Request) -> KnowledgeBaseRead:
    try:
        return _container(request).repository.get_knowledge_base(knowledge_base_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/knowledge-bases/{knowledge_base_id}/documents")
def list_documents(knowledge_base_id: str, request: Request):
    try:
        return _container(request).repository.list_documents(knowledge_base_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=UploadResponse,
    status_code=202,
)
async def upload_documents(
    knowledge_base_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, Markdown, or TXT files")],
    auto_index: Annotated[bool, Form()] = False,
) -> UploadResponse:
    container = _container(request)
    items: list[UploadResult] = []
    for upload in files:
        try:
            content = await upload.read(container.settings.upload_max_mb * 1024 * 1024 + 1)
            document, job = container.ingestion.accept_upload(
                knowledge_base_id=knowledge_base_id,
                filename=upload.filename or "document.txt",
                content_type=upload.content_type or "application/octet-stream",
                content=content,
                auto_index=auto_index,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _schedule(request, background_tasks, job)
        items.append(UploadResult(document=document, job_id=job.id))
    return UploadResponse(items=items)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/index-builds",
    response_model=IndexBuildResponse,
    status_code=202,
)
def create_index_build(
    knowledge_base_id: str,
    payload: IndexBuildRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> IndexBuildResponse:
    container = _container(request)
    try:
        index_id, job = container.ingestion.request_index_build(
            knowledge_base_id=knowledge_base_id,
            contextualize=payload.contextualize,
            activate=payload.activate,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _schedule(request, background_tasks, job)
    return IndexBuildResponse(
        index_version_id=index_id,
        job_id=job.id,
        status=IndexStatus.QUEUED,
    )


@router.get("/index-versions/{index_version_id}")
def get_index_version(index_version_id: str, request: Request):
    try:
        return _container(request).repository.get_index_record(index_version_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: str, request: Request) -> JobRead:
    try:
        return _container(request).repository.get_job(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    container = _container(request)
    try:
        result = await container.retriever.retrieve(
            knowledge_base_id=payload.knowledge_base_id,
            query=payload.query,
            index_version_id=payload.index_version_id,
            top_k=payload.top_k,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RetrieveResponse(
        knowledge_base_id=payload.knowledge_base_id,
        index_version_id=result.index_version_id,
        query=result.query,
        hits=result.hits,
        latency_ms=result.latency_ms,
    )


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    container = _container(request)
    try:
        return await container.agent.query(
            knowledge_base_id=payload.knowledge_base_id,
            question=payload.question,
            index_version_id=payload.index_version_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=RunRead)
def get_run(run_id: str, request: Request) -> RunRead:
    try:
        return _container(request).repository.get_run(run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/evaluations", response_model=EvaluationResponse, status_code=202)
def create_evaluation(
    payload: EvaluationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> EvaluationResponse:
    container = _container(request)
    try:
        evaluation_id = container.repository.create_evaluation(
            payload.knowledge_base_id,
            payload.name,
            [example.model_dump(mode="json") for example in payload.examples],
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job = container.repository.create_job("evaluation", {"evaluation_id": evaluation_id})
    _schedule(request, background_tasks, job)
    return EvaluationResponse(id=evaluation_id, job_id=job.id, status="queued")


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationRead)
def get_evaluation(evaluation_id: str, request: Request) -> EvaluationRead:
    try:
        return _container(request).repository.get_evaluation(evaluation_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
