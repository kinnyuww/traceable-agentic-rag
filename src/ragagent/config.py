from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 8080
    inline_jobs: bool = True
    upload_max_mb: int = 100
    trace_retention_days: int = 30

    embedding_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_dimensions: int = 384

    rerank_provider: Literal["lexical", "http"] = "lexical"
    rerank_endpoint: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""

    llm_enabled: bool = False
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = Field(default="", validation_alias="RAG_LLM_API_KEY")
    llm_timeout_seconds: float = 120.0

    retrieve_dense_k: int = 20
    retrieve_sparse_k: int = 20
    retrieve_fused_k: int = 12
    rerank_k: int = 6
    max_agent_rounds: int = 2
    max_subqueries: int = 4
    retrieval_query_concurrency: int = Field(default=2, ge=1, le=4)
    conversation_memory_turns: int = Field(default=5, ge=0, le=10)
    evidence_threshold: float = 0.52
    context_min_rerank_score: float = 0.02
    context_relative_score: float = 0.10
    evidence_single_hop_limit: int = Field(default=4, ge=1, le=6)
    evidence_multi_hop_limit: int = Field(default=6, ge=1, le=6)

    dense_default_backend: Literal["auto", "exact", "hnsw"] = "auto"
    dense_auto_hnsw_min_chunks: int = Field(default=100000, ge=1)
    hnsw_connectivity: int = Field(default=32, ge=2, le=128)
    hnsw_expansion_add: int = Field(default=512, ge=16, le=1024)
    hnsw_expansion_search: int = Field(default=512, ge=10, le=1024)

    chunk_default_strategy: Literal["auto", "structure", "semantic"] = "auto"
    chunk_target_chars: int = Field(default=1100, ge=128, le=8000)
    chunk_overlap_chars: int = Field(default=160, ge=0, le=2000)
    semantic_breakpoint_percentile: float = Field(default=90.0, ge=50.0, le=99.9)
    semantic_min_chars: int = Field(default=320, ge=64, le=4000)
    semantic_max_chars: int = Field(default=1600, ge=256, le=12000)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "ragagent.sqlite3"

    @property
    def objects_dir(self) -> Path:
        return self.data_dir / "objects"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def vector_indexes_dir(self) -> Path:
        return self.data_dir / "vector-indexes"

    @model_validator(mode="after")
    def validate_model_endpoints(self) -> Settings:
        if self.embedding_provider == "openai" and not self.embedding_base_url:
            raise ValueError("RAG_EMBEDDING_BASE_URL is required for openai embeddings")
        if self.rerank_provider == "http" and not self.rerank_endpoint:
            raise ValueError("RAG_RERANK_ENDPOINT is required for HTTP reranking")
        if self.llm_enabled and (not self.llm_base_url or not self.llm_api_key):
            raise ValueError("LLM is enabled but its base URL or API key is missing")
        if self.evidence_single_hop_limit > self.rerank_k:
            raise ValueError("Single-hop evidence limit cannot exceed rerank_k")
        if self.evidence_multi_hop_limit > self.rerank_k:
            raise ValueError("Multi-hop evidence limit cannot exceed rerank_k")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("Chunk overlap must be smaller than the target size")
        if self.semantic_min_chars > self.chunk_target_chars:
            raise ValueError("Semantic minimum size cannot exceed the chunk target")
        if self.semantic_max_chars < self.chunk_target_chars:
            raise ValueError("Semantic maximum size cannot be smaller than the chunk target")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
