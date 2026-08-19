# Changelog

## 0.1.0 — Traceable and Evaluable Baseline

Initial public baseline.

### Added

- Standalone local Web RAG application and versioned REST/OpenAPI service.
- PDF, DOCX, Markdown and UTF-8 text ingestion with asynchronous job traces.
- Deterministic structure-aware chunking with source offsets.
- Immutable index versions with local dense + BM25 + RRF hybrid retrieval.
- Pluggable embedding, reranking and OpenAI-compatible generation adapters.
- Bounded Agentic query path with at most two retrieval rounds.
- Grounded answers, source citations, evidence gating and extractive fallback.
- Full online traces across retrieval, fusion, reranking, context selection,
  routing and generation.
- Fixed-set evaluation API, public benchmark diagnostics and a Ragas adapter.
- Docker Compose API/worker deployment and Apple Silicon model-runner profile.
- Failure lab demonstrating parser, routing, citation and untrusted-context
  instability even when final answer text remains correct.
- Three supported operation modes: full Web UI, Web-built/API-consumed, and
  API-only knowledge-base lifecycle.

### Explicitly deferred

- feedback and expert-label workflow;
- automatic failure attribution and candidate experiment generation;
- sealed regression gates, HITL promotion, rollback and constrained
  self-tuning;
- MCP adapter, OCR/layout tables, multi-tenancy and RBAC.
