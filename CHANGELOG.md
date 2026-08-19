# Changelog

All notable changes to Traceable Agentic RAG are recorded here. The project
uses semantic versioning for public application/API milestones.

## 0.2.0 - 2026-08-20

### Added

- DeepSeek-backed query understanding with a deterministic fallback,
  canonical query, single-hop/multi-hop classification, intent, entities,
  constraints, clarification routing and atomic multi-hop subqueries.
- Conversation-isolated memory using optional `conversation_id` and bounded
  `memory_turns`; Web controls for starting a new conversation.
- Bounded parallel multi-view recall followed by chunk-ID deduplication and a
  single global rerank against the canonical query.
- Adaptive `auto / structure / semantic` chunking with structure-first routing,
  local-embedding semantic breakpoints and traceable strategy diagnostics.
- Selectable `auto / exact / hnsw` dense search. Exact search caches a
  normalized matrix; HNSW uses a persisted USearch sidecar.
- A browser-readable online retrieval execution diagram and expanded benchmark
  trace support.

### Changed

- Rerank still returns Top 6. Single-hop evidence uses the dynamic score floor
  and keeps at most four chunks; multi-hop retains all valid Top 6 chunks.
- Evidence Gate, answer generation and citations now share the exact same
  immutable Evidence Set. There is no post-Gate context reselection.
- Query-view retrieval is recall-only; all merged candidates are compared once
  on a common canonical-query rerank scale.
- Failure-lab and benchmark scripts read the new global-rerank trace stage.

### Known limitations

- Harness Engineering, feedback adjudication, automatic failure attribution,
  candidate experiments and HITL promotion are not implemented yet.
- OCR, layout-aware tables, multimodal parsing, multi-tenancy/RBAC and MCP are
  outside this release.
- The locally tested Qwen reranker can fail on long pair inputs under the
  current Docker Model Runner batch limit. The visible lexical fallback keeps
  the request functional but can add several seconds of latency.

## 0.1.0 - 2026-08-18

- First traceable baseline: Web application, REST API, Docker API/worker,
  immutable indexes, local embeddings, Dense + BM25 + RRF hybrid retrieval,
  reranking, bounded two-round Agentic RAG, Evidence Gate, citations, run trace,
  offline evaluation and failure-lab datasets.
