#!/usr/bin/env bash
set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-/Applications/Docker.app/Contents/Resources/bin/docker}"
EMBED_MODEL="${RAG_EMBEDDING_MODEL:-ai/qwen3-embedding:0.6B-F16}"
RERANK_MODEL="${RAG_RERANK_MODEL:-ai/qwen3-reranker:0.6B}"

"$DOCKER_BIN" model status
"$DOCKER_BIN" model inspect "$EMBED_MODEL" >/dev/null
"$DOCKER_BIN" model inspect "$RERANK_MODEL" >/dev/null

response=$(curl -fsS --max-time 180 \
  'http://localhost:12434/engines/llama.cpp/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$EMBED_MODEL\",\"input\":[\"traceable rag smoke test\"]}")

python3 -c 'import json,sys; p=json.load(sys.stdin); assert len(p["data"][0]["embedding"]) == 1024; print("embedding smoke test: passed (1024 dimensions)")' <<<"$response"

rerank_response=$(curl -fsS --max-time 180 \
  'http://localhost:12434/rerank' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$RERANK_MODEL\",\"query\":\"capital of France\",\"documents\":[\"Paris is the capital of France.\",\"Berlin is in Germany.\"]}")

python3 -c 'import json,sys; p=json.load(sys.stdin); r=p["results"]; assert r[0]["index"] == 0 and r[0]["relevance_score"] > r[1]["relevance_score"]; print("reranker smoke test: passed")' <<<"$rerank_response"
