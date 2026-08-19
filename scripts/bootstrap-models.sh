#!/usr/bin/env bash
set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-/Applications/Docker.app/Contents/Resources/bin/docker}"
EMBED_MODEL="${RAG_EMBEDDING_MODEL:-ai/qwen3-embedding:0.6B-F16}"
RERANK_MODEL="${RAG_RERANK_MODEL:-ai/qwen3-reranker:0.6B}"

if [[ ! -x "$DOCKER_BIN" ]]; then
  echo "Docker Desktop CLI was not found at: $DOCKER_BIN" >&2
  exit 1
fi

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
  echo "Docker Desktop is not running." >&2
  exit 1
fi

"$DOCKER_BIN" desktop enable model-runner --tcp 12434
"$DOCKER_BIN" model pull "$EMBED_MODEL"
"$DOCKER_BIN" model pull "$RERANK_MODEL"

DOCKER_BIN="$DOCKER_BIN" \
  RAG_EMBEDDING_MODEL="$EMBED_MODEL" \
  RAG_RERANK_MODEL="$RERANK_MODEL" \
  "$(dirname "$0")/verify-models.sh"
