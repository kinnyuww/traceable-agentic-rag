#!/usr/bin/env bash
set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-/Applications/Docker.app/Contents/Resources/bin/docker}"
SECRET_ENV="${RAG_SECRET_ENV:-$HOME/.config/traceable-rag-agent/secrets.env}"

if [[ ! -x "$DOCKER_BIN" ]]; then
  echo "Docker CLI was not found at: $DOCKER_BIN" >&2
  exit 1
fi

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
  echo "Docker Desktop is not running." >&2
  exit 1
fi

if [[ -f "$SECRET_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRET_ENV"
  set +a
fi

exec "$DOCKER_BIN" compose up --build "$@"

