# Local Operations

## Supported first target

- Apple M1 Pro, 32 GB unified memory, macOS
- Docker Desktop 4.87 with Docker Model Runner
- Qwen3-Embedding-0.6B F16 and Qwen3-Reranker-0.6B F16 through llama.cpp/Metal
- ARM64 Python 3.12 application containers

## First-time setup

Start Docker Desktop, enable Model Runner with host TCP access on port 12434,
then run:

```bash
./scripts/bootstrap-models.sh
./scripts/run-docker.sh -d
```

Open <http://127.0.0.1:8080/app/>. OpenAPI is at
<http://127.0.0.1:8080/docs>.

The helper scripts use Docker Desktop's absolute CLI path by default because a
per-user macOS install may not put `docker` on `PATH`. Override `DOCKER_BIN` if
the CLI is installed elsewhere.

## External generation model

Keep secrets outside the repository, for example:

```bash
mkdir -p "$HOME/.config/traceable-rag-agent"
chmod 700 "$HOME/.config/traceable-rag-agent"
cp .env.example "$HOME/.config/traceable-rag-agent/secrets.env"
chmod 600 "$HOME/.config/traceable-rag-agent/secrets.env"
```

Set `RAG_LLM_ENABLED=true` only after the configured DeepSeek official endpoint,
model name and key pass an authenticated smoke test. The verified identifiers
are `https://api.deepseek.com` and `deepseek-v4-flash`.
`scripts/run-docker.sh` reads this external file. A missing/invalid key must not
be copied into the repository, browser, image, logs or trace.

## Developer mode

```bash
UV_CACHE_DIR=/tmp/ragagent-uv-cache uv sync --extra dev --no-editable --python 3.12
PYTHONPATH=src uv run --no-sync uvicorn ragagent.main:app --port 8080
```

The `--no-editable` workaround avoids a Python 3.12/macOS build in this machine
that skips underscore-prefixed editable `.pth` files. Tests also set
`pythonpath = ["src"]`.

## Verification

```bash
./scripts/verify-models.sh
PYTHONPATH=src uv run --no-sync ruff check .
PYTHONPATH=src uv run --no-sync pytest -q
```

Against the running Compose stack, execute the complete black-box flow:

```bash
UV_CACHE_DIR=/tmp/ragagent-uv-cache uv run --no-sync python \
  scripts/verify_deployment.py --base-url http://127.0.0.1:8080
```

This creates a timestamped smoke-test knowledge base in the persistent volume,
uploads and indexes a Markdown fixture, checks a cited query and its six trace
stages, runs two concurrent queries, and submits a two-example evaluation.

Run the fixed public diagnostic after downloading the two datasets-server
`first-rows` JSON files:

```bash
PYTHONPATH=src uv run --no-sync python scripts/run_benchmarks.py \
  --qasper /path/to/qasper-first.json \
  --multihop /path/to/multihop-first.json
```

The checked-in JSON reports are fixed-slice diagnostics rather than official
leaderboard submissions. They include input SHA-256 values and sampling policy
so subsequent model/config experiments can use the same slice.

## Current external-model status

The repository contains no API key. The deployment reads an operator-provided
DeepSeek key from a mode-600 secret file outside the repository. The official
`/models` endpoint returned 200 and
listed `deepseek-v4-flash`; a minimal completion through this application's
actual client also passed with zero retries. Enabling the model means selected
document chunks leave the local machine for DeepSeek, so deployments handling
private material must still obtain the workspace owner's data-egress approval.

## Backup and recovery

Application data is in the named volume `traceable-rag-agent-data`. Index
versions and original objects are local. Stop API and worker containers before
taking a filesystem-consistent manual backup. v0.1 deliberately has no delete
endpoint or automated destructive migration.

## Known scaling boundary

Exact dense search loads the active version's vectors and is appropriate for
the local MVP and diagnostic corpora. Before large production corpora, replace
the dense repository adapter with HNSW/Qdrant/pgvector, add tenant/RBAC policy,
stream uploads, and benchmark concurrency. The REST and trace schemas can stay
stable.
