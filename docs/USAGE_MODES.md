# Three Ways to Use Traceable RAG Agent

The Web UI and every API caller use the same application services, database,
object store, active index and run-trace format. These are three entry paths to
one system, not three separate implementations.

## 1. Web UI builds and queries the knowledge base

Use this when a person wants a complete local RAG application.

1. Start the stack with `./scripts/run-docker.sh -d`.
2. Open <http://127.0.0.1:8080/app/>.
3. Create a knowledge base.
4. Upload PDF, DOCX, Markdown or TXT files.
5. Wait until documents are `ready`, then build and activate an index version.
6. Ask questions, inspect citations and open the run trace.

This path requires no client code.

## 2. Web UI builds; another Agent queries through REST

Use the Web UI as the knowledge-base workbench and let an application or Agent
consume the active knowledge base by ID.

Find the ID:

```bash
curl -s http://127.0.0.1:8080/v1/knowledge-bases
```

Ask through the stable query contract:

```bash
curl -s http://127.0.0.1:8080/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "knowledge_base_id": "kb_REPLACE_ME",
    "question": "这份制度中员工每年有多少天年假？"
  }'
```

Or use the checked-in Python example and resolve by the Web UI name:

```bash
PYTHONPATH=src uv run --no-sync python examples/rest_workflow.py query \
  --knowledge-base-name '粗糙知识库' \
  --question 'ZX-417 的最高工作温度是多少？'
```

The response contains `answer`, `citations`, `route`, `rounds`, `run_id` and
`trace_url`. An upstream Agent should preserve the citations and run ID rather
than flattening the answer into an untraceable string.

## 3. Code builds and queries the knowledge base through REST

Use this for CI ingestion, an application-managed tenant/workspace, scheduled
document updates, or an Agent that owns the whole lifecycle.

The full sequence is:

```text
POST /v1/knowledge-bases
  → POST /v1/knowledge-bases/{id}/documents
  → poll GET /v1/jobs/{parse_job_id}
  → POST /v1/knowledge-bases/{id}/index-builds
  → poll GET /v1/jobs/{index_job_id}
  → POST /v1/query or /v1/retrieve
  → GET /v1/runs/{run_id}
```

The runnable example performs the sequence and optionally asks a first
question:

```bash
PYTHONPATH=src uv run --no-sync python examples/rest_workflow.py build \
  --name '产品手册' \
  --description '由上层 Agent 通过 REST 创建' \
  --question '安装前需要满足什么条件？' \
  ./manual.md ./faq.pdf
```

Parsing and index construction are jobs. Production callers must poll job
status instead of sleeping for a fixed duration. An index becomes the active
knowledge-base version only after a successful atomic build.

## Choosing `/query` or `/retrieve`

- `/v1/query` runs the bounded Agentic path and returns a grounded final answer.
- `/v1/retrieve` returns ranked chunks for an upstream Agent that wants to own
  final generation.
- `/v1/runs/{run_id}` returns the immutable online trace.
- `/v1/evaluations` runs a fixed gold set against the active knowledge base.

OpenAPI documentation is available at <http://127.0.0.1:8080/docs>.

## MCP later

An MCP server can map tools such as `query`, `retrieve`, `get_run`,
`list_knowledge_bases` and `upload_document` to these contracts. Retrieval and
Agent logic remain in the application core, so MCP is a thin transport adapter
rather than a rewrite.
