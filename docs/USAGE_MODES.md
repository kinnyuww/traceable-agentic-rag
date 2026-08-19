# Three Ways to Use Traceable RAG Agent

The Web UI and every API caller use the same application services, database,
object store, active index and run-trace format. These are three entry paths to
one system, not three separate implementations.

| Mode | Knowledge-base owner | Query caller | Stable hand-off |
|---|---|---|---|
| Web application | Human / domain expert | Human | knowledge-base name and UI |
| Web + Agent | Human / domain expert | External Agent | `knowledge_base_id` |
| API lifecycle | Application / CI / Agent | Application / Agent | `knowledge_base_id` + active index |

The `knowledge_base_id` is the long-lived integration handle. Rebuilding and
activating an immutable index does not require the upstream Agent to change its
tool configuration. A caller may optionally pin `index_version_id` when exact
replay is more important than following the active version.

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

A minimal Agent tool wrapper can keep the contract explicit:

```python
import httpx


async def ask_private_knowledge(question: str, knowledge_base_id: str) -> dict:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080") as client:
        response = await client.post(
            "/v1/query",
            json={
                "knowledge_base_id": knowledge_base_id,
                "question": question,
            },
        )
        response.raise_for_status()
        result = response.json()
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "route": result["route"],
        "run_id": result["run_id"],
    }
```

The wrapper deliberately returns route and provenance. For example, an
upstream Agent should not treat `insufficient_evidence` as a normal answer or
drop citations before presenting a factual claim.

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

The build request can freeze both strategy axes:

```json
{
  "activate": true,
  "chunk_strategy": "auto",
  "dense_backend": "auto"
}
```

Use `structure|semantic` and `exact|hnsw` to create controlled A/B index
versions. The Web UI exposes the same choices; both entry paths call this REST
contract.

## Shared data flow

```text
Web UI ──────────────┐
                    ├─ REST/application service
External Agent ─────┤        │
                    │        ├─ document objects + parse jobs
Automation script ──┘        ├─ immutable index versions
                             ├─ hybrid retrieval + Evidence Gate
                             └─ answer, citations and run trace
```

The Web UI does not keep a browser-only copy of uploaded documents or indexes.
It calls the same API and persists to the same service-owned volume, which is
why a knowledge base created in the UI is immediately usable from code.

## Choosing `/query` or `/retrieve`

- `/v1/query` runs the bounded Agentic path and returns a grounded final answer.
- `/v1/retrieve` returns ranked chunks for an upstream Agent that wants to own
  final generation.
- `/v1/runs/{run_id}` returns the immutable online trace.
- `/v1/evaluations` runs a fixed gold set against the active knowledge base.

OpenAPI documentation is available at <http://127.0.0.1:8080/docs>.

## Integration rules for upstream Agents

1. Keep `knowledge_base_id` in Agent configuration; do not resolve by display
   name on every production request.
2. Use `/query` by default so the Evidence Gate, retry budget and abstention
   contract remain active.
3. Preserve `run_id`, route and citations with the answer. They are part of the
   result, not debug-only metadata.
4. Treat document text as untrusted data. Never turn retrieved text into system
   instructions or tool permissions.
5. Set timeouts for model calls and surface `502` as provider degradation rather
   than silently inventing an answer.
6. Pin `index_version_id` for evaluations and replay; omit it in normal use to
   follow the active immutable version.
7. Do not expose the current local service directly to the public internet;
   authentication, tenant isolation and rate limiting are outside v0.1.

## MCP later

An MCP server can map tools such as `query`, `retrieve`, `get_run`,
`list_knowledge_bases` and `upload_document` to these contracts. Retrieval and
Agent logic remain in the application core, so MCP is a thin transport adapter
rather than a rewrite.
