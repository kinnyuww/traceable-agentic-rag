# RAG Instability and Trace Contract

## Principle

A wrong answer is the end of a causal chain, not a useful diagnosis. v0.1
records each boundary where information can be lost or distorted so an operator
can answer: “Was the evidence absent from storage, not parsed, badly chunked,
not recalled, fused too low, reranked away, rejected by the evidence gate, lost
in the prompt, or ignored by generation?”

## Failure-surface matrix

| Layer | Typical instability | Required evidence in v0.1 | Evaluation or alert | Runtime behavior |
|---|---|---|---|---|
| Upload | empty, duplicate, too large, unsupported extension | HTTP status; hash; size; document ID | upload contract tests | reject without overwrite |
| Object store | partial write or path traversal | safe basename; SHA-256 object path | duplicate and filename tests | atomic temporary-file replace |
| PDF/DOCX/TXT parse | corrupt file, encoding error, missing extractable text | job `parsing` event; parser; section/character counts; error | parser fixtures; manual source inspection | document becomes `failed` |
| OCR/table/layout | information exists visually but parser never sees it | zero/low extracted characters and source metadata | operator review; future layout benchmark | explicitly unsupported in v0.1 |
| Chunk boundary | answer split across chunks, oversized/noisy chunks | index config, ordinal, section/page and offsets | gold-evidence hit by chunk; boundary fixtures | structure + sentence boundaries + overlap |
| Context enrichment | summary omits or invents a fact, or doubles cost | index `contextualize` config | contextual-vs-baseline offline experiment | LLM contextual summaries explicitly unavailable in v0.1 |
| Embedding | endpoint unavailable, dimension mismatch, drift, local request accidentally routed through a system proxy | provider/model, dimension, retry count, job error | real-model smoke; dimension and local proxy-bypass assertions | bypass proxy for local hosts; transient retry; otherwise fail index/query |
| Dense retrieval | semantic miss, multilingual weakness, exact-search cost | ordered chunk IDs and cosine scores | Hit@K, MRR@10, Recall@10, nDCG@10 | keep dense candidates for trace |
| Sparse retrieval | synonym miss, tokenizer mismatch, identifier sensitivity | ordered BM25 candidates and scores | same metrics; dense/BM25 ablation | Unicode + Chinese char/bigram terms |
| Fusion | relevant item demoted by rank combination | RRF candidates/scores, k=60 | stage-by-stage metrics | rank-based RRF avoids score calibration |
| Reranker | model cold start/503, truncation, domain mismatch, relevant item demotion | provider, retry count, latency, scores, fallback/error | pre/post-rerank delta | bounded retry; lexical fallback |
| Query understanding | underspecified referent or missed facets | evidence-gate reason and missing facts | clarify/answerability cases | clarify or one rewrite/decomposition round |
| Multi-hop retrieval | only one supporting source found | source diversity, subqueries, second-round candidates | all-evidence Recall@10 | max four subqueries, max two rounds |
| Evidence gate | false accept causes hallucination; false reject hides valid answer | score inputs, method, confidence, decision and error | answerability accuracy; abstention cases | deterministic rule; optional gray-zone LLM |
| Context assembly | duplicate/noisy evidence; useful item in the middle | selected chunk IDs and citation mapping | citation/evidence-set checks | maximum four selected chunks |
| Prompt injection | document tells model to ignore system instructions | untrusted-source delimiters and security prompt | adversarial document fixture | document text never grants instructions |
| Generation | unsupported claim, invalid citation, provider/auth failure | model, usage, selected chunks, citation markers, degraded flag | faithfulness/citation checks when LLM is enabled | sanitize markers; cited extractive fallback |
| External API | invalid key, 401, timeout, data-egress risk | server-side config state; sanitized error; no key in trace | connectivity preflight | disabled by default; 4xx is not retried |
| Agent loop | latency/cost explosion or oscillation | round/subquery budget and stop event | bounded-state tests | two rounds, then abstain |
| Index/version drift | answer cannot be reproduced after document changes | KB, index version, manifest, model and chunk IDs | replay against fixed version | immutable versions, atomic active pointer |
| Evaluation | LLM judge bias or metric gaming | dataset hash, sample policy, per-example run ID | deterministic IR metrics + human gold | Ragas is optional adapter, not authority |
| Concurrency/storage | SQLite lock, worker crash, or a service using the wrong health probe | job/run status and timestamps; API HTTP probe; worker DB probe | two-query load; Compose health | WAL, busy timeout, short connections, service-specific health checks |

## Run trace schema

Ordered stages are intentionally human-readable JSON:

- `query_received`: question, knowledge base, immutable index and budgets.
- `retrieval_round`: query; every dense, sparse, RRF and final candidate;
  chunk ID、document ID、文件名、页/章节位置和分数；counts；model
  providers；retry counts；stage and total latency；reranker degradation。
- `evidence_gate`: answer/retry/clarify, reason, confidence, missing facts,
  deterministic/LLM method and model error.
- `query_plan`: second-round subqueries, reason, planner method and error.
- `context_selection`: absolute/relative score floor and accepted/discarded
  chunks, preventing weak tail candidates from automatically entering prompts.
- `answer_generation`: selected chunks, provider, usage, valid citation markers
  and degradation.
- `stop`: the exhausted budget and remaining evidence.
- `run_failure`: stage, sanitized error class and recoverability.
- `run_completed`: final route, rounds, citation count and latency.

Ingestion and evaluation use `job_events`, preserving queued, starting,
parsing/chunking/embedding/persisting/evaluating, completed or failed states.

## What is not yet automatically diagnosed

v0.1 supplies the evidence needed for diagnosis but does not claim to infer a
single root cause automatically. The next Harness Engineering iteration can
compare gold evidence with each candidate list, assign a failure taxonomy,
propose a controlled intervention, run it against development data, and require
a sealed-test/HITL promotion gate. Keeping observation separate from automatic
mutation avoids converting one bad user feedback item into system-wide drift.
