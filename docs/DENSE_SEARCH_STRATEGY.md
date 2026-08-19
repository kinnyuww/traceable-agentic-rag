# Dense and SQLite Search Strategy

## The short answer

The application has two independent retrieval channels:

```text
query
  ├─ Dense: Exact cosine or persisted USearch HNSW → Top 20
  └─ Sparse: SQLite FTS5 BM25                    → Top 20
        ↓
      RRF(k=60) → Top 12 → reranker → Top 6
```

SQLite is the source of truth for chunks, metadata, embeddings and the BM25
index. HNSW is a rebuildable sidecar artifact for one immutable index version;
it is not a second knowledge base.

## What the SQLite algorithm is

`chunks_fts` is an SQLite FTS5 virtual table. The query is tokenized into Latin
terms plus Chinese characters/bigrams, escaped, and joined with `OR`. SQLite's
`bm25(chunks_fts)` ranks matching rows. FTS5 returns better matches with lower
numeric BM25 values, but this application does not mix the raw BM25 magnitude
with cosine scores. It converts each result list to ranks and combines them
with Reciprocal Rank Fusion:

```text
RRF score(chunk) = Σ 1 / (60 + rank_in_channel)
```

This avoids pretending that cosine and BM25 scores share a calibrated scale.

## Dense modes

| Mode | Implementation | Recall | Main cost | Intended use |
|---|---|---:|---|---|
| `exact` | normalized NumPy matrix dot product | 1.0 against stored vectors | memory and linear work | small/medium KB, evaluation baseline |
| `hnsw` | persisted USearch multilayer graph | approximate | index build, graph memory, tunable recall | large or latency/concurrency-sensitive KB |
| `auto` | exact below 100,000 chunks; HNSW otherwise | mode-dependent | conservative automatic choice | default |

Exact mode no longer reloads and decodes every SQLite vector on every query.
Because index versions are immutable, the normalized matrix is cached safely
after first use. HNSW stores `{index_version_id}.usearch` and an integer-key to
chunk-ID manifest under `data/vector-indexes/`, then memory-maps the graph.

The Auto boundary is configurable with
`RAG_DENSE_AUTO_HNSW_MIN_CHUNKS`. It is deliberately conservative: on the M1
Pro synthetic 20,000 × 384-dimensional benchmark, cached exact search was
faster than HNSW, while HNSW only becomes attractive when linear matrix memory,
tail latency or concurrency is the actual bottleneck. Operators should move
the boundary using their own P95 latency, memory and recall measurements.

## HNSW settings and quality control

The default graph parameters prioritize recall over minimum latency:

```text
connectivity (M) = 32
expansion_add     = 512
expansion_search  = 512
```

Increasing connectivity and construction expansion costs build time and graph
memory. Increasing search expansion costs query latency but usually improves
recall. The checked-in seeded random-vector microbenchmark reported Recall@20
of 0.966 at 20,000 vectors; its full result and warning are in
[`reports/results/dense-backend-synthetic.json`](../reports/results/dense-backend-synthetic.json).
Random vectors are an implementation diagnostic, not evidence of domain RAG
quality. Every real deployment should compare HNSW Top-K against the exact
index on a fixed query set before promotion.

## Why HNSW before IVF

IVF/IVFFlat is the cluster/list family the product discussion referred to. It
trains centroids, routes a query to a subset of lists, and trades the number of
probed lists for speed and recall. It can build faster and use less memory than
HNSW, but needs enough representative data and another tuning surface.

This local release implements HNSW because it needs no training phase and can
be shipped as a small ARM64-compatible sidecar. IVF remains a future adapter
for very large corpora where its build/memory economics are measured to be
better. pgvector, Qdrant or another server is a separate deployment choice,
not required for the local single-workspace product.

## API and trace contract

An index build accepts:

```json
{
  "activate": true,
  "chunk_strategy": "auto",
  "dense_backend": "auto"
}
```

The requested and resolved backends, threshold and HNSW parameters are frozen
in `index_versions.config_json`. Every `retrieval_round` trace contains
`dense_search.backend`, whether the result is exact, number of indexed chunks,
cache status, parameters and dense-stage latency. This lets an evaluation
distinguish model/retrieval changes from an ANN recall miss.

## References

- [pgvector indexing: exact, HNSW and IVFFlat](https://github.com/pgvector/pgvector#indexing)
- [Qdrant vector indexing and full-scan threshold](https://qdrant.tech/documentation/manage-data/indexing/)
- [Qdrant: measuring ANN recall against exact search](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/)
- [USearch](https://github.com/unum-cloud/USearch)
