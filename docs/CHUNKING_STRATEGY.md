# Adaptive Chunking Strategy

## Mental model

Chunking is an offline indexing decision, not an online Agent loop:

```text
parser hard boundary
  → decide section mode
      ├─ meaningful heading: structure/sentence split
      └─ weak structure: embedding-based semantic breakpoint
  → bounded chunk (~1100 chars, max 1600, overlap ~160)
  → original offsets + deterministic source context
  → embedding + BM25
```

The parser still supplies hard parent boundaries: Markdown/DOCX headings, PDF
pages, or a TXT document. The chunker never joins unrelated parser sections and
never replaces source text with generated text.

## Three selectable modes

| Build option | Behavior | Best use |
|---|---|---|
| `structure` | paragraph/sentence boundaries inside every parser section | replayable baseline, well-structured Markdown/DOCX |
| `semantic` | force semantic breakpoint analysis for every long section | weak headings or topic-dense prose experiments |
| `auto` | meaningful headings use structure; generic `Document`/`Page N` sections use semantic analysis | default mixed corpus |

All modes keep sections at or below 1100 characters intact. Long sections use
a 1100-character target and roughly 160 characters of sentence-aligned overlap.
Semantic chunks also have a 320-character minimum and 1600-character maximum.
Every chunk preserves filename, document ID, section/page, ordinal and absolute
character offsets.

## Semantic breakpoint algorithm

For a long weakly structured section:

1. Find multilingual sentence and paragraph boundaries. Chinese `。！？` no
   longer require following whitespace; English periods split only before
   whitespace/end, avoiding decimal points such as `3.14`.
2. Create a one-sentence context buffer on both sides of every sentence.
3. Embed those windows with the same configured local embedding model used by
   retrieval. No additional online LLM call or new model service is required.
4. Compute cosine dissimilarity between adjacent windows.
5. Treat distances at/above the configured 90th percentile as topic-change
   candidates.
6. Choose a semantic candidate nearest the target size while respecting the
   minimum/maximum bounds. If there is no usable candidate, fall back to the
   normal sentence boundary and finally a hard size cap.

The approach follows the same basic pattern as LlamaIndex's semantic splitter:
embed buffered sentences, measure adjacent dissimilarity, and select percentile
breakpoints. The local implementation adds parser hard boundaries and explicit
size limits so a noisy document cannot create unbounded chunks.

## What contextual text means

`contextual_text` remains deterministic source metadata:

```text
Document: handbook.md | Section: Refund policy | Page: 4
```

It is used as a retrieval hint by embedding, BM25 and reranking. Citations and
Evidence Gate source coverage use the original chunk text. The existing
`contextualize=true` switch still fails explicitly because Anthropic-style
generated chunk summaries have a different risk profile: a generated summary
can omit, distort or invent evidence. If added later, it should be a separate
immutable index experiment, labeled as a non-evidence retrieval hint and tested
for recall, cost and hallucinated retrieval cues.

## Trace and reproducibility

The immutable index config stores the requested strategy, target, overlap,
semantic percentile and min/max sizes. The index job result records, per
document and section:

- resolved `structure` or `semantic` mode and reason;
- number of semantic windows embedded and breakpoints selected;
- chunk count and length distribution;
- source section/page and input character count.

This makes a missed answer attributable to parsing, boundary choice, embedding,
retrieval or later stages instead of hiding the split behind a framework.

## Known boundaries

- PDF pages remain hard boundaries; cross-page parent/child expansion is not yet
  implemented.
- DOCX tables, OCR, complex layouts, code ASTs and images still require better
  parsers before chunking can recover their evidence.
- Semantic splitting adds offline embedding work and can be unstable if the
  embedding model changes; that model identity is therefore part of the index.
- A percentile is corpus-relative. It finds unusual transitions but does not
  guarantee that every transition is meaningful.
- Chunk quality still requires a gold evidence set and downstream retrieval
  comparison; visually plausible boundaries are not enough.

## References

- [LlamaIndex semantic splitter API](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/semantic_splitter/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
