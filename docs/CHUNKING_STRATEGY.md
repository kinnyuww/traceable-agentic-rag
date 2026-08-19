# Chunking Strategy in v0.1

## What is implemented

v0.1 uses a deterministic, structure-aware baseline. It is deliberately easy
to replay and diagnose before semantic or LLM-generated chunking is introduced.

```text
document parser
  → PDF page / DOCX heading / Markdown heading / plain-text document
  → section-local split
  → paragraph or sentence boundary when available
  → ~1100-character target with 160-character overlap
  → source offsets + local structural context
  → embedding and BM25 indexing
```

The implementation is in [`src/ragagent/chunking.py`](../src/ragagent/chunking.py).

## Exact rules

1. Parse first, chunk second. The chunker never receives an unstructured byte
   stream; it receives `ParsedSection` objects with document ID, filename,
   section title, page number when available, and character offsets.
2. Never cross a parser section boundary. A Markdown heading, a DOCX heading,
   or a PDF page is therefore a hard parent boundary in v0.1.
3. Keep any section at or below 1100 characters as one chunk.
4. For a longer section, choose the last blank-line or sentence boundary before
   the 1100-character target. If no such boundary exists, use a hard character
   cut.
5. Start the next chunk roughly 160 characters before the previous end, then
   move to the next whitespace when possible. The overlap protects facts that
   straddle a boundary.
6. Persist absolute `start_char` and `end_char` offsets, page/section metadata,
   ordinal, filename and document ID for every chunk.
7. Prefix the searchable representation with deterministic local context such
   as `Document: handbook.md | Section: Leave policy | Page: 2`. Dense
   embedding, BM25 and reranking see this context plus the original chunk, but
   citations still quote the original source text.

The baseline uses characters rather than tokenizer-specific token counts. That
keeps chunk IDs and boundaries stable when a model provider changes, and is
simple for mixed Chinese/English documents. It is not claimed to be optimal.

## What “contextual” does and does not mean here

The `contextual_text` field currently contains source structure only: document,
section and page. v0.1 does **not** call an LLM to write a document-level summary
for every chunk and does not claim to implement Anthropic-style Contextual
Retrieval. An index request with `contextualize=true` fails explicitly instead
of silently pretending the feature exists.

## Known instability surfaces

- Long Chinese prose often has no whitespace after `。！？`. The current
  boundary expression may therefore fall back to a hard 1100-character cut.
- A PDF page is a hard boundary, so evidence split across pages is not joined
  by the chunker.
- DOCX tables, OCR text, layout relations, code syntax trees and images are not
  represented by the baseline parser.
- One target size is used for every document type and query type.
- Overlap can improve recall while increasing duplicate candidates and prompt
  cost.
- Structural prefixes help retrieval but are not a substitute for full
  document-level semantic context.

Each issue is observable through parse jobs, immutable index configuration,
candidate traces, source offsets and fixed evaluations. This makes the baseline
useful for controlled experiments rather than hiding the choices inside a
framework default.

## Candidate experiments for the Harness phase

These are candidates, not v0.1 claims:

| Candidate | Hypothesis | Required evaluation |
|---|---|---|
| Chinese-aware punctuation boundaries | Fewer mid-sentence cuts | gold evidence recall, duplication, chunk length distribution |
| Token-aware size by embedding model | Better use of model context | Recall@K, latency, index size |
| Parent-child retrieval | Small chunks recall, larger parents answer | evidence recall and faithfulness |
| LLM contextual summaries | Better ambiguous/local chunk recall | contextual vs baseline ablation, cost, hallucination audit |
| Semantic breakpoint chunking | Topic-coherent chunks | boundary benchmark and downstream retrieval delta |
| Table/layout-aware parsing | Recover facts such as `HT-8842` | parser coverage and cell-level citations |
| Code AST or domain-specific splitters | Preserve executable/domain units | domain gold set, not generic preference |

No candidate should replace the active strategy based on one user complaint.
The Harness phase should build a candidate index, run development and sealed
regression sets, compare quality/cost/latency, and require human approval before
promotion.
