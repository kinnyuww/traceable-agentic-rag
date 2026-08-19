from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ragagent.documents import ParsedDocument, ParsedSection
from ragagent.models import EmbeddingClient
from ragagent.schemas import SourceLocation

ChunkStrategy = Literal["auto", "structure", "semantic"]


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    text: str
    contextual_text: str
    source: SourceLocation


@dataclass(frozen=True)
class AdaptiveChunkResult:
    chunks: list[ChunkDraft]
    diagnostics: dict[str, Any]


def chunk_document(
    document: ParsedDocument,
    *,
    target_chars: int = 1100,
    overlap_chars: int = 160,
) -> list[ChunkDraft]:
    """Deterministic structure/sentence baseline retained for replay and tests."""
    return _draft_chunks(
        document,
        [
            _split_section(
                section,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
            for section in document.sections
        ],
    )


async def chunk_document_adaptive(
    document: ParsedDocument,
    embedding_client: EmbeddingClient,
    *,
    strategy: ChunkStrategy = "auto",
    target_chars: int = 1100,
    overlap_chars: int = 160,
    semantic_breakpoint_percentile: float = 90.0,
    semantic_min_chars: int = 320,
    semantic_max_chars: int = 1600,
) -> AdaptiveChunkResult:
    """Structure-first chunking with embedding-based fallback for weak structure.

    Parser sections remain hard parent boundaries. In ``auto`` mode, meaningful
    Markdown/DOCX headings use deterministic structure-aware splitting, while
    generic TXT/PDF sections use semantic breakpoints when they are long enough.
    """
    if strategy not in {"auto", "structure", "semantic"}:
        raise ValueError(f"Unsupported chunk strategy: {strategy}")

    section_splits: list[list[tuple[int, str]]] = []
    section_modes: list[dict[str, Any]] = []
    semantic_windows = 0
    semantic_breakpoints = 0
    for section in document.sections:
        mode = _resolve_section_mode(strategy, section, target_chars)
        preferred: set[int] = set()
        if len(section.text.strip()) <= target_chars:
            mode_reason = "below_target_no_split"
        elif strategy == "structure":
            mode_reason = "forced_structure"
        elif strategy == "semantic":
            mode_reason = "forced_semantic"
        else:
            mode_reason = "meaningful_structure" if mode == "structure" else "weak_structure"
        if mode == "semantic":
            preferred, windows = await _semantic_breakpoints(
                section.text,
                embedding_client,
                percentile=semantic_breakpoint_percentile,
            )
            semantic_windows += windows
            semantic_breakpoints += len(preferred)
            if not preferred:
                mode_reason = "semantic_fallback_no_breakpoint"
        splits = _split_section(
            section,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            preferred_boundaries=preferred,
            min_chars=semantic_min_chars if mode == "semantic" else None,
            max_chars=semantic_max_chars if mode == "semantic" else None,
        )
        section_splits.append(splits)
        section_modes.append(
            {
                "section": section.section,
                "page": section.page,
                "characters": len(section.text),
                "mode": mode,
                "reason": mode_reason,
                "semantic_breakpoints": len(preferred),
                "chunks": len(splits),
            }
        )

    chunks = _draft_chunks(document, section_splits)
    lengths = [len(chunk.text) for chunk in chunks]
    return AdaptiveChunkResult(
        chunks=chunks,
        diagnostics={
            "requested_strategy": strategy,
            "resolved_strategy": _resolved_document_strategy(section_modes),
            "target_chars": target_chars,
            "overlap_chars": overlap_chars,
            "semantic_breakpoint_percentile": semantic_breakpoint_percentile,
            "semantic_min_chars": semantic_min_chars,
            "semantic_max_chars": semantic_max_chars,
            "semantic_windows_embedded": semantic_windows,
            "semantic_breakpoints": semantic_breakpoints,
            "sections": section_modes,
            "chunk_count": len(chunks),
            "chunk_length": {
                "min": min(lengths, default=0),
                "max": max(lengths, default=0),
                "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            },
        },
    )


def _draft_chunks(
    document: ParsedDocument,
    section_splits: list[list[tuple[int, str]]],
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for section, splits in zip(document.sections, section_splits, strict=True):
        for relative_start, text in splits:
            chunks.append(
                ChunkDraft(
                    ordinal=len(chunks),
                    text=text,
                    contextual_text=_local_context(document.filename, section),
                    source=SourceLocation(
                        document_id=document.document_id,
                        filename=document.filename,
                        page=section.page,
                        section=section.section,
                        start_char=section.start_char + relative_start,
                        end_char=section.start_char + relative_start + len(text),
                    ),
                )
            )
    return chunks


def _local_context(filename: str, section: ParsedSection) -> str:
    parts = [f"Document: {filename}"]
    if section.section:
        parts.append(f"Section: {section.section}")
    if section.page:
        parts.append(f"Page: {section.page}")
    return " | ".join(parts)


def _resolve_section_mode(
    strategy: ChunkStrategy, section: ParsedSection, target_chars: int
) -> Literal["structure", "semantic"]:
    if strategy == "structure" or len(section.text.strip()) <= target_chars:
        return "structure"
    if strategy == "semantic":
        return "semantic"
    label = (section.section or "").strip().lower()
    generic = not label or label == "document" or bool(re.fullmatch(r"page\s+\d+", label))
    return "semantic" if generic else "structure"


def _resolved_document_strategy(section_modes: list[dict[str, Any]]) -> str:
    modes = {item["mode"] for item in section_modes}
    if len(modes) == 1:
        return modes.pop() if modes else "structure"
    return "hybrid"


async def _semantic_breakpoints(
    text: str,
    embedding_client: EmbeddingClient,
    *,
    percentile: float,
) -> tuple[set[int], int]:
    sentence_spans = _sentence_spans(text.strip())
    if len(sentence_spans) < 3:
        return set(), 0
    windows: list[str] = []
    for index in range(len(sentence_spans)):
        left = max(0, index - 1)
        right = min(len(sentence_spans), index + 2)
        windows.append(" ".join(item[2] for item in sentence_spans[left:right]))
    embedded: list[list[float]] = []
    for start in range(0, len(windows), 32):
        embedded.extend(await embedding_client.embed(windows[start : start + 32]))
    vectors = np.asarray(embedded, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(windows):
        raise RuntimeError("Embedding endpoint returned invalid semantic-window vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    if not len(distances) or np.allclose(distances, distances[0]):
        return set(), len(windows)
    threshold = float(np.percentile(distances, percentile))
    breakpoints = {
        sentence_spans[index][1]
        for index, distance in enumerate(distances)
        if float(distance) >= threshold
    }
    return breakpoints, len(windows)


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    boundaries = _natural_boundaries(text)
    spans: list[tuple[int, int, str]] = []
    start = 0
    for end in boundaries:
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        cleaned = raw.strip()
        if cleaned:
            actual_start = start + leading
            spans.append((actual_start, actual_start + len(cleaned), cleaned))
        start = end
    if start < len(text):
        raw = text[start:]
        leading = len(raw) - len(raw.lstrip())
        cleaned = raw.strip()
        if cleaned:
            actual_start = start + leading
            spans.append((actual_start, actual_start + len(cleaned), cleaned))
    return spans


def _natural_boundaries(text: str) -> list[int]:
    pattern = re.compile(r"[。！？!?]+[”’」』】》]?|(?<=[.])(?=\s|$)|\n\s*\n+")
    return sorted({match.end() for match in pattern.finditer(text)} | {len(text)})


def _split_section(
    section: ParsedSection,
    target_chars: int,
    overlap_chars: int,
    *,
    preferred_boundaries: set[int] | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> list[tuple[int, str]]:
    raw_section = section.text
    leading = len(raw_section) - len(raw_section.lstrip())
    text = raw_section.strip()
    if len(text) <= target_chars:
        return [(leading, text)]

    boundaries = _natural_boundaries(text)
    preferred = preferred_boundaries or set()
    minimum = min_chars or 1
    maximum = max_chars or target_chars
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= target_chars:
            end = len(text)
        else:
            lower = min(len(text), start + minimum)
            upper = min(len(text), start + maximum)
            semantic = [point for point in preferred if lower <= point <= upper]
            if semantic:
                end = min(semantic, key=lambda point: (abs(point - (start + target_chars)), point))
            else:
                before_target = [
                    point for point in boundaries if start < point <= min(upper, start + target_chars)
                ]
                after_target = [
                    point for point in boundaries if start + target_chars < point <= upper
                ]
                if before_target and max(before_target) >= lower:
                    end = max(before_target)
                elif after_target:
                    end = min(after_target)
                else:
                    end = upper
        raw = text[start:end]
        chunk_leading = len(raw) - len(raw.lstrip())
        cleaned = raw.strip()
        if cleaned:
            chunks.append((leading + start + chunk_leading, cleaned))
        if end >= len(text):
            break
        desired_start = max(start + 1, end - overlap_chars)
        sentence_starts = [0, *boundaries[:-1]]
        aligned = [point for point in sentence_starts if desired_start <= point < end]
        next_start = min(aligned) if aligned else desired_start
        start = next_start if next_start > start else end
    return chunks
