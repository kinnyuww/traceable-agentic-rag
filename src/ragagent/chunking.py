from __future__ import annotations

import re
from dataclasses import dataclass

from ragagent.documents import ParsedDocument, ParsedSection
from ragagent.schemas import SourceLocation


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    text: str
    contextual_text: str
    source: SourceLocation


def chunk_document(
    document: ParsedDocument,
    *,
    target_chars: int = 1100,
    overlap_chars: int = 160,
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    ordinal = 0
    for section in document.sections:
        for relative_start, text in _split_section(section, target_chars, overlap_chars):
            contextual = _local_context(document.filename, section)
            chunks.append(
                ChunkDraft(
                    ordinal=ordinal,
                    text=text,
                    contextual_text=contextual,
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
            ordinal += 1
    return chunks


def _local_context(filename: str, section: ParsedSection) -> str:
    parts = [f"Document: {filename}"]
    if section.section:
        parts.append(f"Section: {section.section}")
    if section.page:
        parts.append(f"Page: {section.page}")
    return " | ".join(parts)


def _split_section(
    section: ParsedSection,
    target_chars: int,
    overlap_chars: int,
) -> list[tuple[int, str]]:
    text = section.text.strip()
    if len(text) <= target_chars:
        return [(0, text)]

    boundaries = [match.end() for match in re.finditer(r"(?:\n\s*\n|(?<=[。！？.!?])\s+)", text)]
    boundaries.append(len(text))
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        desired_end = min(len(text), start + target_chars)
        candidates = [boundary for boundary in boundaries if start < boundary <= desired_end]
        end = max(candidates) if candidates else desired_end
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        cleaned = raw.strip()
        if cleaned:
            chunks.append((start + leading, cleaned))
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        whitespace = re.search(r"\s+", text[next_start:end])
        start = next_start + whitespace.end() if whitespace else next_start
    return chunks
