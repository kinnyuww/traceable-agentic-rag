from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragagent.agent import AgentService, EvidenceGate, _build_evidence_set
from ragagent.config import Settings
from ragagent.models import ChatResult
from ragagent.schemas import RetrievalHit, SourceLocation


def make_hit(index: int, score: float, document_id: str = "doc_primary") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk_{index}",
        text=f"evidence text {index}",
        contextual_text=f"Document: source-{document_id}.md | Section: {index}",
        source=SourceLocation(
            document_id=document_id,
            filename=f"source-{document_id}.md",
            section=f"section-{index}",
        ),
        rrf_score=1.0 / (60 + index),
        rerank_score=score,
        query_coverage=0.0,
    )


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_single_hop_evidence_set_selects_at_most_four_chunks() -> None:
    hits = [make_hit(index, 0.9 - index * 0.1) for index in range(6)]

    evidence_set = _build_evidence_set("What is the leave policy?", hits, settings())

    assert evidence_set.question_type == "single-hop"
    assert evidence_set.selection_policy == "dynamic_score_floor"
    assert evidence_set.limit == 4
    assert [hit.chunk_id for hit in evidence_set.hits] == [f"chunk_{index}" for index in range(4)]
    assert [hit.chunk_id for hit in evidence_set.discarded_hits] == ["chunk_4", "chunk_5"]


def test_multi_hop_evidence_set_can_include_rank_five_and_six() -> None:
    hits = [
        make_hit(0, 0.9),
        make_hit(1, 0.8),
        make_hit(2, 0.7),
        make_hit(3, 0.6),
        make_hit(4, 0.01, "doc_second"),
        make_hit(5, 0.001, "doc_second"),
    ]

    evidence_set = _build_evidence_set("Compare alpha and beta", hits, settings())

    assert evidence_set.question_type == "multi-hop"
    assert evidence_set.selection_policy == "retain_rerank_top_k"
    assert evidence_set.limit == 6
    assert evidence_set.score_floor is None
    assert [hit.chunk_id for hit in evidence_set.hits] == [f"chunk_{index}" for index in range(6)]


@pytest.mark.asyncio
async def test_multi_hop_gate_audits_rank_five_even_below_single_hop_floor() -> None:
    hits = [
        make_hit(0, 0.9),
        make_hit(1, 0.8),
        make_hit(2, 0.7),
        make_hit(3, 0.6),
        make_hit(4, 0.08, "doc_second"),
        make_hit(5, 0.07, "doc_second"),
    ]
    test_settings = settings(evidence_threshold=0.10)
    evidence_set = _build_evidence_set("Compare alpha and beta", hits, test_settings)

    decision = await EvidenceGate(test_settings, None).decide(
        "Compare alpha and beta", evidence_set, round_number=1
    )

    assert [hit.chunk_id for hit in evidence_set.hits] == [
        "chunk_0",
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
        "chunk_5",
    ]
    assert decision.decision == "answer"


def test_single_hop_score_floor_discards_low_tail_before_limit() -> None:
    hits = [
        make_hit(0, 0.9),
        make_hit(1, 0.4),
        make_hit(2, 0.08),
        make_hit(3, 0.07),
    ]

    evidence_set = _build_evidence_set("What is alpha?", hits, settings())

    assert evidence_set.score_floor == pytest.approx(0.09)
    assert [hit.chunk_id for hit in evidence_set.hits] == ["chunk_0", "chunk_1"]
    assert [hit.chunk_id for hit in evidence_set.discarded_hits] == [
        "chunk_2",
        "chunk_3",
    ]


@pytest.mark.asyncio
async def test_gate_uses_joint_query_coverage_across_selected_chunks() -> None:
    first = make_hit(0, 0.4, "doc_alpha").model_copy(
        update={"text": "alpha evidence"}
    )
    second = make_hit(1, 0.3, "doc_beta").model_copy(
        update={"text": "beta and evidence"}
    )
    test_settings = settings(evidence_threshold=0.20)
    evidence_set = _build_evidence_set("alpha and beta", [first, second], test_settings)

    decision = await EvidenceGate(test_settings, None).decide(
        "alpha and beta", evidence_set, round_number=1
    )

    assert decision.joint_query_coverage == 1.0
    assert decision.decision == "answer"


@dataclass
class RecordingChat:
    messages: list[dict[str, str]] | None = None

    async def complete(self, messages, *, temperature=0.0, max_tokens=1200):
        self.messages = messages
        return ChatResult(
            '{"decision":"retry","reason":"needs more evidence",'
            '"confidence":0.6,"missing_facts":["beta"]}'
        )


@pytest.mark.asyncio
async def test_gray_zone_llm_receives_all_six_multi_hop_evidence_chunks() -> None:
    hits = [make_hit(index, 0.71 - index * 0.1, f"doc_{index}") for index in range(6)]
    test_settings = settings(evidence_threshold=0.52)
    evidence_set = _build_evidence_set("missing 以及 facts", hits, test_settings)
    chat = RecordingChat()

    decision = await EvidenceGate(test_settings, chat).decide(
        "missing 以及 facts", evidence_set, round_number=1
    )

    assert decision.method == "llm"
    assert chat.messages is not None
    grader_prompt = chat.messages[-1]["content"]
    assert "[S6]" in grader_prompt
    assert "evidence text 5" in grader_prompt
    assert "retrieval_rank=6" in grader_prompt
    assert "rerank_score=0.210000" in grader_prompt
    assert "query-relevance hints only" in chat.messages[0]["content"]
    assert [hit.chunk_id for hit in evidence_set.hits] == [f"chunk_{index}" for index in range(6)]


@dataclass
class RecordingAnswerChat:
    messages: list[dict[str, str]] | None = None

    async def complete(self, messages, *, temperature=0.0, max_tokens=1200):
        self.messages = messages
        return ChatResult("Supported answer [S1] [S2]")


@pytest.mark.asyncio
async def test_answer_prompt_labels_rank_and_score_as_relevance_hints() -> None:
    chat = RecordingAnswerChat()
    service = AgentService(None, None, None, chat, settings())  # type: ignore[arg-type]
    hits = [make_hit(0, 0.9), make_hit(1, 0.4)]

    answer, _ = await service._generate_answer("What is alpha?", hits)

    assert answer == "Supported answer [S1] [S2]"
    assert chat.messages is not None
    assert "query-relevance hints only" in chat.messages[0]["content"]
    prompt = chat.messages[-1]["content"]
    assert 'retrieval_rank="1"' in prompt
    assert 'rerank_score="0.900000"' in prompt
