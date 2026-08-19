import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("benchmark_trace_helpers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_round_candidate_ids = MODULE._round_candidate_ids
_round_reranked_ids = MODULE._round_reranked_ids


def event(stage: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(stage=stage, payload=payload)


def test_benchmark_reads_all_recall_views_and_global_rerank() -> None:
    trace = [
        event(
            "retrieval_round",
            {
                "round": 1,
                "dense_candidates": [{"chunk_id": "a"}, {"chunk_id": "shared"}],
                "reranked_candidates": [],
            },
        ),
        event(
            "retrieval_round",
            {
                "round": 1,
                "dense_candidates": [{"chunk_id": "shared"}, {"chunk_id": "b"}],
                "reranked_candidates": [],
            },
        ),
        event(
            "retrieval_merge_rerank",
            {
                "round": 1,
                "global_rerank": {
                    "candidates": [{"chunk_id": "b"}, {"chunk_id": "a"}]
                },
            },
        ),
    ]

    assert _round_candidate_ids(
        trace,
        round_number=1,
        candidate_group="dense_candidates",
    ) == ["a", "shared", "b"]
    assert _round_reranked_ids(trace, round_number=1) == ["b", "a"]


def test_benchmark_keeps_old_single_retrieval_trace_compatibility() -> None:
    trace = [
        event(
            "retrieval_round",
            {
                "round": 1,
                "reranked_candidates": [{"chunk_id": "legacy"}],
            },
        )
    ]

    assert _round_reranked_ids(trace, round_number=1) == ["legacy"]
