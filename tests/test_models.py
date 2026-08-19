import numpy as np
import pytest

from ragagent.models import (
    DeterministicEmbeddingClient,
    LexicalRerankClient,
    fts_text,
    lexical_terms,
    query_coverage,
    trust_environment_proxy,
)


@pytest.mark.asyncio
async def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    client = DeterministicEmbeddingClient(64)
    first, second = await client.embed(["混合检索 BM25", "混合检索 BM25"])
    assert first == second
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_chinese_terms_include_characters_and_bigrams() -> None:
    terms = lexical_terms("混合检索")
    assert "混" in terms
    assert "混合" in terms
    assert "检索" in terms
    assert "混合" in fts_text("混合检索")


@pytest.mark.asyncio
async def test_lexical_reranker_rewards_query_coverage() -> None:
    client = LexicalRerankClient()
    scores = await client.rerank("年假 天数", ["员工年假天数为十天", "办公地点在上海"])
    assert scores[0] > scores[1]
    assert query_coverage("年假 天数", "员工年假天数为十天") > 0.5


def test_local_model_urls_bypass_environment_proxy() -> None:
    assert trust_environment_proxy("http://localhost:12434/engines/v1") is False
    assert trust_environment_proxy("http://127.0.0.1:9000/v1") is False
    assert trust_environment_proxy("http://model-runner.docker.internal/rerank") is False
    assert trust_environment_proxy("https://api.deepseek.com") is True
