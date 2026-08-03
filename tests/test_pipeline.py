"""End-to-end tests. These run offline: no API key, no model download."""

from pathlib import Path

import pytest

from ragqa import ExtractiveAnswerer, RagPipeline
from ragqa.pipeline import REFUSAL

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def pipeline():
    pipe = RagPipeline(answerer=ExtractiveAnswerer())
    pipe.index_path(SAMPLE_DOCS)
    return pipe


def test_indexing_produces_chunks(pipeline):
    assert len(pipeline.store) > 0


def test_answer_is_grounded_in_a_retrieved_source(pipeline):
    result = pipeline.ask("how long does a rotated API key stay valid?")
    assert result.grounded
    assert result.citations
    assert any("api-guide" in c.source for c in result.citations)


def test_citation_markers_are_sequential_from_one(pipeline):
    result = pipeline.ask("what happens when a card payment fails?")
    assert [c.marker for c in result.citations] == list(range(1, len(result.citations) + 1))


def test_retrieval_finds_the_right_document(pipeline):
    result = pipeline.ask("how long is raw event data retained?")
    assert any("data-policy" in c.source for c in result.citations)


def test_off_topic_question_is_refused_rather_than_answered(pipeline):
    """The guard that separates 'not in these documents' from a confident
    answer assembled out of irrelevant passages."""
    result = pipeline.ask("what is the airspeed velocity of an unladen swallow")
    assert result.grounded is False
    assert result.text == REFUSAL
    assert result.citations == []


def test_k_limits_the_number_of_citations(pipeline):
    assert len(pipeline.ask("billing cycle", k=2).citations) <= 2


def test_context_is_numbered_to_match_citations(pipeline):
    hits = pipeline.store.search("refund policy", k=3)
    context = pipeline.build_context(hits)
    for marker in range(1, len(hits) + 1):
        assert f"[{marker}]" in context


def test_asking_before_indexing_is_an_error():
    with pytest.raises(RuntimeError):
        RagPipeline(answerer=ExtractiveAnswerer()).ask("anything")


def test_empty_question_is_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_indexing_nothing_is_an_error():
    with pytest.raises(ValueError):
        RagPipeline(answerer=ExtractiveAnswerer()).index_documents({})
