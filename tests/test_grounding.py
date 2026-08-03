"""The guard that decides whether the corpus can answer at all.

The two parametrised tests below are the measurement that motivated using
lexical coverage instead of a similarity floor — see ragqa/grounding.py.
"""

from pathlib import Path

import pytest

from ragqa import ExtractiveAnswerer, RagPipeline
from ragqa.grounding import build_vocabulary, content_terms, vocabulary_coverage
from ragqa.pipeline import REFUSAL

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"

ON_TOPIC = [
    "how long does a rotated API key stay valid?",
    "how long is raw event data retained?",
    "what happens when a card payment fails?",
    "can I get a refund",
    "what does a 403 mean",
    "European Union data residency",
    "export to CSV",
    "batch events in one request",
]

OFF_TOPIC = [
    "what is the airspeed velocity of an unladen swallow",
    "recipe for sourdough bread starter",
    "who won the world cup in 1998",
    "best hiking trails in Patagonia",
    "how do I train for a marathon",
    "translate this poem into French",
]


@pytest.fixture(scope="module")
def vocabulary():
    return build_vocabulary(
        [path.read_text(encoding="utf-8") for path in SAMPLE_DOCS.glob("*.md")]
    )


@pytest.fixture(scope="module")
def pipeline():
    pipe = RagPipeline(answerer=ExtractiveAnswerer())
    pipe.index_path(SAMPLE_DOCS)
    return pipe


def test_content_terms_drop_function_words():
    assert content_terms("What is the retention window?") == ["retention", "window"]


def test_prefix_matching_handles_inflections(vocabulary):
    """"refunds" is in the corpus; "refund" and "retention" must still match."""
    assert vocabulary_coverage("refund", vocabulary) == 1.0
    assert vocabulary_coverage("retention", vocabulary) == 1.0


def test_question_with_no_content_terms_scores_zero(vocabulary):
    assert vocabulary_coverage("what is it?", vocabulary) == 0.0


@pytest.mark.parametrize("question", ON_TOPIC)
def test_on_topic_questions_clear_the_threshold(question, vocabulary):
    assert vocabulary_coverage(question, vocabulary) >= 0.5


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_questions_fall_below_the_threshold(question, vocabulary):
    assert vocabulary_coverage(question, vocabulary) < 0.5


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_pipeline_refuses_every_off_topic_question(question, pipeline):
    result = pipeline.ask(question)
    assert result.grounded is False
    assert result.text == REFUSAL


@pytest.mark.parametrize("question", ON_TOPIC)
def test_pipeline_answers_every_on_topic_question(question, pipeline):
    result = pipeline.ask(question)
    assert result.grounded is True
    assert result.citations


def test_similarity_alone_would_not_have_caught_this(pipeline):
    """Documents the measurement: the top cosine score for an off-topic
    question is not meaningfully below that of a real one, which is why the
    guard is lexical. If a future embedder change makes scores discriminative,
    this test fails and the guard can be revisited."""
    relevant = max(s for _, s in pipeline.store.search("how long is raw event data retained?"))
    nonsense = max(s for _, s in pipeline.store.search("airspeed velocity of an unladen swallow"))
    assert nonsense > relevant - 0.15
