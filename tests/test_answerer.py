from ragqa.llm import ExtractiveAnswerer

CONTEXT = (
    "[1] api-guide.md: Rotating a key issues a new secret and keeps the old one "
    "valid for twenty four hours. Revoking a key takes effect immediately.\n"
    "[2] billing.md: Refunds are issued for the current billing period only."
)


def test_answer_draws_from_the_relevant_source():
    answer = ExtractiveAnswerer().answer("how long does a rotated key stay valid?", CONTEXT)
    assert "twenty four hours" in answer


def test_answer_carries_the_citation_marker():
    answer = ExtractiveAnswerer().answer("how long does a rotated key stay valid?", CONTEXT)
    assert "[1]" in answer


def test_unanswerable_question_says_so_instead_of_guessing():
    answer = ExtractiveAnswerer().answer("what colour is the logo", CONTEXT)
    assert "do not contain" in answer.lower()


def test_max_sentences_bounds_the_answer():
    answer = ExtractiveAnswerer(max_sentences=1).answer("key refunds billing period", CONTEXT)
    assert answer.count(".") <= 2


WRAPPED = (
    "[1] api-guide.md: ## Authentication\n"
    "Rotating a key issues a new secret and keeps the old one valid for twenty\n"
    "four hours, which lets you deploy the replacement without downtime."
)


def test_sentence_spanning_a_line_break_is_not_truncated():
    """Source documents are hard-wrapped; splitting line by line would cut
    this sentence at "twenty" and lose the answer."""
    answer = ExtractiveAnswerer().answer("how long does a rotated key stay valid?", WRAPPED)
    assert "twenty four hours" in answer


def test_markdown_headings_are_not_glued_onto_the_sentence():
    answer = ExtractiveAnswerer().answer("how long does a rotated key stay valid?", WRAPPED)
    assert "## Authentication" not in answer


def test_answer_ranks_by_question_coverage_not_brevity():
    """A short sentence sharing one common word must not outrank the sentence
    that actually answers the question."""
    context = (
        "[1] doc.md: Create an API key under Settings. "
        "Rotating an API key keeps the old secret valid for twenty four hours."
    )
    answer = ExtractiveAnswerer(max_sentences=1).answer(
        "how long does a rotated API key stay valid?", context
    )
    assert "twenty four hours" in answer
