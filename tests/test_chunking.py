import pytest

from ragqa.chunking import chunk_text

TEXT = (
    "Alpha paragraph about API keys and how they are created.\n\n"
    "Beta paragraph about rate limits and the 429 response code.\n\n"
    "Gamma paragraph about data retention windows across plans."
)


def test_short_text_is_a_single_chunk():
    assert len(chunk_text("just a sentence.", max_chars=500)) == 1


def test_empty_text_produces_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_carry_source_and_index():
    chunks = chunk_text(TEXT, source="notes.md", max_chars=70, overlap=10)
    assert all(c.source == "notes.md" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_citation_identifies_the_span():
    chunk = chunk_text("hello world", source="a.md")[0]
    assert chunk.citation == "a.md#chunk0"


def test_prefers_paragraph_boundaries_over_mid_sentence_cuts():
    chunks = chunk_text(TEXT, max_chars=70, overlap=0)
    assert any(c.text.startswith("Beta paragraph") for c in chunks)


def test_overlap_carries_context_across_the_boundary():
    with_overlap = chunk_text(TEXT, max_chars=70, overlap=25)
    without = chunk_text(TEXT, max_chars=70, overlap=0)
    assert len(with_overlap[1].text) > len(without[1].text)


def test_no_content_is_dropped():
    joined = " ".join(c.text for c in chunk_text(TEXT, max_chars=70, overlap=0))
    for word in ("Alpha", "Beta", "Gamma"):
        assert word in joined


def test_text_with_no_separators_is_hard_cut_not_dropped():
    chunks = chunk_text("x" * 250, max_chars=100, overlap=0)
    assert len(chunks) >= 3


def test_overlap_at_or_above_max_chars_is_rejected():
    """Otherwise each chunk re-emits the previous one and chunking never advances."""
    with pytest.raises(ValueError):
        chunk_text(TEXT, max_chars=100, overlap=100)


def test_non_positive_max_chars_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(TEXT, max_chars=0)
