import pytest

from ragqa.chunking import chunk_text
from ragqa.store import VectorStore

CHUNKS = (
    chunk_text("API keys are created under Settings, Developers.", source="a.md")
    + chunk_text("Raw event data is retained for twelve months.", source="b.md")
    + chunk_text("Refunds must be requested within fourteen days.", source="c.md")
)


@pytest.fixture(scope="module")
def store():
    s = VectorStore()
    s.index(CHUNKS)
    return s


def test_search_ranks_the_matching_document_first(store):
    top, _ = store.search("where are API keys created", k=1)[0]
    assert top.source == "a.md"


def test_scores_are_sorted_descending(store):
    scores = [score for _, score in store.search("data retention", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_bounds_the_result_count(store):
    assert len(store.search("refund", k=2)) == 2


def test_blank_query_returns_nothing(store):
    assert store.search("   ") == []


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError):
        VectorStore().search("anything")


def test_indexing_an_empty_list_is_an_error():
    with pytest.raises(ValueError):
        VectorStore().index([])


def test_a_single_chunk_corpus_can_be_indexed_and_searched():
    # regression: min_df=2 on character n-grams made a one-chunk corpus unfittable
    from ragqa.chunking import Chunk
    from ragqa.embeddings import LsaEmbedder
    from ragqa.store import VectorStore

    store = VectorStore(LsaEmbedder())
    store.index([Chunk(id="a0", text="Refunds are processed within five days.", source="a.md", index=0, start=0, end=41)])
    hits = store.search("how long do refunds take?", k=3)
    assert len(hits) == 1 and hits[0][0].source == "a.md"
