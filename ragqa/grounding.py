"""Deciding whether the corpus can answer a question at all.

A RAG system that always answers is a RAG system that confabulates on
out-of-scope questions: retrieval returns *something* for every query, and a
generator handed irrelevant passages will still write a confident paragraph.
The guard has to run before generation.

**A similarity floor does not work here, and that is measured, not assumed.**
Over `sample_docs` with the offline LSA embedder, the best-matching chunk
scores 0.907 for "what is the airspeed velocity of an unladen swallow" and
0.905 for "how long is raw event data retained?" — the off-topic question
scores *higher*. Normalising the score against the corpus (z-margin of the top
hit over the mean) does not separate them either: on-topic queries span
1.12–1.78 and off-topic 0.97–1.48, which overlap.

The reason is that character-n-gram TF-IDF plus SVD mostly encodes generic
English text structure, so every query is broadly "similar" to every chunk.

Vocabulary coverage does separate them, because it asks a lexical question and
answers it lexically: do this question's content words appear in the indexed
corpus at all? On the same query set, on-topic queries score 0.75–1.00 and
off-topic 0.00–0.25 — a gap wide enough to put the threshold in the middle.

Matching is prefix-based in both directions so that "refund" matches "refunds"
and "retained" matches "retention" without a stemmer.
"""

from __future__ import annotations

import re

# Function words carry no topical signal; leaving them in would let any
# well-formed English question clear the threshold.
STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from get had has have how i if in
    into is it its may me might must my of on or should so than that the their them then there
    these they this to was we were what when where which who why will with would you your""".split()
)

MIN_TERM_LENGTH = 3
_TOKEN = re.compile(r"[a-z0-9]+")


def content_terms(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in STOPWORDS and len(token) >= MIN_TERM_LENGTH
    ]


def build_vocabulary(texts: list[str]) -> frozenset[str]:
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(_TOKEN.findall(text.lower()))
    return frozenset(vocabulary)


def _stem_prefix(token: str) -> str:
    """Enough of a token to match its inflections without a stemmer."""
    return token[: max(4, len(token) - 2)]


def matches_any(term: str, vocabulary: frozenset[str]) -> bool:
    """Whether `term` appears in `vocabulary`, allowing for inflection."""
    if term in vocabulary:
        return True
    term_prefix = _stem_prefix(term)
    return any(
        word.startswith(term_prefix) or term.startswith(_stem_prefix(word))
        for word in vocabulary
    )


def vocabulary_coverage(question: str, vocabulary: frozenset[str]) -> float:
    """Fraction of the question's content terms that appear in the corpus.

    Returns 0.0 for a question with no content terms at all — a bare "what is
    it?" is unanswerable for the same reason an off-topic question is.
    """
    terms = content_terms(question)
    if not terms:
        return 0.0
    return sum(matches_any(term, vocabulary) for term in terms) / len(terms)
