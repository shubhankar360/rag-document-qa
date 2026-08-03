"""The RAG pipeline: index, retrieve, assemble context, answer, cite.

The design commitment here is that every answer is traceable. Retrieved
passages are numbered, the numbers are what the model is told to cite, and the
same numbers come back to the caller attached to real chunks with real
character offsets. An answer you cannot trace is an answer you cannot check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_text
from .grounding import build_vocabulary, vocabulary_coverage
from .llm import Answerer, default_answerer
from .loaders import load_directory, load_text
from .store import VectorStore

REFUSAL = "I could not find anything in the indexed documents that answers that question."


@dataclass
class Citation:
    marker: int
    source: str
    score: float
    excerpt: str
    start: int
    end: int


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    backend: str = ""

    def __str__(self) -> str:
        if not self.citations:
            return self.text
        sources = "\n".join(f"  [{c.marker}] {c.source} ({c.score:.3f})" for c in self.citations)
        return f"{self.text}\n\nSources:\n{sources}"


class RagPipeline:
    def __init__(
        self,
        *,
        answerer: Answerer | None = None,
        store: VectorStore | None = None,
        chunk_chars: int = 900,
        chunk_overlap: int = 150,
        top_k: int = 4,
        min_coverage: float = 0.5,
        min_score: float = 0.0,
    ):
        self.store = store or VectorStore()
        self.answerer = answerer or default_answerer()
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        # The groundedness guard. See ragqa/grounding.py for why this is
        # lexical coverage and not a similarity floor — the floor was measured
        # and does not separate on-topic from off-topic questions.
        self.min_coverage = min_coverage
        # A similarity floor on top, off by default. It buys nothing with the
        # offline embedder, but becomes a useful second filter with a real
        # neural encoder, where scores are actually discriminative.
        self.min_score = min_score
        self._vocabulary: frozenset[str] = frozenset()

    # ---------- indexing ----------

    def add_text(self, text: str, *, source: str = "document") -> list[Chunk]:
        return chunk_text(
            text, source=source, max_chars=self.chunk_chars, overlap=self.chunk_overlap
        )

    def index_documents(self, documents: dict[str, str]) -> int:
        chunks: list[Chunk] = []
        for source, text in documents.items():
            chunks.extend(self.add_text(text, source=source))
        if not chunks:
            raise ValueError("no indexable text found in the supplied documents")
        self.store.index(chunks)
        self._vocabulary = build_vocabulary([c.text for c in chunks])
        return len(chunks)

    def index_path(self, path: Path | str) -> int:
        path = Path(path)
        documents = load_directory(path) if path.is_dir() else {path.name: load_text(path)}
        return self.index_documents(documents)

    # ---------- querying ----------

    def build_context(self, hits: list[tuple[Chunk, float]]) -> str:
        """Number the passages so the model has something concrete to cite."""
        return "\n\n".join(
            f"[{i}] {chunk.source}: {chunk.text}" for i, (chunk, _) in enumerate(hits, start=1)
        )

    def ask(self, question: str, *, k: int | None = None) -> Answer:
        if self.store.is_empty:
            raise RuntimeError("index documents before asking questions")
        if not question.strip():
            raise ValueError("question must not be empty")

        coverage = vocabulary_coverage(question, self._vocabulary)
        hits = self.store.search(question, k=k or self.top_k)
        hits = [(chunk, score) for chunk, score in hits if score >= self.min_score]

        if coverage < self.min_coverage or not hits:
            return Answer(question=question, text=REFUSAL, grounded=False, backend=self.answerer.name)

        text = self.answerer.answer(question, self.build_context(hits))
        citations = [
            Citation(
                marker=i,
                source=chunk.citation,
                score=score,
                excerpt=chunk.text[:220],
                start=chunk.start,
                end=chunk.end,
            )
            for i, (chunk, score) in enumerate(hits, start=1)
        ]
        return Answer(
            question=question, text=text, citations=citations, backend=self.answerer.name
        )
