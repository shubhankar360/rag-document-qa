"""In-memory vector store.

Vectors are L2-normalised, so cosine similarity is a dot product and the whole
index is one matrix multiply. At document-Q&A scale that is faster than any
approximate index, and it keeps the dependency surface at numpy.
"""

from __future__ import annotations

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, default_embedder


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or default_embedder()
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def index(self, chunks: list[Chunk]) -> None:
        """Build the index. Re-indexes from scratch — the offline embedder is
        fitted on the corpus, so incremental adds would silently mix vectors
        from two different fits."""
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self.chunks = list(chunks)
        self._matrix = self.embedder.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        if not query.strip():
            return []
        sims = self._matrix @ self.embedder.transform([query])[0]
        order = np.argsort(-sims)[: max(0, k)]
        return [(self.chunks[i], float(sims[i])) for i in order]
