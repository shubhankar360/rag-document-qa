"""Embedding backends.

Two implementations behind one interface. The default needs no API key and no
model download, so a clean checkout can be indexed and queried offline; the
sentence-transformers backend is a one-line swap for real semantic retrieval.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class LsaEmbedder:
    """Offline default: word + character TF-IDF reduced by truncated SVD.

    Latent semantic indexing, not a neural encoder — it generalises past exact
    token overlap but cannot resolve a synonym absent from the corpus. Stated
    plainly rather than presented as something it isn't.
    """

    name = "lsa"

    def __init__(self, n_components: int = 192, random_state: int = 0):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
            ]
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._n_components = n_components
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self._vectorizer.fit_transform(texts)
        self._svd.n_components = max(1, min(self._n_components, min(sparse.shape) - 1))
        dense = self._svd.fit_transform(sparse)
        self._fitted = True
        return l2_normalize(dense)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("index the corpus before querying it")
        return l2_normalize(self._svd.transform(self._vectorizer.transform(texts)))


class SentenceTransformerEmbedder:
    """Real neural embeddings, used when sentence-transformers is installed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def default_embedder() -> Embedder:
    """Prefer a real encoder; fall back so a clean checkout still runs."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return LsaEmbedder()
