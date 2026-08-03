"""Citation-grounded question answering over your own documents."""

from .chunking import Chunk, chunk_text
from .embeddings import LsaEmbedder, SentenceTransformerEmbedder, default_embedder
from .llm import ClaudeAnswerer, ExtractiveAnswerer, default_answerer
from .loaders import UnsupportedDocument, load_directory, load_text
from .pipeline import Answer, Citation, RagPipeline
from .store import VectorStore

__all__ = [
    "Chunk", "chunk_text",
    "LsaEmbedder", "SentenceTransformerEmbedder", "default_embedder",
    "ClaudeAnswerer", "ExtractiveAnswerer", "default_answerer",
    "UnsupportedDocument", "load_directory", "load_text",
    "Answer", "Citation", "RagPipeline",
    "VectorStore",
]
