"""Splitting documents into retrievable chunks.

Chunk size is the central tradeoff in a RAG system. Too small and a chunk
loses the context that makes it interpretable; too large and the retrieved
passage is mostly irrelevant text the model has to read and you have to pay
for. Overlap exists so a fact that straddles a boundary survives in at least
one chunk intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tried in order: split on the strongest structural boundary that works, and
# only fall back to cutting mid-word when a single token exceeds the limit.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass(frozen=True)
class Chunk:
    """A retrievable span, with enough provenance to cite it."""

    id: str
    text: str
    source: str
    index: int
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source}#chunk{self.index}"


def _split_recursive(text: str, max_chars: int, separators: list[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    if not separators:
        # No separator left that helps — hard-cut rather than emit an oversized chunk.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, max_chars, rest)

    out: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            out.append(buffer)
        # The part alone may still be too big; recurse with a weaker separator.
        if len(part) > max_chars:
            out.extend(_split_recursive(part, max_chars, rest))
            buffer = ""
        else:
            buffer = part
    if buffer:
        out.append(buffer)
    return [chunk for chunk in out if chunk.strip()]


def chunk_text(
    text: str,
    *,
    source: str = "document",
    max_chars: int = 900,
    overlap: int = 150,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks that respect structure where possible.

    `start`/`end` are character offsets into the original text, which is what
    lets an answer cite an exact span rather than a whole document.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars, or chunking never advances")

    pieces = _split_recursive(text, max_chars, SEPARATORS)

    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start == -1:  # hard-cut path can lose exact byte alignment
            start = cursor
        end = start + len(piece)
        cursor = max(start + 1, end - overlap)

        body = piece
        if overlap and chunks:
            tail = chunks[-1].text[-overlap:]
            body = f"{tail} {piece}"
            start = max(0, start - len(tail) - 1)

        chunks.append(
            Chunk(
                id=f"{source}::{i}",
                text=body,
                source=source,
                index=i,
                start=start,
                end=end,
                metadata=dict(metadata or {}),
            )
        )
    return chunks
