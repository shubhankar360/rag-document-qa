"""Answer generation.

Two implementations behind one interface: Claude when an API key is present,
and an extractive fallback otherwise. The fallback is what lets the pipeline,
the tests and the demo run on a clean checkout with no credentials — the
retrieval half of a RAG system is fully exercisable without a model.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from .grounding import build_vocabulary, content_terms, matches_any

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
- Use only what the sources state. Do not add outside knowledge.
- Cite the sources you used inline as [1], [2], and so on.
- If the sources do not contain the answer, say so plainly and stop. Do not guess.
- Be direct. Lead with the answer, then the supporting detail."""


NO_ANSWER = "The provided sources do not contain an answer to that question."

_SOURCE_HEADER = re.compile(r"^\[(\d+)\][^:\n]*:\s*", re.MULTILINE)


def _split_sources(context: str) -> list[tuple[str, str]]:
    """Split numbered context back into (marker, body) pairs."""
    blocks: list[tuple[str, str]] = []
    matches = list(_SOURCE_HEADER.finditer(context))
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        blocks.append((match.group(1), context[match.end() : stop]))
    return blocks


def _sentences(body: str) -> list[str]:
    """Split a passage into sentences.

    Whitespace is collapsed first: source documents are usually hard-wrapped,
    and splitting the raw text line by line truncates every sentence that
    spans a line break — which is most of them. Markdown headings are dropped
    beforehand, since they carry no terminal punctuation and would otherwise
    be glued onto the front of the sentence that follows.
    """
    prose = [line for line in body.splitlines() if not line.lstrip().startswith("#")]
    flat = " ".join(" ".join(prose).split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) >= 20]


class Answerer(Protocol):
    name: str

    def answer(self, question: str, context: str) -> str: ...


class ExtractiveAnswerer:
    """Zero-dependency fallback: return the sentences that best overlap the
    question, with their source markers intact.

    Not a language model and does not pretend to be one — it exists so the
    retrieval pipeline is verifiable end to end without credentials.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, question: str, context: str) -> str:
        terms = content_terms(question)
        if not terms:
            return NO_ANSWER

        scored: list[tuple[float, float, str]] = []

        for marker, body in _split_sources(context):
            for sentence in _sentences(body):
                words = build_vocabulary([sentence])
                # Prefix-aware, so "rotated" in the question matches
                # "Rotating" in the text without a stemmer — the same
                # matching the groundedness guard uses.
                hits = sum(matches_any(term, words) for term in terms)
                if hits < min(2, len(terms)):
                    continue
                # Rank by how much of the *question* a sentence covers, and
                # only break ties on brevity. Scoring by density alone
                # promotes short sentences that happen to share a common word
                # over the one that actually answers the question.
                coverage = hits / len(terms)
                density = hits / (len(words) ** 0.5)
                scored.append((coverage, density, f"{sentence} [{marker}]"))

        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda row: (-row[0], -row[1]))
        return " ".join(sentence for _, _, sentence in scored[: self.max_sentences])


class ClaudeAnswerer:
    """Generation via the Anthropic Messages API."""

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 1024):
        import anthropic

        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("no Anthropic credential in the environment")

        self.name = f"claude:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def answer(self, question: str, context: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Sources:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        # A refusal returns HTTP 200 with an empty or partial content list, so
        # check stop_reason before indexing into content.
        if message.stop_reason == "refusal":
            return "The request was declined by the model's safety system."
        return "\n".join(block.text for block in message.content if block.type == "text").strip()


def default_answerer() -> Answerer:
    """Use Claude when a credential is present; fall back so the pipeline
    always runs."""
    try:
        return ClaudeAnswerer()
    except Exception:
        return ExtractiveAnswerer()
