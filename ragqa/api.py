"""FastAPI service.

Imported lazily by `serve.py` so the core package installs and tests without
FastAPI present — the pipeline is the product, the HTTP layer is one adapter.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RagPipeline


class IndexRequest(BaseModel):
    path: str = Field(description="File or directory to index")


class AskRequest(BaseModel):
    question: str
    k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    marker: int
    source: str
    score: float
    excerpt: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    grounded: bool
    backend: str
    citations: list[CitationOut]


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="rag-document-qa", version="1.0.0")
    app.state.pipeline = pipeline or RagPipeline()

    @app.get("/health")
    def health() -> dict:
        pipe: RagPipeline = app.state.pipeline
        return {
            "status": "ok",
            "chunks": len(pipe.store),
            "embedder": pipe.store.embedder.name,
            "answerer": pipe.answerer.name,
        }

    @app.post("/index")
    def index(request: IndexRequest) -> dict:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no such path: {request.path}")
        return {"chunks": app.state.pipeline.index_path(path)}

    @app.post("/ask", response_model=AnswerOut)
    def ask(request: AskRequest) -> AnswerOut:
        pipe: RagPipeline = app.state.pipeline
        if pipe.store.is_empty:
            raise HTTPException(status_code=409, detail="index documents before asking")
        result = pipe.ask(request.question, k=request.k)
        return AnswerOut(
            question=result.question,
            answer=result.text,
            grounded=result.grounded,
            backend=result.backend,
            citations=[
                CitationOut(marker=c.marker, source=c.source, score=c.score, excerpt=c.excerpt)
                for c in result.citations
            ],
        )

    return app
