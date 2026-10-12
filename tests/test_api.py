from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ragqa import RagPipeline  # noqa: E402
from ragqa.api import create_app  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "sample_docs"


def client(index_root=None):
    pipe = RagPipeline()
    pipe.index_path(DOCS)
    return TestClient(create_app(pipe, index_root=index_root))


def test_ask_returns_a_cited_answer():
    r = client().post("/ask", json={"question": "how long is raw event data retained?"}).json()
    assert r["grounded"] and r["citations"]


def test_indexing_over_http_is_off_by_default():
    assert client().post("/index", json={"path": str(DOCS)}).status_code == 403


def test_index_root_blocks_escapes(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("Refunds are processed within five days.")
    (tmp_path / "secret.txt").write_text("the vault code is 1234")
    (tmp_path / "docs" / "link.txt").symlink_to(tmp_path / "secret.txt")
    c = client(index_root=tmp_path / "docs")
    assert c.post("/index", json={"path": "a.md"}).json() == {"chunks": 1}
    assert c.post("/index", json={"path": "../secret.txt"}).status_code == 403
    assert c.post("/index", json={"path": str(tmp_path / "secret.txt")}).status_code == 403
    assert c.post("/index", json={"path": "link.txt"}).status_code == 403
