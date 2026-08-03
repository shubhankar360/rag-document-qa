#!/usr/bin/env python3
"""Streamlit chat UI.

    pip install streamlit && streamlit run app_streamlit.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ragqa import RagPipeline

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Ask your documents")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline()
    st.session_state.history = []

pipeline: RagPipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload .txt, .md or .pdf", accept_multiple_files=True, type=["txt", "md", "pdf"]
    )
    if uploads and st.button("Index", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp:
            for upload in uploads:
                (Path(tmp) / upload.name).write_bytes(upload.getvalue())
            count = pipeline.index_path(tmp)
        st.success(f"Indexed {count} chunks from {len(uploads)} file(s)")

    if not pipeline.store.is_empty:
        st.caption(f"{len(pipeline.store)} chunks indexed")
    st.caption(f"Embedder: `{pipeline.store.embedder.name}`")
    st.caption(f"Answerer: `{pipeline.answerer.name}`")

for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer.text)
        if answer.citations:
            with st.expander(f"{len(answer.citations)} sources"):
                for citation in answer.citations:
                    st.markdown(f"**[{citation.marker}] {citation.source}** · {citation.score:.3f}")
                    st.caption(citation.excerpt)

if question := st.chat_input("Ask a question about your documents"):
    if pipeline.store.is_empty:
        st.warning("Upload and index a document first.")
    else:
        st.session_state.history.append((question, pipeline.ask(question)))
        st.rerun()
