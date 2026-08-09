"""
core/vector_store.py
─────────────────────
ChromaDB vector store management for RECALL — AI Meeting Intelligence.

Builds and retrieves vector stores from meeting transcript text.
All paths and model names come from core.config so nothing is hardcoded.

Known limitation (Phase 1):
  All meetings share a single ChromaDB collection ("meeting_transcript").
  Chunks from different meetings can contaminate each other's retrieval.
  Per-meeting isolation will be addressed in a future phase.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import PATHS, EmbeddingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)

_emb_cfg = EmbeddingConfig()
_vs_cfg = VectorStoreConfig()


# ── Streamlit-safe cache decorator ────────────────────────────────────────────
def _cache_resource(fn):
    """Apply @st.cache_resource inside Streamlit; otherwise return fn as-is."""
    try:
        import streamlit as st
        return st.cache_resource(fn)
    except Exception:
        return fn


@_cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model.

    Loading all-MiniLM-L6-v2 is slow (~2–5 s); caching avoids reloading
    on every Streamlit rerun.
    """
    logger.debug("Loading embedding model: %s", _emb_cfg.model_name)
    return HuggingFaceEmbeddings(
        model_name=_emb_cfg.model_name,
        model_kwargs={"device": _emb_cfg.device},
    )


def build_vector_store(transcript: str) -> Chroma:
    """Chunk *transcript* and index it into ChromaDB.

    Parameters
    ----------
    transcript:
        Plain-text meeting transcript.

    Returns
    -------
    Chroma
        A ready-to-query vector store instance.
    """
    logger.debug("Building vector store from transcript (%d chars).", len(transcript))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_vs_cfg.chunk_size,
        chunk_overlap=_vs_cfg.chunk_overlap,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=_vs_cfg.collection_name,
        persist_directory=str(PATHS.vector_db),
    )
    logger.debug("Indexed %d chunks into ChromaDB collection '%s'.", len(docs), _vs_cfg.collection_name)
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """Return a similarity-search retriever over *vector_store*."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
