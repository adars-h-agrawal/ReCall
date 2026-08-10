"""
core/vector_store.py
─────────────────────
ChromaDB vector store management for RECALL — AI Meeting Intelligence.

Phase 3A changes
----------------
- HuggingFaceEmbeddings now imported from langchain_huggingface (not the
  deprecated langchain_community path).
- Canonical ingestion path: add_meeting_segments(meeting_id, segments)
  accepts TranscriptSegment objects and stores each as a Chroma Document
  with full provenance metadata.
- Every stored document carries:
    meeting_id, segment_id, start, end, speaker, language, source
  so a retrieved chunk can always be traced back to its source segment
  and ultimately to its Meeting.
- If a segment is long enough to require sub-splitting for retrieval
  quality, the sub-chunks preserve the parent segment's provenance
  (meeting_id, segment_id, start, end, source).
- The old build_vector_store(transcript: str) is kept as a backward-
  compatible adapter used by the legacy rag_engine path.

Phase 3B changes
----------------
- get_retriever() now accepts an explicit meeting_id parameter.
  When meeting_id is provided (non-empty string), a Chroma metadata
  filter {"meeting_id": meeting_id} is applied so retrieval is scoped
  to documents belonging only to that meeting.
- The canonical path (build_rag_chain_from_meeting) ALWAYS supplies
  meeting_id.  The legacy path (build_rag_chain) passes None and retains
  the previous unscoped behavior with a documented warning.
- Pre-Phase-3A documents that lack the meeting_id field will NOT be
  returned by a meeting-scoped query, providing natural isolation from
  legacy data.

Known limitation (Phase 3B):
  All meetings still share a single ChromaDB collection.  The per-meeting
  filter provides retrieval isolation but not storage isolation.  A
  collection-per-meeting architecture is a future consideration.

Legacy data compatibility:
  Documents indexed before Phase 3A lack the meeting_id field.  They
  will not be returned by meeting-scoped queries.  To include them in
  a meeting's context, re-index the meeting via the canonical pipeline.

Data model
----------
Chroma stores Documents with these metadata fields:

  meeting_id  : str   — UUID of the meeting (never empty in canonical path)
  segment_id  : str   — UUID of the TranscriptSegment
  start       : float — segment start in seconds
  end         : float — segment end in seconds
  speaker     : str   — speaker label or "" if unknown/None
  language    : str   — BCP-47 output language tag (e.g. "en")
  source      : str   — original media URL or file path
  chunk_index : int   — position within this segment (0 if no split)
  is_split    : bool  — True if this document is a sub-chunk of a segment

Note on None values: ChromaDB metadata does not support None.  All None
fields are stored as empty strings ("") to preserve round-trip fidelity.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import PATHS, EmbeddingConfig, VectorStoreConfig
from core.models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

_emb_cfg = EmbeddingConfig()
_vs_cfg = VectorStoreConfig()

# Maximum characters for a single Chroma document.
# Segments longer than this are sub-split while preserving provenance.
_MAX_DOC_CHARS = 800
_DOC_OVERLAP_CHARS = 80


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

    Imported from langchain_huggingface (the dedicated, non-deprecated
    package).  Loading all-MiniLM-L6-v2 is slow (~2–5 s); caching avoids
    reloading on every Streamlit rerun.
    """
    logger.debug("Loading embedding model: %s", _emb_cfg.model_name)
    return HuggingFaceEmbeddings(
        model_name=_emb_cfg.model_name,
        model_kwargs={"device": _emb_cfg.device},
    )


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _none_to_str(value: Optional[str]) -> str:
    """Chroma metadata does not support None; convert to empty string."""
    return value if value is not None else ""


def _segment_to_base_metadata(seg: TranscriptSegment) -> dict:
    """Build the provenance metadata dict for a TranscriptSegment."""
    return {
        "meeting_id": seg.meeting_id,
        "segment_id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "speaker": _none_to_str(seg.speaker),
        "language": seg.language,
        "source": seg.source,
        "chunk_index": 0,
        "is_split": False,
    }


# ── Segment → Documents conversion ───────────────────────────────────────────

def segments_to_documents(segments: list[TranscriptSegment]) -> list[Document]:
    """Convert TranscriptSegments to Chroma Documents with full provenance.

    Short segments (≤ _MAX_DOC_CHARS) become a single Document.
    Long segments are sub-split; each sub-chunk inherits the parent
    segment's meeting_id, segment_id, start, end, and source so
    provenance is never lost.

    Parameters
    ----------
    segments:
        Ordered list of TranscriptSegment objects.

    Returns
    -------
    list[Document]
        Ready-to-embed Documents carrying full provenance metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_MAX_DOC_CHARS,
        chunk_overlap=_DOC_OVERLAP_CHARS,
        separators=[". ", " ", ""],
    )

    docs: list[Document] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        base_meta = _segment_to_base_metadata(seg)

        if len(text) <= _MAX_DOC_CHARS:
            docs.append(Document(page_content=text, metadata=base_meta))
        else:
            # Sub-split long segments while preserving provenance.
            sub_chunks = splitter.split_text(text)
            for i, sub in enumerate(sub_chunks):
                meta = dict(base_meta)
                meta["chunk_index"] = i
                meta["is_split"] = True
                docs.append(Document(page_content=sub, metadata=meta))

    return docs


# ── Canonical ingestion API ───────────────────────────────────────────────────

def add_meeting_segments(
    meeting_id: str,
    segments: list[TranscriptSegment],
) -> Chroma:
    """Index TranscriptSegments for *meeting_id* into ChromaDB.

    This is the canonical ingestion path for Phase 3A+.
    Every stored document carries meeting_id and full segment provenance.

    Parameters
    ----------
    meeting_id:
        UUID of the meeting.  Stored on every document for future
        per-meeting filtering.
    segments:
        Ordered list of TranscriptSegment objects from the pipeline.

    Returns
    -------
    Chroma
        A ready-to-query vector store instance.
    """
    logger.info(
        "Indexing %d segments for meeting %s into ChromaDB.",
        len(segments), meeting_id,
    )

    docs = segments_to_documents(segments)
    if not docs:
        raise ValueError(
            f"No documents produced from segments for meeting {meeting_id}. "
            "Check that segments contain non-empty text."
        )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=_vs_cfg.collection_name,
        persist_directory=str(PATHS.vector_db),
    )
    logger.info(
        "Indexed %d documents for meeting %s into collection '%s'.",
        len(docs), meeting_id, _vs_cfg.collection_name,
    )
    return vector_store


# ── Legacy adapter ────────────────────────────────────────────────────────────

def build_vector_store(transcript: str) -> Chroma:
    """Legacy adapter: chunk a plain transcript string and index it.

    This path is preserved for backward compatibility with the existing
    rag_engine.build_rag_chain(transcript) call in app.py and main.py.
    It stores documents with a minimal metadata schema (only chunk_index)
    because no TranscriptSegment provenance is available here.

    Prefer add_meeting_segments() for all new code.

    NOTE: Documents indexed via this path carry meeting_id="" and will not
    support per-meeting filtering.  They should be considered legacy data.
    """
    logger.debug(
        "build_vector_store (legacy): indexing transcript (%d chars).",
        len(transcript),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_vs_cfg.chunk_size,
        chunk_overlap=_vs_cfg.chunk_overlap,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "meeting_id": "",
                "segment_id": "",
                "start": 0.0,
                "end": 0.0,
                "speaker": "",
                "language": "",
                "source": "",
                "chunk_index": i,
                "is_split": False,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=_vs_cfg.collection_name,
        persist_directory=str(PATHS.vector_db),
    )
    logger.debug(
        "Legacy: indexed %d chunks into collection '%s'.",
        len(docs), _vs_cfg.collection_name,
    )
    return vector_store


# ── Retriever ─────────────────────────────────────────────────────────────────

def get_retriever(
    vector_store: Chroma,
    meeting_id: Optional[str] = None,
    k: int = 4,
):
    """Return a similarity-search retriever over *vector_store*.

    Parameters
    ----------
    vector_store:
        A Chroma instance returned by add_meeting_segments() or
        build_vector_store().
    meeting_id:
        When provided (non-empty string), a Chroma metadata filter
        ``{"meeting_id": meeting_id}`` is applied so ONLY documents
        belonging to that meeting are retrieved.

        The canonical Meeting path MUST always supply this argument.

        Pass None (or omit) only for the legacy plain-transcript path
        that has no meeting identity.  In that case retrieval is
        unscoped and may return documents from any meeting — this is a
        documented limitation of the legacy adapter.
    k:
        Number of documents to retrieve per query.

    Returns
    -------
    VectorStoreRetriever
        A retriever that applies the meeting-scoped filter when
        meeting_id is provided.
    """
    search_kwargs: dict = {"k": k}

    if meeting_id:
        # Apply Chroma metadata filter to restrict retrieval to this meeting.
        # Documents from other meetings (or legacy documents without a
        # meeting_id) will not be returned.
        search_kwargs["filter"] = {"meeting_id": meeting_id}
        logger.debug(
            "Retriever scoped to meeting_id=%s (k=%d).", meeting_id, k
        )
    else:
        logger.debug(
            "Retriever UNSCOPED (legacy path, no meeting_id supplied, k=%d).", k
        )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
