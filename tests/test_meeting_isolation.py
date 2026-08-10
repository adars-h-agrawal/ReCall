"""
tests/test_meeting_isolation.py
────────────────────────────────
Phase 3B: meeting isolation tests for RECALL — AI Meeting Intelligence.

These tests verify that the meeting-scoped retriever ONLY returns
documents belonging to the requested meeting, even when multiple meetings
share the same Chroma collection.

Design
------
- A real (in-memory, non-persistent) Chroma collection is used so the
  actual Chroma filter logic is exercised.
- HuggingFace model weights are NOT loaded.  A deterministic fake
  embedding function returns fixed vectors so tests run without GPU,
  network, or model download.
- No Mistral, Whisper, Sarvam, or YouTube calls are made.

Coverage
--------
1. Correct meeting — retriever scoped to meeting-a returns only meeting-a.
2. Wrong meeting — retriever scoped to meeting-b never returns meeting-a.
3. Unknown meeting — retriever scoped to nonexistent ID returns no docs.
4. Both meetings coexist — neither contaminates the other.
5. Retrieved document metadata — all docs have the requested meeting_id.
6. get_retriever filter structure — meeting_id applied as Chroma filter.
7. Legacy unscoped retriever — no filter when meeting_id is None.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document


# ── Deterministic fake embeddings ─────────────────────────────────────────────
# Returns a fixed-length unit vector for every text input.
# This is sufficient for Chroma to accept the documents and perform
# metadata-filtered retrieval; actual semantic similarity is irrelevant
# for isolation tests.

class _FakeEmbeddings:
    """Deterministic embedding stub — returns a fixed 8-dim vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ── In-memory Chroma fixture ──────────────────────────────────────────────────

def _build_shared_collection() -> Chroma:
    """Create an in-memory Chroma collection with documents from two meetings.

    Meeting A: "The team selected PostgreSQL."
    Meeting B: "The team selected MongoDB."

    Both use the same collection, simulating the shared-collection
    architecture of RECALL.
    """
    docs = [
        Document(
            page_content="The team selected PostgreSQL.",
            metadata={
                "meeting_id": "meeting-a",
                "segment_id": "seg-a-1",
                "start": 10.0,
                "end": 15.0,
                "speaker": "",
                "language": "en",
                "source": "meeting_a.mp4",
                "chunk_index": 0,
                "is_split": False,
            },
        ),
        Document(
            page_content="The team selected MongoDB.",
            metadata={
                "meeting_id": "meeting-b",
                "segment_id": "seg-b-1",
                "start": 20.0,
                "end": 25.0,
                "speaker": "",
                "language": "en",
                "source": "meeting_b.mp4",
                "chunk_index": 0,
                "is_split": False,
            },
        ),
    ]
    # In-memory collection: no persist_directory → ephemeral, no disk writes.
    return Chroma.from_documents(
        documents=docs,
        embedding=_FakeEmbeddings(),
        collection_name="test_isolation",
    )


# ── Helper: run a scoped retrieval synchronously ─────────────────────────────

def _retrieve(vector_store: Chroma, meeting_id: str | None, query: str = "database") -> list[Document]:
    """Run a retrieval using the meeting-scoped get_retriever."""
    from core.vector_store import get_retriever
    retriever = get_retriever(vector_store, meeting_id=meeting_id, k=10)
    return retriever.invoke(query)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMeetingIsolation:
    """Core isolation invariant: one meeting's retriever must never return
    documents from a different meeting."""

    @pytest.fixture(scope="class")
    def shared_store(self):
        """Single Chroma collection shared by all tests in this class."""
        return _build_shared_collection()

    def test_meeting_a_retriever_returns_meeting_a_document(self, shared_store):
        """A retriever scoped to meeting-a must return meeting-a's document."""
        docs = _retrieve(shared_store, "meeting-a")
        assert len(docs) >= 1
        texts = [d.page_content for d in docs]
        assert any("PostgreSQL" in t for t in texts), (
            f"Expected PostgreSQL in results for meeting-a, got: {texts}"
        )

    def test_meeting_a_retriever_never_returns_meeting_b(self, shared_store):
        """A retriever scoped to meeting-a must NOT return meeting-b's document."""
        docs = _retrieve(shared_store, "meeting-a")
        for doc in docs:
            assert doc.metadata["meeting_id"] == "meeting-a", (
                f"Cross-meeting contamination: got meeting_id="
                f"'{doc.metadata['meeting_id']}' in meeting-a retriever"
            )

    def test_meeting_b_retriever_returns_meeting_b_document(self, shared_store):
        """A retriever scoped to meeting-b must return meeting-b's document."""
        docs = _retrieve(shared_store, "meeting-b")
        assert len(docs) >= 1
        texts = [d.page_content for d in docs]
        assert any("MongoDB" in t for t in texts), (
            f"Expected MongoDB in results for meeting-b, got: {texts}"
        )

    def test_meeting_b_retriever_never_returns_meeting_a(self, shared_store):
        """A retriever scoped to meeting-b must NOT return meeting-a's document."""
        docs = _retrieve(shared_store, "meeting-b")
        for doc in docs:
            assert doc.metadata["meeting_id"] == "meeting-b", (
                f"Cross-meeting contamination: got meeting_id="
                f"'{doc.metadata['meeting_id']}' in meeting-b retriever"
            )

    def test_nonexistent_meeting_returns_no_documents(self, shared_store):
        """A retriever scoped to an unknown meeting ID must return no documents."""
        docs = _retrieve(shared_store, "meeting-does-not-exist")
        assert docs == [], (
            f"Expected empty results for unknown meeting, got: "
            f"{[d.page_content for d in docs]}"
        )

    def test_all_returned_docs_have_requested_meeting_id(self, shared_store):
        """Every document returned by a scoped retriever must carry the
        requested meeting_id in its metadata — the isolation invariant."""
        for mid in ("meeting-a", "meeting-b"):
            docs = _retrieve(shared_store, mid)
            for doc in docs:
                assert doc.metadata.get("meeting_id") == mid, (
                    f"Retrieved doc has wrong meeting_id: "
                    f"expected '{mid}', got '{doc.metadata.get('meeting_id')}'"
                )

    def test_both_meetings_coexist_independently(self, shared_store):
        """Both meetings can be queried independently from the same collection
        without either contaminating the other."""
        docs_a = _retrieve(shared_store, "meeting-a")
        docs_b = _retrieve(shared_store, "meeting-b")

        ids_a = {d.metadata["meeting_id"] for d in docs_a}
        ids_b = {d.metadata["meeting_id"] for d in docs_b}

        assert ids_a == {"meeting-a"}, f"meeting-a contained foreign IDs: {ids_a}"
        assert ids_b == {"meeting-b"}, f"meeting-b contained foreign IDs: {ids_b}"
        # The two result sets share no documents.
        seg_ids_a = {d.metadata["segment_id"] for d in docs_a}
        seg_ids_b = {d.metadata["segment_id"] for d in docs_b}
        assert seg_ids_a.isdisjoint(seg_ids_b), (
            f"Overlapping segment IDs between meeting-a and meeting-b: "
            f"{seg_ids_a & seg_ids_b}"
        )


# ── Unit tests for get_retriever filter structure ─────────────────────────────

class TestGetRetrieverFilterStructure:
    """Verify the filter is wired correctly without running real similarity search."""

    def test_scoped_retriever_has_meeting_id_filter(self):
        """When meeting_id is supplied, search_kwargs must contain the filter."""
        from core.vector_store import get_retriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        get_retriever(mock_vs, meeting_id="mtg-xyz", k=4)

        call_kwargs = mock_vs.as_retriever.call_args
        search_kwargs = call_kwargs.kwargs.get("search_kwargs", {})
        assert search_kwargs.get("filter") == {"meeting_id": "mtg-xyz"}, (
            f"Expected filter {{'meeting_id': 'mtg-xyz'}}, got: {search_kwargs}"
        )

    def test_scoped_retriever_preserves_k(self):
        """k must be preserved in search_kwargs when meeting_id is supplied."""
        from core.vector_store import get_retriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        get_retriever(mock_vs, meeting_id="mtg-abc", k=6)

        search_kwargs = mock_vs.as_retriever.call_args.kwargs["search_kwargs"]
        assert search_kwargs["k"] == 6

    def test_unscoped_retriever_has_no_filter(self):
        """When meeting_id is None, search_kwargs must NOT contain a filter."""
        from core.vector_store import get_retriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        get_retriever(mock_vs, meeting_id=None, k=4)

        search_kwargs = mock_vs.as_retriever.call_args.kwargs["search_kwargs"]
        assert "filter" not in search_kwargs, (
            f"Unscoped retriever should have no filter, got: {search_kwargs}"
        )

    def test_empty_string_meeting_id_treated_as_unscoped(self):
        """An empty-string meeting_id (legacy sentinel) must not apply a filter."""
        from core.vector_store import get_retriever
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        get_retriever(mock_vs, meeting_id="", k=4)

        search_kwargs = mock_vs.as_retriever.call_args.kwargs["search_kwargs"]
        assert "filter" not in search_kwargs


# ── RAG engine wiring ─────────────────────────────────────────────────────────

class TestRagEngineIsolationWiring:
    """Verify build_rag_chain_from_meeting passes meeting.id to the retriever."""

    def test_build_rag_chain_from_meeting_passes_meeting_id_to_retriever(self):
        """The retriever created inside build_rag_chain_from_meeting must
        receive meeting.id as meeting_id — confirmed by capturing the
        get_retriever call arguments."""
        from core.rag_engine import build_rag_chain_from_meeting
        from core.models.meeting import Meeting
        from core.models.transcript import TranscriptSegment

        meeting = Meeting(source="test.mp4")
        seg = TranscriptSegment(
            meeting_id=meeting.id,
            text="We chose PostgreSQL for the database.",
            start=0.0,
            end=5.0,
        )
        meeting.segments = [seg]

        captured = {}

        def fake_get_retriever(vector_store, meeting_id=None, k=4):
            captured["meeting_id"] = meeting_id
            # Return a trivial mock retriever that the chain can hold.
            mock_ret = MagicMock()
            mock_ret.invoke = MagicMock(return_value=[])
            return mock_ret

        with patch("core.rag_engine.add_meeting_segments", return_value=MagicMock()):
            with patch("core.rag_engine.get_retriever", side_effect=fake_get_retriever):
                with patch("core.rag_engine.get_llm_default", return_value=MagicMock()):
                    build_rag_chain_from_meeting(meeting)

        assert captured.get("meeting_id") == meeting.id, (
            f"Expected meeting_id='{meeting.id}', "
            f"got '{captured.get('meeting_id')}'"
        )

    def test_build_rag_chain_legacy_passes_none_meeting_id(self):
        """The legacy build_rag_chain must pass meeting_id=None so the
        retriever remains unscoped."""
        from core.rag_engine import build_rag_chain

        captured = {}

        def fake_get_retriever(vector_store, meeting_id=None, k=4):
            captured["meeting_id"] = meeting_id
            mock_ret = MagicMock()
            mock_ret.invoke = MagicMock(return_value=[])
            return mock_ret

        with patch("core.rag_engine.build_vector_store", return_value=MagicMock()):
            with patch("core.rag_engine.get_retriever", side_effect=fake_get_retriever):
                with patch("core.rag_engine.get_llm_default", return_value=MagicMock()):
                    build_rag_chain("Some transcript text.")

        assert captured.get("meeting_id") is None, (
            f"Legacy path should pass meeting_id=None, "
            f"got '{captured.get('meeting_id')}'"
        )
