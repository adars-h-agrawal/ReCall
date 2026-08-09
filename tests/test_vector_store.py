"""
tests/test_vector_store.py
───────────────────────────
Unit tests for Phase 3A RAG data foundation — RECALL AI Meeting Intelligence.

These tests do NOT:
  - Call real Mistral, Whisper, Sarvam, or YouTube APIs.
  - Perform real vector similarity search.
  - Write to the persistent vector_db directory.
  - Load the actual HuggingFace model weights.

They verify:
  1. HuggingFace embedding import comes from langchain_huggingface.
  2. get_embeddings() is Streamlit-cache-resource decorated (or returns fn).
  3. segments_to_documents(): metadata schema correctness.
  4. segments_to_documents(): None speaker stored as empty string.
  5. segments_to_documents(): all required metadata keys present.
  6. segments_to_documents(): multiple segments retain distinct provenance.
  7. segments_to_documents(): meeting_id present on every document.
  8. segments_to_documents(): long segment is sub-split with provenance.
  9. segments_to_documents(): sub-chunks inherit parent segment_id/start/end.
  10. segments_to_documents(): empty-text segments are skipped.
  11. add_meeting_segments(): raises ValueError on empty segment list.
  12. add_meeting_segments(): passes correct meeting_id to documents.
  13. build_vector_store (legacy): documents carry empty-string meeting_id.
  14. Document provenance trace: segment_id links document back to segment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.transcript import TranscriptSegment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_seg(
    text: str,
    start: float,
    end: float,
    meeting_id: str = "meeting-test",
    speaker: str | None = None,
    language: str = "en",
    source: str = "test.mp4",
) -> TranscriptSegment:
    return TranscriptSegment(
        meeting_id=meeting_id,
        text=text,
        start=start,
        end=end,
        speaker=speaker,
        language=language,
        source=source,
    )


# ── 1. HuggingFace import source ──────────────────────────────────────────────

class TestHuggingFaceImport:
    def test_embeddings_class_from_langchain_huggingface(self):
        """HuggingFaceEmbeddings must come from langchain_huggingface,
        not the deprecated langchain_community path."""
        from langchain_huggingface import HuggingFaceEmbeddings as LHF
        from core.vector_store import get_embeddings
        import inspect

        # Read the source of vector_store to confirm the import statement.
        import core.vector_store as vs_module
        source = inspect.getsource(vs_module)
        assert "from langchain_huggingface import HuggingFaceEmbeddings" in source
        assert "langchain_community.embeddings" not in source

    def test_get_embeddings_is_callable(self):
        from core.vector_store import get_embeddings
        assert callable(get_embeddings)


# ── 2. segments_to_documents: metadata schema ─────────────────────────────────

class TestSegmentsToDocuments:
    REQUIRED_KEYS = {
        "meeting_id", "segment_id", "start", "end",
        "speaker", "language", "source", "chunk_index", "is_split",
    }

    def test_single_segment_produces_one_document(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Hello world.", 0.0, 2.0)
        docs = segments_to_documents([seg])
        assert len(docs) == 1

    def test_all_required_metadata_keys_present(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Hello world.", 0.0, 2.0)
        docs = segments_to_documents([seg])
        for key in self.REQUIRED_KEYS:
            assert key in docs[0].metadata, f"Missing metadata key: {key}"

    def test_meeting_id_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0, meeting_id="meeting-123")
        docs = segments_to_documents([seg])
        assert docs[0].metadata["meeting_id"] == "meeting-123"

    def test_segment_id_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0)
        docs = segments_to_documents([seg])
        assert docs[0].metadata["segment_id"] == seg.id

    def test_start_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 42.5, 48.2)
        docs = segments_to_documents([seg])
        assert docs[0].metadata["start"] == 42.5

    def test_end_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 42.5, 48.2)
        docs = segments_to_documents([seg])
        assert docs[0].metadata["end"] == 48.2

    def test_language_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0, language="en")
        docs = segments_to_documents([seg])
        assert docs[0].metadata["language"] == "en"

    def test_source_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0, source="meeting.mp4")
        docs = segments_to_documents([seg])
        assert docs[0].metadata["source"] == "meeting.mp4"

    def test_none_speaker_stored_as_empty_string(self):
        """ChromaDB does not support None in metadata.
        speaker=None must be stored as ""."""
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0, speaker=None)
        docs = segments_to_documents([seg])
        assert docs[0].metadata["speaker"] == ""
        assert docs[0].metadata["speaker"] is not None

    def test_known_speaker_preserved(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Text.", 0.0, 1.0, speaker="SPEAKER_01")
        docs = segments_to_documents([seg])
        assert docs[0].metadata["speaker"] == "SPEAKER_01"

    def test_short_segment_is_not_split(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("Short text.", 0.0, 1.0)
        docs = segments_to_documents([seg])
        assert len(docs) == 1
        assert docs[0].metadata["is_split"] is False
        assert docs[0].metadata["chunk_index"] == 0

    def test_empty_text_segment_skipped(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("   ", 0.0, 1.0)  # whitespace only
        docs = segments_to_documents([seg])
        assert docs == []

    def test_multiple_segments_distinct_segment_ids(self):
        from core.vector_store import segments_to_documents
        seg1 = _make_seg("First segment text.", 0.0, 5.0)
        seg2 = _make_seg("Second segment text.", 5.0, 10.0)
        docs = segments_to_documents([seg1, seg2])
        assert len(docs) == 2
        ids = [d.metadata["segment_id"] for d in docs]
        assert ids[0] != ids[1]

    def test_multiple_segments_distinct_timestamps(self):
        from core.vector_store import segments_to_documents
        seg1 = _make_seg("First.", 0.0, 5.0)
        seg2 = _make_seg("Second.", 10.0, 15.0)
        docs = segments_to_documents([seg1, seg2])
        assert docs[0].metadata["start"] == 0.0
        assert docs[1].metadata["start"] == 10.0

    def test_meeting_id_on_every_document(self):
        from core.vector_store import segments_to_documents
        segs = [
            _make_seg("One.", 0.0, 1.0, meeting_id="mtg-abc"),
            _make_seg("Two.", 1.0, 2.0, meeting_id="mtg-abc"),
            _make_seg("Three.", 2.0, 3.0, meeting_id="mtg-abc"),
        ]
        docs = segments_to_documents(segs)
        for doc in docs:
            assert doc.metadata["meeting_id"] == "mtg-abc"


# ── 3. Long segment sub-splitting with provenance ─────────────────────────────

class TestLongSegmentSplitting:
    def test_long_segment_produces_multiple_documents(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)  # well over the threshold
        seg = _make_seg(long_text, 10.0, 70.0)
        docs = segments_to_documents([seg])
        assert len(docs) > 1

    def test_sub_chunks_inherit_parent_segment_id(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)
        seg = _make_seg(long_text, 10.0, 70.0)
        docs = segments_to_documents([seg])
        for doc in docs:
            assert doc.metadata["segment_id"] == seg.id

    def test_sub_chunks_inherit_parent_meeting_id(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)
        seg = _make_seg(long_text, 10.0, 70.0, meeting_id="mtg-long")
        docs = segments_to_documents([seg])
        for doc in docs:
            assert doc.metadata["meeting_id"] == "mtg-long"

    def test_sub_chunks_inherit_parent_timestamps(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)
        seg = _make_seg(long_text, 42.0, 102.0)
        docs = segments_to_documents([seg])
        for doc in docs:
            # All sub-chunks carry the parent segment's original time range.
            assert doc.metadata["start"] == 42.0
            assert doc.metadata["end"] == 102.0

    def test_sub_chunks_marked_is_split_true(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)
        seg = _make_seg(long_text, 0.0, 60.0)
        docs = segments_to_documents([seg])
        for doc in docs:
            assert doc.metadata["is_split"] is True

    def test_sub_chunks_have_sequential_chunk_index(self):
        from core.vector_store import segments_to_documents, _MAX_DOC_CHARS
        long_text = "word " * (_MAX_DOC_CHARS // 4)
        seg = _make_seg(long_text, 0.0, 60.0)
        docs = segments_to_documents([seg])
        indices = [d.metadata["chunk_index"] for d in docs]
        assert indices == list(range(len(docs)))


# ── 4. add_meeting_segments (mocked Chroma) ───────────────────────────────────

class TestAddMeetingSegments:
    """Test add_meeting_segments without calling real ChromaDB or embeddings."""

    def _mock_chroma(self):
        """Patch Chroma.from_documents to avoid real DB/embedding calls."""
        mock_vs = MagicMock()
        return patch(
            "core.vector_store.Chroma.from_documents",
            return_value=mock_vs,
        ), mock_vs

    def test_raises_on_empty_segment_list(self):
        from core.vector_store import add_meeting_segments
        with pytest.raises(ValueError, match="No documents"):
            add_meeting_segments("meeting-empty", [])

    def test_raises_on_all_empty_text_segments(self):
        from core.vector_store import add_meeting_segments
        segs = [_make_seg("   ", 0.0, 1.0)]
        with pytest.raises(ValueError, match="No documents"):
            add_meeting_segments("meeting-empty", segs)

    def test_calls_chroma_from_documents(self):
        from core.vector_store import add_meeting_segments
        segs = [_make_seg("Hello world.", 0.0, 2.0, meeting_id="mtg-1")]
        patch_chroma, mock_vs = self._mock_chroma()
        with patch_chroma:
            with patch("core.vector_store.get_embeddings", return_value=MagicMock()):
                result = add_meeting_segments("mtg-1", segs)
        assert result is mock_vs

    def test_documents_passed_to_chroma_carry_meeting_id(self):
        from core.vector_store import add_meeting_segments
        segs = [
            _make_seg("Segment one.", 0.0, 5.0, meeting_id="mtg-check"),
            _make_seg("Segment two.", 5.0, 10.0, meeting_id="mtg-check"),
        ]
        captured_docs = []

        def fake_from_documents(documents, **kwargs):
            captured_docs.extend(documents)
            return MagicMock()

        with patch("core.vector_store.Chroma.from_documents", side_effect=fake_from_documents):
            with patch("core.vector_store.get_embeddings", return_value=MagicMock()):
                add_meeting_segments("mtg-check", segs)

        assert len(captured_docs) == 2
        for doc in captured_docs:
            assert doc.metadata["meeting_id"] == "mtg-check"


# ── 5. Legacy build_vector_store produces consistent metadata schema ──────────

class TestLegacyBuildVectorStore:
    """The legacy path should still produce documents with the full metadata
    schema (even if most fields are empty), so retrieval code can rely on
    consistent key presence."""

    def test_legacy_documents_carry_empty_meeting_id(self):
        from core.vector_store import build_vector_store
        captured_docs = []

        def fake_from_documents(documents, **kwargs):
            captured_docs.extend(documents)
            return MagicMock()

        with patch("core.vector_store.Chroma.from_documents", side_effect=fake_from_documents):
            with patch("core.vector_store.get_embeddings", return_value=MagicMock()):
                build_vector_store("Short transcript for testing purposes.")

        assert len(captured_docs) >= 1
        for doc in captured_docs:
            assert "meeting_id" in doc.metadata
            assert doc.metadata["meeting_id"] == ""

    def test_legacy_documents_carry_all_schema_keys(self):
        from core.vector_store import build_vector_store
        required = {
            "meeting_id", "segment_id", "start", "end",
            "speaker", "language", "source", "chunk_index", "is_split",
        }
        captured_docs = []

        def fake_from_documents(documents, **kwargs):
            captured_docs.extend(documents)
            return MagicMock()

        with patch("core.vector_store.Chroma.from_documents", side_effect=fake_from_documents):
            with patch("core.vector_store.get_embeddings", return_value=MagicMock()):
                build_vector_store("Short transcript for testing purposes.")

        for doc in captured_docs:
            for key in required:
                assert key in doc.metadata, f"Missing metadata key: {key}"


# ── 6. Provenance trace: document → segment ───────────────────────────────────

class TestProvenanceTrace:
    """Verify a retrieved document's metadata can be used to trace it back
    to the originating TranscriptSegment."""

    def test_document_segment_id_matches_source_segment(self):
        from core.vector_store import segments_to_documents
        seg = _make_seg("We agreed to use Python 3.12.", 55.0, 60.0, meeting_id="mtg-trace")
        docs = segments_to_documents([seg])

        # Simulate retrieval: use metadata to look up the source segment.
        doc = docs[0]
        assert doc.metadata["segment_id"] == seg.id
        assert doc.metadata["meeting_id"] == seg.meeting_id
        assert doc.metadata["start"] == seg.start
        assert doc.metadata["end"] == seg.end

    def test_two_segments_two_distinct_provenance_chains(self):
        from core.vector_store import segments_to_documents
        seg_a = _make_seg("Decision A was made.", 10.0, 15.0, meeting_id="mtg-x")
        seg_b = _make_seg("Decision B was made.", 20.0, 25.0, meeting_id="mtg-x")
        docs = segments_to_documents([seg_a, seg_b])

        assert docs[0].metadata["segment_id"] == seg_a.id
        assert docs[1].metadata["segment_id"] == seg_b.id
        assert docs[0].metadata["start"] == 10.0
        assert docs[1].metadata["start"] == 20.0
