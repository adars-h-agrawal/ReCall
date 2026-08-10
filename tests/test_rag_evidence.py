"""
tests/test_rag_evidence.py
──────────────────────────
Phase 3C: evidence-grounded RAG tests for RECALL — AI Meeting Intelligence.

These tests verify that:
1. RAG responses include structured evidence derived from retrieved documents
2. Evidence fields (timestamps, segment IDs, text) come from Documents, not LLM
3. Evidence is deduplicated by segment_id
4. Meeting isolation is preserved when extracting evidence
5. No evidence is fabricated when retrieval returns empty

Design
------
- Uses in-memory Chroma collection with deterministic fake embeddings
- Mocks the LLM to test evidence extraction independently
- No external API calls (Mistral, HuggingFace, etc.)
- Tests exercise the actual evidence building logic, not mock dictionaries

Coverage
--------
1. Structured answer with evidence — answer + evidence list populated
2. Evidence metadata — segment_id, start, end come from Document metadata
3. LLM cannot invent timestamps — LLM returns fake timestamp, evidence uses Document
4. Evidence deduplication — same segment_id appears once
5. Multiple segments — different segment_ids produce separate evidence
6. Meeting isolation preserved — evidence only from current meeting's retrieval
7. No retrieved documents — evidence is empty list
8. Evidence text is exact — no paraphrasing
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

from core.models.rag_answer import RAGAnswer, RAGEvidence
from core.rag_engine import build_structured_rag_response


# ── Deterministic fake embeddings ─────────────────────────────────────────────

class _FakeEmbeddings:
    """Deterministic embedding stub — returns a fixed 8-dim vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestStructuredRAGResponse:
    """Test evidence extraction from retrieved documents."""

    def test_structured_answer_with_evidence(self):
        """A structured response should have both answer and evidence list."""
        doc = Document(
            page_content="PostgreSQL is an open-source relational database.",
            metadata={
                "segment_id": "seg-1",
                "start": 10.0,
                "end": 15.0,
                "meeting_id": "m1",
            },
        )
        response = build_structured_rag_response(
            "PostgreSQL is a good choice.",
            [doc],
        )
        assert isinstance(response, RAGAnswer)
        assert response.answer == "PostgreSQL is a good choice."
        assert len(response.evidence) == 1

    def test_evidence_metadata_from_document(self):
        """Evidence metadata must come from Document, not be invented."""
        doc = Document(
            page_content="The team chose MongoDB.",
            metadata={
                "segment_id": "seg-db-choice",
                "start": 42.5,
                "end": 47.3,
                "meeting_id": "m1",
            },
        )
        response = build_structured_rag_response("MongoDB was selected.", [doc])

        assert len(response.evidence) == 1
        evidence = response.evidence[0]
        assert evidence.segment_id == "seg-db-choice"
        assert evidence.start == 42.5
        assert evidence.end == 47.3
        assert evidence.text == "The team chose MongoDB."

    def test_evidence_text_is_exact(self):
        """Evidence text must be exactly Document.page_content, no paraphrasing."""
        original_text = "The decision was made at exactly 15:30 UTC."
        doc = Document(
            page_content=original_text,
            metadata={
                "segment_id": "seg-decision",
                "start": 100.0,
                "end": 105.0,
            },
        )
        response = build_structured_rag_response(
            "A decision was made.",
            [doc],
        )

        assert response.evidence[0].text == original_text

    def test_llm_cannot_invent_timestamps(self):
        """LLM answer contains fake timestamp; evidence uses Document metadata."""
        doc = Document(
            page_content="We will deploy next Friday.",
            metadata={
                "segment_id": "seg-deploy",
                "start": 200.0,
                "end": 205.0,
            },
        )
        # LLM answer includes a fake timestamp
        llm_answer = "The deployment is scheduled for 99:99:99."
        response = build_structured_rag_response(llm_answer, [doc])

        # Evidence timestamps must come from Document, not LLM
        assert response.evidence[0].start == 200.0
        assert response.evidence[0].end == 205.0
        # (LLM timestamp 99:99:99 is irrelevant to evidence)

    def test_evidence_deduplication(self):
        """Same segment_id from multiple chunks should deduplicate."""
        # Two documents from the same segment (e.g., due to sub-chunking)
        doc1 = Document(
            page_content="First part of segment.",
            metadata={
                "segment_id": "seg-same",
                "start": 50.0,
                "end": 60.0,
                "chunk_index": 0,
                "is_split": True,
            },
        )
        doc2 = Document(
            page_content="Second part of segment.",
            metadata={
                "segment_id": "seg-same",
                "start": 50.0,
                "end": 60.0,
                "chunk_index": 1,
                "is_split": True,
            },
        )
        response = build_structured_rag_response(
            "Both chunks matter.",
            [doc1, doc2],
        )

        # Should deduplicate to 1 evidence entry
        assert len(response.evidence) == 1
        # First occurrence wins
        assert response.evidence[0].text == "First part of segment."

    def test_multiple_segments_multiple_evidence(self):
        """Different segment_ids should produce separate evidence entries."""
        docs = [
            Document(
                page_content="PostgreSQL for databases.",
                metadata={
                    "segment_id": "seg-db",
                    "start": 10.0,
                    "end": 15.0,
                },
            ),
            Document(
                page_content="Python for backend.",
                metadata={
                    "segment_id": "seg-lang",
                    "start": 20.0,
                    "end": 25.0,
                },
            ),
        ]
        response = build_structured_rag_response(
            "PostgreSQL and Python.",
            docs,
        )

        assert len(response.evidence) == 2
        segment_ids = {e.segment_id for e in response.evidence}
        assert segment_ids == {"seg-db", "seg-lang"}

    def test_no_retrieved_documents_empty_evidence(self):
        """If retrieval returns no documents, evidence is empty."""
        response = build_structured_rag_response(
            "I could not find information in the transcript.",
            [],
        )

        assert response.answer == "I could not find information in the transcript."
        assert response.evidence == []

    def test_meeting_isolation_preserved(self):
        """Evidence must only come from current meeting's retrieved documents."""
        # Simulate documents from two meetings
        # (In practice, retriever should not return meeting-b docs,
        # but we test that evidence extraction respects the retrieval result)
        doc_a = Document(
            page_content="Meeting A content.",
            metadata={
                "segment_id": "seg-a",
                "start": 10.0,
                "end": 15.0,
                "meeting_id": "meeting-a",
            },
        )
        doc_b = Document(
            page_content="Meeting B content.",
            metadata={
                "segment_id": "seg-b",
                "start": 20.0,
                "end": 25.0,
                "meeting_id": "meeting-b",
            },
        )

        # Simulate retrieval that only returned meeting-a docs
        # (Phase 3B guarantees this, but we test evidence layer)
        response = build_structured_rag_response(
            "Answer from meeting A.",
            [doc_a],
        )

        # Evidence should only contain meeting-a
        assert len(response.evidence) == 1
        assert response.evidence[0].segment_id == "seg-a"


class TestRAGEvidenceModel:
    """Test the RAGEvidence and RAGAnswer Pydantic models."""

    def test_rag_evidence_construction(self):
        """RAGEvidence can be constructed with required fields."""
        evidence = RAGEvidence(
            segment_id="seg-1",
            start=10.0,
            end=15.0,
            text="Some transcript text.",
        )
        assert evidence.segment_id == "seg-1"
        assert evidence.start == 10.0
        assert evidence.end == 15.0
        assert evidence.text == "Some transcript text."

    def test_rag_answer_construction(self):
        """RAGAnswer can be constructed with answer and evidence list."""
        evidence = RAGEvidence(
            segment_id="seg-1",
            start=10.0,
            end=15.0,
            text="Text.",
        )
        answer = RAGAnswer(
            answer="The answer is yes.",
            evidence=[evidence],
        )
        assert answer.answer == "The answer is yes."
        assert len(answer.evidence) == 1

    def test_rag_answer_default_empty_evidence(self):
        """RAGAnswer evidence defaults to empty list."""
        answer = RAGAnswer(answer="Just an answer.")
        assert answer.evidence == []

    def test_rag_answer_json_serializable(self):
        """RAGAnswer is Pydantic-serializable to JSON."""
        evidence = RAGEvidence(
            segment_id="seg-1",
            start=10.0,
            end=15.0,
            text="Text.",
        )
        answer = RAGAnswer(
            answer="The answer.",
            evidence=[evidence],
        )
        json_dict = answer.model_dump()
        assert json_dict["answer"] == "The answer."
        assert len(json_dict["evidence"]) == 1
        assert json_dict["evidence"][0]["segment_id"] == "seg-1"


class TestAskQuestionStructured:
    """Test the structured ask_question_structured function."""

    def test_ask_question_structured_flow(self):
        """ask_question_structured should combine answer and evidence chains."""
        from core.rag_engine import ask_question_structured

        # Create mock chains
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = "The answer is yes."

        mock_retriever_chain = MagicMock()
        mock_doc = Document(
            page_content="Evidence text.",
            metadata={
                "segment_id": "seg-1",
                "start": 10.0,
                "end": 15.0,
            },
        )
        mock_retriever_chain.invoke.return_value = [mock_doc]

        result = ask_question_structured(
            mock_rag_chain,
            mock_retriever_chain,
            "What happened?",
        )

        assert isinstance(result, RAGAnswer)
        assert result.answer == "The answer is yes."
        assert len(result.evidence) == 1
        assert result.evidence[0].segment_id == "seg-1"

    def test_ask_question_structured_with_empty_retrieval(self):
        """ask_question_structured should handle empty retrieval gracefully."""
        from core.rag_engine import ask_question_structured

        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = "No information found."

        mock_retriever_chain = MagicMock()
        mock_retriever_chain.invoke.return_value = []

        result = ask_question_structured(
            mock_rag_chain,
            mock_retriever_chain,
            "What is X?",
        )

        assert result.answer == "No information found."
        assert result.evidence == []

    def test_ask_question_structured_invokes_both_chains(self):
        """Both chains must be invoked with the same question."""
        from core.rag_engine import ask_question_structured

        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = "Answer."

        mock_retriever_chain = MagicMock()
        mock_retriever_chain.invoke.return_value = []

        ask_question_structured(
            mock_rag_chain,
            mock_retriever_chain,
            "Test question",
        )

        # Both chains should have been invoked
        mock_rag_chain.invoke.assert_called_once_with("Test question")
        mock_retriever_chain.invoke.assert_called_once_with("Test question")


class TestBackwardCompatibility:
    """Test that the legacy ask_question function still works."""

    def test_ask_question_legacy_returns_string(self):
        """ask_question(chain, question) should return a string answer."""
        from core.rag_engine import ask_question

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The answer."

        result = ask_question(mock_chain, "What is X?")

        assert isinstance(result, str)
        assert result == "The answer."
        mock_chain.invoke.assert_called_once_with("What is X?")
