"""
core/rag_engine.py
───────────────────
Retrieval-Augmented Generation (RAG) chat over meeting transcripts
for RECALL — AI Meeting Intelligence.

Phase 3A changes
----------------
- build_rag_chain_from_meeting(meeting) is the canonical entry point.
  It uses add_meeting_segments() to index TranscriptSegments with full
  provenance metadata, then returns an LCEL RAG chain.
- build_rag_chain(transcript) is preserved as a backward-compatible
  adapter.

Phase 3B changes
----------------
- _build_chain() now accepts meeting_id and passes it to get_retriever()
  so the retriever applies a Chroma metadata filter:
      {"meeting_id": meeting_id}
  This ensures ONLY documents belonging to the current meeting are
  retrieved.  Cross-meeting contamination is blocked at retrieval time —
  not in the LLM prompt.
- build_rag_chain_from_meeting(meeting) passes meeting.id through the
  entire chain.
- build_rag_chain(transcript) retains unscoped behavior (meeting_id=None)
  with a documented warning about lack of isolation.

Retrieval isolation guarantee (canonical path):
  Meeting → add_meeting_segments(meeting.id, segments)
          → get_retriever(vector_store, meeting_id=meeting.id, k=4)
          → filter: {"meeting_id": meeting.id}
          → only this meeting's documents reach the LLM

Phase 3C changes
----------------
- build_structured_rag_response(answer_text, retrieved_docs) converts
  retrieved Chroma Documents into RAGEvidence entries.
- All evidence timestamps, segment IDs, and text come from Document
  metadata and content — the LLM cannot invent these.
- Evidence is deduplicated by segment_id so the same transcript segment
  does not appear multiple times due to sub-chunking or retrieval overlap.
- ask_question_structured(rag_chain, question) orchestrates retrieval,
  LLM invocation, and evidence extraction into a RAGAnswer.
- ask_question(rag_chain, question) remains backward compatible,
  returning only the answer string.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm_default
from core.models.rag_answer import RAGAnswer, RAGEvidence
from core.vector_store import add_meeting_segments, build_vector_store, get_retriever

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = """You are RECALL, an evidence-grounded meeting intelligence assistant.

Answer the user's question based on the meeting transcript context provided below.

IMPORTANT:
- If the retrieved context contains information that directly or indirectly answers the question, synthesize a clear, concise answer from that context.
- Always draw your answer from the retrieved context — do not rely on general knowledge.
- If relevant information exists in the context, answer the question. Do NOT say the information is unavailable when the context clearly contains an answer.
- Only say "I could not find this information in the meeting transcript." when the retrieved context genuinely does not contain an answer.
- Be concise and precise. Do not invent facts, names, timestamps, or details.
- When synthesizing multiple context excerpts, be clear about what was discussed.

Context from meeting transcript:
{context}"""


def _format_docs(docs) -> str:
    """Format documents for LLM context."""
    return "\n\n".join(doc.page_content for doc in docs)


def _is_no_answer(response_text: str) -> bool:
    """Check if the LLM response indicates no answer was found.

    The canonical no-answer response is:
        "I could not find this information in the meeting transcript."

    Parameters
    ----------
    response_text : str
        The LLM-generated answer text.

    Returns
    -------
    bool
        True if response indicates no answer found, False otherwise.
    """
    canonical_no_answer = "I could not find this information in the meeting transcript."
    return canonical_no_answer in response_text.strip()


def build_structured_rag_response(
    answer_text: str,
    retrieved_docs: list[Document],
) -> RAGAnswer:
    """Convert retrieved documents into structured RAG response with evidence.

    Parameters
    ----------
    answer_text:
        The natural-language answer from the LLM.
    retrieved_docs:
        The list of Chroma Documents returned by the retriever.
        Each carries metadata: segment_id, start, end, meeting_id, etc.

    Returns
    -------
    RAGAnswer
        A structured response with answer and deduplicated evidence.
        If the answer indicates no information was found, evidence is empty.

    Deduplication
    --------
    The same TranscriptSegment may appear multiple times due to:
    - Sub-chunking (long segments split into multiple documents)
    - Retrieval overlap (k=4 returns multiple chunks from the same segment)

    We deduplicate by segment_id, preserving the original segment's
    timestamp range (start, end).  The first occurrence wins.

    No-Answer Guarantee
    -------------------
    If the LLM answer is the canonical no-answer response, evidence is empty.
    This prevents displaying irrelevant retrieved documents as "supporting"
    an answer that does not exist.
    """
    # If there's no answer, return empty evidence
    if _is_no_answer(answer_text):
        logger.debug("No-answer response detected; returning empty evidence")
        return RAGAnswer(answer=answer_text, evidence=[])

    # Extract evidence from each retrieved document
    # (before deduplication, so we capture all context)
    evidence_list: list[RAGEvidence] = []
    seen_segment_ids: set[str] = set()

    for doc in retrieved_docs:
        meta = doc.metadata
        segment_id = meta.get("segment_id", "")
        start = meta.get("start", 0.0)
        end = meta.get("end", 0.0)

        # Deduplicate by segment_id: only create evidence for the first
        # occurrence of each segment
        if segment_id and segment_id not in seen_segment_ids:
            seen_segment_ids.add(segment_id)
            evidence_list.append(
                RAGEvidence(
                    segment_id=segment_id,
                    start=start,
                    end=end,
                    text=doc.page_content,
                )
            )

    return RAGAnswer(answer=answer_text, evidence=evidence_list)


def _build_chain(vector_store, meeting_id: str | None = None):
    """Build the LCEL RAG chain over *vector_store*.

    Parameters
    ----------
    vector_store:
        A Chroma instance already populated with meeting documents.
    meeting_id:
        When provided (non-empty), the retriever is scoped to this
        meeting via a Chroma metadata filter so other meetings' documents
        are never retrieved.  Pass None only for the legacy unscoped path.
    """
    retriever = get_retriever(vector_store, meeting_id=meeting_id, k=4)
    llm = get_llm_default()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def _build_retriever_chain(vector_store, meeting_id: str | None = None):
    """Build an LCEL chain that returns retrieved documents (not the answer).

    Used internally by ask_question_structured to get the raw documents
    for evidence extraction while still using the same meeting-scoped
    retriever as the main chain.

    Parameters
    ----------
    vector_store:
        A Chroma instance already populated with meeting documents.
    meeting_id:
        When provided (non-empty), the retriever is scoped to this
        meeting via a Chroma metadata filter.

    Returns
    -------
    Runnable
        An LCEL chain that accepts a question and returns retrieved documents.
    """
    retriever = get_retriever(vector_store, meeting_id=meeting_id, k=4)
    return retriever


def build_rag_chain_from_meeting(meeting) -> object:
    """Index a Meeting's TranscriptSegments and return a meeting-scoped RAG chain.

    Canonical entry point for Phase 3A+.  The retriever is filtered to
    ``meeting.id`` so only documents belonging to this meeting are
    retrieved — cross-meeting contamination is blocked at retrieval time.

    Parameters
    ----------
    meeting : Meeting
        A fully populated Meeting object with non-empty segments.

    Returns
    -------
    Runnable
        An LCEL chain that accepts a question string and returns an answer.
    """
    logger.info(
        "Building RAG chain from Meeting %s (%d segments).",
        meeting.id, len(meeting.segments),
    )
    vector_store = add_meeting_segments(
        meeting_id=meeting.id,
        segments=meeting.segments,
    )
    return _build_chain(vector_store, meeting_id=meeting.id)


def build_rag_chains_from_meeting(meeting) -> tuple[object, object]:
    """Index a Meeting and return both answer and retriever chains.

    Phase 3C: structured RAG requires both chains — one for generating
    answers and one for independently extracting evidence.  Both use
    the same meeting_id scoping for consistent isolation.

    Parameters
    ----------
    meeting : Meeting
        A fully populated Meeting object with non-empty segments.

    Returns
    -------
    tuple[Runnable, Runnable]
        A pair of LCEL chains:
        - rag_chain: generates natural-language answers
        - retriever_chain: returns the raw retrieved Documents for evidence

    Notes
    -----
    - Use with ask_question_structured() to get structured RAG responses.
    - Both chains are scoped to meeting.id, preserving Phase 3B isolation.
    """
    logger.info(
        "Building structured RAG chains from Meeting %s (%d segments).",
        meeting.id, len(meeting.segments),
    )
    vector_store = add_meeting_segments(
        meeting_id=meeting.id,
        segments=meeting.segments,
    )
    rag_chain = _build_chain(vector_store, meeting_id=meeting.id)
    retriever_chain = _build_retriever_chain(vector_store, meeting_id=meeting.id)
    return rag_chain, retriever_chain


def build_rag_chain(transcript: str) -> object:
    """Legacy adapter: index a plain transcript string and return a RAG chain.

    WARNING: This path has NO meeting isolation.  The retriever is
    unscoped (no metadata filter applied) and may return documents from
    other meetings stored in the same Chroma collection.  Use only for
    backward compatibility.

    Prefer build_rag_chain_from_meeting() for all new code.

    Parameters
    ----------
    transcript:
        The full meeting transcript as a plain string.

    Returns
    -------
    Runnable
        An LCEL chain that accepts a question string and returns an answer.
    """
    logger.warning(
        "build_rag_chain (legacy): unscoped retrieval — "
        "no meeting isolation applied.  %d chars.",
        len(transcript),
    )
    vector_store = build_vector_store(transcript)
    # meeting_id=None → unscoped retriever (legacy behavior)
    return _build_chain(vector_store, meeting_id=None)


def ask_question(rag_chain, question: str) -> str:
    """Legacy: invoke *rag_chain* with *question* and return the answer string.

    This function is backward compatible and continues to work with
    existing application code that expects a simple string response.

    Parameters
    ----------
    rag_chain:
        An LCEL chain returned by build_rag_chain_from_meeting() or
        build_rag_chain().
    question:
        The user's question.

    Returns
    -------
    str
        The natural-language answer (without evidence metadata).
    """
    logger.debug("RAG question (legacy): %s", question)
    answer = rag_chain.invoke(question)
    logger.debug("RAG answer length: %d chars", len(answer))
    return answer


def ask_question_structured(rag_chain, retriever_chain, question: str) -> RAGAnswer:
    """Invoke *rag_chain* and *retriever_chain* to return a structured RAG response.

    This is the canonical Phase 3C+ path for applications that need
    evidence-grounded answers.  The retriever is guaranteed to be scoped
    to the current meeting (via Phase 3B isolation), so evidence is
    sourced only from the intended meeting.

    Parameters
    ----------
    rag_chain:
        An LCEL chain returned by build_rag_chain_from_meeting() that
        generates natural-language answers.
    retriever_chain:
        An LCEL chain returned by _build_retriever_chain() that returns
        the actual Chroma Documents for evidence extraction.
    question:
        The user's question.

    Returns
    -------
    RAGAnswer
        A structured response with:
        - answer: the natural-language response
        - evidence: deduplicated transcript excerpts supporting the answer

    Notes
    -----
    - Both chains must use the same meeting_id scoping.
    - If retrieval returns 0 documents, evidence is empty.
    - Timestamps and segment IDs are never influenced by LLM output.
    """
    logger.debug("RAG question (structured): %s", question)

    # Invoke the answer chain
    answer_text = rag_chain.invoke(question)
    logger.debug("RAG answer length: %d chars", len(answer_text))

    # Independently retrieve and convert to evidence
    retrieved_docs = retriever_chain.invoke(question)
    logger.debug("Retrieved %d documents for evidence", len(retrieved_docs))

    # Build the structured response
    structured_response = build_structured_rag_response(answer_text, retrieved_docs)
    logger.debug(
        "Structured response: answer=%d chars, %d evidence items",
        len(structured_response.answer), len(structured_response.evidence),
    )

    return structured_response
