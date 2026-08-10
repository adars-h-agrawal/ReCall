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
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm_default
from core.vector_store import add_meeting_segments, build_vector_store, get_retriever

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question \
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: \
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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
    """Invoke *rag_chain* with *question* and return the answer string."""
    logger.debug("RAG question: %s", question)
    answer = rag_chain.invoke(question)
    logger.debug("RAG answer length: %d chars", len(answer))
    return answer
