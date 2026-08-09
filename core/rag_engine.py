"""
core/rag_engine.py
───────────────────
Retrieval-Augmented Generation (RAG) chat over meeting transcripts
for RECALL — AI Meeting Intelligence.

Phase 3A changes
----------------
- build_rag_chain_from_meeting(meeting) is the new canonical entry point.
  It uses add_meeting_segments() to index TranscriptSegments with full
  provenance metadata, then returns an LCEL RAG chain.
- build_rag_chain(transcript) is preserved as a backward-compatible
  adapter used by app.py and main.py until they are migrated.

Known limitation (Phase 3A):
  ChromaDB still uses a single shared collection for all meetings.
  Retrieved documents may include chunks from other meetings.
  Per-meeting retrieval filtering will be added in Phase 3B.
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


def _build_chain(vector_store):
    """Build the LCEL RAG chain over an existing vector store."""
    retriever = get_retriever(vector_store, k=4)
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
    """Index a Meeting's TranscriptSegments and return a RAG chain.

    This is the canonical RAG entry point for Phase 3A+.
    Uses add_meeting_segments() so every indexed document carries full
    provenance (meeting_id, segment_id, start, end, speaker, language,
    source).

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
    return _build_chain(vector_store)


def build_rag_chain(transcript: str) -> object:
    """Legacy adapter: index a plain transcript string and return a RAG chain.

    Preserved for backward compatibility with app.py and main.py.
    Documents indexed via this path carry empty provenance metadata.

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
    logger.debug(
        "build_rag_chain (legacy): indexing transcript (%d chars).",
        len(transcript),
    )
    vector_store = build_vector_store(transcript)
    return _build_chain(vector_store)


def ask_question(rag_chain, question: str) -> str:
    """Invoke *rag_chain* with *question* and return the answer string."""
    logger.debug("RAG question: %s", question)
    answer = rag_chain.invoke(question)
    logger.debug("RAG answer length: %d chars", len(answer))
    return answer
