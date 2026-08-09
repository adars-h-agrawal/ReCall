"""
core/rag_engine.py
───────────────────
Retrieval-Augmented Generation (RAG) chat over meeting transcripts
for RECALL — AI Meeting Intelligence.

build_rag_chain() indexes the transcript into ChromaDB and returns an
LCEL chain that can be used to answer questions via ask_question().

Known limitation (Phase 1):
  ChromaDB uses a single shared collection for all meetings.  Meeting
  isolation will be addressed in a future phase.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm_default
from core.vector_store import build_vector_store, get_retriever

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


def build_rag_chain(transcript: str):
    """Index *transcript* into ChromaDB and return a ready-to-use RAG chain.

    Parameters
    ----------
    transcript:
        The full meeting transcript as a plain string.

    Returns
    -------
    Runnable
        An LCEL chain that accepts a question string and returns an answer string.
    """
    logger.debug("Building RAG chain from transcript (%d chars).", len(transcript))

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm_default()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """Invoke *rag_chain* with *question* and return the answer string."""
    logger.debug("RAG question: %s", question)
    answer = rag_chain.invoke(question)
    logger.debug("RAG answer length: %d chars", len(answer))
    return answer
