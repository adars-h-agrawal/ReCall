"""
core/extractor.py
──────────────────
Structured extraction from meeting transcripts for RECALL — AI Meeting Intelligence.

Extracts three categories:
  - Action items  (task, owner, deadline)
  - Key decisions
  - Open / unresolved questions

All three functions return formatted strings (numbered lists).
Structured Pydantic output is planned for a future phase.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm_extraction

logger = logging.getLogger(__name__)


def _build_chain(system_prompt: str):
    """Return an LCEL chain that accepts a transcript string and returns a string."""
    llm = get_llm_extraction()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items from the transcript as a numbered list."""
    chain = _build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from the transcript as a numbered list."""
    chain = _build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions and follow-up topics as a numbered list."""
    chain = _build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
