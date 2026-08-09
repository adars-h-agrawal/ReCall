"""
core/summarizer.py
──────────────────
Meeting summarization and title generation for RECALL — AI Meeting Intelligence.

Uses a map-reduce approach for long transcripts:
  1. Split transcript into overlapping chunks.
  2. Summarize each chunk independently (map).
  3. Combine chunk summaries into one final summary (reduce).
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm_default

logger = logging.getLogger(__name__)


def split_transcript(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Produce a bullet-point meeting summary via map-reduce over the transcript."""
    llm = get_llm_default()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    logger.debug("Summarizing %d transcript chunk(s).", len(chunks))
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    """Generate a short professional meeting title (max 8 words)."""
    llm = get_llm_default()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    # Only the first 2000 characters are needed for a representative title.
    return title_chain.invoke(transcript[:2000])
