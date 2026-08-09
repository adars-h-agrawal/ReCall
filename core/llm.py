"""
core/llm.py
───────────
Shared Mistral LLM factory for RECALL — AI Meeting Intelligence.

All modules that need a ChatMistralAI instance must import from here.
This is the single place where:
  - the model name is read from configuration
  - the API key is retrieved (via core.config, never hardcoded)
  - temperature semantics are defined

Two temperatures are used across the pipeline:
  - DEFAULT (0.3)  — summarization, RAG chat
  - EXTRACTION (0.2) — action items, decisions, questions

Streamlit resource caching
--------------------------
get_llm_default() and get_llm_extraction() are decorated with
@st.cache_resource so Streamlit reuses the same client across reruns.
When imported outside Streamlit (CLI, tests) the decorator is a no-op
wrapper, so behaviour is identical.
"""

from __future__ import annotations

import logging

from langchain_mistralai import ChatMistralAI

from core.config import LLMConfig, get_mistral_api_key

logger = logging.getLogger(__name__)

# ── Resolve config once at import time ───────────────────────────────────────
_cfg = LLMConfig()


# ── Streamlit-safe cache decorator ───────────────────────────────────────────
def _cache_resource(fn):
    """Apply @st.cache_resource when running inside Streamlit; otherwise
    return the function unchanged so CLI / test usage is unaffected."""
    try:
        import streamlit as st
        return st.cache_resource(fn)
    except Exception:  # streamlit not available or not in a server context
        return fn


# ── LLM factories ─────────────────────────────────────────────────────────────

@_cache_resource
def get_llm_default() -> ChatMistralAI:
    """Return a cached ChatMistralAI instance at the default temperature (0.3).

    Used by: summarizer, rag_engine.
    """
    logger.debug("Creating ChatMistralAI (default temperature=%.1f)", _cfg.temperature_default)
    return ChatMistralAI(
        model=_cfg.model_name,
        mistral_api_key=get_mistral_api_key(),
        temperature=_cfg.temperature_default,
    )


@_cache_resource
def get_llm_extraction() -> ChatMistralAI:
    """Return a cached ChatMistralAI instance at extraction temperature (0.2).

    Used by: extractor.
    """
    logger.debug("Creating ChatMistralAI (extraction temperature=%.1f)", _cfg.temperature_extraction)
    return ChatMistralAI(
        model=_cfg.model_name,
        mistral_api_key=get_mistral_api_key(),
        temperature=_cfg.temperature_extraction,
    )
