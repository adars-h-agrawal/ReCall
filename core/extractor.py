"""
core/extractor.py
──────────────────
Structured intelligence extraction for RECALL — AI Meeting Intelligence.

Public API — structured (canonical, Phase 2B+)
-----------------------------------------------
extract_structured(transcript, meeting_id, segments)
    → tuple[list[ActionItem], list[Decision], list[OpenQuestion]]
    Uses Mistral with_structured_output to produce validated Pydantic objects.
    Evidence-links extracted items back to TranscriptSegments by text match.

Public API — string adapters (backward-compatible, used by app.py/main.py)
---------------------------------------------------------------------------
extract_action_items(transcript)   → str   (formatted numbered list)
extract_key_decisions(transcript)  → str
extract_questions(transcript)      → str

These call the structured extraction and format the results as strings so
the existing UI continues to work without modification.

Transcript length guard
-----------------------
Mistral context is finite.  Transcripts over MAX_EXTRACTION_CHARS are
chunked before extraction and results are merged + deduplicated.
The threshold is conservative (12 000 chars ≈ 3 000 tokens).
Deduplication is exact case-insensitive text matching — items with
identical text (after lowercasing) from overlapping chunks are dropped.

Evidence linking
----------------
Timestamps are NOT produced by the LLM.
After extraction, we search each item's text against the transcript
segments using a simple substring / fuzzy match to find the best candidate
segment.  This is deterministic — the LLM cannot hallucinate timestamps.
If no match is found, timestamp_ref stays None.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from core.llm import get_llm_extraction
from core.models.intelligence import ActionItem, Decision, OpenQuestion
from core.models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Characters per extraction call.  ~12 k chars ≈ 3 k tokens — well within
# Mistral's window while leaving room for the system prompt and JSON output.
MAX_EXTRACTION_CHARS = 12_000
CHUNK_OVERLAP_CHARS = 300


# ── Intermediate LLM schemas ──────────────────────────────────────────────────
# These schemas are what the LLM fills in.  They deliberately omit meeting_id
# and id (which we assign deterministically) and timestamp_ref (which we
# derive via text matching, never from the LLM).

class _LLMActionItem(BaseModel):
    task: str = Field(description="Description of the task or commitment.")
    owner: Optional[str] = Field(
        default=None,
        description=(
            "Name of the person or team responsible for the task, "
            "exactly as mentioned in the transcript. "
            "If not explicitly identified, return null."
        ),
    )
    deadline: Optional[str] = Field(
        default=None,
        description=(
            "Deadline as stated in the transcript (e.g. 'next Friday', 'by EOD'). "
            "If not mentioned, return null."
        ),
    )


class _LLMActionItemList(BaseModel):
    items: list[_LLMActionItem] = Field(
        default_factory=list,
        description="All action items found. Empty list if none.",
    )


class _LLMDecision(BaseModel):
    decision: str = Field(
        description=(
            "A firm decision that was made and agreed upon. "
            "Do NOT include suggestions, possibilities, or unresolved topics."
        ),
    )


class _LLMDecisionList(BaseModel):
    items: list[_LLMDecision] = Field(
        default_factory=list,
        description="All confirmed decisions. Empty list if none.",
    )


class _LLMOpenQuestion(BaseModel):
    question: str = Field(
        description=(
            "A question or topic that remains unresolved or needs follow-up. "
            "Do NOT include questions that were answered during the meeting."
        ),
    )


class _LLMOpenQuestionList(BaseModel):
    items: list[_LLMOpenQuestion] = Field(
        default_factory=list,
        description="All unresolved questions. Empty list if none.",
    )


# ── Evidence linking ──────────────────────────────────────────────────────────

def _find_timestamp_ref(
    text: str,
    segments: list[TranscriptSegment],
) -> Optional[float]:
    """Return the start time of the segment that best matches *text*.

    Strategy: case-insensitive substring search against each segment.
    If no single segment contains the phrase, fall back to the segment
    whose text has the longest common token overlap with *text*.
    Returns None if *segments* is empty or no reasonable match is found.

    This is deterministic — no LLM involved.
    """
    if not segments or not text.strip():
        return None

    needle = text.strip().lower()

    # Pass 1: exact substring match
    for seg in segments:
        if needle in seg.text.lower():
            return seg.start

    # Pass 2: token overlap (words in common)
    needle_tokens = set(needle.split())
    if not needle_tokens:
        return None

    best_seg = None
    best_score = 0
    for seg in segments:
        seg_tokens = set(seg.text.lower().split())
        score = len(needle_tokens & seg_tokens)
        if score > best_score:
            best_score = score
            best_seg = seg

    # Require at least 3 tokens in common to avoid spurious matches.
    if best_seg is not None and best_score >= 3:
        return best_seg.start

    return None


# ── Transcript chunking for large transcripts ─────────────────────────────────

def _split_for_extraction(transcript: str) -> list[str]:
    """Split *transcript* into chunks suitable for a single extraction call."""
    if len(transcript) <= MAX_EXTRACTION_CHARS:
        return [transcript]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_EXTRACTION_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(transcript)


def _dedup_action_items(items: list[ActionItem]) -> list[ActionItem]:
    """Remove exact-duplicate action items (same task text, case-insensitive)."""
    seen: set[str] = set()
    result: list[ActionItem] = []
    for item in items:
        key = item.task.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedup_decisions(items: list[Decision]) -> list[Decision]:
    seen: set[str] = set()
    result: list[Decision] = []
    for item in items:
        key = item.decision.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedup_questions(items: list[OpenQuestion]) -> list[OpenQuestion]:
    seen: set[str] = set()
    result: list[OpenQuestion] = []
    for item in items:
        key = item.question.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ── Core extraction per chunk ─────────────────────────────────────────────────

_ACTION_SYSTEM = (
    "You are a precise meeting analyst. Extract ONLY explicit action items — "
    "tasks or commitments that someone agreed to do.\n"
    "Rules:\n"
    "- owner: only set if the transcript explicitly names who is responsible. "
    "Do NOT invent or guess an owner. Return null if not stated.\n"
    "- deadline: only set if the transcript explicitly states a deadline. "
    "Return null if not stated.\n"
    "- Do not include suggestions, ideas, or things people might do.\n"
    "Transcript portion:\n{text}"
)

_DECISION_SYSTEM = (
    "You are a precise meeting analyst. Extract ONLY confirmed decisions — "
    "things that were agreed upon or resolved.\n"
    "Rules:\n"
    "- Do NOT include suggestions, possibilities, ideas, or questions.\n"
    "- A decision must be something that was explicitly agreed or concluded.\n"
    "Transcript portion:\n{text}"
)

_QUESTION_SYSTEM = (
    "You are a precise meeting analyst. Extract ONLY unresolved questions "
    "or topics that require follow-up after the meeting.\n"
    "Rules:\n"
    "- Do NOT include questions that were answered during the meeting.\n"
    "- Do NOT include rhetorical questions.\n"
    "- Only include items that genuinely need further action or clarification.\n"
    "Transcript portion:\n{text}"
)


def _extract_chunk_action_items(chunk: str) -> list[_LLMActionItem]:
    llm = get_llm_extraction()
    structured = llm.with_structured_output(_LLMActionItemList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _ACTION_SYSTEM),
        ("human", "Extract action items from the transcript above."),
    ])
    chain = prompt | structured
    try:
        result: _LLMActionItemList = chain.invoke({"text": chunk})
        return result.items
    except Exception as exc:
        logger.error("Action item extraction failed for chunk: %s", exc)
        raise


def _extract_chunk_decisions(chunk: str) -> list[_LLMDecision]:
    llm = get_llm_extraction()
    structured = llm.with_structured_output(_LLMDecisionList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _DECISION_SYSTEM),
        ("human", "Extract confirmed decisions from the transcript above."),
    ])
    chain = prompt | structured
    try:
        result: _LLMDecisionList = chain.invoke({"text": chunk})
        return result.items
    except Exception as exc:
        logger.error("Decision extraction failed for chunk: %s", exc)
        raise


def _extract_chunk_questions(chunk: str) -> list[_LLMOpenQuestion]:
    llm = get_llm_extraction()
    structured = llm.with_structured_output(_LLMOpenQuestionList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _QUESTION_SYSTEM),
        ("human", "Extract unresolved questions from the transcript above."),
    ])
    chain = prompt | structured
    try:
        result: _LLMOpenQuestionList = chain.invoke({"text": chunk})
        return result.items
    except Exception as exc:
        logger.error("Open question extraction failed for chunk: %s", exc)
        raise


# ── Canonical structured extraction ──────────────────────────────────────────

def extract_structured(
    transcript: str,
    meeting_id: str,
    segments: Optional[list[TranscriptSegment]] = None,
) -> tuple[list[ActionItem], list[Decision], list[OpenQuestion]]:
    """Extract structured intelligence from *transcript*.

    This is the canonical extraction function.  All extraction logic
    flows through here — there is no duplicate implementation.

    Parameters
    ----------
    transcript:
        Full meeting transcript as a plain string.
    meeting_id:
        UUID of the meeting, stamped on every extracted object.
    segments:
        Optional list of TranscriptSegments used for evidence linking.
        If provided, timestamp_ref is populated via text matching.
        If None or empty, timestamp_ref stays None.

    Returns
    -------
    tuple[list[ActionItem], list[Decision], list[OpenQuestion]]
        Validated Pydantic objects ready to attach to Meeting.

    Raises
    ------
    RuntimeError
        If the LLM call fails.  The error message is safe (no API keys).
    """
    segs = segments or []
    chunks = _split_for_extraction(transcript)
    logger.info(
        "Extracting intelligence from %d chunk(s) (transcript %d chars).",
        len(chunks), len(transcript),
    )

    raw_actions: list[_LLMActionItem] = []
    raw_decisions: list[_LLMDecision] = []
    raw_questions: list[_LLMOpenQuestion] = []

    for i, chunk in enumerate(chunks):
        logger.debug("Extraction chunk %d/%d …", i + 1, len(chunks))
        try:
            raw_actions.extend(_extract_chunk_action_items(chunk))
            raw_decisions.extend(_extract_chunk_decisions(chunk))
            raw_questions.extend(_extract_chunk_questions(chunk))
        except Exception as exc:
            raise RuntimeError(
                f"Intelligence extraction failed on chunk {i + 1}/{len(chunks)}. "
                "Please try again."
            ) from exc

    # Promote to full models with meeting_id and evidence linking.
    action_items = _dedup_action_items([
        ActionItem(
            meeting_id=meeting_id,
            task=a.task,
            owner=a.owner,
            deadline=a.deadline,
            timestamp_ref=_find_timestamp_ref(a.task, segs),
        )
        for a in raw_actions
    ])

    decisions = _dedup_decisions([
        Decision(
            meeting_id=meeting_id,
            decision=d.decision,
            timestamp_ref=_find_timestamp_ref(d.decision, segs),
        )
        for d in raw_decisions
    ])

    open_questions = _dedup_questions([
        OpenQuestion(
            meeting_id=meeting_id,
            question=q.question,
            timestamp_ref=_find_timestamp_ref(q.question, segs),
        )
        for q in raw_questions
    ])

    logger.info(
        "Extraction complete: %d action items, %d decisions, %d questions.",
        len(action_items), len(decisions), len(open_questions),
    )
    return action_items, decisions, open_questions


# ── String formatting helpers ─────────────────────────────────────────────────

def _format_action_items(items: list[ActionItem]) -> str:
    if not items:
        return "No action items found."
    lines = []
    for i, item in enumerate(items, 1):
        owner_str = f" — Owner: {item.owner}" if item.owner else ""
        deadline_str = f" — Deadline: {item.deadline}" if item.deadline else ""
        lines.append(f"{i}. {item.task}{owner_str}{deadline_str}")
    return "\n".join(lines)


def _format_decisions(items: list[Decision]) -> str:
    if not items:
        return "No key decisions found."
    return "\n".join(f"{i}. {d.decision}" for i, d in enumerate(items, 1))


def _format_questions(items: list[OpenQuestion]) -> str:
    if not items:
        return "No open questions found."
    return "\n".join(f"{i}. {q.question}" for i, q in enumerate(items, 1))


# ── Backward-compatible string API (used by app.py / main.py) ─────────────────
# These are thin adapters over extract_structured so there is exactly one
# extraction implementation.  The meeting_id is a placeholder here because
# these functions are called outside the Meeting context.  app.py and main.py
# will be migrated in a future phase to use extract_structured() directly.

_PLACEHOLDER_MEETING_ID = "legacy"


def extract_action_items(transcript: str) -> str:
    """Extract action items and return as a formatted numbered list string."""
    try:
        items, _, _ = extract_structured(transcript, _PLACEHOLDER_MEETING_ID)
        return _format_action_items(items)
    except Exception as exc:
        logger.error("extract_action_items failed: %s", exc)
        raise


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions and return as a formatted numbered list string."""
    try:
        _, decisions, _ = extract_structured(transcript, _PLACEHOLDER_MEETING_ID)
        return _format_decisions(decisions)
    except Exception as exc:
        logger.error("extract_key_decisions failed: %s", exc)
        raise


def extract_questions(transcript: str) -> str:
    """Extract open questions and return as a formatted numbered list string."""
    try:
        _, _, questions = extract_structured(transcript, _PLACEHOLDER_MEETING_ID)
        return _format_questions(questions)
    except Exception as exc:
        logger.error("extract_questions failed: %s", exc)
        raise
