"""
core/pipeline.py
─────────────────
Canonical meeting processing pipeline for RECALL — AI Meeting Intelligence.

build_meeting() is the single entry point that takes raw audio chunks
and returns a fully populated Meeting object.

Flow
----
chunks + metadata
    ↓
transcribe_all_structured()  →  list[TranscriptSegment]
    ↓
Meeting(id, source, source_type, language, segments, ...)
    ↓
generate_title()             →  meeting.title
    ↓
summarize()                  →  summary string (returned separately)
    ↓
extract_structured()         →  action_items, decisions, open_questions
    ↓
Meeting (fully populated)

The Meeting object is the canonical result.
The summary string is returned alongside it because it is not yet a
structured model — that is addressed in a future phase.

Backward compatibility
----------------------
app.py and main.py still call the individual extractor/summarizer
functions with plain strings.  They continue to work unchanged.
build_meeting() is the new canonical path used going forward.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from core.extractor import extract_structured
from core.models.meeting import Meeting, SourceType
from core.models.transcript import segments_to_text
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all_structured
from utils.audio_processor import _is_youtube_url

logger = logging.getLogger(__name__)


def _infer_source_type(source: str) -> SourceType:
    return SourceType.YOUTUBE if _is_youtube_url(source) else SourceType.LOCAL_FILE


def build_meeting(
    chunks: list[str],
    source: str,
    language: str = "english",
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Meeting, str]:
    """Build a fully populated Meeting from transcription-ready audio chunks.

    Parameters
    ----------
    chunks:
        Ordered WAV file paths from audio_processor.process_input().
    source:
        The original YouTube URL or local file path.
    language:
        'english' (Whisper) or 'hinglish' (Sarvam).
    on_progress:
        Optional callback called with a stage name string as each stage
        completes.  Useful for UI progress updates.
        Stage names: 'transcription', 'title', 'summary', 'extraction'.

    Returns
    -------
    tuple[Meeting, str]
        (meeting, summary) where meeting is fully populated and summary
        is the plain-text bullet-point summary string.

    Raises
    ------
    ValueError
        If transcription produces no usable text.
    RuntimeError
        If any LLM stage fails.
    """
    def _progress(stage: str) -> None:
        if on_progress:
            on_progress(stage)

    # ── Stage 1: create Meeting shell ─────────────────────────────────────
    meeting = Meeting(
        source=source,
        source_type=_infer_source_type(source),
        language=language,
    )
    logger.info("Building meeting %s from source: %s", meeting.id, source)

    # ── Stage 2: structured transcription ─────────────────────────────────
    segments = transcribe_all_structured(
        chunks=chunks,
        language=language,
        meeting_id=meeting.id,
        source=source,
    )
    meeting.segments = segments

    # Derive duration from the last segment's end time if available.
    if segments:
        meeting.duration = segments[-1].end

    _progress("transcription")
    logger.info(
        "Meeting %s: %d segments, duration %.1fs.",
        meeting.id, len(segments), meeting.duration or 0,
    )

    # Derive plain transcript for LLM stages (single source of truth).
    transcript = meeting.plain_transcript()

    # ── Stage 3: title generation ──────────────────────────────────────────
    meeting.title = generate_title(transcript)
    _progress("title")
    logger.debug("Meeting %s title: %s", meeting.id, meeting.title)

    # ── Stage 4: summarization ─────────────────────────────────────────────
    summary = summarize(transcript)
    _progress("summary")

    # ── Stage 5: structured extraction ────────────────────────────────────
    action_items, decisions, open_questions = extract_structured(
        transcript=transcript,
        meeting_id=meeting.id,
        segments=segments,
    )
    meeting.action_items = action_items
    meeting.decisions = decisions
    meeting.open_questions = open_questions
    _progress("extraction")

    logger.info(
        "Meeting %s complete: %d action items, %d decisions, %d questions.",
        meeting.id,
        len(meeting.action_items),
        len(meeting.decisions),
        len(meeting.open_questions),
    )
    return meeting, summary
