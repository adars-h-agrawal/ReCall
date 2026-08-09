"""
core/models/transcript.py
──────────────────────────
Transcript data models for RECALL — AI Meeting Intelligence.

TranscriptSegment is the canonical unit of transcribed speech.
It preserves timing, language, and speaker information alongside text.

Timestamps
----------
All timestamps are stored as seconds (float) — the native unit from
Whisper and the most useful form for downstream processing.
Use format_timestamp() to convert to a human-readable "MM:SS" string.

Speaker
-------
Speaker diarization is NOT implemented.  The speaker field defaults to
None and must not be fabricated.  It will be populated in a future phase.

Language
--------
- Whisper segments carry the detected language code (e.g. "en").
- Sarvam translates Hindi/Hinglish to English.  The output language is
  "en" but the source language is recorded separately so the translation
  origin is never misrepresented.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


def format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to a human-readable 'MM:SS' string.

    Examples
    --------
    >>> format_timestamp(142.3)
    '02:22'
    >>> format_timestamp(0.0)
    '00:00'
    >>> format_timestamp(3661.0)
    '61:01'
    """
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


class TranscriptSegment(BaseModel):
    """A single timed segment of transcribed (or translated) speech.

    Fields
    ------
    id : str
        Unique identifier for this segment (UUID).
    meeting_id : str
        The UUID of the meeting this segment belongs to.
    text : str
        The transcribed (or translated) text for this segment.
    start : float
        Segment start time in seconds from the beginning of the audio.
    end : float
        Segment end time in seconds from the beginning of the audio.
    speaker : Optional[str]
        Speaker identifier.  None until diarization is implemented.
    language : str
        BCP-47 language tag of the *output* text (e.g. "en").
    source_language : Optional[str]
        Original spoken language if translation occurred (e.g. "hi" for
        Sarvam-translated Hindi).  None when no translation took place.
    chunk_index : int
        Which audio chunk this segment originated from (0-based).
    source : str
        The original media source — URL or file path.
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    meeting_id: str
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    language: str = "en"
    source_language: Optional[str] = None
    chunk_index: int = 0
    source: str = ""

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "TranscriptSegment":
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < 0:
            raise ValueError(f"end must be >= 0, got {self.end}")
        if self.end < self.start:
            raise ValueError(
                f"end ({self.end}) must be >= start ({self.start})"
            )
        return self

    @property
    def start_formatted(self) -> str:
        """Start time as 'MM:SS' string."""
        return format_timestamp(self.start)

    @property
    def end_formatted(self) -> str:
        """End time as 'MM:SS' string."""
        return format_timestamp(self.end)

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    """Concatenate segment texts into a single plain-text transcript string.

    This is the backward-compatibility bridge used by summarizer, extractor,
    and RAG engine until those layers are updated to consume segments directly.

    Parameters
    ----------
    segments : list[TranscriptSegment]
        Ordered list of transcript segments.

    Returns
    -------
    str
        A single space-joined string of all segment texts, stripped of
        leading/trailing whitespace.
    """
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
