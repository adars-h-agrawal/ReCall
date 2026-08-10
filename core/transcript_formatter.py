"""
core/transcript_formatter.py
─────────────────────────────
Utilities for formatting and displaying TranscriptSegments.
"""

from __future__ import annotations

from core.models.transcript import TranscriptSegment, format_timestamp


def format_transcript_for_display(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments as structured HTML for display.

    Parameters
    ----------
    segments : list[TranscriptSegment]
        Ordered list of transcript segments.

    Returns
    -------
    str
        HTML string with timestamps and speaker info (if available).
    """
    lines = []
    for seg in segments:
        timestamp = format_timestamp(seg.start)
        line_parts = [f"<strong>[{timestamp}]</strong>"]

        # Add speaker if available
        if seg.speaker:
            line_parts.append(f"<em>{seg.speaker}:</em>")

        line_parts.append(seg.text)
        lines.append(" ".join(line_parts))

    return "<br/>".join(lines)


def format_transcript_for_plain_text(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments as plain text.

    Parameters
    ----------
    segments : list[TranscriptSegment]
        Ordered list of transcript segments.

    Returns
    -------
    str
        Plain text with timestamps and speaker info.
    """
    lines = []
    for seg in segments:
        timestamp = format_timestamp(seg.start)
        parts = [f"[{timestamp}]"]

        if seg.speaker:
            parts.append(f"{seg.speaker}:")

        parts.append(seg.text)
        lines.append(" ".join(parts))

    return "\n".join(lines)
