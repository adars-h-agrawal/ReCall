"""
core/transcriber.py
────────────────────
Speech-to-text transcription for RECALL — AI Meeting Intelligence.

Routes each audio chunk to the appropriate engine:
  - English  → local OpenAI Whisper model
  - Hinglish → Sarvam AI speech-to-text-translate API (outputs English)

Public API
----------
transcribe_all(chunks, language) -> str
    Legacy flat-string interface.  Preserved for backward compatibility.
    Used by summarizer, extractor, RAG engine, and existing tests.

transcribe_all_structured(chunks, language, meeting_id, source) -> list[TranscriptSegment]
    New structured interface.  Returns one TranscriptSegment per Whisper
    segment (or per Sarvam piece) with timing and language metadata.
    This is the canonical output going forward.

Timestamp strategy
------------------
Whisper:  result["segments"] provides precise start/end per segment.
          Each segment maps directly to one TranscriptSegment.

Sarvam:   The API returns one text blob per ≤25 s piece.  We do not have
          word-level timing.  We use the piece boundary as start/end:
            chunk_offset + piece_index * piece_seconds
          This is honest chunk-level timing — not fabricated precision.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
import whisper
from pydub import AudioSegment

from core.config import WhisperConfig, SarvamConfig, get_sarvam_api_key
from core.models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

_w_cfg = WhisperConfig()
_s_cfg = SarvamConfig()


# ── Streamlit-safe cache decorator ────────────────────────────────────────────
def _cache_resource(fn):
    try:
        import streamlit as st
        return st.cache_resource(fn)
    except Exception:
        return fn


# ── Whisper model (cached) ────────────────────────────────────────────────────

@_cache_resource
def _load_whisper_model() -> whisper.Whisper:
    """Load and return the Whisper model.  Cached across Streamlit reruns."""
    logger.info("Loading Whisper model '%s' …", _w_cfg.model_name)
    model = whisper.load_model(_w_cfg.model_name)
    logger.info("Whisper model loaded.")
    return model


# ── Whisper transcription ─────────────────────────────────────────────────────

def _transcribe_whisper_raw(chunk_path: str) -> dict:
    """Run Whisper on *chunk_path* and return the full result dict.

    Returns the raw Whisper result including 'text' and 'segments'.
    """
    model = _load_whisper_model()
    return model.transcribe(chunk_path, task="transcribe")


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe a single WAV chunk with the local Whisper model.
    Returns the flat transcript text only (legacy interface).
    """
    return _transcribe_whisper_raw(chunk_path)["text"]


def _whisper_segments_to_models(
    raw_segments: list[dict],
    chunk_offset: float,
    chunk_index: int,
    meeting_id: str,
    source: str,
    language: str,
) -> list[TranscriptSegment]:
    """Convert Whisper's raw segment dicts to TranscriptSegment models.

    Parameters
    ----------
    raw_segments:
        The 'segments' list from whisper.transcribe() result.
    chunk_offset:
        Time offset (seconds) of the start of this chunk within the
        full recording.  Added to each segment's start/end.
    chunk_index:
        Which audio chunk these segments come from.
    meeting_id, source, language:
        Metadata passed through to each segment.
    """
    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(TranscriptSegment(
            meeting_id=meeting_id,
            text=text,
            start=round(chunk_offset + seg["start"], 3),
            end=round(chunk_offset + seg["end"], 3),
            language=language,
            source_language=None,  # No translation for Whisper
            chunk_index=chunk_index,
            source=source,
        ))
    return segments


# ── Sarvam transcription ──────────────────────────────────────────────────────

def _send_to_sarvam(piece_path: str, api_key: str) -> str:
    """POST one ≤30 s WAV file to the Sarvam STT-translate endpoint."""
    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": _s_cfg.model, "with_diarization": "false"}
        response = requests.post(
            _s_cfg.stt_translate_url,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        logger.error("Sarvam API returned HTTP %d.", response.status_code)
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Transcribe a Hindi/Hinglish WAV chunk via the Sarvam API.
    Returns flat text (legacy interface).
    """
    api_key = get_sarvam_api_key()
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = _s_cfg.piece_seconds * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    full_text = ""
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            logger.debug("Sarvam piece %d/%d …", i + 1, total_pieces)
            full_text += _send_to_sarvam(piece_path, api_key) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def _transcribe_sarvam_structured(
    chunk_path: str,
    chunk_offset: float,
    chunk_index: int,
    meeting_id: str,
    source: str,
) -> list[TranscriptSegment]:
    """Transcribe a Hinglish chunk via Sarvam, returning structured segments.

    Timing: Sarvam returns one text blob per ≤25 s piece.
    We assign start/end based on the piece boundary within the chunk,
    plus the chunk offset within the full recording.
    This is honest chunk-level timing — not fabricated word precision.
    """
    api_key = get_sarvam_api_key()
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = _s_cfg.piece_seconds * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    chunk_duration_s = len(audio) / 1000.0

    segments: list[TranscriptSegment] = []
    for i, start_ms in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start_ms: start_ms + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            logger.debug("Sarvam piece %d/%d …", i + 1, total_pieces)
            text = _send_to_sarvam(piece_path, api_key).strip()
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

        if not text:
            continue

        piece_start_s = start_ms / 1000.0
        piece_end_s = min(piece_start_s + _s_cfg.piece_seconds, chunk_duration_s)

        segments.append(TranscriptSegment(
            meeting_id=meeting_id,
            text=text,
            start=round(chunk_offset + piece_start_s, 3),
            end=round(chunk_offset + piece_end_s, 3),
            language="en",           # Sarvam outputs English
            source_language="hi",    # Source was Hindi/Hinglish
            chunk_index=chunk_index,
            source=source,
        ))

    return segments


# ── Routing ───────────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Route one chunk to Whisper (english) or Sarvam (hinglish).
    Returns flat text (legacy interface).
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


# ── Legacy flat-string interface (backward-compatible) ───────────────────────

def transcribe_all(chunks: list[str], language: str = "english") -> str:
    """Transcribe all chunks and return the joined transcript string.

    This is the legacy interface used by app.py, main.py, summarizer,
    extractor, and RAG engine.  It is preserved unchanged.

    Parameters
    ----------
    chunks:
        Ordered list of WAV file paths produced by audio_processor.
    language:
        'english' uses Whisper; 'hinglish' uses Sarvam AI.

    Returns
    -------
    str
        The full transcript as a single plain-text string.

    Raises
    ------
    ValueError
        If *chunks* is empty or the resulting transcript is blank.
    """
    if not chunks:
        raise ValueError("No audio chunks provided for transcription.")

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    logger.info("Transcribing %d chunk(s) with %s.", len(chunks), engine)

    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        logger.debug("Transcribing chunk %d/%d …", i + 1, len(chunks))
        text = transcribe_chunk(chunk, language=language)
        parts.append(text)

    transcript = " ".join(parts).strip()

    if not transcript:
        raise ValueError(
            "Transcription produced no text. "
            "The audio may be silent, corrupted, or in an unsupported format."
        )

    logger.info("Transcription complete (%d characters).", len(transcript))
    return transcript


# ── Structured interface (new canonical output) ───────────────────────────────

def transcribe_all_structured(
    chunks: list[str],
    language: str = "english",
    meeting_id: str = "",
    source: str = "",
) -> list[TranscriptSegment]:
    """Transcribe all chunks and return a list of TranscriptSegment objects.

    This is the new canonical interface.  It preserves all timing and
    language metadata available from the transcription engine.

    Parameters
    ----------
    chunks:
        Ordered list of WAV file paths produced by audio_processor.
    language:
        'english' uses Whisper (with per-segment timestamps).
        'hinglish' uses Sarvam AI (with piece-level timestamps).
    meeting_id:
        UUID of the meeting these segments belong to.
    source:
        Original media source (URL or file path) for provenance.

    Returns
    -------
    list[TranscriptSegment]
        Ordered segments covering the full recording.

    Raises
    ------
    ValueError
        If *chunks* is empty or the resulting transcript is blank.
    """
    if not chunks:
        raise ValueError("No audio chunks provided for transcription.")

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    logger.info("Structured transcription: %d chunk(s) with %s.", len(chunks), engine)

    all_segments: list[TranscriptSegment] = []
    # Track the time offset so segments from chunk N start after chunk N-1 ends.
    chunk_offset: float = 0.0

    for i, chunk_path in enumerate(chunks):
        logger.debug("Transcribing chunk %d/%d …", i + 1, len(chunks))

        if language.lower() == "hinglish":
            segs = _transcribe_sarvam_structured(
                chunk_path=chunk_path,
                chunk_offset=chunk_offset,
                chunk_index=i,
                meeting_id=meeting_id,
                source=source,
            )
            # Advance offset by the actual audio duration of this chunk.
            chunk_duration = len(AudioSegment.from_wav(chunk_path)) / 1000.0
            chunk_offset += chunk_duration
        else:
            raw = _transcribe_whisper_raw(chunk_path)
            segs = _whisper_segments_to_models(
                raw_segments=raw.get("segments", []),
                chunk_offset=chunk_offset,
                chunk_index=i,
                meeting_id=meeting_id,
                source=source,
                language=raw.get("language", "en"),
            )
            # Advance offset by the last segment's end time if available,
            # otherwise by the raw text length approximation from the chunk.
            if segs:
                # Use the end time of the last segment (relative to chunk start)
                # to compute the true chunk duration.
                last_raw_end = raw["segments"][-1]["end"] if raw.get("segments") else 0.0
                chunk_offset += last_raw_end
            else:
                # Fallback: derive duration from audio file.
                chunk_duration = len(AudioSegment.from_wav(chunk_path)) / 1000.0
                chunk_offset += chunk_duration

        all_segments.extend(segs)

    if not all_segments:
        raise ValueError(
            "Transcription produced no text. "
            "The audio may be silent, corrupted, or in an unsupported format."
        )

    logger.info(
        "Structured transcription complete: %d segment(s), %.1f s total.",
        len(all_segments),
        all_segments[-1].end if all_segments else 0.0,
    )
    return all_segments
