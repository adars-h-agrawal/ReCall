"""
core/transcriber.py
────────────────────
Speech-to-text transcription for RECALL — AI Meeting Intelligence.

Routes each audio chunk to the appropriate engine:
  - English  → local OpenAI Whisper model
  - Hinglish → Sarvam AI speech-to-text-translate API (outputs English)

The Whisper model is loaded once and cached (via Streamlit cache_resource
when running inside Streamlit, or as a module-level singleton otherwise).

Sarvam note: the sync API rejects audio longer than 30 s.  Each 10-minute
chunk is therefore sub-divided into 25-second pieces before sending.
"""

from __future__ import annotations

import logging
import os

import requests
import whisper
from pydub import AudioSegment

from core.config import WhisperConfig, SarvamConfig, get_sarvam_api_key

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

def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe a single WAV chunk with the local Whisper model."""
    model = _load_whisper_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


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

    Splits the chunk into ≤25 s pieces (Sarvam's 30 s limit minus margin),
    sends each piece, and joins the results.
    """
    # Validate key presence before doing any audio work.
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


# ── Routing ───────────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Route one chunk to Whisper (english) or Sarvam (hinglish)."""
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    """Transcribe all chunks and return the joined transcript string.

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
