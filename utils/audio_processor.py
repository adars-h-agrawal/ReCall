"""
utils/audio_processor.py
─────────────────────────
Audio acquisition and preparation for RECALL — AI Meeting Intelligence.

Responsibilities
----------------
- Download audio from YouTube URLs via yt-dlp.
- Convert any supported audio/video format to 16 kHz mono WAV via pydub.
- Split WAV files into fixed-length chunks for transcription.
- Manage temporary working directories and clean them up after use.

All temporary files are placed under a session-scoped temp directory
(inside <project_root>/temp/).  The caller receives a cleanup function
alongside the chunk list; callers SHOULD invoke it when they are done
with the chunks (even on error).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import yt_dlp
from pydub import AudioSegment

from core.config import PATHS

logger = logging.getLogger(__name__)

# Supported local file extensions (ffmpeg/pydub handles many more, but we
# validate against this list to catch obvious mistakes early).
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".flac", ".aac",
    ".mkv", ".mov", ".avi",
}


# ── Validation helpers ────────────────────────────────────────────────────────

def _is_youtube_url(source: str) -> bool:
    """Return True if *source* looks like a YouTube (or yt-dlp-compatible) URL."""
    return source.startswith("http://") or source.startswith("https://")


def validate_source(source: str) -> None:
    """Raise ValueError with a safe message if *source* is not usable.

    Does NOT make any network requests.
    """
    source = source.strip()
    if not source:
        raise ValueError("No source provided. Enter a YouTube URL or a local file path.")

    if _is_youtube_url(source):
        # Basic structural check — yt-dlp will do the real validation.
        if len(source) < 12 or " " in source:
            raise ValueError(
                "The URL does not look valid. "
                "Please provide a full YouTube URL (e.g. https://youtube.com/watch?v=...)."
            )
        return

    # Local file checks
    path = Path(source)
    if not path.exists():
        raise ValueError(f"File not found: {source}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {source}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


# ── Temp-directory management ─────────────────────────────────────────────────

def _make_temp_dir() -> Path:
    """Create and return a unique temp directory under PATHS.temp_dir."""
    PATHS.temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=PATHS.temp_dir))
    logger.debug("Created temp directory: %s", tmp)
    return tmp


def _cleanup_temp_dir(tmp_dir: Path) -> None:
    """Remove *tmp_dir* and all its contents silently."""
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            logger.debug("Removed temp directory: %s", tmp_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not remove temp directory %s: %s", tmp_dir, exc)


# ── YouTube download ──────────────────────────────────────────────────────────

def download_youtube_audio(url: str, tmp_dir: Path) -> str:
    """Download the best audio stream from *url* and return the WAV path.

    The output file is written inside *tmp_dir*.

    Parameters
    ----------
    url:
        A yt-dlp-compatible URL.
    tmp_dir:
        Temporary directory that owns the output file.

    Returns
    -------
    str
        Absolute path to the downloaded WAV file.
    """
    output_template = str(tmp_dir / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp renames the file after post-processing; find the actual WAV.
        expected = Path(ydl.prepare_filename(info)).with_suffix(".wav")

    if not expected.exists():
        # Fallback: find the first WAV in the temp dir
        wav_files = list(tmp_dir.glob("*.wav"))
        if not wav_files:
            raise RuntimeError(
                "Audio download completed but no WAV file was found in the temp directory."
            )
        expected = wav_files[0]
        logger.debug("Used fallback WAV detection: %s", expected)

    logger.info("Downloaded audio: %s", expected.name)
    return str(expected)


# ── Local file → WAV ──────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, tmp_dir: Path) -> str:
    """Convert any audio/video file to 16 kHz mono WAV.

    The converted file is written inside *tmp_dir* so the original is
    never modified.

    Parameters
    ----------
    input_path:
        Path to the source audio/video file.
    tmp_dir:
        Temporary directory that owns the output file.

    Returns
    -------
    str
        Absolute path to the converted WAV file.
    """
    stem = Path(input_path).stem
    output_path = tmp_dir / f"{stem}_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16 kHz mono for Whisper
    audio.export(str(output_path), format="wav")

    logger.info("Converted to WAV: %s", output_path.name)
    return str(output_path)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """Split a WAV file into fixed-length chunks.

    Chunks are written into the same directory as *wav_path* (which is
    already inside a temp directory managed by the caller).

    Parameters
    ----------
    wav_path:
        Path to the source WAV file.
    chunk_minutes:
        Maximum length of each chunk in minutes.

    Returns
    -------
    list[str]
        Ordered list of paths to the chunk WAV files.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks: list[str] = []

    base = Path(wav_path)
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = base.parent / f"{base.stem}_chunk_{i}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))

    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

def process_input(source: str) -> tuple[list[str], Callable[[], None]]:
    """Process a YouTube URL or local file into transcription-ready WAV chunks.

    Creates a private temp directory, performs all conversion/chunking
    inside it, and returns both the chunk list and a cleanup callable.

    The caller is responsible for calling ``cleanup()`` when the chunks
    are no longer needed — ideally in a ``finally`` block.

    Parameters
    ----------
    source:
        YouTube URL or absolute/relative path to a local media file.

    Returns
    -------
    tuple[list[str], Callable[[], None]]
        ``(chunks, cleanup)`` where ``chunks`` is a non-empty list of WAV
        file paths and ``cleanup`` removes the temp directory.

    Raises
    ------
    ValueError
        If *source* fails basic validation before any processing begins.
    RuntimeError
        If audio download or conversion fails.
    """
    validate_source(source)

    tmp_dir = _make_temp_dir()
    cleanup: Callable[[], None] = lambda: _cleanup_temp_dir(tmp_dir)

    try:
        if _is_youtube_url(source):
            logger.info("Detected YouTube URL — downloading audio…")
            wav_path = download_youtube_audio(source, tmp_dir)
        else:
            logger.info("Detected local file — converting to WAV…")
            wav_path = convert_to_wav(source, tmp_dir)

        chunks = chunk_audio(wav_path)

        if not chunks:
            raise RuntimeError("Audio chunking produced no output chunks.")

        logger.info("Audio ready — %d chunk(s) created.", len(chunks))
        return chunks, cleanup

    except Exception:
        # If anything goes wrong before the caller can take ownership,
        # clean up immediately so temp files don't accumulate.
        cleanup()
        raise
