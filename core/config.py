"""
core/config.py
──────────────
Central configuration for RECALL — AI Meeting Intelligence.

Responsibilities
----------------
- Resolve all application paths relative to the project root so the app
  works regardless of the current working directory.
- Validate required environment variables at startup and raise a clear,
  secret-free error when they are absent.
- Expose a single PATHS object and a single get_config() call so every
  other module imports from one place instead of sprinkling os.getenv /
  hardcoded strings across the codebase.

Rules
-----
- Never log or print the value of any secret.
- Validation errors mention only that a key is *missing*, not its value.
- os.getenv calls for secrets happen here and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# This file lives at <project_root>/core/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ── Application paths ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AppPaths:
    """All filesystem paths the application uses, resolved to absolute paths."""

    project_root: Path = PROJECT_ROOT

    # Persistent storage
    vector_db: Path = field(default_factory=lambda: PROJECT_ROOT / "vector_db")

    # Temporary working directories (created on demand, cleaned up after use)
    temp_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "temp")


PATHS = AppPaths()


# ── Model / service configuration ────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    """Mistral LLM defaults.  Model name is overridable via env var."""

    model_name: str = field(
        default_factory=lambda: os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    )
    temperature_default: float = 0.3
    temperature_extraction: float = 0.2


@dataclass(frozen=True)
class WhisperConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "small")
    )


@dataclass(frozen=True)
class SarvamConfig:
    stt_translate_url: str = "https://api.sarvam.ai/speech-to-text-translate"
    model: str = field(
        default_factory=lambda: os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
    )
    piece_seconds: int = 25  # Sarvam sync API rejects audio > 30 s


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass(frozen=True)
class VectorStoreConfig:
    collection_name: str = "meeting_transcript"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4


@dataclass(frozen=True)
class AppConfig:
    paths: AppPaths = field(default_factory=AppPaths)
    llm: LLMConfig = field(default_factory=LLMConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    sarvam: SarvamConfig = field(default_factory=SarvamConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


def get_config() -> AppConfig:
    """Return the application configuration (paths + model settings).

    Does NOT validate secrets — call validate_secrets() separately when
    you need to confirm that API keys are present before making network
    calls.
    """
    return AppConfig()


# ── Secret accessors ──────────────────────────────────────────────────────────
# Each function returns the raw value for internal use only.
# Callers must never log or display the returned value.

def get_mistral_api_key() -> str:
    """Return the Mistral API key.

    Raises
    ------
    EnvironmentError
        If MISTRAL_API_KEY is not set.  The error message does not
        contain the key value.
    """
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return key


def get_sarvam_api_key() -> str:
    """Return the Sarvam API key.

    Raises
    ------
    EnvironmentError
        If SARVAM_API_KEY is not set.  The error message does not
        contain the key value.
    """
    key = os.getenv("SARVAM_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "SARVAM_API_KEY is not set. "
            "Add it to your .env file (see .env.example). "
            "It is required for Hindi/Hinglish transcription."
        )
    return key


def validate_secrets(require_sarvam: bool = False) -> None:
    """Eagerly validate that required API keys are present.

    Parameters
    ----------
    require_sarvam:
        Pass True when the Sarvam API will actually be called (i.e. the
        user has selected the 'hinglish' language).

    Raises
    ------
    EnvironmentError
        If a required key is missing.  The error message never contains
        the key value itself.
    """
    get_mistral_api_key()
    if require_sarvam:
        get_sarvam_api_key()
