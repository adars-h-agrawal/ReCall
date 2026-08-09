"""
tests/test_foundation.py
─────────────────────────
Lightweight foundation tests for RECALL — AI Meeting Intelligence.

These tests do NOT:
  - Download any YouTube audio
  - Make real Mistral API calls
  - Make real Sarvam API calls
  - Load the Whisper model

They verify:
  1. Configuration loads correctly and exposes expected structure.
  2. Missing API keys produce safe, secret-free errors.
  3. Input validation catches bad sources before any processing.
  4. Temporary directory creation and cleanup work correctly.
  5. Transcript validation catches empty transcription output.
  6. All application modules import without error.
  7. __init__.py files exist for core/ and utils/ packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path when running from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 1. Configuration ──────────────────────────────────────────────────────────

class TestConfig:
    def test_get_config_returns_expected_structure(self):
        from core.config import get_config, AppConfig
        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_paths_are_absolute(self):
        from core.config import PATHS
        assert PATHS.project_root.is_absolute()
        assert PATHS.vector_db.is_absolute()
        assert PATHS.temp_dir.is_absolute()

    def test_vector_db_path_under_project_root(self):
        from core.config import PATHS
        assert str(PATHS.vector_db).startswith(str(PATHS.project_root))

    def test_llm_config_defaults(self):
        from core.config import LLMConfig
        cfg = LLMConfig()
        assert cfg.model_name == "mistral-small-latest"
        assert cfg.temperature_default == 0.3
        assert cfg.temperature_extraction == 0.2

    def test_whisper_config_default(self):
        from core.config import WhisperConfig
        cfg = WhisperConfig()
        assert cfg.model_name == "small"

    def test_sarvam_config_default(self):
        from core.config import SarvamConfig
        cfg = SarvamConfig()
        assert cfg.piece_seconds == 25


# ── 2. Secret validation — safe errors ───────────────────────────────────────

class TestSecretValidation:
    def test_missing_mistral_key_raises_environment_error(self):
        from core.config import get_mistral_api_key
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
            with pytest.raises(EnvironmentError) as exc_info:
                get_mistral_api_key()
        # Error must mention the key NAME, not any value.
        assert "MISTRAL_API_KEY" in str(exc_info.value)

    def test_missing_mistral_key_error_does_not_contain_secret(self):
        """The error message must never contain an actual secret value."""
        from core.config import get_mistral_api_key
        fake_key = "sk-super-secret-test-key-12345"
        with patch.dict(os.environ, {"MISTRAL_API_KEY": fake_key}, clear=False):
            # Key is present — confirm it works first.
            result = get_mistral_api_key()
            assert result == fake_key
        # Now confirm missing-key error does NOT expose any fake value.
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
            with pytest.raises(EnvironmentError) as exc_info:
                get_mistral_api_key()
        assert fake_key not in str(exc_info.value)

    def test_missing_sarvam_key_raises_environment_error(self):
        from core.config import get_sarvam_api_key
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=False):
            with pytest.raises(EnvironmentError) as exc_info:
                get_sarvam_api_key()
        assert "SARVAM_API_KEY" in str(exc_info.value)

    def test_validate_secrets_passes_when_mistral_set(self):
        from core.config import validate_secrets
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key-value"}, clear=False):
            # Should not raise
            validate_secrets(require_sarvam=False)

    def test_validate_secrets_requires_sarvam_when_flagged(self):
        from core.config import validate_secrets
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key", "SARVAM_API_KEY": ""}, clear=False):
            with pytest.raises(EnvironmentError):
                validate_secrets(require_sarvam=True)


# ── 3. Input validation ───────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_source_raises(self):
        from utils.audio_processor import validate_source
        with pytest.raises(ValueError, match="No source provided"):
            validate_source("")

    def test_whitespace_only_source_raises(self):
        from utils.audio_processor import validate_source
        with pytest.raises(ValueError, match="No source provided"):
            validate_source("   ")

    def test_nonexistent_file_raises(self):
        from utils.audio_processor import validate_source
        with pytest.raises(ValueError, match="File not found"):
            validate_source("/nonexistent/path/to/file.mp4")

    def test_unsupported_extension_raises(self):
        from utils.audio_processor import validate_source
        # Create a temp file with an unsupported extension to test the check.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                validate_source(tmp)
        finally:
            os.unlink(tmp)

    def test_valid_https_url_passes(self):
        from utils.audio_processor import validate_source
        # Should not raise for a well-formed URL.
        validate_source("https://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_valid_http_url_passes(self):
        from utils.audio_processor import validate_source
        validate_source("http://example.com/video.mp4")

    def test_existing_supported_file_passes(self):
        from utils.audio_processor import validate_source
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            validate_source(tmp)  # Should not raise
        finally:
            os.unlink(tmp)


# ── 4. Temp directory creation and cleanup ────────────────────────────────────

class TestTempDirectoryHandling:
    def test_make_temp_dir_creates_directory(self):
        from utils.audio_processor import _make_temp_dir, _cleanup_temp_dir
        tmp = _make_temp_dir()
        assert tmp.exists()
        assert tmp.is_dir()
        _cleanup_temp_dir(tmp)
        assert not tmp.exists()

    def test_cleanup_is_idempotent(self):
        from utils.audio_processor import _make_temp_dir, _cleanup_temp_dir
        tmp = _make_temp_dir()
        _cleanup_temp_dir(tmp)
        # Calling cleanup again on a missing directory must not raise.
        _cleanup_temp_dir(tmp)

    def test_cleanup_removes_contents(self):
        from utils.audio_processor import _make_temp_dir, _cleanup_temp_dir
        tmp = _make_temp_dir()
        (tmp / "dummy.wav").write_bytes(b"fake")
        assert (tmp / "dummy.wav").exists()
        _cleanup_temp_dir(tmp)
        assert not tmp.exists()


# ── 5. Transcript validation ──────────────────────────────────────────────────

class TestTranscriptValidation:
    def test_empty_chunks_raises_value_error(self):
        from unittest.mock import patch as _patch
        from core.transcriber import transcribe_all
        with pytest.raises(ValueError, match="No audio chunks"):
            transcribe_all([])

    def test_blank_transcript_raises_value_error(self):
        """If all chunks transcribe to whitespace, raise a clear error."""
        from core.transcriber import transcribe_all
        with patch("core.transcriber.transcribe_chunk", return_value="   "):
            with pytest.raises(ValueError, match="no text"):
                transcribe_all(["fake_chunk.wav"])


# ── 6. Module imports ─────────────────────────────────────────────────────────

class TestModuleImports:
    """Verify all application modules import without raising exceptions."""

    def test_import_core_config(self):
        import core.config  # noqa: F401

    def test_import_core_llm(self):
        import core.llm  # noqa: F401

    def test_import_core_transcriber(self):
        import core.transcriber  # noqa: F401

    def test_import_core_summarizer(self):
        import core.summarizer  # noqa: F401

    def test_import_core_extractor(self):
        import core.extractor  # noqa: F401

    def test_import_core_vector_store(self):
        import core.vector_store  # noqa: F401

    def test_import_core_rag_engine(self):
        import core.rag_engine  # noqa: F401

    def test_import_utils_audio_processor(self):
        import utils.audio_processor  # noqa: F401


# ── 7. Package structure ──────────────────────────────────────────────────────

class TestPackageStructure:
    def test_core_init_exists(self):
        assert (PROJECT_ROOT / "core" / "__init__.py").exists()

    def test_utils_init_exists(self):
        assert (PROJECT_ROOT / "utils" / "__init__.py").exists()

    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.example").exists()

    def test_env_example_has_no_real_keys(self):
        """Ensure .env.example contains only placeholder text, not real keys."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        # Placeholders should contain 'your_' prefix, not look like real keys.
        assert "your_mistral_api_key_here" in content
        assert "your_sarvam_api_key_here" in content

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_no_pyc_tracked_by_git(self):
        """No .pyc files should be tracked in the git index."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "*.pyc"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        # git ls-files --error-unmatch exits non-zero if no files match — that's what we want.
        assert result.returncode != 0, "Found .pyc files tracked by git"
