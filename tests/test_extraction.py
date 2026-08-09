"""
tests/test_extraction.py
─────────────────────────
Unit tests for Phase 2B structured extraction — RECALL AI Meeting Intelligence.

These tests do NOT call real Mistral, Sarvam, Whisper, YouTube, or ChromaDB.
All LLM calls are mocked.

Coverage:
  - _find_timestamp_ref: evidence linking
  - _split_for_extraction: transcript length guard
  - _dedup_*: deduplication helpers
  - _format_*: string formatting adapters
  - extract_structured: full extraction with mocked LLM
      - action items: task, optional owner, optional deadline
      - decisions: only confirmed decisions
      - open questions: only unresolved questions
      - meeting_id stamped on all results
  - extract_structured: large transcript triggers chunking
  - extract_structured: LLM failure raises RuntimeError (no API key exposed)
  - backward-compat string functions: extract_action_items, extract_key_decisions, extract_questions
  - pipeline.build_meeting: Meeting is populated correctly (mocked transcription + LLM)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.intelligence import ActionItem, Decision, OpenQuestion
from core.models.transcript import TranscriptSegment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_seg(text: str, start: float, end: float, meeting_id: str = "m") -> TranscriptSegment:
    return TranscriptSegment(meeting_id=meeting_id, text=text, start=start, end=end)


def _make_action_item_llm_result(items):
    """Build a mock _LLMActionItemList."""
    from core.extractor import _LLMActionItemList, _LLMActionItem
    return _LLMActionItemList(items=[_LLMActionItem(**i) for i in items])


def _make_decision_llm_result(items):
    from core.extractor import _LLMDecisionList, _LLMDecision
    return _LLMDecisionList(items=[_LLMDecision(**i) for i in items])


def _make_question_llm_result(items):
    from core.extractor import _LLMOpenQuestionList, _LLMOpenQuestion
    return _LLMOpenQuestionList(items=[_LLMOpenQuestion(**i) for i in items])


# ── Evidence linking ──────────────────────────────────────────────────────────

class TestFindTimestampRef:
    def test_exact_substring_match(self):
        from core.extractor import _find_timestamp_ref
        segs = [
            _make_seg("Alice will update the documentation by Friday.", 10.0, 15.0),
            _make_seg("Bob will send the report.", 20.0, 25.0),
        ]
        result = _find_timestamp_ref("update the documentation", segs)
        assert result == 10.0

    def test_second_segment_matched(self):
        from core.extractor import _find_timestamp_ref
        segs = [
            _make_seg("We discussed the roadmap.", 0.0, 5.0),
            _make_seg("Bob will send the report.", 20.0, 25.0),
        ]
        result = _find_timestamp_ref("send the report", segs)
        assert result == 20.0

    def test_case_insensitive(self):
        from core.extractor import _find_timestamp_ref
        segs = [_make_seg("ALICE WILL UPDATE THE DOCS.", 5.0, 10.0)]
        result = _find_timestamp_ref("alice will update the docs", segs)
        assert result == 5.0

    def test_token_overlap_fallback(self):
        from core.extractor import _find_timestamp_ref
        segs = [
            _make_seg("We need to schedule a budget review meeting soon.", 30.0, 35.0),
            _make_seg("The launch date is confirmed.", 40.0, 45.0),
        ]
        # "schedule a budget review meeting" has 5 tokens overlapping with first segment
        result = _find_timestamp_ref("schedule a budget review meeting before end of quarter", segs)
        assert result == 30.0

    def test_no_match_returns_none(self):
        from core.extractor import _find_timestamp_ref
        segs = [_make_seg("Hello world.", 0.0, 1.0)]
        result = _find_timestamp_ref("completely unrelated xyz abc", segs)
        assert result is None

    def test_empty_segments_returns_none(self):
        from core.extractor import _find_timestamp_ref
        assert _find_timestamp_ref("anything", []) is None

    def test_empty_text_returns_none(self):
        from core.extractor import _find_timestamp_ref
        segs = [_make_seg("Some text.", 0.0, 1.0)]
        assert _find_timestamp_ref("", segs) is None

    def test_too_few_token_overlap_returns_none(self):
        from core.extractor import _find_timestamp_ref
        # Only 1 word in common — below the threshold of 3.
        segs = [_make_seg("The budget.", 0.0, 1.0)]
        result = _find_timestamp_ref("The completely different sentence here.", segs)
        # "The" is only 1 token — should not match.
        assert result is None


# ── Transcript chunking ───────────────────────────────────────────────────────

class TestSplitForExtraction:
    def test_short_transcript_returns_single_chunk(self):
        from core.extractor import _split_for_extraction, MAX_EXTRACTION_CHARS
        short = "Hello world. " * 10
        result = _split_for_extraction(short)
        assert len(result) == 1
        assert result[0] == short

    def test_long_transcript_splits(self):
        from core.extractor import _split_for_extraction, MAX_EXTRACTION_CHARS
        long_text = "Word " * (MAX_EXTRACTION_CHARS // 4)
        result = _split_for_extraction(long_text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= MAX_EXTRACTION_CHARS + 500  # small tolerance for splitter

    def test_chunks_cover_full_content(self):
        from core.extractor import _split_for_extraction, MAX_EXTRACTION_CHARS
        # Every word should appear in at least one chunk.
        words = [f"word{i}" for i in range(3000)]
        long_text = " ".join(words)
        chunks = _split_for_extraction(long_text)
        combined = " ".join(chunks)
        # Spot-check a few words.
        assert "word0" in combined
        assert "word2999" in combined


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_dedup_action_items_removes_duplicates(self):
        from core.extractor import _dedup_action_items
        items = [
            ActionItem(meeting_id="m", task="Update the docs"),
            ActionItem(meeting_id="m", task="Update the docs"),
            ActionItem(meeting_id="m", task="Send the report"),
        ]
        result = _dedup_action_items(items)
        assert len(result) == 2

    def test_dedup_case_insensitive(self):
        from core.extractor import _dedup_action_items
        items = [
            ActionItem(meeting_id="m", task="Update docs"),
            ActionItem(meeting_id="m", task="UPDATE DOCS"),
        ]
        result = _dedup_action_items(items)
        assert len(result) == 1

    def test_dedup_decisions(self):
        from core.extractor import _dedup_decisions
        items = [
            Decision(meeting_id="m", decision="We will use Python."),
            Decision(meeting_id="m", decision="We will use Python."),
        ]
        result = _dedup_decisions(items)
        assert len(result) == 1

    def test_dedup_questions(self):
        from core.extractor import _dedup_questions
        items = [
            OpenQuestion(meeting_id="m", question="Who owns the rollout?"),
            OpenQuestion(meeting_id="m", question="Who owns the rollout?"),
            OpenQuestion(meeting_id="m", question="When is the deadline?"),
        ]
        result = _dedup_questions(items)
        assert len(result) == 2


# ── String formatters ─────────────────────────────────────────────────────────

class TestStringFormatters:
    def test_format_action_items_empty(self):
        from core.extractor import _format_action_items
        assert _format_action_items([]) == "No action items found."

    def test_format_action_items_with_owner_and_deadline(self):
        from core.extractor import _format_action_items
        items = [ActionItem(meeting_id="m", task="Write report", owner="Alice", deadline="Friday")]
        result = _format_action_items(items)
        assert "Write report" in result
        assert "Alice" in result
        assert "Friday" in result

    def test_format_action_items_owner_absent(self):
        from core.extractor import _format_action_items
        items = [ActionItem(meeting_id="m", task="Write report")]
        result = _format_action_items(items)
        assert "Write report" in result
        assert "Owner" not in result

    def test_format_decisions_empty(self):
        from core.extractor import _format_decisions
        assert _format_decisions([]) == "No key decisions found."

    def test_format_decisions_with_items(self):
        from core.extractor import _format_decisions
        items = [Decision(meeting_id="m", decision="We will use Python.")]
        result = _format_decisions(items)
        assert "We will use Python." in result

    def test_format_questions_empty(self):
        from core.extractor import _format_questions
        assert _format_questions([]) == "No open questions found."

    def test_format_questions_with_items(self):
        from core.extractor import _format_questions
        items = [OpenQuestion(meeting_id="m", question="Who leads the rollout?")]
        result = _format_questions(items)
        assert "Who leads the rollout?" in result


# ── extract_structured (mocked LLM) ──────────────────────────────────────────

class TestExtractStructured:
    """All tests mock the LLM — no real API calls made."""

    def _mock_llm_responses(self, action_items, decisions, questions):
        """Return a context manager that patches _extract_chunk_* functions."""
        action_result = _make_action_item_llm_result(action_items)
        decision_result = _make_decision_llm_result(decisions)
        question_result = _make_question_llm_result(questions)

        patches = [
            patch("core.extractor._extract_chunk_action_items", return_value=action_result.items),
            patch("core.extractor._extract_chunk_decisions", return_value=decision_result.items),
            patch("core.extractor._extract_chunk_questions", return_value=question_result.items),
        ]
        return patches

    def _apply_patches(self, patches):
        for p in patches:
            p.start()
        return patches

    def _stop_patches(self, patches):
        for p in patches:
            p.stop()

    def test_action_item_task_extracted(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Write the report"}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Alice will write the report.", "meeting-1")
            assert len(items) == 1
            assert items[0].task == "Write the report"
        finally:
            self._stop_patches(patches)

    def test_action_item_owner_extracted_when_stated(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Prepare slides", "owner": "Alice"}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Alice will prepare slides.", "meeting-1")
            assert items[0].owner == "Alice"
        finally:
            self._stop_patches(patches)

    def test_action_item_owner_none_when_not_stated(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Update the docs", "owner": None}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Someone should update the docs.", "meeting-1")
            assert items[0].owner is None
        finally:
            self._stop_patches(patches)

    def test_action_item_deadline_none_when_not_stated(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Fix the bug", "owner": None, "deadline": None}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Fix the bug.", "meeting-1")
            assert items[0].deadline is None
        finally:
            self._stop_patches(patches)

    def test_action_item_deadline_extracted_when_stated(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Submit report", "deadline": "next Friday"}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Submit the report by next Friday.", "meeting-1")
            assert items[0].deadline == "next Friday"
        finally:
            self._stop_patches(patches)

    def test_meeting_id_stamped_on_action_items(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "Task A"}], decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("Do task A.", "my-meeting-uuid")
            assert items[0].meeting_id == "my-meeting-uuid"
        finally:
            self._stop_patches(patches)

    def test_decision_extracted(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[],
            decisions=[{"decision": "We will adopt Python 3.12."}],
            questions=[],
        ))
        try:
            _, decisions, _ = extract_structured("We decided to adopt Python 3.12.", "m")
            assert len(decisions) == 1
            assert "Python 3.12" in decisions[0].decision
        finally:
            self._stop_patches(patches)

    def test_meeting_id_stamped_on_decisions(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[],
            decisions=[{"decision": "Go with option A."}],
            questions=[],
        ))
        try:
            _, decisions, _ = extract_structured("text", "uuid-123")
            assert decisions[0].meeting_id == "uuid-123"
        finally:
            self._stop_patches(patches)

    def test_open_question_extracted(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[], decisions=[],
            questions=[{"question": "Who will own the deployment?"}],
        ))
        try:
            _, _, questions = extract_structured("text", "m")
            assert len(questions) == 1
            assert "deployment" in questions[0].question
        finally:
            self._stop_patches(patches)

    def test_meeting_id_stamped_on_questions(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[], decisions=[],
            questions=[{"question": "Who leads?"}],
        ))
        try:
            _, _, questions = extract_structured("text", "mtg-abc")
            assert questions[0].meeting_id == "mtg-abc"
        finally:
            self._stop_patches(patches)

    def test_empty_results_on_no_extractions(self):
        from core.extractor import extract_structured
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[], decisions=[], questions=[],
        ))
        try:
            items, decisions, questions = extract_structured("Short meeting.", "m")
            assert items == []
            assert decisions == []
            assert questions == []
        finally:
            self._stop_patches(patches)

    def test_llm_failure_raises_runtime_error(self):
        from core.extractor import extract_structured
        with patch("core.extractor._extract_chunk_action_items", side_effect=Exception("API error")):
            with pytest.raises(RuntimeError, match="extraction failed"):
                extract_structured("some transcript", "m")

    def test_runtime_error_does_not_expose_api_key(self):
        from core.extractor import extract_structured
        fake_key = "sk-secret-key-should-not-appear"
        with patch("core.extractor._extract_chunk_action_items",
                   side_effect=Exception(f"unauthorized: {fake_key}")):
            with pytest.raises(RuntimeError) as exc_info:
                extract_structured("some transcript", "m")
        # The RuntimeError message we raise should not contain the key.
        assert fake_key not in str(exc_info.value)

    def test_evidence_linking_populates_timestamp_ref(self):
        from core.extractor import extract_structured
        segs = [
            _make_seg("Alice will write the technical report.", 42.0, 48.0, "m"),
        ]
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "write the technical report", "owner": "Alice"}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured(
                "Alice will write the technical report.",
                "m",
                segments=segs,
            )
            assert items[0].timestamp_ref == 42.0
        finally:
            self._stop_patches(patches)

    def test_timestamp_ref_none_when_no_match(self):
        from core.extractor import extract_structured
        segs = [_make_seg("Completely different content.", 10.0, 15.0, "m")]
        patches = self._apply_patches(self._mock_llm_responses(
            action_items=[{"task": "xyz abc def ghi jkl mno"}],
            decisions=[], questions=[],
        ))
        try:
            items, _, _ = extract_structured("xyz abc def ghi jkl mno", "m", segments=segs)
            # No match expected — timestamp_ref should be None.
            assert items[0].timestamp_ref is None
        finally:
            self._stop_patches(patches)


# ── Large transcript triggers chunking ───────────────────────────────────────

class TestLargeTranscriptChunking:
    def test_large_transcript_calls_extraction_multiple_times(self):
        from core.extractor import extract_structured, MAX_EXTRACTION_CHARS
        long_transcript = "word " * (MAX_EXTRACTION_CHARS // 4)

        call_count = {"n": 0}

        def fake_actions(chunk):
            call_count["n"] += 1
            from core.extractor import _LLMActionItemList
            return _LLMActionItemList(items=[]).items

        with patch("core.extractor._extract_chunk_action_items", side_effect=fake_actions):
            with patch("core.extractor._extract_chunk_decisions", return_value=[]):
                with patch("core.extractor._extract_chunk_questions", return_value=[]):
                    extract_structured(long_transcript, "m")

        assert call_count["n"] > 1


# ── Backward-compatible string functions ──────────────────────────────────────

class TestBackwardCompatStringFunctions:
    def _patch_extract_structured(self, items, decisions, questions):
        return patch(
            "core.extractor.extract_structured",
            return_value=(items, decisions, questions),
        )

    def test_extract_action_items_returns_string(self):
        from core.extractor import extract_action_items
        item = ActionItem(meeting_id="legacy", task="Do something", owner="Bob")
        with self._patch_extract_structured([item], [], []):
            result = extract_action_items("transcript text")
        assert isinstance(result, str)
        assert "Do something" in result
        assert "Bob" in result

    def test_extract_key_decisions_returns_string(self):
        from core.extractor import extract_key_decisions
        dec = Decision(meeting_id="legacy", decision="Use Python.")
        with self._patch_extract_structured([], [dec], []):
            result = extract_key_decisions("transcript text")
        assert isinstance(result, str)
        assert "Use Python." in result

    def test_extract_questions_returns_string(self):
        from core.extractor import extract_questions
        q = OpenQuestion(meeting_id="legacy", question="Who leads?")
        with self._patch_extract_structured([], [], [q]):
            result = extract_questions("transcript text")
        assert isinstance(result, str)
        assert "Who leads?" in result

    def test_empty_extraction_returns_safe_string(self):
        from core.extractor import extract_action_items
        with self._patch_extract_structured([], [], []):
            result = extract_action_items("nothing here")
        assert result == "No action items found."


# ── build_meeting integration (fully mocked) ─────────────────────────────────

class TestBuildMeeting:
    """Test pipeline.build_meeting() with all I/O mocked."""

    def test_meeting_populated_from_pipeline(self):
        from core.pipeline import build_meeting
        from core.models.meeting import Meeting, SourceType

        fake_seg = _make_seg("We agreed on a Python migration.", 0.0, 5.0, "")
        fake_items = [ActionItem(meeting_id="", task="Start migration")]
        fake_decisions = [Decision(meeting_id="", decision="Use Python 3.12.")]
        fake_questions = [OpenQuestion(meeting_id="", question="When do we start?")]

        with patch("core.pipeline.transcribe_all_structured", return_value=[fake_seg]) as mock_t:
            with patch("core.pipeline.generate_title", return_value="Python Migration Planning"):
                with patch("core.pipeline.summarize", return_value="• Decided to migrate."):
                    with patch("core.pipeline.extract_structured",
                               return_value=(fake_items, fake_decisions, fake_questions)):
                        meeting, summary = build_meeting(
                            chunks=["chunk1.wav"],
                            source="https://youtube.com/watch?v=test",
                            language="english",
                        )

        assert isinstance(meeting, Meeting)
        assert meeting.title == "Python Migration Planning"
        assert summary == "• Decided to migrate."
        assert meeting.source == "https://youtube.com/watch?v=test"
        assert meeting.source_type == SourceType.YOUTUBE
        assert meeting.language == "english"
        assert len(meeting.segments) == 1
        assert len(meeting.action_items) == 1
        assert len(meeting.decisions) == 1
        assert len(meeting.open_questions) == 1

    def test_meeting_id_is_unique(self):
        from core.pipeline import build_meeting
        fake_seg = _make_seg("Hello.", 0.0, 1.0, "")

        ids = set()
        for _ in range(5):
            with patch("core.pipeline.transcribe_all_structured", return_value=[fake_seg]):
                with patch("core.pipeline.generate_title", return_value="T"):
                    with patch("core.pipeline.summarize", return_value="S"):
                        with patch("core.pipeline.extract_structured", return_value=([], [], [])):
                            meeting, _ = build_meeting(["c.wav"], "file.mp4")
                            ids.add(meeting.id)
        assert len(ids) == 5

    def test_source_type_local_file(self):
        from core.pipeline import build_meeting
        from core.models.meeting import SourceType
        fake_seg = _make_seg("Hello.", 0.0, 1.0, "")
        with patch("core.pipeline.transcribe_all_structured", return_value=[fake_seg]):
            with patch("core.pipeline.generate_title", return_value="T"):
                with patch("core.pipeline.summarize", return_value="S"):
                    with patch("core.pipeline.extract_structured", return_value=([], [], [])):
                        meeting, _ = build_meeting(["c.wav"], "/local/file.mp4")
        assert meeting.source_type == SourceType.LOCAL_FILE

    def test_duration_set_from_last_segment(self):
        from core.pipeline import build_meeting
        segs = [
            _make_seg("Hello.", 0.0, 5.0, ""),
            _make_seg("World.", 5.0, 120.5, ""),
        ]
        with patch("core.pipeline.transcribe_all_structured", return_value=segs):
            with patch("core.pipeline.generate_title", return_value="T"):
                with patch("core.pipeline.summarize", return_value="S"):
                    with patch("core.pipeline.extract_structured", return_value=([], [], [])):
                        meeting, _ = build_meeting(["c.wav"], "file.mp4")
        assert meeting.duration == 120.5

    def test_progress_callback_called(self):
        from core.pipeline import build_meeting
        fake_seg = _make_seg("Hello.", 0.0, 1.0, "")
        progress_calls = []

        with patch("core.pipeline.transcribe_all_structured", return_value=[fake_seg]):
            with patch("core.pipeline.generate_title", return_value="T"):
                with patch("core.pipeline.summarize", return_value="S"):
                    with patch("core.pipeline.extract_structured", return_value=([], [], [])):
                        build_meeting(
                            ["c.wav"], "file.mp4",
                            on_progress=progress_calls.append,
                        )

        assert "transcription" in progress_calls
        assert "title" in progress_calls
        assert "summary" in progress_calls
        assert "extraction" in progress_calls

    def test_meeting_id_propagated_to_all_items(self):
        from core.pipeline import build_meeting
        fake_seg = _make_seg("Agreed to use Python.", 0.0, 5.0, "")

        captured_meeting_id = {}

        def fake_extract(transcript, meeting_id, segments=None):
            captured_meeting_id["id"] = meeting_id
            item = ActionItem(meeting_id=meeting_id, task="Task")
            dec = Decision(meeting_id=meeting_id, decision="Decision")
            q = OpenQuestion(meeting_id=meeting_id, question="Question?")
            return [item], [dec], [q]

        with patch("core.pipeline.transcribe_all_structured", return_value=[fake_seg]):
            with patch("core.pipeline.generate_title", return_value="T"):
                with patch("core.pipeline.summarize", return_value="S"):
                    with patch("core.pipeline.extract_structured", side_effect=fake_extract):
                        meeting, _ = build_meeting(["c.wav"], "file.mp4")

        mid = meeting.id
        assert captured_meeting_id["id"] == mid
        assert meeting.action_items[0].meeting_id == mid
        assert meeting.decisions[0].meeting_id == mid
        assert meeting.open_questions[0].meeting_id == mid


# ── Overlapping-chunk deduplication ──────────────────────────────────────────

class TestOverlapDeduplication:
    """Verify that the same ActionItem extracted from two overlapping chunks
    appears only once in the final result."""

    def test_duplicate_action_item_from_overlapping_chunks_deduped_to_one(self):
        """Simulates two extraction chunks both returning the same action item
        (as happens when chunks overlap at boundaries).  The final list must
        contain exactly one copy.
        """
        from core.extractor import extract_structured, _LLMActionItem

        # Both chunks return the same task — identical text, identical casing.
        duplicate_item = _LLMActionItem(task="Update the deployment documentation")

        call_count = {"n": 0}

        def fake_actions(chunk):
            call_count["n"] += 1
            return [duplicate_item]  # same item returned for every chunk

        with patch("core.extractor._extract_chunk_action_items", side_effect=fake_actions):
            with patch("core.extractor._extract_chunk_decisions", return_value=[]):
                with patch("core.extractor._extract_chunk_questions", return_value=[]):
                    # Force two chunks by patching _split_for_extraction.
                    with patch(
                        "core.extractor._split_for_extraction",
                        return_value=["chunk one text", "chunk two text"],
                    ):
                        items, _, _ = extract_structured("ignored", "meeting-dedup")

        # Two chunks were processed.
        assert call_count["n"] == 2
        # But deduplication must have collapsed both to one.
        assert len(items) == 1
        assert items[0].task == "Update the deployment documentation"
        assert items[0].meeting_id == "meeting-dedup"

    def test_different_action_items_from_overlapping_chunks_both_kept(self):
        """Distinct items from different chunks must not be deduplicated."""
        from core.extractor import extract_structured, _LLMActionItem

        items_per_call = [
            [_LLMActionItem(task="Task Alpha")],
            [_LLMActionItem(task="Task Beta")],
        ]
        call_iter = iter(items_per_call)

        with patch("core.extractor._extract_chunk_action_items", side_effect=lambda c: next(call_iter)):
            with patch("core.extractor._extract_chunk_decisions", return_value=[]):
                with patch("core.extractor._extract_chunk_questions", return_value=[]):
                    with patch(
                        "core.extractor._split_for_extraction",
                        return_value=["chunk one", "chunk two"],
                    ):
                        items, _, _ = extract_structured("ignored", "m")

        assert len(items) == 2
        tasks = {i.task for i in items}
        assert tasks == {"Task Alpha", "Task Beta"}


# ── Evidence linking is scoped to the supplied segment list ──────────────────

class TestEvidenceLinkingScope:
    """Verify that _find_timestamp_ref only matches within the supplied
    segment list — it cannot accidentally pick up segments from elsewhere."""

    def test_match_found_only_in_supplied_segments(self):
        """The supplied segment list contains a match.  It is found."""
        from core.extractor import _find_timestamp_ref

        supplied = [
            _make_seg("Alice will update the deployment pipeline.", 55.0, 60.0, "meeting-A"),
        ]
        result = _find_timestamp_ref("update the deployment pipeline", supplied)
        assert result == 55.0

    def test_unrelated_segment_not_considered(self):
        """A segment belonging to a different meeting is NOT in the supplied
        list.  _find_timestamp_ref has no way to reach it, so it returns None.

        This confirms that matching is scoped entirely to the list passed in —
        the function has no global segment store to accidentally query.
        """
        from core.extractor import _find_timestamp_ref

        # Segment that WOULD match — but it is not in the supplied list.
        # We intentionally pass a list that does NOT contain it.
        unrelated_seg = _make_seg(
            "Update the deployment pipeline immediately.",
            100.0, 105.0, "meeting-B",
        )

        # Supplied list is completely unrelated.
        supplied = [
            _make_seg("We discussed quarterly targets.", 0.0, 5.0, "meeting-A"),
        ]

        result = _find_timestamp_ref("Update the deployment pipeline", supplied)
        # The matching segment (unrelated_seg) was never passed in.
        # The supplied segment has no useful overlap → None.
        assert result is None

    def test_only_first_matching_segment_start_is_returned(self):
        """If multiple supplied segments match, the first one (earliest in the
        list) is returned — consistent with the substring pass-1 logic."""
        from core.extractor import _find_timestamp_ref

        supplied = [
            _make_seg("Send the quarterly report to the board.", 10.0, 15.0, "m"),
            _make_seg("Send the quarterly report as well.", 20.0, 25.0, "m"),
        ]
        result = _find_timestamp_ref("Send the quarterly report", supplied)
        # First match wins.
        assert result == 10.0
