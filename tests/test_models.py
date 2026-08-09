"""
tests/test_models.py
─────────────────────
Unit tests for RECALL Phase 2A data models.

These tests do NOT call Whisper, Sarvam, Mistral, YouTube, or ChromaDB.
They test only the data model layer in core/models/.

Coverage:
  - format_timestamp()
  - TranscriptSegment: valid construction, timestamp validation, properties
  - segments_to_text(): backward-compatibility helper
  - ActionItem: required/optional fields
  - Decision: required/optional fields
  - OpenQuestion: required/optional fields
  - Meeting: ID generation, relationships, plain_transcript()
  - Serialization: model_dump() round-trip
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── format_timestamp ──────────────────────────────────────────────────────────

class TestFormatTimestamp:
    def test_zero(self):
        from core.models.transcript import format_timestamp
        assert format_timestamp(0.0) == "00:00"

    def test_under_one_minute(self):
        from core.models.transcript import format_timestamp
        assert format_timestamp(42.0) == "00:42"

    def test_exact_one_minute(self):
        from core.models.transcript import format_timestamp
        assert format_timestamp(60.0) == "01:00"

    def test_typical_value(self):
        from core.models.transcript import format_timestamp
        assert format_timestamp(142.3) == "02:22"

    def test_over_one_hour(self):
        from core.models.transcript import format_timestamp
        assert format_timestamp(3661.0) == "61:01"

    def test_fractional_seconds_truncated(self):
        from core.models.transcript import format_timestamp
        # 90.9 seconds → 01:30 (fractional part truncated, not rounded)
        assert format_timestamp(90.9) == "01:30"


# ── TranscriptSegment ─────────────────────────────────────────────────────────

class TestTranscriptSegment:
    def _make(self, **kwargs):
        from core.models.transcript import TranscriptSegment
        defaults = dict(
            meeting_id="test-meeting-id",
            text="Hello world.",
            start=0.0,
            end=5.0,
        )
        defaults.update(kwargs)
        return TranscriptSegment(**defaults)

    def test_valid_construction(self):
        seg = self._make()
        assert seg.text == "Hello world."
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.meeting_id == "test-meeting-id"

    def test_id_is_auto_generated(self):
        seg = self._make()
        assert seg.id is not None
        assert len(seg.id) > 0

    def test_ids_are_unique(self):
        seg1 = self._make()
        seg2 = self._make()
        assert seg1.id != seg2.id

    def test_speaker_defaults_to_none(self):
        seg = self._make()
        assert seg.speaker is None

    def test_language_defaults_to_en(self):
        seg = self._make()
        assert seg.language == "en"

    def test_source_language_defaults_to_none(self):
        seg = self._make()
        assert seg.source_language is None

    def test_chunk_index_defaults_to_zero(self):
        seg = self._make()
        assert seg.chunk_index == 0

    def test_negative_start_raises(self):
        from core.models.transcript import TranscriptSegment
        with pytest.raises(ValidationError, match="start must be"):
            TranscriptSegment(meeting_id="m", text="x", start=-1.0, end=5.0)

    def test_negative_end_raises(self):
        from core.models.transcript import TranscriptSegment
        with pytest.raises(ValidationError, match="end must be"):
            TranscriptSegment(meeting_id="m", text="x", start=0.0, end=-1.0)

    def test_end_before_start_raises(self):
        from core.models.transcript import TranscriptSegment
        with pytest.raises(ValidationError, match="end .* must be >= start"):
            TranscriptSegment(meeting_id="m", text="x", start=10.0, end=5.0)

    def test_end_equal_to_start_is_valid(self):
        # Zero-duration segment is allowed (e.g. a single timestamp marker).
        seg = self._make(start=5.0, end=5.0)
        assert seg.duration == 0.0

    def test_duration_property(self):
        seg = self._make(start=10.0, end=15.5)
        assert abs(seg.duration - 5.5) < 1e-9

    def test_start_formatted_property(self):
        seg = self._make(start=142.3, end=148.0)
        assert seg.start_formatted == "02:22"

    def test_end_formatted_property(self):
        seg = self._make(end=200.0)
        assert seg.end_formatted == "03:20"

    def test_source_language_set_for_translated(self):
        seg = self._make(language="en", source_language="hi")
        assert seg.language == "en"
        assert seg.source_language == "hi"

    def test_speaker_can_be_set(self):
        seg = self._make(speaker="SPEAKER_01")
        assert seg.speaker == "SPEAKER_01"

    def test_serialization_round_trip(self):
        seg = self._make(start=10.0, end=20.0, text="Test segment.")
        data = seg.model_dump()
        from core.models.transcript import TranscriptSegment
        restored = TranscriptSegment(**data)
        assert restored.text == seg.text
        assert restored.start == seg.start
        assert restored.end == seg.end
        assert restored.id == seg.id


# ── segments_to_text ──────────────────────────────────────────────────────────

class TestSegmentsToText:
    def _seg(self, text, meeting_id="m"):
        from core.models.transcript import TranscriptSegment
        return TranscriptSegment(meeting_id=meeting_id, text=text, start=0.0, end=1.0)

    def test_empty_list_returns_empty_string(self):
        from core.models.transcript import segments_to_text
        assert segments_to_text([]) == ""

    def test_single_segment(self):
        from core.models.transcript import segments_to_text
        assert segments_to_text([self._seg("Hello.")]) == "Hello."

    def test_multiple_segments_joined(self):
        from core.models.transcript import segments_to_text
        segs = [self._seg("Hello."), self._seg("World.")]
        result = segments_to_text(segs)
        assert "Hello." in result
        assert "World." in result

    def test_whitespace_only_segments_skipped(self):
        from core.models.transcript import segments_to_text
        segs = [self._seg("Hello."), self._seg("   "), self._seg("World.")]
        result = segments_to_text(segs)
        assert result == "Hello. World."


# ── ActionItem ────────────────────────────────────────────────────────────────

class TestActionItem:
    def test_required_fields(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Write the report")
        assert item.task == "Write the report"
        assert item.meeting_id == "m"

    def test_id_auto_generated(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Do something")
        assert item.id is not None and len(item.id) > 0

    def test_owner_optional_defaults_none(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Task")
        assert item.owner is None

    def test_deadline_optional_defaults_none(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Task")
        assert item.deadline is None

    def test_timestamp_ref_optional_defaults_none(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Task")
        assert item.timestamp_ref is None

    def test_all_optional_fields_set(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(
            meeting_id="m",
            task="Prepare slides",
            owner="Alice",
            deadline="next Friday",
            timestamp_ref=142.3,
        )
        assert item.owner == "Alice"
        assert item.deadline == "next Friday"
        assert item.timestamp_ref == 142.3

    def test_task_required_raises_without_it(self):
        from core.models.intelligence import ActionItem
        with pytest.raises(ValidationError):
            ActionItem(meeting_id="m")

    def test_serialization(self):
        from core.models.intelligence import ActionItem
        item = ActionItem(meeting_id="m", task="Task", owner="Bob")
        data = item.model_dump()
        assert data["task"] == "Task"
        assert data["owner"] == "Bob"
        assert data["deadline"] is None


# ── Decision ──────────────────────────────────────────────────────────────────

class TestDecision:
    def test_required_fields(self):
        from core.models.intelligence import Decision
        d = Decision(meeting_id="m", decision="We will use Python.")
        assert d.decision == "We will use Python."

    def test_id_auto_generated(self):
        from core.models.intelligence import Decision
        d = Decision(meeting_id="m", decision="Decision text")
        assert d.id is not None and len(d.id) > 0

    def test_timestamp_ref_optional(self):
        from core.models.intelligence import Decision
        d = Decision(meeting_id="m", decision="x")
        assert d.timestamp_ref is None

    def test_decision_required_raises(self):
        from core.models.intelligence import Decision
        with pytest.raises(ValidationError):
            Decision(meeting_id="m")

    def test_serialization(self):
        from core.models.intelligence import Decision
        d = Decision(meeting_id="m", decision="Go ahead.", timestamp_ref=30.0)
        data = d.model_dump()
        assert data["decision"] == "Go ahead."
        assert data["timestamp_ref"] == 30.0


# ── OpenQuestion ──────────────────────────────────────────────────────────────

class TestOpenQuestion:
    def test_required_fields(self):
        from core.models.intelligence import OpenQuestion
        q = OpenQuestion(meeting_id="m", question="Who handles the rollout?")
        assert q.question == "Who handles the rollout?"

    def test_id_auto_generated(self):
        from core.models.intelligence import OpenQuestion
        q = OpenQuestion(meeting_id="m", question="x")
        assert q.id is not None and len(q.id) > 0

    def test_timestamp_ref_optional(self):
        from core.models.intelligence import OpenQuestion
        q = OpenQuestion(meeting_id="m", question="x")
        assert q.timestamp_ref is None

    def test_question_required_raises(self):
        from core.models.intelligence import OpenQuestion
        with pytest.raises(ValidationError):
            OpenQuestion(meeting_id="m")

    def test_serialization(self):
        from core.models.intelligence import OpenQuestion
        q = OpenQuestion(meeting_id="m", question="What next?")
        data = q.model_dump()
        assert data["question"] == "What next?"


# ── Meeting ───────────────────────────────────────────────────────────────────

class TestMeeting:
    def _make(self, **kwargs):
        from core.models.meeting import Meeting
        defaults = dict(source="https://youtube.com/watch?v=test")
        defaults.update(kwargs)
        return Meeting(**defaults)

    def test_id_auto_generated(self):
        m = self._make()
        assert m.id is not None and len(m.id) > 0

    def test_ids_are_unique(self):
        m1 = self._make()
        m2 = self._make()
        assert m1.id != m2.id

    def test_id_is_not_derived_from_title(self):
        m1 = self._make(title="Same Title")
        m2 = self._make(title="Same Title")
        assert m1.id != m2.id

    def test_required_source_field(self):
        from core.models.meeting import Meeting
        with pytest.raises(ValidationError):
            Meeting()  # source is required

    def test_title_defaults_empty(self):
        m = self._make()
        assert m.title == ""

    def test_language_defaults_english(self):
        m = self._make()
        assert m.language == "english"

    def test_duration_defaults_none(self):
        m = self._make()
        assert m.duration is None

    def test_created_at_is_utc_datetime(self):
        m = self._make()
        assert isinstance(m.created_at, datetime)
        assert m.created_at.tzinfo is not None

    def test_source_type_default(self):
        from core.models.meeting import SourceType
        m = self._make()
        assert m.source_type == SourceType.UNKNOWN

    def test_source_type_youtube(self):
        from core.models.meeting import SourceType
        m = self._make(source_type=SourceType.YOUTUBE)
        assert m.source_type == SourceType.YOUTUBE

    def test_segments_default_empty(self):
        m = self._make()
        assert m.segments == []
        assert m.segment_count == 0
        assert not m.has_transcript

    def test_action_items_default_empty(self):
        m = self._make()
        assert m.action_items == []

    def test_decisions_default_empty(self):
        m = self._make()
        assert m.decisions == []

    def test_open_questions_default_empty(self):
        m = self._make()
        assert m.open_questions == []

    def test_segments_belong_to_meeting(self):
        from core.models.transcript import TranscriptSegment
        m = self._make()
        seg = TranscriptSegment(
            meeting_id=m.id, text="Hello.", start=0.0, end=2.0
        )
        m.segments.append(seg)
        assert m.segment_count == 1
        assert m.has_transcript
        assert m.segments[0].meeting_id == m.id

    def test_plain_transcript_empty_when_no_segments(self):
        m = self._make()
        assert m.plain_transcript() == ""

    def test_plain_transcript_joins_segments(self):
        from core.models.transcript import TranscriptSegment
        m = self._make()
        m.segments.append(TranscriptSegment(meeting_id=m.id, text="Hello.", start=0.0, end=2.0))
        m.segments.append(TranscriptSegment(meeting_id=m.id, text="World.", start=2.0, end=4.0))
        text = m.plain_transcript()
        assert "Hello." in text
        assert "World." in text

    def test_meeting_id_propagated_to_action_items(self):
        from core.models.intelligence import ActionItem
        m = self._make()
        item = ActionItem(meeting_id=m.id, task="Do something")
        m.action_items.append(item)
        assert m.action_items[0].meeting_id == m.id

    def test_meeting_id_propagated_to_decisions(self):
        from core.models.intelligence import Decision
        m = self._make()
        d = Decision(meeting_id=m.id, decision="Go with option A")
        m.decisions.append(d)
        assert m.decisions[0].meeting_id == m.id

    def test_meeting_id_propagated_to_questions(self):
        from core.models.intelligence import OpenQuestion
        m = self._make()
        q = OpenQuestion(meeting_id=m.id, question="Who leads?")
        m.open_questions.append(q)
        assert m.open_questions[0].meeting_id == m.id

    def test_serialization_round_trip(self):
        from core.models.meeting import Meeting
        m = self._make(title="Q3 Planning", language="english")
        data = m.model_dump()
        restored = Meeting(**data)
        assert restored.id == m.id
        assert restored.title == m.title
        assert restored.source == m.source

    def test_two_meetings_have_independent_segment_lists(self):
        """Ensure segment lists are not shared between Meeting instances."""
        from core.models.transcript import TranscriptSegment
        m1 = self._make()
        m2 = self._make()
        seg = TranscriptSegment(meeting_id=m1.id, text="Only in m1.", start=0.0, end=1.0)
        m1.segments.append(seg)
        assert len(m1.segments) == 1
        assert len(m2.segments) == 0


# ── Cross-model: meeting ID consistency ───────────────────────────────────────

class TestMeetingIdConsistency:
    def test_all_related_objects_carry_same_meeting_id(self):
        from core.models.meeting import Meeting
        from core.models.transcript import TranscriptSegment
        from core.models.intelligence import ActionItem, Decision, OpenQuestion

        m = Meeting(source="test.mp4")
        mid = m.id

        seg = TranscriptSegment(meeting_id=mid, text="Text.", start=0.0, end=1.0)
        item = ActionItem(meeting_id=mid, task="Task")
        dec = Decision(meeting_id=mid, decision="Decided.")
        q = OpenQuestion(meeting_id=mid, question="Question?")

        assert seg.meeting_id == mid
        assert item.meeting_id == mid
        assert dec.meeting_id == mid
        assert q.meeting_id == mid

    def test_ids_from_different_meetings_do_not_collide(self):
        from core.models.meeting import Meeting
        ids = {Meeting(source="x").id for _ in range(50)}
        assert len(ids) == 50  # All 50 must be unique
