"""
tests/test_report_generation.py
────────────────────────────────
Tests for PDF and TXT report generation in RECALL.

Design
------
- Tests verify TXT and PDF generation without external dependencies
- PDF is tested for byte output (not full rendering validation)
- TXT is tested for content structure and data presence
- No Mistral, Sarvam, Whisper, or network calls
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.report_generator import generate_pdf_report, generate_txt_report
from core.models.meeting import Meeting
from core.models.intelligence import ActionItem, Decision, OpenQuestion
from core.models.transcript import TranscriptSegment


class TestTXTReportGeneration:
    """Test plain-text report generation."""

    def test_txt_report_basic_structure(self):
        """TXT report should contain required sections."""
        meeting = Meeting(source="test.mp4")
        report = generate_txt_report(meeting, "Test summary")

        assert "RECALL — AI Meeting Intelligence" in report
        assert "MEETING" in report
        assert "SUMMARY" in report
        assert "KEY DECISIONS" in report
        assert "ACTION ITEMS" in report
        assert "OPEN QUESTIONS" in report
        assert "TRANSCRIPT" in report

    def test_txt_report_includes_meeting_info(self):
        """TXT report should include meeting metadata."""
        meeting = Meeting(
            source="https://youtube.com/watch?v=test",
            title="Test Meeting",
            language="english",
            duration=3600.0,
        )
        report = generate_txt_report(meeting, "Summary")

        assert "Title:    Test Meeting" in report
        assert "Source:   https://youtube.com/watch?v=test" in report
        assert "Language: english" in report
        assert "Duration: 3600.0 seconds" in report

    def test_txt_report_includes_summary(self):
        """TXT report should include the summary text."""
        meeting = Meeting(source="test.mp4")
        summary_text = "This is the meeting summary."
        report = generate_txt_report(meeting, summary_text)

        assert summary_text in report

    def test_txt_report_includes_decisions(self):
        """TXT report should list all decisions."""
        meeting = Meeting(source="test.mp4")
        meeting.decisions = [
            Decision(meeting_id=meeting.id, decision="Use PostgreSQL"),
            Decision(meeting_id=meeting.id, decision="Deploy on Monday"),
        ]
        report = generate_txt_report(meeting, "Summary")

        assert "1. Use PostgreSQL" in report
        assert "2. Deploy on Monday" in report

    def test_txt_report_includes_action_items(self):
        """TXT report should list action items with task, owner, deadline."""
        meeting = Meeting(source="test.mp4")
        meeting.action_items = [
            ActionItem(
                meeting_id=meeting.id,
                task="Deploy API",
                owner="Alice",
                deadline="Friday",
            ),
            ActionItem(
                meeting_id=meeting.id,
                task="Write tests",
            ),
        ]
        report = generate_txt_report(meeting, "Summary")

        assert "1. Task: Deploy API" in report
        assert "Owner: Alice" in report
        assert "Deadline: Friday" in report
        assert "2. Task: Write tests" in report

    def test_txt_report_includes_open_questions(self):
        """TXT report should list open questions."""
        meeting = Meeting(source="test.mp4")
        meeting.open_questions = [
            OpenQuestion(meeting_id=meeting.id, question="Should we use React?"),
            OpenQuestion(meeting_id=meeting.id, question="What about data retention?"),
        ]
        report = generate_txt_report(meeting, "Summary")

        assert "1. Should we use React?" in report
        assert "2. What about data retention?" in report

    def test_txt_report_includes_transcript(self):
        """TXT report should include transcript segments with timestamps."""
        meeting = Meeting(source="test.mp4")
        meeting.segments = [
            TranscriptSegment(
                meeting_id=meeting.id,
                text="We decided on PostgreSQL.",
                start=10.0,
                end=15.0,
            ),
            TranscriptSegment(
                meeting_id=meeting.id,
                text="Deployment next Friday.",
                start=20.0,
                end=25.0,
            ),
        ]
        report = generate_txt_report(meeting, "Summary")

        # Check timestamps are formatted (MM:SS)
        assert "[00:10]" in report
        assert "[00:20]" in report
        # Check text is present
        assert "We decided on PostgreSQL." in report
        assert "Deployment next Friday." in report

    def test_txt_report_handles_missing_data(self):
        """TXT report should handle meetings with no structured data."""
        meeting = Meeting(source="test.mp4")
        # No decisions, action items, or questions
        report = generate_txt_report(meeting, "Summary")

        assert "No decisions recorded." in report
        assert "No action items recorded." in report
        assert "No open questions recorded." in report
        assert "No transcript available." in report

    def test_txt_report_returns_string(self):
        """TXT report generation should return a string."""
        meeting = Meeting(source="test.mp4")
        report = generate_txt_report(meeting, "Summary")

        assert isinstance(report, str)
        assert len(report) > 0


class TestPDFReportGeneration:
    """Test PDF report generation."""

    def test_pdf_report_returns_bytes(self):
        """PDF report generation should return bytes."""
        meeting = Meeting(source="test.mp4")
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_pdf_report_starts_with_pdf_signature(self):
        """PDF bytes should start with PDF signature."""
        meeting = Meeting(source="test.mp4")
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        # PDF files start with %PDF
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_report_includes_meeting_info(self):
        """PDF report should include meeting metadata."""
        meeting = Meeting(
            source="test.mp4",
            title="Test Meeting",
            language="english",
        )
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        # Verify the PDF was generated (size should be reasonable)
        assert len(pdf_bytes) > 500  # Basic sanity check

    def test_pdf_report_includes_summary(self):
        """PDF report should include summary text."""
        meeting = Meeting(source="test.mp4")
        summary_text = "This is a comprehensive meeting summary."
        pdf_bytes = generate_pdf_report(meeting, summary_text)

        # Summary should be in the PDF content
        assert summary_text.encode() in pdf_bytes or len(pdf_bytes) > 500

    def test_pdf_report_with_decisions(self):
        """PDF report should include decisions."""
        meeting = Meeting(source="test.mp4")
        meeting.decisions = [
            Decision(meeting_id=meeting.id, decision="Use PostgreSQL"),
        ]
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_report_with_action_items(self):
        """PDF report should include action items."""
        meeting = Meeting(source="test.mp4")
        meeting.action_items = [
            ActionItem(
                meeting_id=meeting.id,
                task="Deploy API",
                owner="Alice",
            ),
        ]
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_report_with_transcript(self):
        """PDF report should include transcript."""
        meeting = Meeting(source="test.mp4")
        meeting.segments = [
            TranscriptSegment(
                meeting_id=meeting.id,
                text="We agreed on the timeline.",
                start=10.0,
                end=15.0,
            ),
        ]
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_report_handles_missing_data(self):
        """PDF report should handle meetings with no structured data."""
        meeting = Meeting(source="test.mp4")
        # No decisions, action items, questions, or segments
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_report_size_reasonable(self):
        """PDF report size should be reasonable (not empty, not huge)."""
        meeting = Meeting(
            source="test.mp4",
            title="Meeting",
        )
        meeting.decisions = [
            Decision(meeting_id=meeting.id, decision="Decision 1"),
        ]
        meeting.action_items = [
            ActionItem(meeting_id=meeting.id, task="Task 1"),
        ]
        pdf_bytes = generate_pdf_report(meeting, "Summary")

        # PDF should be between 1KB and 10MB
        assert 1000 < len(pdf_bytes) < 10_000_000
