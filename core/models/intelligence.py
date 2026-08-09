"""
core/models/intelligence.py
────────────────────────────
Extracted meeting intelligence models for RECALL — AI Meeting Intelligence.

These models define the canonical shape of structured data extracted from
meeting transcripts.  In Phase 2A they are established but not yet
populated by the LLM pipeline — that migration happens in Phase 2B.

ActionItem
----------
Represents a committed task with an optional owner and deadline.
Owner and deadline are Optional because the LLM may not be able to
identify them from the transcript.  They must never be invented.

Decision
--------
A key decision made during the meeting.

OpenQuestion
------------
An unresolved question or topic that needs follow-up.

Timestamp reference
-------------------
Each model carries an optional timestamp_ref (start seconds) pointing
to the approximate location in the transcript where the item was found.
This will support evidence linking in a future phase.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """A task or commitment arising from the meeting.

    Fields
    ------
    id : str
        Unique identifier (UUID hex).
    meeting_id : str
        UUID of the meeting this item belongs to.
    task : str
        Description of the work to be done.  Required.
    owner : Optional[str]
        Person or team responsible.  None if not identified.
    deadline : Optional[str]
        Deadline as mentioned in the transcript (free-form text, e.g.
        'next Friday', '2024-09-01').  None if not mentioned.
    timestamp_ref : Optional[float]
        Approximate position in the transcript (seconds) where this
        item was identified.  None if unknown.
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    meeting_id: str
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    timestamp_ref: Optional[float] = None


class Decision(BaseModel):
    """A key decision made during the meeting.

    Fields
    ------
    id : str
        Unique identifier (UUID hex).
    meeting_id : str
        UUID of the meeting this decision belongs to.
    decision : str
        The decision that was made.  Required.
    timestamp_ref : Optional[float]
        Approximate position in the transcript (seconds).  None if unknown.
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    meeting_id: str
    decision: str
    timestamp_ref: Optional[float] = None


class OpenQuestion(BaseModel):
    """An unresolved question or follow-up topic from the meeting.

    Fields
    ------
    id : str
        Unique identifier (UUID hex).
    meeting_id : str
        UUID of the meeting this question belongs to.
    question : str
        The question or topic.  Required.
    timestamp_ref : Optional[float]
        Approximate position in the transcript (seconds).  None if unknown.
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    meeting_id: str
    question: str
    timestamp_ref: Optional[float] = None
