"""
Submission model and status enumeration for SmartEval.
Tracks student PDF submissions, document page extraction results, and processing states.
"""

from datetime import datetime, timezone
from enum import Enum
from app.extensions import db


class SubmissionStatus(str, Enum):
    """Supported submission lifecycle states."""
    PENDING = "Pending"
    UPLOADED = "Uploaded"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    REVIEW_REQUIRED = "Review Required"
    FACULTY_REVIEWED = "Faculty Reviewed"
    PUBLISHED = "Published"
    FAILED = "Failed"

    @classmethod
    def choices(cls):
        return [(status.value, status.value) for status in cls]


class Submission(db.Model):
    """Student submission entity storing uploaded PDF details, extracted pages, and status."""
    __tablename__ = "submission"

    submission_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    assignment_id = db.Column(
        db.Integer,
        db.ForeignKey("assignment.assignment_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    student_id = db.Column(
        db.Integer,
        db.ForeignKey("student.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    file_path = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size_bytes = db.Column(db.Integer, nullable=False, default=0)
    submission_date = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    status = db.Column(
        db.String(50),
        default=SubmissionStatus.UPLOADED.value,
        nullable=False,
        index=True
    )
    evaluation_notes = db.Column(db.Text, nullable=True)  # Processing error or review notes

    # Phase 2: Relationship to processed document pages
    pages = db.relationship(
        "DocumentPage",
        backref="submission",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="DocumentPage.page_number"
    )

    # Phase 3A: Relationship to segmented and matched answers
    answers = db.relationship(
        "SubmissionAnswer",
        backref="submission",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="SubmissionAnswer.answer_id"
    )

    # Phase 3B: Relationship to AI and faculty evaluations
    evaluations = db.relationship(
        "AnswerEvaluation",
        backref="submission",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="AnswerEvaluation.evaluation_id"
    )

    # Phase 4: Relationship to finalized, publishable submission result
    result = db.relationship(
        "SubmissionResult",
        backref="submission",
        uselist=False,
        cascade="all, delete-orphan",
        lazy=True,
    )

    @property
    def matched_answers(self):
        """List of answers successfully or tentatively mapped to a question."""
        return [a for a in self.answers if a.question_id is not None]

    @property
    def unmatched_answers(self):
        """List of orphan or unmapped extracted text segments."""
        return [a for a in self.answers if a.question_id is None]

    @property
    def has_unmatched_answers(self) -> bool:
        """Check if any extracted text segment could not be mapped."""
        return len(self.unmatched_answers) > 0

    @property
    def total_ai_score(self) -> float:
        """Calculate total marks awarded by AI across evaluated questions."""
        valid_scores = [e.ai_score for e in self.evaluations if e.ai_score is not None]
        return round(sum(valid_scores), 2) if valid_scores else 0.0

    @property
    def total_final_score(self) -> float:
        """Calculate total final marks (faculty approved/modified or AI preliminary)."""
        if self.result and self.result.total_obtained_marks is not None:
            return round(self.result.total_obtained_marks, 2)
        valid_scores = [e.final_score for e in self.evaluations if e.final_score is not None]
        return round(sum(valid_scores), 2) if valid_scores else 0.0

    @property
    def total_maximum_marks(self) -> float:
        """Calculate total maximum marks for the assignment."""
        if self.result and self.result.total_maximum_marks is not None:
            return round(self.result.total_maximum_marks, 2)
        if self.assignment:
            return round(self.assignment.total_marks, 2)
        return round(sum(e.maximum_marks for e in self.evaluations), 2)

    @property
    def percentage(self) -> float:
        """Calculate final percentage awarded."""
        if self.result and self.result.percentage is not None:
            return round(self.result.percentage, 2)
        max_marks = self.total_maximum_marks
        if max_marks <= 0:
            return 0.0
        return round((self.total_final_score / max_marks) * 100, 2)

    @property
    def is_published(self) -> bool:
        """Check if final results are published to the student."""
        if self.result:
            return bool(self.result.is_published)
        return self.status == SubmissionStatus.PUBLISHED.value

    @property
    def is_reviewed(self) -> bool:
        """Check if all evaluations have been approved or modified by faculty."""
        if not self.evaluations:
            return False
        return all(e.evaluation_status in ("Approved", "Modified") for e in self.evaluations)

    @property
    def is_fully_evaluated(self) -> bool:
        """Check if all assignment questions have been evaluated successfully."""
        if not self.assignment or not self.assignment.questions:
            return False
        return len(self.evaluations) >= len(self.assignment.questions) and all(e.evaluation_status != "Failed" and e.final_score is not None for e in self.evaluations)

    @property
    def has_pending_faculty_review(self) -> bool:
        """Check if any evaluation requires faculty review."""
        return any(e.needs_faculty_review for e in self.evaluations)

    @property
    def page_count(self) -> int:
        """Total number of processed pages."""
        return len(self.pages)

    @property
    def average_confidence(self) -> float:
        """Calculate the average confidence percentage across all extracted pages."""
        valid_pages = [p for p in self.pages if p.confidence is not None]
        if not valid_pages:
            return 0.0
        avg = sum(p.confidence_percentage for p in valid_pages) / len(valid_pages)
        return round(avg, 1)

    @property
    def full_extracted_text(self) -> str:
        """Concatenate extracted text from all pages in order."""
        texts = []
        for p in self.pages:
            if p.extracted_text:
                texts.append(f"--- Page {p.page_number} ---\n{p.extracted_text.strip()}")
        return "\n\n".join(texts)

    @property
    def file_size_display(self) -> str:
        """Format file size in human-readable units."""
        if not self.file_size_bytes:
            return "0 KB"
        kb = self.file_size_bytes / 1024
        if kb < 1024:
            return f"{kb:.1f} KB"
        mb = kb / 1024
        return f"{mb:.2f} MB"

    @property
    def is_review_required(self) -> bool:
        return self.status == SubmissionStatus.REVIEW_REQUIRED.value

    def to_dict(self, include_pages: bool = False, include_answers: bool = False) -> dict:
        data = {
            "submission_id": self.submission_id,
            "assignment_id": self.assignment_id,
            "assignment_title": self.assignment.title if self.assignment else None,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "original_filename": self.original_filename,
            "file_size_bytes": self.file_size_bytes,
            "file_size_display": self.file_size_display,
            "submission_date": (
                self.submission_date.isoformat() if self.submission_date else None
            ),
            "status": self.status,
            "page_count": self.page_count,
            "average_confidence": self.average_confidence,
            "evaluation_notes": self.evaluation_notes,
            "matched_answer_count": len(self.matched_answers),
            "unmatched_answer_count": len(self.unmatched_answers),
        }
        if include_pages:
            data["pages"] = [p.to_dict() for p in self.pages]
        if include_answers:
            data["answers"] = [a.to_dict() for a in self.answers]
        return data

    def __repr__(self) -> str:
        return (
            f"<Submission {self.submission_id}: Assignment {self.assignment_id}, "
            f"Student {self.student_id}, Status='{self.status}', Pages={self.page_count}>"
        )
