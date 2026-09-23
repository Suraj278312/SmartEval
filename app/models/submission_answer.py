"""
SubmissionAnswer model for SmartEval Phase 3A.
Stores segmented student answer blocks mapped to assignment questions,
tracking source page ranges, matching technique, confidence score, and review status.
"""

from datetime import datetime, timezone
from app.extensions import db


class SubmissionAnswer(db.Model):
    """
    Represents a segmented answer block extracted from student submission pages
    and mapped to an assignment question.
    """
    __tablename__ = "submission_answer"

    answer_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    submission_id = db.Column(
        db.Integer,
        db.ForeignKey("submission.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id = db.Column(
        db.Integer,
        db.ForeignKey("question.question_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    extracted_text = db.Column(db.Text, nullable=False)
    detected_label = db.Column(db.String(50), nullable=True)  # e.g., 'Q1', '1.', 'Question 2', 'Ans 3'
    page_start = db.Column(db.Integer, nullable=False, default=1)
    page_end = db.Column(db.Integer, nullable=False, default=1)
    match_method = db.Column(
        db.String(50),
        nullable=False,
        default="explicit_numbering",
    )  # 'explicit_numbering', 'text_similarity', 'semantic_similarity', 'unmatched'
    match_confidence = db.Column(db.Float, nullable=False, default=0.0)  # 0.0 to 1.0
    status = db.Column(
        db.String(50),
        default="Matched",
        nullable=False,
        index=True,
    )  # 'Matched', 'Review Required', 'Unmatched'
    review_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    @property
    def match_confidence_percentage(self) -> float:
        """Return match confidence as a percentage (0.0 to 100.0)."""
        if self.match_confidence is None:
            return 0.0
        if self.match_confidence <= 1.0:
            return round(self.match_confidence * 100, 1)
        return round(self.match_confidence, 1)

    @property
    def is_low_confidence(self) -> bool:
        """Flag matches with confidence below 75% as uncertain/requiring review."""
        if self.match_confidence is None:
            return True
        val = self.match_confidence if self.match_confidence > 1.0 else self.match_confidence * 100
        return val < 75.0

    @property
    def page_span_display(self) -> str:
        """Format human-readable page span (e.g., 'Page 1' or 'Pages 1-2')."""
        if self.page_start == self.page_end:
            return f"Page {self.page_start}"
        return f"Pages {self.page_start}–{self.page_end}"

    @property
    def match_method_display(self) -> str:
        """Format match method for clean UI badge presentation."""
        mapping = {
            "explicit_numbering": "Explicit Numbering",
            "text_similarity": "Text Similarity",
            "semantic_similarity": "Semantic Similarity",
            "unmatched": "Unmatched / Orphan",
        }
        return mapping.get(self.match_method, self.match_method.replace("_", " ").title())

    def to_dict(self, include_supportive_answer: bool = False) -> dict:
        """
        Serialize answer mapping to dict.
        Note: Supportive answers are strictly excluded for student-facing contexts.
        """
        return {
            "answer_id": self.answer_id,
            "submission_id": self.submission_id,
            "question_id": self.question_id,
            "question_number": self.question.question_number if self.question else None,
            "question_text": self.question.question_text if self.question else None,
            "extracted_text": self.extracted_text,
            "detected_label": self.detected_label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_span_display": self.page_span_display,
            "match_method": self.match_method,
            "match_method_display": self.match_method_display,
            "match_confidence": self.match_confidence,
            "match_confidence_percentage": self.match_confidence_percentage,
            "is_low_confidence": self.is_low_confidence,
            "status": self.status,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        q_label = f"Q{self.question.question_number}" if self.question else "Unmatched"
        return (
            f"<SubmissionAnswer {self.answer_id}: Sub {self.submission_id} -> {q_label}, "
            f"Method='{self.match_method}', Conf={self.match_confidence_percentage}%, Status='{self.status}'>"
        )
