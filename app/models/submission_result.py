"""
SubmissionResult model for SmartEval Phase 4.
Stores calculated final scores, percentage, letter grades, grounded overall feedback,
publication metadata, and audit timestamps.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.extensions import db


class SubmissionResult(db.Model):
    """
    Represents the finalized, calculated, and publishable result of a student submission.
    Maintains complete audit separation between preliminary AI scoring and faculty final decision.
    """
    __tablename__ = "submission_result"

    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    submission_id = db.Column(
        db.Integer,
        db.ForeignKey("submission.submission_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    total_maximum_marks = db.Column(db.Float, nullable=False, default=0.0)
    total_obtained_marks = db.Column(db.Float, nullable=False, default=0.0)
    percentage = db.Column(db.Float, nullable=False, default=0.0)
    grade_letter = db.Column(db.String(10), nullable=True)

    # Structured Grounded Overall Feedback
    overall_strengths = db.Column(db.JSON, nullable=True)  # List of string strengths grounded in Q evaluations
    overall_improvements = db.Column(db.JSON, nullable=True)  # List of string missing concepts
    overall_feedback = db.Column(db.Text, nullable=True)  # Grounded synthesis summary
    faculty_summary_feedback = db.Column(db.Text, nullable=True)  # Optional instructor overall remarks

    # Publication & Lifecycle Audit State
    is_published = db.Column(db.Boolean, default=False, nullable=False, index=True)
    published_at = db.Column(db.DateTime, nullable=True)
    faculty_reviewed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    @property
    def formatted_percentage(self) -> str:
        """Return formatted percentage string."""
        return f"{self.percentage:.2f}%"

    @property
    def strengths_list(self) -> List[str]:
        """Return strengths as a list of strings."""
        if isinstance(self.overall_strengths, list):
            return self.overall_strengths
        return []

    @property
    def improvements_list(self) -> List[str]:
        """Return improvement areas as a list of strings."""
        if isinstance(self.overall_improvements, list):
            return self.overall_improvements
        return []

    @staticmethod
    def calculate_grade_letter(percentage: float) -> str:
        """Determine letter grade band from percentage."""
        if percentage >= 90.0:
            return "A+"
        elif percentage >= 80.0:
            return "A"
        elif percentage >= 70.0:
            return "B"
        elif percentage >= 60.0:
            return "C"
        elif percentage >= 50.0:
            return "D"
        else:
            return "F"

    def to_dict(self, include_private_info: bool = False) -> Dict[str, Any]:
        """
        Serialize result to dict.
        Note: Internal audit metadata can be included for faculty views, but excluded for student.
        """
        data = {
            "result_id": self.result_id,
            "submission_id": self.submission_id,
            "total_maximum_marks": round(self.total_maximum_marks, 2),
            "total_obtained_marks": round(self.total_obtained_marks, 2),
            "percentage": round(self.percentage, 2),
            "formatted_percentage": self.formatted_percentage,
            "grade_letter": self.grade_letter,
            "overall_strengths": self.strengths_list,
            "overall_improvements": self.improvements_list,
            "overall_feedback": self.overall_feedback,
            "faculty_summary_feedback": self.faculty_summary_feedback,
            "is_published": self.is_published,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_private_info:
            data["faculty_reviewed_at"] = (
                self.faculty_reviewed_at.isoformat() if self.faculty_reviewed_at else None
            )
        return data

    def __repr__(self) -> str:
        return (
            f"<SubmissionResult {self.result_id}: Submission {self.submission_id} -> "
            f"{self.total_obtained_marks}/{self.total_maximum_marks} ({self.percentage}%), "
            f"Published={self.is_published}>"
        )
