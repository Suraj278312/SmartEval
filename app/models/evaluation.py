"""
AnswerEvaluation model for SmartEval Phase 3B.
Stores structured AI evaluation outputs, rubric criteria scoring, missing concepts,
strengths, feedback, and preserves separate faculty approval and modification overrides.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.extensions import db


class AnswerEvaluation(db.Model):
    """
    Represents an AI-generated and/or faculty-approved semantic grading evaluation
    for an individual question answer in a submission.
    """
    __tablename__ = "answer_evaluation"

    evaluation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    submission_id = db.Column(
        db.Integer,
        db.ForeignKey("submission.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id = db.Column(
        db.Integer,
        db.ForeignKey("question.question_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    answer_id = db.Column(
        db.Integer,
        db.ForeignKey("submission_answer.answer_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    extracted_answer = db.Column(db.Text, nullable=False)
    ai_score = db.Column(db.Float, nullable=True, default=None)
    maximum_marks = db.Column(db.Float, nullable=False, default=10.0)
    criteria_scores = db.Column(db.JSON, nullable=True)  # List of criteria dicts
    strengths = db.Column(db.JSON, nullable=True)  # List of string strengths
    missing_elements = db.Column(db.JSON, nullable=True)  # List of string missing concepts
    feedback = db.Column(db.Text, nullable=True)
    ai_confidence = db.Column(db.Float, nullable=False, default=0.85)  # 0.0 to 1.0
    semantic_similarity_score = db.Column(db.Float, nullable=True, default=0.0)
    evaluation_status = db.Column(
        db.String(50),
        default="Evaluated",
        nullable=False,
        index=True,
    )  # 'Pending', 'Evaluating', 'Evaluated', 'Review Required', 'Approved', 'Modified', 'Failed'
    needs_faculty_review = db.Column(db.Boolean, default=True, nullable=False)
    faculty_score = db.Column(db.Float, nullable=True)
    faculty_feedback = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
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

    # Relationships
    answer = db.relationship("SubmissionAnswer", foreign_keys=[answer_id], lazy=True)


    @property
    def final_score(self) -> Optional[float]:
        """Return faculty score if modified/approved with override, otherwise AI score."""
        if self.faculty_score is not None:
            return round(self.faculty_score, 2)
        if self.ai_score is not None:
            return round(self.ai_score, 2)
        return None

    @property
    def final_feedback(self) -> str:
        """Return faculty feedback if provided, otherwise AI generated feedback."""
        if self.faculty_feedback and self.faculty_feedback.strip():
            return self.faculty_feedback
        return self.feedback or ""

    @property
    def score_percentage(self) -> Optional[float]:
        """Return the percentage awarded against maximum marks."""
        if not self.maximum_marks or self.maximum_marks <= 0 or self.final_score is None:
            return None
        return round((self.final_score / self.maximum_marks) * 100, 1)

    @property
    def ai_score_percentage(self) -> Optional[float]:
        """Return the percentage calculated by the AI engine."""
        if not self.maximum_marks or self.maximum_marks <= 0 or self.ai_score is None:
            return None
        return round((self.ai_score / self.maximum_marks) * 100, 1)

    @property
    def is_failed(self) -> bool:
        return self.evaluation_status == "Failed"

    @property
    def is_approved(self) -> bool:
        return self.evaluation_status == "Approved"

    @property
    def is_modified(self) -> bool:
        return self.evaluation_status == "Modified"

    @property
    def is_reviewed(self) -> bool:
        return self.evaluation_status in ("Approved", "Modified")

    @property
    def criteria_list(self) -> List[Dict[str, Any]]:
        """Return parsed criteria breakdown as a list."""
        if isinstance(self.criteria_scores, list):
            return self.criteria_scores
        return []

    @property
    def strengths_list(self) -> List[str]:
        """Return strengths as a list of strings."""
        if isinstance(self.strengths, list):
            return self.strengths
        return []

    @property
    def missing_elements_list(self) -> List[str]:
        """Return missing elements as a list of strings."""
        if isinstance(self.missing_elements, list):
            return self.missing_elements
        return []

    def to_dict(self, include_supportive_answer: bool = False) -> dict:
        """
        Serialize evaluation to dictionary.
        CRITICAL: Reference supportive answers are strictly excluded for student serialization.
        """
        data = {
            "evaluation_id": self.evaluation_id,
            "submission_id": self.submission_id,
            "question_id": self.question_id,
            "question_number": self.question.question_number if self.question else None,
            "question_text": self.question.question_text if self.question else None,
            "extracted_answer": self.extracted_answer,
            "ai_score": self.ai_score,
            "faculty_score": self.faculty_score,
            "final_score": self.final_score,
            "maximum_marks": self.maximum_marks,
            "score_percentage": self.score_percentage,
            "criteria_scores": self.criteria_list,
            "strengths": self.strengths_list,
            "missing_elements": self.missing_elements_list,
            "feedback": self.feedback,
            "faculty_feedback": self.faculty_feedback,
            "final_feedback": self.final_feedback,
            "ai_confidence": round(self.ai_confidence * 100, 1) if self.ai_confidence <= 1.0 else self.ai_confidence,
            "evaluation_status": self.evaluation_status,
            "needs_faculty_review": self.needs_faculty_review,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_supportive_answer and self.question:
            data["supportive_answer"] = self.question.supportive_answer
        return data

    def __repr__(self) -> str:
        q_label = f"Q{self.question.question_number}" if self.question else f"Q_ID:{self.question_id}"
        return (
            f"<AnswerEvaluation {self.evaluation_id}: Sub {self.submission_id} -> {q_label}, "
            f"AI_Score={self.ai_score}/{self.maximum_marks}, Final={self.final_score}, Status='{self.evaluation_status}'>"
        )
