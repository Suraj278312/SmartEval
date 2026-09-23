"""
Assignment and Question models for SmartEval.
Supports multi-question assignments, reference supportive answers, and rubrics.
"""

from datetime import datetime, timezone
from app.extensions import db


class Assignment(db.Model):
    """Assignment created by a faculty member."""
    __tablename__ = "assignment"

    assignment_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    faculty_id = db.Column(
        db.Integer,
        db.ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    deadline = db.Column(db.DateTime, nullable=False)
    published = db.Column(db.Boolean, default=False, nullable=False, index=True)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    questions = db.relationship(
        "Question",
        backref="assignment",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="Question.question_number"
    )
    submissions = db.relationship(
        "Submission",
        backref="assignment",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="desc(Submission.submission_date)"
    )

    @property
    def total_marks(self) -> float:
        """Calculate the sum of maximum marks across all questions."""
        return sum(q.maximum_marks for q in self.questions if q.maximum_marks)

    @property
    def question_count(self) -> int:
        """Get the number of questions in this assignment."""
        return len(self.questions)

    @property
    def is_past_deadline(self) -> bool:
        """Check if the assignment deadline has elapsed."""
        if not self.deadline:
            return False
        # Normalize comparison to naive or timezone-aware
        now = datetime.now(timezone.utc)
        if self.deadline.tzinfo is None:
            return datetime.now(timezone.utc).replace(tzinfo=None) > self.deadline
        return now > self.deadline

    def to_dict(self, include_supportive_answers: bool = False) -> dict:
        """
        Serialize assignment to dictionary.
        Security Note: include_supportive_answers is strictly False for student serialization.
        """
        return {
            "assignment_id": self.assignment_id,
            "faculty_id": self.faculty_id,
            "faculty_name": self.faculty.name if self.faculty else None,
            "title": self.title,
            "subject": self.subject,
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "published": self.published,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "total_marks": self.total_marks,
            "question_count": self.question_count,
            "is_past_deadline": self.is_past_deadline,
            "questions": [
                q.to_dict(include_supportive_answer=include_supportive_answers)
                for q in self.questions
            ],
        }

    def __repr__(self) -> str:
        return f"<Assignment {self.assignment_id}: '{self.title}' (Published={self.published})>"


class Question(db.Model):
    """Question belonging to an Assignment."""
    __tablename__ = "question"

    question_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    assignment_id = db.Column(
        db.Integer,
        db.ForeignKey("assignment.assignment_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    question_number = db.Column(db.Integer, nullable=False, default=1)
    question_text = db.Column(db.Text, nullable=False)
    supportive_answer = db.Column(db.Text, nullable=True)  # Strictly Faculty/AI only
    maximum_marks = db.Column(db.Float, nullable=False, default=10.0)
    rubric = db.Column(db.Text, nullable=True)  # Scoring guidelines

    # Phase 3A: Relationship to student answer mappings
    answers = db.relationship(
        "SubmissionAnswer",
        backref="question",
        lazy=True,
    )

    # Phase 3B: Relationship to AI & faculty evaluations
    evaluations = db.relationship(
        "AnswerEvaluation",
        backref="question",
        cascade="all, delete-orphan",
        lazy=True,
    )

    def to_dict(self, include_supportive_answer: bool = False) -> dict:
        """
        Serialize question to dict.
        NEVER includes supportive_answer when accessed by a student.
        """
        data = {
            "question_id": self.question_id,
            "assignment_id": self.assignment_id,
            "question_number": self.question_number,
            "question_text": self.question_text,
            "maximum_marks": self.maximum_marks,
            "rubric": self.rubric,
        }
        if include_supportive_answer:
            data["supportive_answer"] = self.supportive_answer
        return data

    def __repr__(self) -> str:
        return f"<Question {self.question_id} (#{self.question_number}) for Assignment {self.assignment_id}>"
