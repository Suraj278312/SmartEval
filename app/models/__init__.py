"""
Database models package for SmartEval.
Exports Faculty, Student, Assignment, Question, Submission, DocumentPage, and SubmissionStatus.
"""

from app.models.user import Faculty, Student
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.submission_result import SubmissionResult

__all__ = [
    "Faculty",
    "Student",
    "Assignment",
    "Question",
    "Submission",
    "DocumentPage",
    "SubmissionStatus",
    "SubmissionAnswer",
    "AnswerEvaluation",
    "SubmissionResult",
]
