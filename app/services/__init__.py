"""
Services package for SmartEval business logic.
"""

from app.services.auth_service import AuthService, login_required, role_required
from app.services.assignment_service import AssignmentService
from app.services.submission_service import SubmissionService
from app.services.evaluation_service import EvaluationService
from app.services.result_service import ResultService

__all__ = [
    "AuthService",
    "AssignmentService",
    "SubmissionService",
    "EvaluationService",
    "ResultService",
    "login_required",
    "role_required",
]

