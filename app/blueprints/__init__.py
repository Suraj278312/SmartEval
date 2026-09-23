"""
Blueprints package for SmartEval.
"""

from app.blueprints.auth import auth_bp
from app.blueprints.faculty import faculty_bp
from app.blueprints.student import student_bp
from app.blueprints.api import api_bp

__all__ = ["auth_bp", "faculty_bp", "student_bp", "api_bp"]
