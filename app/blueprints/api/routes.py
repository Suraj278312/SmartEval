"""
API Blueprint: JSON endpoints for health, assignment status, and lightweight REST interactions.
"""

from flask import Blueprint, jsonify, session
from app.models.assignment import Assignment
from app.models.submission import Submission
from app.services.auth_service import AuthService
from app.services.assignment_service import AssignmentService

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "SmartEval Core Foundation",
        "phase": 1,
        "ai_pipeline_attached": False,
    }), 200


@api_bp.route("/assignments/<int:assignment_id>", methods=["GET"])
def get_assignment(assignment_id: int):
    """
    Get assignment JSON.
    Respects student vs faculty data isolation: supportive answers are omitted for non-faculty.
    """
    role = session.get("role")
    user_id = session.get("user_id")

    if role == "faculty":
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=user_id
        ).first()
        if not assignment:
            return jsonify({"error": "Assignment not found or unauthorized"}), 404
        return jsonify(assignment.to_dict(include_supportive_answers=True)), 200

    # Student or public view (only published, NO supportive answers)
    assignment = Assignment.query.filter_by(
        assignment_id=assignment_id, published=True
    ).first()
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404

    return jsonify(assignment.to_dict(include_supportive_answers=False)), 200
