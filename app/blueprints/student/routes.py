"""
Student Blueprint: Routes for dashboard, published assignments, question viewer, and PDF submission upload.
"""

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    abort,
    jsonify,
    current_app,
)
from app.extensions import db
from app.models.submission import Submission, SubmissionStatus
from app.models.user import Student
from app.services.auth_service import role_required, AuthService
from app.services.assignment_service import AssignmentService
from app.services.submission_service import SubmissionService
from app.services.result_service import ResultService

student_bp = Blueprint("student", __name__, url_prefix="/student")


@student_bp.route("/dashboard")
@role_required("student")
def dashboard():
    """Student dashboard displaying available assignments and submission history."""
    student_id = session.get("user_id")
    student = db.get_or_404(Student, student_id)

    available_assignments = AssignmentService.get_student_available_assignments(student_id)
    total_available = len(available_assignments)
    submitted_count = sum(1 for item in available_assignments if item["has_submitted"])
    pending_count = total_available - submitted_count

    recent_submissions = SubmissionService.get_student_submissions(student_id)[:5]

    return render_template(
        "student/dashboard.html",
        student=student,
        assignments=available_assignments[:5],
        total_available=total_available,
        submitted_count=submitted_count,
        pending_count=pending_count,
        recent_submissions=recent_submissions,
    )


@student_bp.route("/assignments")
@role_required("student")
def assignment_list():
    """List all published assignments for the student."""
    student_id = session.get("user_id")
    assignments_data = AssignmentService.get_student_available_assignments(student_id)
    return render_template("student/assignment_list.html", assignments_data=assignments_data)


@student_bp.route("/assignments/<int:assignment_id>")
@role_required("student")
def assignment_view(assignment_id: int):
    """
    View assignment questions, deadline, and upload portal.
    CRITICAL: Reference / supportive answers are never shown to the student.
    """
    student_id = session.get("user_id")
    assignment, submission, error = AssignmentService.get_student_assignment_detail(
        assignment_id, student_id
    )

    if error:
        flash(error, "danger")
        return redirect(url_for("student.assignment_list"))

    return render_template(
        "student/assignment_view.html",
        assignment=assignment,
        submission=submission,
        max_file_size_mb=current_app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024),
    )


@student_bp.route("/assignments/<int:assignment_id>/submit-scan", methods=["POST"])
@role_required("student")
def submit_scanned_assignment(assignment_id: int):
    """
    Handle in-app camera scanner submission with multiple captured pages and provenance tracking.
    Server compiles the pages into a single multi-page PDF and initiates the HTR pipeline.
    """
    student_id = session.get("user_id")
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    pages_folder = current_app.config.get("PAGES_FOLDER")
    max_bytes = current_app.config["MAX_CONTENT_LENGTH"]

    pages_data = []
    session_id = None

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        pages_data = payload.get("pages", [])
        session_id = payload.get("session_id")
    else:
        # Check form data
        import json
        raw_pages = request.form.get("pages")
        if raw_pages:
            try:
                pages_data = json.loads(raw_pages)
            except Exception:
                pages_data = []
        session_id = request.form.get("session_id")

    if not pages_data:
        err_msg = "No scanned pages were provided. Please capture at least one page before submitting."
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": err_msg}), 400
        flash(err_msg, "danger")
        return redirect(url_for("student.assignment_view", assignment_id=assignment_id)), 400

    submission, error = SubmissionService.create_or_update_scanned_submission(
        assignment_id=assignment_id,
        student_id=student_id,
        pages_data=pages_data,
        session_id=session_id,
        upload_folder=upload_folder,
        pages_folder=pages_folder,
        max_bytes=max_bytes,
    )

    if error:
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": error}), 400
        flash(error, "danger")
        return redirect(url_for("student.assignment_view", assignment_id=assignment_id))

    redirect_target = url_for("student.submission_view", submission_id=submission.submission_id)
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "success": True,
            "submission_id": submission.submission_id,
            "redirect_url": redirect_target,
            "message": f"Successfully compiled {len(pages_data)} scanned page(s) into assignment submission.",
        }), 200

    flash(
        f"Your handwritten scan ({len(pages_data)} pages) has been successfully submitted and queued for evaluation.",
        "success",
    )
    return redirect(redirect_target)


@student_bp.route("/assignments/<int:assignment_id>/submit", methods=["POST"])
@role_required("student")
def submit_assignment(assignment_id: int):
    """
    Primary submission endpoint for students.
    Direct PDF file uploads are disabled in favor of the in-app camera scanner.
    """
    # Check if a legacy direct file upload was attempted
    if "submission_file" in request.files:
        error_msg = (
            "Direct PDF file upload is disabled for students. "
            "Submissions must be captured and verified using the In-App Camera Scanner."
        )
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "danger")
        return redirect(url_for("student.assignment_view", assignment_id=assignment_id))

    # If payload is scanner data sent to standard submit endpoint, delegate to scanner handler
    if request.is_json or request.form.get("pages"):
        return submit_scanned_assignment(assignment_id)

    # Empty / invalid request
    error_msg = "No scan data received. Please use the In-App Camera Scanner to capture your assignment."
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": False, "error": error_msg}), 400
    flash(error_msg, "danger")
    return redirect(url_for("student.assignment_view", assignment_id=assignment_id))



@student_bp.route("/submissions/<int:submission_id>")
@role_required("student")
def submission_view(submission_id: int):
    """View submission details and status tracker for student."""
    student_id = session.get("user_id")
    submission = Submission.query.filter_by(
        submission_id=submission_id, student_id=student_id
    ).first()

    if not submission:
        flash("Submission not found or unauthorized.", "danger")
        return redirect(url_for("student.dashboard"))

    return render_template("student/submission_view.html", submission=submission)


@student_bp.route("/submissions/<int:submission_id>/download")
@role_required("student")
def download_submission(submission_id: int):
    """Download/view own submitted PDF file."""
    user = AuthService.get_current_user()
    filepath, original_name, error = SubmissionService.get_submission_file_for_user(
        submission_id, user
    )

    if error:
        flash(error, "danger")
        return redirect(url_for("student.dashboard"))

    return send_file(
        filepath,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=original_name,
    )


@student_bp.route("/pages/<int:page_id>/image/<image_type>")
@role_required("student")
def stream_page_image(page_id: int, image_type: str):
    """
    Stream original or preprocessed page image for student's own submission.
    image_type: 'original' or 'processed'
    """
    user = AuthService.get_current_user()
    filepath, error = SubmissionService.get_page_image_for_user(page_id, image_type, user)

    if error or not filepath:
        abort(404)

    return send_file(filepath, mimetype="image/png")


# =========================================================================
# Phase 4: Student Published Result Scorecard View
# =========================================================================

@student_bp.route("/submissions/<int:submission_id>/result")
@role_required("student")
def submission_result(submission_id: int):
    """
    View published final evaluation result and constructive feedback scorecard.
    Strictly blocked if results are unpublished or owned by another student.
    """
    student_id = session.get("user_id")
    submission = db.session.get(Submission, submission_id)

    if not submission:
        flash("Submission record not found.", "danger")
        return redirect(url_for("student.dashboard"))

    if submission.student_id != student_id:
        flash("Unauthorized: You may only access results for your own submissions.", "danger")
        return redirect(url_for("student.dashboard"))

    result_data, error = ResultService.get_student_result(
        submission_id=submission_id, student_id=student_id
    )

    if error or not result_data:
        flash(error or "Results have not been published by your instructor yet.", "info")
        return redirect(url_for("student.submission_view", submission_id=submission_id))

    return render_template(
        "student/result_view.html",
        data=result_data,
        submission=submission,
        assignment=submission.assignment,
    )


