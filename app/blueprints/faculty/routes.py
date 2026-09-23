"""
Faculty Blueprint: Routes for dashboard, assignment management, and submission reviews.
"""

from datetime import datetime, timezone
import os
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
    current_app,
)
from app.extensions import db
from app.models.assignment import Assignment
from app.models.submission import Submission, SubmissionStatus
from app.models.user import Faculty
from app.services.auth_service import role_required, AuthService
from app.services.assignment_service import AssignmentService
from app.services.submission_service import SubmissionService
from app.services.evaluation_service import EvaluationService
from app.services.result_service import ResultService

faculty_bp = Blueprint("faculty", __name__, url_prefix="/faculty")



@faculty_bp.route("/dashboard")
@role_required("faculty")
def dashboard():
    """Faculty dashboard displaying high-level metrics and active assignments."""
    faculty_id = session.get("user_id")
    faculty = db.get_or_404(Faculty, faculty_id)

    assignments = AssignmentService.get_faculty_assignments(faculty_id)
    total_assignments = len(assignments)
    published_count = sum(1 for a in assignments if a.published)
    draft_count = total_assignments - published_count

    # Calculate total submissions received across all faculty assignments
    all_assignment_ids = [a.assignment_id for a in assignments]
    total_submissions = (
        Submission.query.filter(Submission.assignment_id.in_(all_assignment_ids)).count()
        if all_assignment_ids
        else 0
    )
    pending_review = (
        Submission.query.filter(
            Submission.assignment_id.in_(all_assignment_ids),
            Submission.status.in_([
                SubmissionStatus.UPLOADED.value,
                SubmissionStatus.REVIEW_REQUIRED.value,
            ]),
        ).count()
        if all_assignment_ids
        else 0
    )

    recent_submissions = (
        Submission.query.filter(Submission.assignment_id.in_(all_assignment_ids))
        .order_by(Submission.submission_date.desc())
        .limit(5)
        .all()
        if all_assignment_ids
        else []
    )

    return render_template(
        "faculty/dashboard.html",
        faculty=faculty,
        assignments=assignments[:5],
        total_assignments=total_assignments,
        published_count=published_count,
        draft_count=draft_count,
        total_submissions=total_submissions,
        pending_review=pending_review,
        recent_submissions=recent_submissions,
    )


@faculty_bp.route("/assignments")
@role_required("faculty")
def assignment_list():
    """List all assignments owned by the faculty."""
    faculty_id = session.get("user_id")
    assignments = AssignmentService.get_faculty_assignments(faculty_id)
    return render_template("faculty/assignment_list.html", assignments=assignments)


@faculty_bp.route("/assignments/new", methods=["GET", "POST"])
@role_required("faculty")
def create_assignment():
    """Create a new assignment with questions and supportive answers."""
    faculty_id = session.get("user_id")

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        subject = request.form.get("subject", "").strip()
        description = request.form.get("description", "").strip()
        deadline_str = request.form.get("deadline", "").strip()
        action = request.form.get("action", "save_draft")
        published = action == "publish"

        # Parse deadline
        try:
            deadline = datetime.fromisoformat(deadline_str)
        except (ValueError, TypeError):
            flash("Invalid deadline format. Please provide a valid date and time.", "danger")
            return render_template("faculty/assignment_form.html", is_edit=False)

        # Parse questions from form
        # Form arrays: question_text[], maximum_marks[], supportive_answer[], rubric[]
        q_texts = request.form.getlist("question_text[]")
        q_marks = request.form.getlist("maximum_marks[]")
        q_answers = request.form.getlist("supportive_answer[]")
        q_rubrics = request.form.getlist("rubric[]")

        questions_data = []
        for i in range(len(q_texts)):
            if q_texts[i].strip():
                try:
                    marks = float(q_marks[i]) if i < len(q_marks) and q_marks[i] else 10.0
                except ValueError:
                    marks = 10.0

                questions_data.append({
                    "question_text": q_texts[i].strip(),
                    "maximum_marks": marks,
                    "supportive_answer": q_answers[i].strip() if i < len(q_answers) else "",
                    "rubric": q_rubrics[i].strip() if i < len(q_rubrics) else "",
                })

        assignment, error = AssignmentService.create_assignment(
            faculty_id=faculty_id,
            title=title,
            subject=subject,
            description=description,
            deadline=deadline,
            questions_data=questions_data,
            published=published,
        )

        if error:
            flash(error, "danger")
            return render_template(
                "faculty/assignment_form.html",
                is_edit=False,
                form_data=request.form,
                questions_data=questions_data,
            )

        status_msg = "published and open for submissions" if published else "saved as draft"
        flash(f"Assignment '{assignment.title}' was successfully {status_msg}.", "success")
        return redirect(url_for("faculty.assignment_list"))

    return render_template("faculty/assignment_form.html", is_edit=False, assignment=None)


@faculty_bp.route("/assignments/<int:assignment_id>/edit", methods=["GET", "POST"])
@role_required("faculty")
def edit_assignment(assignment_id: int):
    """Edit existing assignment and update questions."""
    faculty_id = session.get("user_id")
    assignment = AssignmentService.get_faculty_assignment_detail(assignment_id, faculty_id)

    if not assignment:
        flash("Assignment not found or unauthorized.", "danger")
        return redirect(url_for("faculty.assignment_list"))

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        subject = request.form.get("subject", "").strip()
        description = request.form.get("description", "").strip()
        deadline_str = request.form.get("deadline", "").strip()
        action = request.form.get("action", "save")
        published = True if action == "publish" else (False if action == "unpublish" else assignment.published)

        try:
            deadline = datetime.fromisoformat(deadline_str)
        except (ValueError, TypeError):
            flash("Invalid deadline date.", "danger")
            return render_template("faculty/assignment_form.html", is_edit=True, assignment=assignment)

        q_texts = request.form.getlist("question_text[]")
        q_marks = request.form.getlist("maximum_marks[]")
        q_answers = request.form.getlist("supportive_answer[]")
        q_rubrics = request.form.getlist("rubric[]")

        questions_data = []
        for i in range(len(q_texts)):
            if q_texts[i].strip():
                try:
                    marks = float(q_marks[i]) if i < len(q_marks) and q_marks[i] else 10.0
                except ValueError:
                    marks = 10.0

                questions_data.append({
                    "question_text": q_texts[i].strip(),
                    "maximum_marks": marks,
                    "supportive_answer": q_answers[i].strip() if i < len(q_answers) else "",
                    "rubric": q_rubrics[i].strip() if i < len(q_rubrics) else "",
                })

        updated, error = AssignmentService.update_assignment(
            assignment_id=assignment_id,
            faculty_id=faculty_id,
            title=title,
            subject=subject,
            description=description,
            deadline=deadline,
            questions_data=questions_data,
            published=published,
        )

        if error:
            flash(error, "danger")
            return render_template("faculty/assignment_form.html", is_edit=True, assignment=assignment)

        flash("Assignment successfully updated.", "success")
        return redirect(url_for("faculty.assignment_list"))

    return render_template("faculty/assignment_form.html", is_edit=True, assignment=assignment)


@faculty_bp.route("/assignments/<int:assignment_id>/toggle-publish", methods=["POST"])
@role_required("faculty")
def toggle_publish(assignment_id: int):
    """Toggle publish/unpublish status for an assignment."""
    faculty_id = session.get("user_id")
    new_state, error = AssignmentService.toggle_publish(assignment_id, faculty_id)

    if error:
        flash(error, "danger")
    else:
        state_text = "published" if new_state else "unpublished (saved as draft)"
        flash(f"Assignment status updated to {state_text}.", "success")

    return redirect(url_for("faculty.assignment_list"))


@faculty_bp.route("/assignments/<int:assignment_id>/delete", methods=["POST"])
@role_required("faculty")
def delete_assignment(assignment_id: int):
    """Delete an assignment."""
    faculty_id = session.get("user_id")
    success, error = AssignmentService.delete_assignment(assignment_id, faculty_id)

    if error:
        flash(error, "danger")
    else:
        flash("Assignment was deleted successfully.", "success")

    return redirect(url_for("faculty.assignment_list"))


@faculty_bp.route("/assignments/<int:assignment_id>/submissions")
@role_required("faculty")
def assignment_submissions(assignment_id: int):
    """View all student submissions for a given assignment."""
    faculty_id = session.get("user_id")
    assignment, submissions, error = SubmissionService.get_assignment_submissions_for_faculty(
        assignment_id, faculty_id
    )

    if error:
        flash(error, "danger")
        return redirect(url_for("faculty.assignment_list"))

    return render_template(
        "faculty/submissions.html",
        assignment=assignment,
        submissions=submissions,
    )


@faculty_bp.route("/submissions/<int:submission_id>/download")
@role_required("faculty")
def download_submission(submission_id: int):
    """Download or view a student's submitted PDF."""
    user = AuthService.get_current_user()
    filepath, original_name, error = SubmissionService.get_submission_file_for_user(
        submission_id, user
    )

    if error:
        flash(error, "danger")
        return redirect(url_for("faculty.dashboard"))

    return send_file(
        filepath,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=original_name,
    )


@faculty_bp.route("/submissions/<int:submission_id>/review")
@role_required("faculty")
def review_submission(submission_id: int):
    """
    Detailed extraction and HTR review page for faculty.
    Presents side-by-side page images, raw recognized text, and confidence ratings.
    """
    faculty_id = session.get("user_id")
    submission, error = SubmissionService.get_submission_for_review(submission_id, faculty_id)

    if error:
        flash(error, "danger")
        return redirect(url_for("faculty.assignment_list"))

    return render_template(
        "faculty/review_submission.html",
        submission=submission,
        assignment=submission.assignment,
        student=submission.student,
        pages=submission.pages,
        matched_answers=submission.matched_answers,
        unmatched_answers=submission.unmatched_answers,
        questions=submission.assignment.questions,
        evaluations=submission.evaluations,
    )


@faculty_bp.route("/submissions/<int:submission_id>/retry-processing", methods=["POST"])
@role_required("faculty")
def retry_processing(submission_id: int):
    """Trigger re-processing of a submission PDF through the HTR pipeline."""
    user = AuthService.get_current_user()
    pages_folder = current_app.config.get("PAGES_FOLDER")

    success, error = SubmissionService.retry_submission_processing(
        submission_id=submission_id,
        user=user,
        pages_folder=pages_folder,
    )

    if not success and error:
        flash(f"Processing failed: {error}", "danger")
    else:
        flash("Document was successfully reprocessed.", "success")

    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/submissions/<int:submission_id>/evaluate", methods=["POST"])
@role_required("faculty")
def evaluate_submission(submission_id: int):
    """Trigger AI semantic evaluation for all questions in a submission."""
    faculty_id = session.get("user_id")
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=submission_id,
        faculty_id=faculty_id,
    )

    if error:
        flash(f"Evaluation encountered an issue: {error}", "danger")
    else:
        flash(f"AI evaluation completed successfully for {len(evaluations)} question(s).", "success")

    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/evaluations/<int:evaluation_id>/approve", methods=["POST"])
@role_required("faculty")
def approve_evaluation(evaluation_id: int):
    """Faculty quick approve: confirms AI evaluation score."""
    faculty_id = session.get("user_id")
    evaluation, error = EvaluationService.approve_evaluation(
        evaluation_id=evaluation_id,
        faculty_id=faculty_id,
    )

    if error:
        flash(f"Failed to approve evaluation: {error}", "danger")
        return redirect(url_for("faculty.assignment_list"))

    flash(f"Evaluation for Q{evaluation.question.question_number} was approved.", "success")
    return redirect(url_for("faculty.review_submission", submission_id=evaluation.submission_id))


@faculty_bp.route("/evaluations/<int:evaluation_id>/modify", methods=["POST"])
@role_required("faculty")
def modify_evaluation(evaluation_id: int):
    """Faculty override: update score and feedback."""
    faculty_id = session.get("user_id")
    score_str = request.form.get("faculty_score", "").strip()
    feedback = request.form.get("faculty_feedback", "").strip()

    try:
        score_val = float(score_str)
    except (ValueError, TypeError):
        flash("Invalid score entered. Please provide a valid numeric score.", "danger")
        return redirect(url_for("faculty.dashboard"))

    evaluation, error = EvaluationService.modify_evaluation(
        evaluation_id=evaluation_id,
        faculty_score=score_val,
        faculty_feedback=feedback if feedback else None,
        faculty_id=faculty_id,
    )

    if error:
        flash(f"Score modification failed: {error}", "danger")
        if evaluation:
            return redirect(url_for("faculty.review_submission", submission_id=evaluation.submission_id))
        return redirect(url_for("faculty.assignment_list"))

    flash(f"Marks and feedback for Q{evaluation.question.question_number} updated to {evaluation.faculty_score}/{evaluation.maximum_marks}.", "success")
    return redirect(url_for("faculty.review_submission", submission_id=evaluation.submission_id))


@faculty_bp.route("/evaluations/<int:evaluation_id>/retry", methods=["POST"])
@role_required("faculty")
def retry_evaluation(evaluation_id: int):
    """Re-evaluate an individual question with AI."""
    faculty_id = session.get("user_id")
    evaluation, error = EvaluationService.re_evaluate_answer(
        evaluation_id=evaluation_id,
        faculty_id=faculty_id,
    )

    if error:
        flash(f"Re-evaluation failed: {error}", "danger")
    else:
        flash(f"Question Q{evaluation.question.question_number} was re-evaluated by AI.", "success")

    if evaluation:
        return redirect(url_for("faculty.review_submission", submission_id=evaluation.submission_id))
    return redirect(url_for("faculty.assignment_list"))


@faculty_bp.route("/pages/<int:page_id>/image/<image_type>")
@role_required("faculty")
def stream_page_image(page_id: int, image_type: str):
    """
    Stream original or preprocessed page image for secure UI rendering.
    image_type: 'original' or 'processed'
    """
    user = AuthService.get_current_user()
    filepath, error = SubmissionService.get_page_image_for_user(page_id, image_type, user)

    if error or not filepath:
        abort(404)

    return send_file(filepath, mimetype="image/png")


# =========================================================================
# Phase 4: Results Dashboard, Publication, Batch Review & Reports
# =========================================================================

@faculty_bp.route("/results")
@role_required("faculty")
def results_dashboard():
    """
    Centralized faculty-facing results dashboard.
    Displays submissions, total marks, percentages, publication status, and filters.
    """
    faculty_id = session.get("user_id")
    faculty = db.get_or_404(Faculty, faculty_id)

    assignment_id_raw = request.args.get("assignment_id")
    assignment_id = int(assignment_id_raw) if assignment_id_raw and assignment_id_raw.isdigit() else None
    status_filter = request.args.get("status")
    search_query = request.args.get("q")
    sort_by = request.args.get("sort", "recent")

    overview = ResultService.get_faculty_results_overview(
        faculty_id=faculty_id,
        assignment_id=assignment_id,
        status_filter=status_filter,
        search_query=search_query,
        sort_by=sort_by,
    )

    return render_template(
        "faculty/results_dashboard.html",
        faculty=faculty,
        submissions=overview["submissions"],
        assignments=overview["assignments"],
        total_submissions=overview["total_submissions"],
        published_count=overview["published_count"],
        pending_review_count=overview["pending_review_count"],
        selected_assignment_id=assignment_id,
        selected_status=status_filter,
        search_query=search_query,
        selected_sort=sort_by,
    )


@faculty_bp.route("/assignments/<int:assignment_id>/results")
@role_required("faculty")
def assignment_results(assignment_id: int):
    """Assignment-specific results view."""
    return redirect(url_for("faculty.results_dashboard", assignment_id=assignment_id))


@faculty_bp.route("/submissions/<int:submission_id>/approve-all", methods=["POST"])
@role_required("faculty")
def approve_all_evaluations(submission_id: int):
    """Approve all question evaluations for a submission in one click."""
    faculty_id = session.get("user_id")
    evaluations, error = EvaluationService.approve_all_evaluations(
        submission_id=submission_id, faculty_id=faculty_id
    )

    if error:
        flash(f"Failed to approve all scores: {error}", "danger")
    else:
        flash(f"Successfully approved AI scores for all {len(evaluations)} question(s).", "success")

    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/evaluations/<int:evaluation_id>/flag-review", methods=["POST"])
@role_required("faculty")
def flag_evaluation_review(evaluation_id: int):
    """Mark a question evaluation as requiring manual human review."""
    faculty_id = session.get("user_id")
    evaluation, error = EvaluationService.flag_for_review(
        evaluation_id=evaluation_id, faculty_id=faculty_id
    )

    if error:
        flash(f"Failed to flag question: {error}", "danger")
        return redirect(url_for("faculty.dashboard"))

    flash(f"Question Q{evaluation.question.question_number} has been flagged for manual review.", "info")
    return redirect(url_for("faculty.review_submission", submission_id=evaluation.submission_id))


@faculty_bp.route("/submissions/<int:submission_id>/publish", methods=["POST"])
@role_required("faculty")
def publish_submission_result(submission_id: int):
    """Publish the finalized result to the student."""
    faculty_id = session.get("user_id")
    result, error = ResultService.publish_result(
        submission_id=submission_id, faculty_id=faculty_id
    )

    if error:
        flash(f"Publication failed: {error}", "danger")
    else:
        flash(f"Result for submission #{submission_id} has been published successfully! The student can now view their scorecard and feedback.", "success")

    next_url = request.form.get("next") or request.referrer
    if next_url and ("results" in next_url or "submissions" in next_url):
        return redirect(next_url)
    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/submissions/<int:submission_id>/unpublish", methods=["POST"])
@role_required("faculty")
def unpublish_submission_result(submission_id: int):
    """Unpublish result, reverting visibility from the student."""
    faculty_id = session.get("user_id")
    result, error = ResultService.unpublish_result(
        submission_id=submission_id, faculty_id=faculty_id
    )

    if error:
        flash(f"Failed to unpublish result: {error}", "danger")
    else:
        flash(f"Result for submission #{submission_id} has been unpublished.", "info")

    next_url = request.form.get("next") or request.referrer
    if next_url and ("results" in next_url or "submissions" in next_url):
        return redirect(next_url)
    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/assignments/<int:assignment_id>/publish-all", methods=["POST"])
@role_required("faculty")
def publish_all_results(assignment_id: int):
    """Bulk publish all evaluated results for an assignment."""
    faculty_id = session.get("user_id")
    count, error = ResultService.publish_all_results_for_assignment(
        assignment_id=assignment_id, faculty_id=faculty_id
    )

    if error:
        flash(f"Bulk publication failed: {error}", "danger")
    else:
        flash(f"Successfully published {count} submission result(s) to students.", "success")

    return redirect(url_for("faculty.assignment_submissions", assignment_id=assignment_id))


@faculty_bp.route("/submissions/<int:submission_id>/summary-feedback", methods=["POST"])
@role_required("faculty")
def save_summary_feedback(submission_id: int):
    """Save overall instructor summary comments on the submission."""
    faculty_id = session.get("user_id")
    feedback = request.form.get("faculty_summary_feedback", "").strip()

    result, error = ResultService.update_faculty_summary_feedback(
        submission_id=submission_id,
        faculty_id=faculty_id,
        summary_feedback=feedback,
    )

    if error:
        flash(f"Failed to update instructor feedback: {error}", "danger")
    else:
        flash("Overall instructor feedback updated successfully.", "success")

    return redirect(url_for("faculty.review_submission", submission_id=submission_id))


@faculty_bp.route("/submissions/<int:submission_id>/report")
@role_required("faculty")
def submission_report(submission_id: int):
    """
    Generate clean, printable, formal grade report for faculty records and exports.
    """
    faculty_id = session.get("user_id")
    report_data, error = ResultService.get_faculty_result_detail(
        submission_id=submission_id, faculty_id=faculty_id
    )

    if error or not report_data:
        flash(error or "Report not available.", "danger")
        return redirect(url_for("faculty.assignment_list"))

    submission = db.session.get(Submission, submission_id)

    return render_template(
        "faculty/submission_report.html",
        data=report_data,
        submission=submission,
        current_time=datetime.now(timezone.utc),
    )



