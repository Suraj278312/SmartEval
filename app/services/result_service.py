"""
Result Service for SmartEval Phase 4.
Handles final score aggregation, percentage calculation, grounded feedback synthesis,
result publication/unpublication workflows, student scorecard generation, and faculty reports.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any, Union
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.evaluation import AnswerEvaluation
from app.models.submission_result import SubmissionResult
from app.models.user import Faculty, Student

logger = logging.getLogger(__name__)


class ResultService:
    """Service orchestrating final score calculations, publication, grounded feedback, and reports."""

    @classmethod
    def calculate_submission_result(
        cls,
        submission_id: int,
        faculty_id: Optional[int] = None,
    ) -> Tuple[Optional[SubmissionResult], Optional[str]]:
        """
        Calculate and persist the final result for a submission.
        Derives totals strictly from question maximum marks and faculty-approved/overridden scores.
        Grounded overall feedback is generated deterministically from question evaluation details.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if faculty_id is not None:
            if not submission.assignment or submission.assignment.faculty_id != faculty_id:
                return None, "Unauthorized: You do not own the assignment for this submission."

        assignment = submission.assignment
        evaluations: List[AnswerEvaluation] = (
            AnswerEvaluation.query.filter_by(submission_id=submission_id)
            .join(Question, AnswerEvaluation.question_id == Question.question_id)
            .order_by(Question.question_number)
            .all()
        )

        # 1. Total Maximum Marks Calculation
        if assignment and assignment.questions:
            total_max = sum(q.maximum_marks for q in assignment.questions if q.maximum_marks)
        elif evaluations:
            total_max = sum(e.maximum_marks for e in evaluations if e.maximum_marks)
        else:
            total_max = 0.0

        # 2. Total Obtained Marks Calculation (derived from approved/overridden final_score)
        valid_scores = [e.final_score for e in evaluations if e.final_score is not None] if evaluations else []
        total_obtained = sum(valid_scores) if valid_scores else 0.0
        total_obtained = max(0.0, min(total_obtained, total_max if total_max > 0 else total_obtained))
        total_obtained = round(total_obtained, 2)

        # 3. Percentage Calculation
        if total_max > 0:
            percentage = round((total_obtained / total_max) * 100.0, 2)
        else:
            percentage = 0.0

        grade_letter = SubmissionResult.calculate_grade_letter(percentage)

        # 4. Grounded Overall Feedback Generation
        overall_strengths = cls._compile_grounded_strengths(evaluations)
        overall_improvements = cls._compile_grounded_improvements(evaluations)
        overall_feedback = cls._synthesize_grounded_feedback(
            evaluations=evaluations,
            total_obtained=total_obtained,
            total_max=total_max,
            percentage=percentage,
            strengths=overall_strengths,
            improvements=overall_improvements,
        )

        # 5. Create or Update SubmissionResult record
        result = SubmissionResult.query.filter_by(submission_id=submission_id).first()
        is_all_reviewed = (
            len(evaluations) > 0 and all(e.is_reviewed for e in evaluations)
        )
        now = datetime.now(timezone.utc)

        try:
            if result:
                result.total_maximum_marks = total_max
                result.total_obtained_marks = total_obtained
                result.percentage = percentage
                result.grade_letter = grade_letter
                result.overall_strengths = overall_strengths
                result.overall_improvements = overall_improvements
                result.overall_feedback = overall_feedback
                if is_all_reviewed and not result.faculty_reviewed_at:
                    result.faculty_reviewed_at = now
            else:
                result = SubmissionResult(
                    submission_id=submission_id,
                    total_maximum_marks=total_max,
                    total_obtained_marks=total_obtained,
                    percentage=percentage,
                    grade_letter=grade_letter,
                    overall_strengths=overall_strengths,
                    overall_improvements=overall_improvements,
                    overall_feedback=overall_feedback,
                    is_published=False,
                    faculty_reviewed_at=now if is_all_reviewed else None,
                )
                db.session.add(result)

            # Update submission review state if all questions are reviewed
            if is_all_reviewed and submission.status not in (
                SubmissionStatus.PUBLISHED.value,
                SubmissionStatus.FAILED.value,
            ):
                submission.status = SubmissionStatus.FACULTY_REVIEWED.value

            db.session.commit()
            return result, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to calculate submission result {submission_id}: {str(e)}", exc_info=True)
            return None, f"Calculation failed: {str(e)}"

    @classmethod
    def publish_result(
        cls, submission_id: int, faculty_id: int
    ) -> Tuple[Optional[SubmissionResult], Optional[str]]:
        """
        Publish the final evaluation result for a student submission.
        Enforces faculty ownership and recalculates the final result before publishing.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this submission."

        if not submission.evaluations:
            return None, "Cannot publish: Submission has no evaluations to publish."

        result, err = cls.calculate_submission_result(submission_id=submission_id, faculty_id=faculty_id)
        if err or not result:
            return None, err or "Failed to calculate result before publishing."

        try:
            result.is_published = True
            result.published_at = datetime.now(timezone.utc)
            if not result.faculty_reviewed_at:
                result.faculty_reviewed_at = datetime.now(timezone.utc)

            submission.status = SubmissionStatus.PUBLISHED.value
            db.session.commit()
            return result, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to publish result for submission {submission_id}: {str(e)}", exc_info=True)
            return None, f"Publication failed: {str(e)}"

    @classmethod
    def unpublish_result(
        cls, submission_id: int, faculty_id: int
    ) -> Tuple[Optional[SubmissionResult], Optional[str]]:
        """
        Unpublish a previously published result, reverting visibility from the student.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this submission."

        result = submission.result
        if not result:
            return None, "No result record exists for this submission."

        try:
            result.is_published = False
            submission.status = SubmissionStatus.FACULTY_REVIEWED.value
            db.session.commit()
            return result, None

        except Exception as e:
            db.session.rollback()
            return None, f"Unpublish failed: {str(e)}"

    @classmethod
    def publish_all_results_for_assignment(
        cls, assignment_id: int, faculty_id: int
    ) -> Tuple[int, Optional[str]]:
        """
        Bulk publish all evaluated submissions for a faculty assignment.
        Returns (count_published, error_message).
        """
        assignment = db.session.get(Assignment, assignment_id)
        if not assignment or assignment.faculty_id != faculty_id:
            return 0, "Unauthorized: Assignment not found or access denied."

        submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
        published_count = 0

        for sub in submissions:
            if sub.evaluations:
                res, err = cls.publish_result(submission_id=sub.submission_id, faculty_id=faculty_id)
                if res and not err:
                    published_count += 1

        return published_count, None

    @classmethod
    def get_student_result(
        cls, submission_id: int, student_id: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Retrieve sanitized result scorecard for student view.
        Enforces strict student ownership and publication check.
        CRITICAL: Never leaks supportive/reference answers, AI prompts, or internal logs.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if submission.student_id != student_id:
            return None, "Unauthorized: You may only access your own submission results."

        if not submission.is_published:
            return None, "Results have not been published by your instructor yet."

        result = submission.result
        if not result:
            # Fallback calculation if result record wasn't pre-saved
            result, _ = cls.calculate_submission_result(submission_id)

        assignment = submission.assignment
        evaluations: List[AnswerEvaluation] = (
            AnswerEvaluation.query.filter_by(submission_id=submission_id)
            .join(Question, AnswerEvaluation.question_id == Question.question_id)
            .order_by(Question.question_number)
            .all()
        )

        question_breakdown = []
        for ev in evaluations:
            q = ev.question
            question_breakdown.append({
                "question_number": q.question_number if q else 1,
                "question_text": q.question_text if q else "",
                "maximum_marks": ev.maximum_marks,
                "final_score": ev.final_score,
                "score_percentage": ev.score_percentage,
                "extracted_answer": ev.extracted_answer,
                "feedback": ev.final_feedback,
                "strengths": ev.strengths_list,
                "missing_elements": ev.missing_elements_list,
                "criteria_scores": ev.criteria_list,
                # STRICT PRIVACY: supportive_answer, prompts, and raw API details are omitted
            })

        data = {
            "submission_id": submission.submission_id,
            "assignment_id": assignment.assignment_id if assignment else None,
            "assignment_title": assignment.title if assignment else "Assignment",
            "subject": assignment.subject if assignment else "",
            "submission_date": submission.submission_date.isoformat() if submission.submission_date else None,
            "published_at": result.published_at.isoformat() if (result and result.published_at) else None,
            "total_maximum_marks": result.total_maximum_marks if result else submission.total_maximum_marks,
            "total_obtained_marks": result.total_obtained_marks if result else submission.total_final_score,
            "percentage": result.percentage if result else submission.percentage,
            "formatted_percentage": result.formatted_percentage if result else f"{submission.percentage:.2f}%",
            "grade_letter": result.grade_letter if result else "N/A",
            "overall_strengths": result.strengths_list if result else [],
            "overall_improvements": result.improvements_list if result else [],
            "overall_feedback": result.overall_feedback if result else "",
            "faculty_summary_feedback": result.faculty_summary_feedback if result else None,
            "questions": question_breakdown,
        }
        return data, None

    @classmethod
    def get_faculty_result_detail(
        cls, submission_id: int, faculty_id: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Retrieve complete result details and metadata for faculty review and report generation.
        Includes AI score vs Final Faculty score comparison and reference answers for faculty.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this submission."

        result = submission.result
        if not result:
            result, _ = cls.calculate_submission_result(submission_id, faculty_id)

        assignment = submission.assignment
        student = submission.student
        evaluations: List[AnswerEvaluation] = (
            AnswerEvaluation.query.filter_by(submission_id=submission_id)
            .join(Question, AnswerEvaluation.question_id == Question.question_id)
            .order_by(Question.question_number)
            .all()
        )

        questions_detail = []
        for ev in evaluations:
            q = ev.question
            questions_detail.append({
                "evaluation_id": ev.evaluation_id,
                "question_number": q.question_number if q else 1,
                "question_text": q.question_text if q else "",
                "supportive_answer": q.supportive_answer if q else "",
                "maximum_marks": ev.maximum_marks,
                "ai_score": ev.ai_score,
                "faculty_score": ev.faculty_score,
                "final_score": ev.final_score,
                "score_percentage": ev.score_percentage,
                "extracted_answer": ev.extracted_answer,
                "ai_feedback": ev.feedback,
                "faculty_feedback": ev.faculty_feedback,
                "final_feedback": ev.final_feedback,
                "evaluation_status": ev.evaluation_status,
                "needs_faculty_review": ev.needs_faculty_review,
                "strengths": ev.strengths_list,
                "missing_elements": ev.missing_elements_list,
                "criteria_scores": ev.criteria_list,
            })

        data = {
            "submission_id": submission.submission_id,
            "original_filename": submission.original_filename,
            "submission_date": submission.submission_date,
            "status": submission.status,
            "is_published": result.is_published if result else False,
            "published_at": result.published_at if result else None,
            "faculty_reviewed_at": result.faculty_reviewed_at if result else None,
            "student": {
                "student_id": student.student_id,
                "name": student.name,
                "email": student.email,
                "class_name": student.class_name,
            } if student else {},
            "assignment": {
                "assignment_id": assignment.assignment_id,
                "title": assignment.title,
                "subject": assignment.subject,
                "total_marks": assignment.total_marks,
                "deadline": assignment.deadline,
            } if assignment else {},
            "result": result.to_dict(include_private_info=True) if result else {},
            "total_maximum_marks": result.total_maximum_marks if result else submission.total_maximum_marks,
            "total_obtained_marks": result.total_obtained_marks if result else submission.total_final_score,
            "percentage": result.percentage if result else submission.percentage,
            "grade_letter": result.grade_letter if result else "N/A",
            "overall_strengths": result.strengths_list if result else [],
            "overall_improvements": result.improvements_list if result else [],
            "overall_feedback": result.overall_feedback if result else "",
            "faculty_summary_feedback": result.faculty_summary_feedback if result else None,
            "questions": questions_detail,
        }
        return data, None

    @classmethod
    def update_faculty_summary_feedback(
        cls, submission_id: int, faculty_id: int, summary_feedback: str
    ) -> Tuple[Optional[SubmissionResult], Optional[str]]:
        """
        Update the overall instructor feedback comment on a submission result.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this submission."

        result = submission.result
        if not result:
            result, _ = cls.calculate_submission_result(submission_id, faculty_id)

        try:
            result.faculty_summary_feedback = summary_feedback.strip() if summary_feedback else None
            db.session.commit()
            return result, None
        except Exception as e:
            db.session.rollback()
            return None, f"Failed to save instructor feedback: {str(e)}"

    @classmethod
    def get_faculty_results_overview(
        cls,
        faculty_id: int,
        assignment_id: Optional[int] = None,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None,
        sort_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query and aggregate submissions and results across faculty assignments with filtering and sorting.
        """
        query = (
            Submission.query.join(Assignment, Submission.assignment_id == Assignment.assignment_id)
            .filter(Assignment.faculty_id == faculty_id)
        )

        if assignment_id:
            query = query.filter(Submission.assignment_id == assignment_id)

        if status_filter:
            if status_filter == "Published":
                query = query.join(SubmissionResult, Submission.submission_id == SubmissionResult.submission_id, isouter=True)
                query = query.filter(
                    (SubmissionResult.is_published.is_(True)) | (Submission.status == SubmissionStatus.PUBLISHED.value)
                )
            elif status_filter == "Unpublished":
                query = query.join(SubmissionResult, Submission.submission_id == SubmissionResult.submission_id, isouter=True)
                query = query.filter(
                    (SubmissionResult.is_published.is_(False)) | (SubmissionResult.is_published.is_(None))
                )
            elif status_filter == "Review Required":
                query = query.filter(Submission.status == SubmissionStatus.REVIEW_REQUIRED.value)
            elif status_filter == "Completed":
                query = query.filter(Submission.status.in_([
                    SubmissionStatus.COMPLETED.value,
                    SubmissionStatus.FACULTY_REVIEWED.value,
                    SubmissionStatus.PUBLISHED.value,
                ]))

        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            query = query.join(Student, Submission.student_id == Student.student_id).filter(
                (Student.name.ilike(term)) | (Student.email.ilike(term)) | (Student.class_name.ilike(term))
            )

        # Sorting
        if sort_by == "student_name":
            query = query.join(Student, Submission.student_id == Student.student_id).order_by(Student.name.asc())
        elif sort_by == "marks_high":
            query = query.order_by(Submission.submission_date.desc())
        else:
            query = query.order_by(Submission.submission_date.desc())

        submissions = query.all()

        # Summary Metrics
        all_faculty_assignments = Assignment.query.filter_by(faculty_id=faculty_id).all()
        all_assignment_ids = [a.assignment_id for a in all_faculty_assignments]
        total_sub_count = (
            Submission.query.filter(Submission.assignment_id.in_(all_assignment_ids)).count()
            if all_assignment_ids
            else 0
        )
        published_count = (
            Submission.query.join(SubmissionResult)
            .filter(
                Submission.assignment_id.in_(all_assignment_ids),
                SubmissionResult.is_published.is_(True),
            )
            .count()
            if all_assignment_ids
            else 0
        )
        pending_review_count = (
            Submission.query.filter(
                Submission.assignment_id.in_(all_assignment_ids),
                Submission.status == SubmissionStatus.REVIEW_REQUIRED.value,
            ).count()
            if all_assignment_ids
            else 0
        )

        return {
            "submissions": submissions,
            "assignments": all_faculty_assignments,
            "total_submissions": total_sub_count,
            "published_count": published_count,
            "pending_review_count": pending_review_count,
            "selected_assignment_id": assignment_id,
            "selected_status": status_filter,
            "selected_sort": sort_by,
        }

    # =========================================================================
    # Internal Helpers for Grounded Overall Feedback
    # =========================================================================

    @staticmethod
    def _compile_grounded_strengths(evaluations: List[AnswerEvaluation]) -> List[str]:
        """Compile a clean, deduplicated list of accurate concepts from evaluations."""
        strengths = []
        seen = set()
        for ev in evaluations:
            for s in ev.strengths_list:
                cleaned = s.strip()
                if cleaned and cleaned.lower() not in seen:
                    strengths.append(cleaned)
                    seen.add(cleaned.lower())
        return strengths[:6]  # Top key strengths

    @staticmethod
    def _compile_grounded_improvements(evaluations: List[AnswerEvaluation]) -> List[str]:
        """Compile a clean, deduplicated list of missing elements and criteria gaps."""
        improvements = []
        seen = set()
        for ev in evaluations:
            for m in ev.missing_elements_list:
                cleaned = m.strip()
                if cleaned and cleaned.lower() not in seen:
                    improvements.append(cleaned)
                    seen.add(cleaned.lower())
            # Also capture reasons from partial or failed criteria
            for c in ev.criteria_list:
                if c.get("status") in ("partial", "none") and c.get("reason"):
                    reason = c["reason"].strip()
                    if reason and reason.lower() not in seen and len(reason) < 140:
                        improvements.append(f"{c.get('name', 'Concept')}: {reason}")
                        seen.add(reason.lower())
        return improvements[:6]  # Top key improvement areas

    @staticmethod
    def _synthesize_grounded_feedback(
        evaluations: List[AnswerEvaluation],
        total_obtained: float,
        total_max: float,
        percentage: float,
        strengths: List[str],
        improvements: List[str],
    ) -> str:
        """
        Synthesize grounded overall evaluation summary strictly from question facts.
        No hallucinated student characteristics.
        """
        if not evaluations:
            return "No evaluations have been recorded for this submission yet."

        parts = []
        # Score Performance Overview
        if percentage >= 85.0:
            parts.append(
                f"Excellent performance across the assignment, scoring {total_obtained}/{total_max} ({percentage}%)."
            )
        elif percentage >= 70.0:
            parts.append(
                f"Solid comprehension demonstrated, achieving {total_obtained}/{total_max} ({percentage}%)."
            )
        elif percentage >= 50.0:
            parts.append(
                f"Satisfactory baseline established with {total_obtained}/{total_max} ({percentage}%)."
            )
        else:
            parts.append(
                f"Significant conceptual gaps identified in this submission ({total_obtained}/{total_max}, {percentage}%)."
            )

        # Highlight Question-Level Patterns
        if strengths:
            top_strengths_str = "; ".join(strengths[:3])
            parts.append(f"Demonstrated proficiency in: {top_strengths_str}.")

        if improvements:
            top_impr_str = "; ".join(improvements[:3])
            parts.append(f"Key areas for further review: {top_impr_str}.")

        return " ".join(parts)
