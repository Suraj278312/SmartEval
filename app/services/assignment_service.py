"""
Assignment management service for SmartEval.
Handles assignment creation, question management, publishing, and student data isolation.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission


class AssignmentService:
    """Service handling assignment operations for Faculty and Students."""

    @staticmethod
    def create_assignment(
        faculty_id: int,
        title: str,
        subject: str,
        description: str,
        deadline: datetime,
        questions_data: List[Dict[str, Any]],
        published: bool = False
    ) -> Tuple[Optional[Assignment], Optional[str]]:
        """Create a new assignment with attached questions."""
        if not title or not subject or not deadline:
            return None, "Title, subject, and deadline are required."

        if not questions_data:
            return None, "An assignment must contain at least one question."

        try:
            assignment = Assignment(
                faculty_id=faculty_id,
                title=title.strip(),
                subject=subject.strip(),
                description=description.strip() if description else "",
                deadline=deadline,
                published=bool(published),
            )
            db.session.add(assignment)
            db.session.flush()  # Generate assignment_id

            for idx, q in enumerate(questions_data, start=1):
                q_text = q.get("question_text", "").strip()
                if not q_text:
                    db.session.rollback()
                    return None, f"Question #{idx} text cannot be empty."

                try:
                    max_marks = float(q.get("maximum_marks", 10.0))
                    if max_marks <= 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    db.session.rollback()
                    return None, f"Question #{idx} must have a valid positive mark value."

                question = Question(
                    assignment_id=assignment.assignment_id,
                    question_number=idx,
                    question_text=q_text,
                    supportive_answer=q.get("supportive_answer", "").strip() or None,
                    maximum_marks=max_marks,
                    rubric=q.get("rubric", "").strip() or None,
                )
                db.session.add(question)

            db.session.commit()
            return assignment, None

        except Exception as e:
            db.session.rollback()
            return None, f"Failed to create assignment: {str(e)}"

    @staticmethod
    def update_assignment(
        assignment_id: int,
        faculty_id: int,
        title: str,
        subject: str,
        description: str,
        deadline: datetime,
        questions_data: List[Dict[str, Any]],
        published: Optional[bool] = None
    ) -> Tuple[Optional[Assignment], Optional[str]]:
        """Update existing assignment and synchronize its questions."""
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=faculty_id
        ).first()

        if not assignment:
            return None, "Assignment not found or access unauthorized."

        if not title or not subject or not deadline:
            return None, "Title, subject, and deadline are required."

        if not questions_data:
            return None, "An assignment must contain at least one question."

        try:
            assignment.title = title.strip()
            assignment.subject = subject.strip()
            assignment.description = description.strip() if description else ""
            assignment.deadline = deadline
            if published is not None:
                assignment.published = bool(published)

            # Clear existing questions and rebuild with new order & data
            Question.query.filter_by(assignment_id=assignment.assignment_id).delete()
            db.session.flush()

            for idx, q in enumerate(questions_data, start=1):
                q_text = q.get("question_text", "").strip()
                if not q_text:
                    db.session.rollback()
                    return None, f"Question #{idx} text cannot be empty."

                try:
                    max_marks = float(q.get("maximum_marks", 10.0))
                    if max_marks <= 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    db.session.rollback()
                    return None, f"Question #{idx} must have a valid positive mark value."

                question = Question(
                    assignment_id=assignment.assignment_id,
                    question_number=idx,
                    question_text=q_text,
                    supportive_answer=q.get("supportive_answer", "").strip() or None,
                    maximum_marks=max_marks,
                    rubric=q.get("rubric", "").strip() or None,
                )
                db.session.add(question)

            db.session.commit()
            return assignment, None

        except Exception as e:
            db.session.rollback()
            return None, f"Failed to update assignment: {str(e)}"

    @staticmethod
    def toggle_publish(assignment_id: int, faculty_id: int) -> Tuple[bool, Optional[str]]:
        """Toggle published state of an assignment."""
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=faculty_id
        ).first()

        if not assignment:
            return False, "Assignment not found or unauthorized."

        try:
            assignment.published = not assignment.published
            db.session.commit()
            return assignment.published, None
        except Exception as e:
            db.session.rollback()
            return False, f"Failed to update publish state: {str(e)}"

    @staticmethod
    def delete_assignment(assignment_id: int, faculty_id: int) -> Tuple[bool, Optional[str]]:
        """Delete an assignment owned by the faculty."""
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=faculty_id
        ).first()

        if not assignment:
            return False, "Assignment not found or unauthorized."

        try:
            db.session.delete(assignment)
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, f"Failed to delete assignment: {str(e)}"

    @staticmethod
    def get_faculty_assignments(faculty_id: int) -> List[Assignment]:
        """Fetch all assignments created by a specific faculty member."""
        return (
            Assignment.query.filter_by(faculty_id=faculty_id)
            .order_by(Assignment.created_at.desc())
            .all()
        )

    @staticmethod
    def get_faculty_assignment_detail(
        assignment_id: int, faculty_id: int
    ) -> Optional[Assignment]:
        """Fetch full assignment details for faculty (includes supportive answers)."""
        return Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=faculty_id
        ).first()

    @staticmethod
    def get_student_available_assignments(student_id: int) -> List[Dict[str, Any]]:
        """
        Fetch all published assignments for students along with student submission status.
        Excludes reference answers.
        """
        assignments = (
            Assignment.query.filter_by(published=True)
            .order_by(Assignment.deadline.asc())
            .all()
        )

        # Get all submissions for this student indexed by assignment_id
        submissions = {
            s.assignment_id: s
            for s in Submission.query.filter_by(student_id=student_id).all()
        }

        results = []
        for a in assignments:
            submission = submissions.get(a.assignment_id)
            results.append({
                "assignment": a,
                "submission": submission,
                "has_submitted": submission is not None,
                "submission_status": submission.status if submission else None,
                "is_overdue": a.is_past_deadline and not submission,
            })
        return results

    @staticmethod
    def get_student_assignment_detail(
        assignment_id: int, student_id: int
    ) -> Tuple[Optional[Assignment], Optional[Submission], Optional[str]]:
        """
        Fetch assignment details for student.
        Guarantees assignment is published and supportive answers are not exposed in queries.
        """
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, published=True
        ).first()

        if not assignment:
            return None, None, "Assignment not found or is currently unpublished."

        submission = Submission.query.filter_by(
            assignment_id=assignment_id, student_id=student_id
        ).first()

        return assignment, submission, None
