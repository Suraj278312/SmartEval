"""
Unit and integration tests for Assignment Management & Question Isolation.
"""

from datetime import datetime, timedelta, timezone
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.services.assignment_service import AssignmentService
from tests.conftest import login_session


def test_create_assignment_with_questions(app, sample_faculty):
    """Test creating an assignment with multi-part questions and supportive answers."""
    with app.app_context():
        deadline = datetime.now(timezone.utc) + timedelta(days=7)
        questions_data = [
            {
                "question_text": "Solve differential equation y' + 2y = 0.",
                "maximum_marks": 10.0,
                "supportive_answer": "General solution is y(t) = C * e^(-2t).",
                "rubric": "5 pts integrating factor, 5 pts final solution",
            },
            {
                "question_text": "Explain eigenvalue decomposition.",
                "maximum_marks": 15.0,
                "supportive_answer": "A = V * Lambda * V^(-1) where V contains eigenvectors.",
                "rubric": "10 pts formulation, 5 pts conditions",
            },
        ]

        assignment, error = AssignmentService.create_assignment(
            faculty_id=sample_faculty.faculty_id,
            title="Calculus & Linear Algebra Quiz",
            subject="MATH201",
            description="Show all step-by-step working.",
            deadline=deadline,
            questions_data=questions_data,
            published=True,
        )

        assert error is None
        assert assignment is not None
        assert assignment.title == "Calculus & Linear Algebra Quiz"
        assert assignment.total_marks == 25.0
        assert assignment.question_count == 2
        assert len(assignment.questions) == 2
        assert assignment.questions[0].supportive_answer == "General solution is y(t) = C * e^(-2t)."


def test_toggle_publish_status(app, sample_faculty, sample_assignment):
    """Test toggling an assignment between published and draft states."""
    with app.app_context():
        # Initially published
        assert sample_assignment.published is True

        # Toggle to draft
        new_state, error = AssignmentService.toggle_publish(
            sample_assignment.assignment_id, sample_faculty.faculty_id
        )
        assert error is None
        assert new_state is False

        # Verify DB updated
        a = db.session.get(Assignment, sample_assignment.assignment_id)
        assert a.published is False

        # Toggle back to published
        new_state, error = AssignmentService.toggle_publish(
            sample_assignment.assignment_id, sample_faculty.faculty_id
        )
        assert error is None
        assert new_state is True


def test_edit_assignment(app, sample_faculty, sample_assignment):
    """Test updating assignment metadata and questions."""
    with app.app_context():
        new_deadline = datetime.now(timezone.utc) + timedelta(days=14)
        updated_questions = [
            {
                "question_text": "Updated Question 1",
                "maximum_marks": 20.0,
                "supportive_answer": "Updated Ref 1",
                "rubric": "Updated Rubric 1",
            }
        ]

        updated, error = AssignmentService.update_assignment(
            assignment_id=sample_assignment.assignment_id,
            faculty_id=sample_faculty.faculty_id,
            title="Updated Title",
            subject="CS301-Advanced",
            description="Updated description",
            deadline=new_deadline,
            questions_data=updated_questions,
            published=True,
        )

        assert error is None
        assert updated.title == "Updated Title"
        assert updated.total_marks == 20.0
        assert updated.question_count == 1


def test_delete_assignment_cascades_questions(app, sample_faculty, sample_assignment):
    """Test deleting an assignment cascades and deletes associated questions."""
    with app.app_context():
        assignment_id = sample_assignment.assignment_id
        q_count_before = Question.query.filter_by(assignment_id=assignment_id).count()
        assert q_count_before == 2

        success, error = AssignmentService.delete_assignment(
            assignment_id, sample_faculty.faculty_id
        )
        assert success is True
        assert error is None

        # Verify assignment and its questions are gone
        assert db.session.get(Assignment, assignment_id) is None
        assert Question.query.filter_by(assignment_id=assignment_id).count() == 0


def test_student_cannot_access_unpublished_draft(client, sample_student, sample_faculty):
    """Test that students cannot access unpublished draft assignments."""
    with client.application.app_context():
        draft = Assignment(
            faculty_id=sample_faculty.faculty_id,
            title="Confidential Draft",
            subject="CS100",
            deadline=datetime.now(timezone.utc) + timedelta(days=5),
            published=False,  # DRAFT
        )
        db.session.add(draft)
        db.session.commit()
        draft_id = draft.assignment_id

    login_session(client, sample_student.student_id, "student", sample_student.name)
    resp = client.get(f"/student/assignments/{draft_id}", follow_redirects=True)
    assert b"Assignment not found or is currently unpublished" in resp.data
