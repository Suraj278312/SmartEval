"""
Comprehensive test suite for Phase 4: Final Results, Faculty Approval, Student Feedback & Reports.
Tests score approval, score modification, bounds validation, final total/percentage calculation,
AI score preservation, student publication isolation, security authorization, report generation,
and edge case handling.
"""

import os
from datetime import datetime, timezone, timedelta
import pytest
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.submission_result import SubmissionResult
from app.models.user import Faculty, Student
from app.services.evaluation_service import EvaluationService
from app.services.result_service import ResultService
from tests.conftest import login_session


@pytest.fixture
def setup_evaluated_submission(app, sample_faculty, sample_student, sample_assignment):
    """
    Create a submission with 2 evaluated questions.
    Question 1: Max 10.0 marks, AI score 8.0
    Question 2: Max 15.0 marks, AI score 12.0
    Total AI Score = 20.0 / 25.0 (80.0%)
    """
    sub = Submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file_path="/tmp/fake_sub.pdf",
        original_filename="midterm_sol.pdf",
        file_size_bytes=1024,
        status=SubmissionStatus.COMPLETED.value,
    )
    db.session.add(sub)
    db.session.flush()

    q1, q2 = sample_assignment.questions[0], sample_assignment.questions[1]

    ev1 = AnswerEvaluation(
        submission_id=sub.submission_id,
        question_id=q1.question_id,
        extracted_answer="T(n) = 2T(n/2) + O(n). By Master theorem Case 2, T(n) = Theta(n log n).",
        ai_score=8.0,
        maximum_marks=10.0,
        criteria_scores=[
            {"name": "Formulation", "max_marks": 5.0, "awarded_marks": 4.5, "status": "full", "reason": "Correct parameters"},
            {"name": "Final Bound", "max_marks": 5.0, "awarded_marks": 3.5, "status": "partial", "reason": "Minor notation omission"}
        ],
        strengths=["Accurate Master Theorem identification"],
        missing_elements=["Base case boundary check"],
        feedback="Good mathematical derivation.",
        ai_confidence=0.9,
        evaluation_status="Evaluated",
        needs_faculty_review=False,
    )

    ev2 = AnswerEvaluation(
        submission_id=sub.submission_id,
        question_id=q2.question_id,
        extracted_answer="QuickSort worst case happens when array is sorted, leading to O(n^2) recursion depth.",
        ai_score=12.0,
        maximum_marks=15.0,
        criteria_scores=[
            {"name": "Pivot Explanation", "max_marks": 8.0, "awarded_marks": 7.0, "status": "full", "reason": "Clear explanation of poor pivot"},
            {"name": "Recursion Depth", "max_marks": 7.0, "awarded_marks": 5.0, "status": "partial", "reason": "Recursion tree could be more detailed"}
        ],
        strengths=["Clear worst-case pivot explanation"],
        missing_elements=["Diagrammatic recursion tree"],
        feedback="Solid conceptual understanding.",
        ai_confidence=0.88,
        evaluation_status="Evaluated",
        needs_faculty_review=False,
    )

    db.session.add_all([ev1, ev2])
    db.session.commit()
    db.session.refresh(sub)
    return sub, ev1, ev2


# =========================================================================
# 1. Faculty Approval & Score Modification Tests
# =========================================================================

def test_faculty_can_approve_ai_score(app, sample_faculty, setup_evaluated_submission):
    """Test 1: Faculty can explicitly approve AI score, which sets faculty_score to ai_score."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    approved_ev, err = EvaluationService.approve_evaluation(
        evaluation_id=ev1.evaluation_id, faculty_id=sample_faculty.faculty_id
    )
    assert err is None
    assert approved_ev is not None
    assert approved_ev.evaluation_status == "Approved"
    assert approved_ev.faculty_score == 8.0
    assert approved_ev.final_score == 8.0
    assert approved_ev.needs_faculty_review is False


def test_faculty_can_modify_score_and_feedback(app, sample_faculty, setup_evaluated_submission):
    """Test 2 & 3: Faculty can override score and feedback; modified score becomes final score."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    modified_ev, err = EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=9.5,
        faculty_feedback="Instructor bump for clean handwriting and rigorous proof.",
        faculty_id=sample_faculty.faculty_id,
    )
    assert err is None
    assert modified_ev.evaluation_status == "Modified"
    assert modified_ev.faculty_score == 9.5
    assert modified_ev.final_score == 9.5
    assert modified_ev.final_feedback == "Instructor bump for clean handwriting and rigorous proof."


def test_score_validation_bounds_cannot_exceed_max_marks(app, sample_faculty, setup_evaluated_submission):
    """Test 4: Score validation rejects faculty scores exceeding question maximum marks."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    # ev1 has maximum_marks = 10.0, attempt to award 11.5
    modified_ev, err = EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=11.5,
        faculty_feedback="Too generous",
        faculty_id=sample_faculty.faculty_id,
    )
    assert modified_ev is None
    assert "between 0.0 and maximum marks" in err
    
    # Confirm DB record was not modified
    db_ev = db.session.get(AnswerEvaluation, ev1.evaluation_id)
    assert db_ev.faculty_score is None


def test_score_validation_bounds_cannot_be_negative(app, sample_faculty, setup_evaluated_submission):
    """Test 5: Score validation rejects negative marks."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    modified_ev, err = EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=-2.0,
        faculty_feedback="Penalty",
        faculty_id=sample_faculty.faculty_id,
    )
    assert modified_ev is None
    assert "between 0.0 and maximum marks" in err


def test_ai_score_preserved_after_modification(app, sample_faculty, setup_evaluated_submission):
    """Test 8: Original AI score remains preserved in the database for auditing after faculty modification."""
    sub, ev1, ev2 = setup_evaluated_submission
    original_ai_score = ev1.ai_score
    
    EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=4.0,
        faculty_feedback="Deducted for missing step",
        faculty_id=sample_faculty.faculty_id,
    )
    
    db_ev = db.session.get(AnswerEvaluation, ev1.evaluation_id)
    assert db_ev.faculty_score == 4.0
    assert db_ev.ai_score == original_ai_score  # AI score intact


# =========================================================================
# 2. Final Result Calculation & Grounded Feedback Tests
# =========================================================================

def test_total_marks_and_percentage_calculation(app, sample_faculty, setup_evaluated_submission):
    """Test 6 & 7: Total maximum marks, total obtained marks, and percentage calculate correctly."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    # Modify ev1 (9.0 / 10.0), Approve ev2 (12.0 / 15.0)
    EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=9.0,
        faculty_feedback="Good",
        faculty_id=sample_faculty.faculty_id,
    )
    EvaluationService.approve_evaluation(
        evaluation_id=ev2.evaluation_id,
        faculty_id=sample_faculty.faculty_id,
    )
    
    result, err = ResultService.calculate_submission_result(
        submission_id=sub.submission_id, faculty_id=sample_faculty.faculty_id
    )
    assert err is None
    assert result.total_maximum_marks == 25.0
    assert result.total_obtained_marks == 21.0  # 9.0 + 12.0
    assert result.percentage == 84.0  # (21 / 25) * 100
    assert result.grade_letter == "A"


def test_grounded_overall_feedback_generation(app, sample_faculty, setup_evaluated_submission):
    """Test overall feedback generation is strictly grounded in question evaluations without external AI."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    result, err = ResultService.calculate_submission_result(
        submission_id=sub.submission_id, faculty_id=sample_faculty.faculty_id
    )
    assert err is None
    assert len(result.strengths_list) > 0
    assert "Accurate Master Theorem identification" in result.strengths_list
    assert len(result.improvements_list) > 0
    assert "Base case boundary check" in result.improvements_list
    assert "80.0%" in result.overall_feedback


# =========================================================================
# 3. Student Publication & Security Isolation Tests
# =========================================================================

def test_student_cannot_view_unpublished_result(client, sample_student, setup_evaluated_submission):
    """Test 9: Student cannot access unpublished evaluation result."""
    sub, ev1, ev2 = setup_evaluated_submission
    login_session(client, user_id=sample_student.student_id, role="student")
    
    response = client.get(f"/student/submissions/{sub.submission_id}/result", follow_redirects=True)
    # Should redirect with message that result is not published
    assert b"Results have not been published" in response.data or response.status_code == 302


def test_student_can_view_published_result(client, sample_faculty, sample_student, setup_evaluated_submission):
    """Test 10: Student can view evaluation results once faculty publishes them."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    # Faculty approves and publishes
    EvaluationService.approve_all_evaluations(sub.submission_id, sample_faculty.faculty_id)
    ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)
    
    login_session(client, user_id=sample_student.student_id, role="student")
    response = client.get(f"/student/submissions/{sub.submission_id}/result")
    assert response.status_code == 200
    assert b"Official Published Result" in response.data
    assert b"20.0" in response.data  # Total marks
    assert b"80.0%" in response.data


def test_student_cannot_access_other_student_result(client, sample_faculty, sample_student, setup_evaluated_submission):
    """Test 11: Student A cannot access Student B's published result."""
    sub, ev1, ev2 = setup_evaluated_submission
    ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)
    
    # Create another student
    other_student = Student(
        name="Grace Hopper",
        email="other_student@smarteval.edu",
        class_name="CS-2026-B",
    )
    other_student.set_password("Student@123")
    db.session.add(other_student)
    db.session.commit()
    
    login_session(client, user_id=other_student.student_id, role="student")
    response = client.get(f"/student/submissions/{sub.submission_id}/result", follow_redirects=True)
    assert b"Unauthorized" in response.data or response.status_code in (302, 403, 404)


def test_student_never_receives_supportive_reference_answers(client, sample_faculty, sample_student, setup_evaluated_submission):
    """Test 12: Student result view strictly omits teacher supportive / reference answers."""
    sub, ev1, ev2 = setup_evaluated_submission
    ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)
    
    login_session(client, user_id=sample_student.student_id, role="student")
    response = client.get(f"/student/submissions/{sub.submission_id}/result")
    assert response.status_code == 200
    # Reference answer text from conftest: "Case 2 applies: a=2, b=2"
    assert b"Case 2 applies: a=2, b=2" not in response.data


def test_faculty_cannot_modify_other_faculty_evaluations(app, sample_faculty, setup_evaluated_submission):
    """Test 13: Faculty member B cannot approve or modify evaluations belonging to Faculty member A."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    # Create other faculty
    other_faculty = Faculty(
        name="Prof. John von Neumann",
        email="other_faculty@smarteval.edu",
        department="Mathematics",
    )
    other_faculty.set_password("Faculty@123")
    db.session.add(other_faculty)
    db.session.commit()
    
    approved_ev, err = EvaluationService.approve_evaluation(
        evaluation_id=ev1.evaluation_id, faculty_id=other_faculty.faculty_id
    )
    assert approved_ev is None
    assert "Unauthorized" in err


# =========================================================================
# 4. Publication Lifecycle & Report Generation Tests
# =========================================================================

def test_publication_and_unpublication_workflow(app, sample_faculty, setup_evaluated_submission):
    """Test 14: Publication lifecycle (publish, check status, unpublish)."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    # 1. Publish
    res, err = ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)
    assert err is None
    assert res.is_published is True
    assert sub.is_published is True
    assert sub.status == SubmissionStatus.PUBLISHED.value
    
    # 2. Unpublish
    res_unpub, err2 = ResultService.unpublish_result(sub.submission_id, sample_faculty.faculty_id)
    assert err2 is None
    assert res_unpub.is_published is False
    assert sub.is_published is False


def test_faculty_submission_report_endpoint(client, sample_faculty, setup_evaluated_submission):
    """Test 15: Printable report generates correct final marks and academic formatting."""
    sub, ev1, ev2 = setup_evaluated_submission
    EvaluationService.modify_evaluation(
        evaluation_id=ev1.evaluation_id,
        faculty_score=8.5,
        faculty_feedback="Well presented.",
        faculty_id=sample_faculty.faculty_id,
    )
    EvaluationService.approve_evaluation(
        evaluation_id=ev2.evaluation_id,
        faculty_id=sample_faculty.faculty_id,
    )
    ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)
    
    login_session(client, user_id=sample_faculty.faculty_id, role="faculty")
    response = client.get(f"/faculty/submissions/{sub.submission_id}/report")
    assert response.status_code == 200
    assert b"Official Student Grade Report" in response.data
    assert b"20.5" in response.data  # 8.5 + 12.0
    assert b"82.0%" in response.data


# =========================================================================
# 5. Edge Cases Tests
# =========================================================================

def test_edge_cases_zero_max_marks_and_empty_assignment(app, sample_faculty, sample_student):
    """Test 16: Zero max marks and edge case divisions do not crash."""
    # Assignment with 0 maximum marks question
    now = datetime.now(timezone.utc)
    empty_assignment = Assignment(
        faculty_id=sample_faculty.faculty_id,
        title="Zero Marks Test",
        subject="TEST101",
        deadline=now + timedelta(days=2),
        published=True,
    )
    db.session.add(empty_assignment)
    db.session.commit()
    
    sub = Submission(
        assignment_id=empty_assignment.assignment_id,
        student_id=sample_student.student_id,
        file_path="/tmp/test_empty.pdf",
        original_filename="test_empty.pdf",
        status=SubmissionStatus.COMPLETED.value,
    )
    db.session.add(sub)
    db.session.commit()
    
    result, err = ResultService.calculate_submission_result(sub.submission_id, sample_faculty.faculty_id)
    assert err is None
    assert result.total_maximum_marks == 0.0
    assert result.total_obtained_marks == 0.0
    assert result.percentage == 0.0
    assert result.grade_letter == "F"


def test_bulk_approve_all_evaluations(app, sample_faculty, setup_evaluated_submission):
    """Test bulk approving all question evaluations for a submission in one click."""
    sub, ev1, ev2 = setup_evaluated_submission
    
    approved_list, err = EvaluationService.approve_all_evaluations(sub.submission_id, sample_faculty.faculty_id)
    assert err is None
    assert len(approved_list) == 2
    assert all(e.is_approved for e in approved_list)
    assert sub.is_reviewed is True
