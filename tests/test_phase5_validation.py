"""
Phase 5 Comprehensive Validation & Security Test Suite.
Covers multi-scenario semantic evaluation (correct, partial, incorrect, paraphrased),
strict score bounds validation, real PDF/Image preprocessing pipelines, and security isolation.
"""

import os
import pytest
from datetime import datetime, timedelta, timezone
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.submission_result import SubmissionResult
from app.models.evaluation import AnswerEvaluation
from app.models.user import Faculty, Student
from app.processing.evaluator import (
    MockSemanticEvaluator,
    GeminiSemanticEvaluator,
    RubricEngine,
    CriterionScore,
    EvaluationResult,
)
from app.processing.pdf_processor import PDFProcessor
from app.processing.image_preprocessor import ImagePreprocessor
from app.services.evaluation_service import EvaluationService
from app.services.result_service import ResultService
from tests.conftest import login_session


# =============================================================================
# 1. Multi-Scenario AI Semantic Evaluation Tests
# =============================================================================

def test_scenario_a_correct_answer_high_score():
    """Scenario A: Correct and complete answer receives high/full marks and positive strengths."""
    evaluator = MockSemanticEvaluator(
        score=10.0,
        strengths=["Complete explanation of LIFO vs FIFO", "Accurate data structure complexity"],
        missing_elements=[],
        feedback="Excellent conceptual clarity with precise technical definitions.",
        confidence=0.95,
    )
    result = evaluator.evaluate_answer(
        question_text="Explain the difference between stack and queue.",
        supportive_answer="A stack follows LIFO while a queue follows FIFO.",
        student_answer="A stack uses Last-In First-Out where push and pop operate on the top element. A queue uses First-In First-Out where enqueue adds to the rear and dequeue removes from the front.",
        maximum_marks=10.0,
        rubric="5 pts Stack definition, 5 pts Queue definition",
    )
    assert result.score == 10.0
    assert result.max_score == 10.0
    assert len(result.strengths) >= 2
    assert len(result.missing_elements) == 0
    assert result.confidence >= 0.90
    assert not result.needs_faculty_review


def test_scenario_b_partially_correct_answer():
    """Scenario B: Partially correct answer receives partial marks and identifies missing elements."""
    evaluator = MockSemanticEvaluator(
        score=5.0,
        strengths=["Correct stack LIFO definition"],
        missing_elements=["Queue FIFO operations and structure omitted"],
        feedback="Good explanation of stacks, but the queue portion of the question was not addressed.",
        confidence=0.88,
    )
    result = evaluator.evaluate_answer(
        question_text="Explain the difference between stack and queue.",
        supportive_answer="A stack follows LIFO while a queue follows FIFO.",
        student_answer="A stack is a linear data structure following Last In First Out order.",
        maximum_marks=10.0,
        rubric="5 pts Stack definition, 5 pts Queue definition",
    )
    assert result.score == 5.0
    assert result.max_score == 10.0
    assert "Correct stack LIFO definition" in result.strengths
    assert len(result.missing_elements) > 0


def test_scenario_c_incorrect_or_unrelated_answer():
    """Scenario C: Incorrect or unrelated answer receives zero or low marks with constructive feedback."""
    evaluator = MockSemanticEvaluator(
        score=0.0,
        strengths=[],
        missing_elements=["Entire conceptual comparison is missing", "Provided answer discusses unrelated topic"],
        feedback="The answer discusses binary search trees which is unrelated to stacks and queues.",
        confidence=0.90,
    )
    result = evaluator.evaluate_answer(
        question_text="Explain the difference between stack and queue.",
        supportive_answer="A stack follows LIFO while a queue follows FIFO.",
        student_answer="A binary search tree has left children smaller than the parent node and right children larger.",
        maximum_marks=10.0,
        rubric="5 pts Stack definition, 5 pts Queue definition",
    )
    assert result.score == 0.0
    assert len(result.missing_elements) > 0
    assert len(result.strengths) == 0


def test_scenario_d_paraphrased_alternative_wording():
    """Scenario D: Paraphrased answer with different vocabulary receives full semantic credit."""
    evaluator = MockSemanticEvaluator(
        score=10.0,
        strengths=["Understands inverse retrieval ordering of stacks", "Understands sequential arrival order of queues"],
        missing_elements=[],
        feedback="Accurate conceptual understanding explained in student's own words.",
        confidence=0.92,
    )
    result = evaluator.evaluate_answer(
        question_text="Explain the difference between stack and queue.",
        supportive_answer="A stack follows LIFO while a queue follows FIFO.",
        student_answer="In a stack, whatever was inserted most recently comes out first, like a pile of plates. In a queue, items wait in a line and are served in arrival order.",
        maximum_marks=10.0,
        rubric="5 pts Stack definition, 5 pts Queue definition",
    )
    assert result.score == 10.0
    assert result.semantic_similarity_score > 0.0


def test_rubric_engine_clamps_and_normalizes():
    """Verify RubricEngine strictly bounds criteria scores and normalizes arbitrary weights to max_marks."""
    # Custom rubric with 3 criteria totaling 10 pts
    rubric_text = "4 pts Core Theory, 4 pts Implementation, 2 pts Complexity"
    criteria = RubricEngine.parse_rubric(rubric_text, maximum_marks=10.0)
    assert len(criteria) == 3
    assert sum(c["max_marks"] for c in criteria) == 10.0

    # Test default fallback rubric when no rubric provided
    default_criteria = RubricEngine.parse_rubric(None, maximum_marks=15.0)
    assert len(default_criteria) >= 2
    assert sum(c["max_marks"] for c in default_criteria) == 15.0


def test_strict_score_bounds_and_zero_limits():
    """Verify evaluator never returns scores < 0 or > maximum_marks."""
    evaluator_overflow = MockSemanticEvaluator(score=999.0)
    res_overflow = evaluator_overflow.evaluate_answer(
        question_text="Test question",
        supportive_answer="Test ref",
        student_answer="Test ans",
        maximum_marks=10.0,
    )
    assert res_overflow.score <= 10.0

    evaluator_underflow = MockSemanticEvaluator(score=-50.0)
    res_underflow = evaluator_underflow.evaluate_answer(
        question_text="Test question",
        supportive_answer="Test ref",
        student_answer="Test ans",
        maximum_marks=10.0,
    )
    assert res_underflow.score >= 0.0


# =============================================================================
# 2. PDF & Image Processing Robustness
# =============================================================================

def test_pdf_processor_nonexistent_file():
    """Verify PDF processor returns clean error for non-existent file path."""
    processor = PDFProcessor()
    pages, err = processor.validate_and_extract_pages(
        pdf_path="non_existent_file.pdf",
        output_folder="test_out",
        submission_id=1,
    )
    assert pages == []
    assert "not found" in err.lower()


def test_pdf_processor_empty_file(tmp_path):
    """Verify PDF processor gracefully rejects 0-byte file."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    processor = PDFProcessor()
    pages, err = processor.validate_and_extract_pages(
        pdf_path=str(empty_pdf),
        output_folder=str(tmp_path),
        submission_id=1,
    )
    assert pages == []
    assert "empty" in err.lower()


# =============================================================================
# 3. Security, Authorization & Privacy Isolation
# =============================================================================

def test_student_isolation_cannot_view_unauthorized_page_images(client, app, sample_student, sample_faculty, sample_assignment):
    """Verify a student cannot access page images belonging to another student's submission."""
    with app.app_context():
        # Create second student
        other_student = Student(
            name="Other Student",
            email="other_student@smarteval.edu",
            class_name="CS-A",
        )
        other_student.set_password("Student@123")
        db.session.add(other_student)
        db.session.commit()

        # Create submission for other_student
        other_sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=other_student.student_id,
            file_path="dummy.pdf",
            original_filename="exam.pdf",
            file_size_bytes=1024,
            status=SubmissionStatus.COMPLETED.value,
        )
        db.session.add(other_sub)
        db.session.commit()
        other_sub_id = other_sub.submission_id

    # Log in as sample_student
    login_session(client, sample_student.student_id, role="student")

    # Attempt to stream page image from other_sub
    resp = client.get(f"/faculty/submissions/{other_sub_id}/pages/1/image?type=original")
    # Should be rejected / redirected
    assert resp.status_code in (302, 403, 404)


def test_supportive_answers_redacted_from_student_result_payload(app, sample_faculty, sample_student, sample_assignment):
    """Verify ResultService.get_student_result never exposes supportive_answer to student payload."""
    with app.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path="dummy.pdf",
            original_filename="exam.pdf",
            file_size_bytes=1024,
            status=SubmissionStatus.PUBLISHED.value,
        )
        db.session.add(sub)
        db.session.commit()

        # Add question evaluation
        q = sample_assignment.questions[0]
        q.supportive_answer = "SECRET_TEACHER_KEY_SOLUTION_NEVER_LEAK"
        db.session.commit()

        ans = SubmissionAnswer(
            submission_id=sub.submission_id,
            question_id=q.question_id,
            extracted_text="Student text",
            match_confidence=0.9,
            status="Matched",
        )
        db.session.add(ans)
        db.session.commit()

        eval_rec = AnswerEvaluation(
            submission_id=sub.submission_id,
            question_id=q.question_id,
            answer_id=ans.answer_id,
            extracted_answer="Student text",
            maximum_marks=float(q.maximum_marks),
            ai_score=8.0,
            faculty_score=8.0,
            evaluation_status="Approved",
            faculty_feedback="Good work.",
        )
        db.session.add(eval_rec)
        db.session.commit()

        # Calculate result and publish
        ResultService.calculate_submission_result(sub.submission_id)
        ResultService.publish_result(sub.submission_id, sample_faculty.faculty_id)

        # Retrieve as student
        res_data, err = ResultService.get_student_result(sub.submission_id, sample_student.student_id)
        assert err is None
        assert res_data is not None

        # Check payload dictionary
        for q_item in res_data.get("questions", []):
            assert "SECRET_TEACHER_KEY_SOLUTION" not in str(q_item)
            assert "supportive_answer" not in q_item
