"""
Comprehensive test suite for Phase 3B: AI Semantic Answer Evaluation.
Tests rubric parsing, semantic evaluation engine, scoring safety clamping,
mock deterministic evaluation, database persistence, faculty review/override workflows,
student reference answer privacy isolation, and HTTP endpoints.
"""

import os
import pytest
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.document_page import DocumentPage
from app.models.user import Faculty, Student
from app.processing.evaluator import (
    RubricEngine,
    CriterionScore,
    EvaluationResult,
    MockSemanticEvaluator,
    GeminiSemanticEvaluator,
)
from app.services.evaluation_service import EvaluationService
from tests.conftest import login_session


# =========================================================================
# 1. Unit Tests: Rubric Parsing & Normalization Engine
# =========================================================================

def test_rubric_engine_parsing_and_normalization():
    """Test rubric parsing from text strings and normalization to match max marks."""
    # Test 1: Standard 'X pts Description' format
    rubric_text = "3 pts Definition; 4 pts Derivation; 3 pts Example"
    criteria = RubricEngine.parse_rubric(rubric_text, 10.0)
    assert len(criteria) == 3
    assert sum(c["max_marks"] for c in criteria) == pytest.approx(10.0)
    assert criteria[0]["name"] == "Definition"
    assert criteria[0]["max_marks"] == pytest.approx(3.0)

    # Test 2: Scaling normalization when criteria sum doesn't match maximum marks
    rubric_text_unscaled = "2 pts Concept, 2 pts Code"  # Sum = 4, but max = 10
    criteria_scaled = RubricEngine.parse_rubric(rubric_text_unscaled, 10.0)
    assert sum(c["max_marks"] for c in criteria_scaled) == pytest.approx(10.0)
    assert criteria_scaled[0]["max_marks"] == pytest.approx(5.0)
    assert criteria_scaled[1]["max_marks"] == pytest.approx(5.0)

    # Test 3: Numbered list format '1. Concept (5 marks)'
    rubric_numbered = "1. Time Complexity (5 marks)\n2. Space Complexity (5 marks)"
    criteria_num = RubricEngine.parse_rubric(rubric_numbered, 10.0)
    assert len(criteria_num) == 2
    assert sum(c["max_marks"] for c in criteria_num) == pytest.approx(10.0)

    # Test 4: Default fallback rubric when rubric is empty
    default_criteria = RubricEngine.parse_rubric(None, 15.0)
    assert len(default_criteria) == 3
    assert sum(c["max_marks"] for c in default_criteria) == pytest.approx(15.0)


# =========================================================================
# 2. Unit Tests: Evaluator Data Contracts & Serialization
# =========================================================================

def test_evaluation_result_dataclass_and_dict():
    """Test EvaluationResult serialization and properties."""
    c1 = CriterionScore(
        name="Core Concept",
        max_marks=5.0,
        awarded_marks=4.5,
        status="full",
        reason="Clear explanation",
    )
    res = EvaluationResult(
        score=9.0,
        max_score=10.0,
        criteria=[c1],
        strengths=["Great intuition"],
        missing_elements=["Minor edge case"],
        feedback="Well done overall.",
        confidence=0.92,
        needs_faculty_review=False,
        semantic_similarity_score=0.88,
    )

    d = res.to_dict()
    assert d["score"] == 9.0
    assert d["max_score"] == 10.0
    assert len(d["criteria"]) == 1
    assert d["criteria"][0]["name"] == "Core Concept"
    assert d["criteria"][0]["awarded_marks"] == 4.5
    assert d["strengths"] == ["Great intuition"]
    assert d["missing_elements"] == ["Minor edge case"]
    assert d["confidence"] == 0.92
    assert d["needs_faculty_review"] is False


# =========================================================================
# 3. Unit Tests: Mock Semantic Evaluator Deterministic Modes
# =========================================================================

def test_mock_evaluator_deterministic_scores():
    """Test mock evaluator with preset percentage, score, and rubric parsing."""
    # Preset percentage (80% of 15 marks = 12.0)
    evaluator_pct = MockSemanticEvaluator(percentage=0.8)
    res = evaluator_pct.evaluate_answer(
        question_text="Explain Dijkstra",
        supportive_answer="Shortest path for non-negative weights.",
        student_answer="Dijkstra finds shortest path using priority queue.",
        maximum_marks=15.0,
        rubric="5 pts priority queue, 10 pts algorithm steps",
    )
    assert res.score == 12.0
    assert res.max_score == 15.0
    assert len(res.criteria) == 2
    assert sum(c.awarded_marks for c in res.criteria) == pytest.approx(12.0)

    # Preset absolute score
    evaluator_score = MockSemanticEvaluator(score=7.5)
    res2 = evaluator_score.evaluate_answer(
        question_text="What is quicksort?",
        supportive_answer="Divide and conquer algorithm.",
        student_answer="Quicksort partitions the array around a pivot.",
        maximum_marks=10.0,
    )
    assert res2.score == 7.5
    assert res2.max_score == 10.0


def test_mock_evaluator_empty_student_answer():
    """Test that empty or blank student answer receives zero marks automatically."""
    evaluator = MockSemanticEvaluator()
    res = evaluator.evaluate_answer(
        question_text="Derive recurrence",
        supportive_answer="T(n) = 2T(n/2) + O(n)",
        student_answer="",
        maximum_marks=10.0,
    )
    assert res.score == 0.0
    assert res.max_score == 10.0
    assert "Entire answer is missing." in res.missing_elements


def test_mock_evaluator_simulated_failure():
    """Test evaluator failure handling and flag propagation."""
    evaluator = MockSemanticEvaluator(should_fail=True, error_message="Rate limit exceeded")
    res = evaluator.evaluate_answer(
        question_text="Explain BFS",
        supportive_answer="Queue-based graph traversal",
        student_answer="BFS uses a queue",
        maximum_marks=10.0,
    )
    assert res.score is None
    assert res.confidence == 0.0
    assert res.needs_faculty_review is True
    assert "Rate limit exceeded" in res.error_message


# =========================================================================
# 4. Integration Tests: EvaluationService & Database Persistence
# =========================================================================

def _setup_test_submission(app, sample_assignment, sample_student):
    """Helper to construct a fully prepared submission with answers."""
    now = sample_assignment.deadline
    sub = Submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file_path="mock/path/submission.pdf",
        original_filename="submission.pdf",
        file_size_bytes=1024,
        status=SubmissionStatus.COMPLETED.value,
    )
    db.session.add(sub)
    db.session.flush()

    # Add mock document page
    page = DocumentPage(
        submission_id=sub.submission_id,
        page_number=1,
        extracted_text="Q1: Master Theorem Case 2 applies. T(n) = Theta(n log n).\nQ2: QuickSort worst case is O(n^2) when already sorted.",
        confidence=0.92,
        processing_status="Completed",
    )
    db.session.add(page)
    db.session.flush()

    # Add segmented answers for Q1 and Q2
    q1 = sample_assignment.questions[0]
    q2 = sample_assignment.questions[1]

    ans1 = SubmissionAnswer(
        submission_id=sub.submission_id,
        question_id=q1.question_id,
        extracted_text="Master Theorem Case 2 applies. a=2, b=2, so T(n) = Theta(n log n).",
        detected_label="Q1",
        page_start=1,
        page_end=1,
        match_method="explicit_numbering",
        match_confidence=0.95,
        status="Matched",
    )
    ans2 = SubmissionAnswer(
        submission_id=sub.submission_id,
        question_id=q2.question_id,
        extracted_text="QuickSort worst case happens when the array is sorted and pivot is first element.",
        detected_label="Q2",
        page_start=1,
        page_end=1,
        match_method="explicit_numbering",
        match_confidence=0.95,
        status="Matched",
    )
    db.session.add_all([ans1, ans2])
    db.session.commit()
    db.session.refresh(sub)
    return sub


def test_evaluation_service_evaluate_submission(app, sample_faculty, sample_student, sample_assignment):
    """Test full evaluation pipeline creates AnswerEvaluation records for all questions."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)

    # Evaluate submission using MockSemanticEvaluator (90% score)
    mock_eval = MockSemanticEvaluator(percentage=0.9)
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval,
    )

    assert error is None
    assert len(evaluations) == 2

    # Check Q1 evaluation (10 marks * 0.9 = 9.0)
    ev1 = next(e for e in evaluations if e.question_id == sample_assignment.questions[0].question_id)
    assert ev1.ai_score == pytest.approx(9.0)
    assert ev1.maximum_marks == 10.0
    assert ev1.final_score == pytest.approx(9.0)
    assert ev1.evaluation_status == "Evaluated"
    assert len(ev1.criteria_scores) == 2

    # Check Q2 evaluation (15 marks * 0.9 = 13.5)
    ev2 = next(e for e in evaluations if e.question_id == sample_assignment.questions[1].question_id)
    assert ev2.ai_score == pytest.approx(13.5)
    assert ev2.maximum_marks == 15.0
    assert ev2.final_score == pytest.approx(13.5)

    # Check submission helper properties
    assert sub.total_ai_score == pytest.approx(22.5)
    assert sub.total_final_score == pytest.approx(22.5)
    assert sub.is_fully_evaluated is True


def test_evaluation_service_handles_missing_questions(app, sample_faculty, sample_student, sample_assignment):
    """Test that questions with no student answers receive zero marks gracefully."""
    sub = Submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file_path="mock/path/sub.pdf",
        original_filename="sub.pdf",
        status=SubmissionStatus.COMPLETED.value,
    )
    db.session.add(sub)
    db.session.flush()

    # Only answer Q1
    q1 = sample_assignment.questions[0]
    ans1 = SubmissionAnswer(
        submission_id=sub.submission_id,
        question_id=q1.question_id,
        extracted_text="Master Theorem explanation...",
        match_confidence=0.90,
    )
    db.session.add(ans1)
    db.session.commit()

    mock_eval = MockSemanticEvaluator(percentage=0.8)
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval,
    )

    assert error is None
    assert len(evaluations) == 2

    # Q2 should be evaluated as empty (0 marks)
    q2 = sample_assignment.questions[1]
    ev2 = next(e for e in evaluations if e.question_id == q2.question_id)
    assert ev2.ai_score == 0.0
    assert "Entire answer is missing." in ev2.missing_elements


def test_evaluation_service_faculty_authorization(app, sample_faculty, sample_student, sample_assignment):
    """Test that faculty cannot evaluate submissions belonging to another faculty."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)

    # Create other faculty
    other_fac = Faculty(name="Other", email="other@smarteval.edu", department="Math")
    other_fac.set_password("Pass@123")
    db.session.add(other_fac)
    db.session.commit()

    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=other_fac.faculty_id,
    )
    assert len(evaluations) == 0
    assert "Unauthorized" in error


def test_evaluation_service_approve_evaluation(app, sample_faculty, sample_student, sample_assignment):
    """Test faculty quick approve marks evaluation status Approved."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(score=8.0)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval
    )

    ev = evaluations[0]
    approved_ev, error = EvaluationService.approve_evaluation(
        evaluation_id=ev.evaluation_id, faculty_id=sample_faculty.faculty_id
    )

    assert error is None
    assert approved_ev.evaluation_status == "Approved"
    assert approved_ev.needs_faculty_review is False
    assert approved_ev.faculty_score == pytest.approx(8.0)
    assert approved_ev.final_score == pytest.approx(8.0)


def test_evaluation_service_modify_evaluation(app, sample_faculty, sample_student, sample_assignment):
    """Test faculty override score and custom feedback."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(score=5.0)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval
    )

    ev = evaluations[0]
    modified_ev, error = EvaluationService.modify_evaluation(
        evaluation_id=ev.evaluation_id,
        faculty_score=9.5,
        faculty_feedback="Excellent derivation with clear step-by-step logic.",
        faculty_id=sample_faculty.faculty_id,
    )

    assert error is None
    assert modified_ev.evaluation_status == "Modified"
    assert modified_ev.faculty_score == pytest.approx(9.5)
    assert modified_ev.ai_score == pytest.approx(5.0)  # Original AI score preserved
    assert modified_ev.final_score == pytest.approx(9.5)
    assert modified_ev.final_feedback == "Excellent derivation with clear step-by-step logic."
    assert modified_ev.needs_faculty_review is False


def test_evaluation_service_modify_evaluation_score_bounds(app, sample_faculty, sample_student, sample_assignment):
    """Test score boundary validation (cannot be < 0 or > max_marks)."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(score=5.0)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval
    )
    ev = evaluations[0]  # max_marks = 10.0

    # Negative score
    _, error_neg = EvaluationService.modify_evaluation(
        evaluation_id=ev.evaluation_id, faculty_score=-2.0, faculty_feedback="", faculty_id=sample_faculty.faculty_id
    )
    assert error_neg is not None

    # Score exceeding maximum marks
    _, error_excess = EvaluationService.modify_evaluation(
        evaluation_id=ev.evaluation_id, faculty_score=15.0, faculty_feedback="", faculty_id=sample_faculty.faculty_id
    )
    assert error_excess is not None


def test_evaluation_service_re_evaluate_answer(app, sample_faculty, sample_student, sample_assignment):
    """Test re-evaluating a single answer."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval1 = MockSemanticEvaluator(score=4.0)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval1
    )
    ev = evaluations[0]
    assert ev.ai_score == pytest.approx(4.0)

    # Re-evaluate with new evaluator giving 8.0
    mock_eval2 = MockSemanticEvaluator(score=8.0)
    updated_ev, error = EvaluationService.re_evaluate_answer(
        evaluation_id=ev.evaluation_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval2,
    )
    assert error is None
    assert updated_ev.ai_score == pytest.approx(8.0)


# =========================================================================
# 5. Security & Privacy: Student Reference Answer Isolation
# =========================================================================

def test_student_cannot_view_supportive_reference_answer(app, sample_faculty, sample_student, sample_assignment):
    """Verify that AnswerEvaluation serialization for students never contains teacher supportive answers."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(percentage=0.85)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval
    )
    ev = evaluations[0]

    # Student-facing dictionary representation
    student_dict = ev.to_dict(include_supportive_answer=False)
    assert "supportive_answer" not in student_dict
    assert student_dict["ai_score"] is not None
    assert student_dict["final_score"] is not None
    assert student_dict["feedback"] is not None

    # Faculty-facing dictionary representation
    faculty_dict = ev.to_dict(include_supportive_answer=True)
    assert "supportive_answer" in faculty_dict
    assert faculty_dict["supportive_answer"] == sample_assignment.questions[0].supportive_answer


# =========================================================================
# 6. HTTP Endpoint Tests: Faculty Evaluation Routes
# =========================================================================

def test_faculty_routes_evaluate_endpoint(client, sample_faculty, sample_student, sample_assignment):
    """Test POST /faculty/submissions/<id>/evaluate route using mocked deterministic evaluator."""
    from unittest.mock import patch
    sub = _setup_test_submission(client.application, sample_assignment, sample_student)
    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    mock_eval = MockSemanticEvaluator(percentage=0.9)
    with patch("app.services.evaluation_service.GeminiSemanticEvaluator", return_value=mock_eval):
        response = client.post(
            f"/faculty/submissions/{sub.submission_id}/evaluate",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"AI Semantic Answer Evaluation" in response.data or b"Handwriting Extraction Review" in response.data


def test_faculty_routes_approve_and_modify_endpoints(client, sample_faculty, sample_student, sample_assignment):
    """Test POST approve and modify routes through HTTP client."""
    sub = _setup_test_submission(client.application, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(score=7.0)
    evaluations, _ = EvaluationService.evaluate_submission_answers(
        sub.submission_id, sample_faculty.faculty_id, evaluator=mock_eval
    )
    ev = evaluations[0]
    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    # 1. Approve
    resp_app = client.post(
        f"/faculty/evaluations/{ev.evaluation_id}/approve",
        follow_redirects=True,
    )
    assert resp_app.status_code == 200
    assert b"approved" in resp_app.data.lower()

    # 2. Modify
    resp_mod = client.post(
        f"/faculty/evaluations/{ev.evaluation_id}/modify",
        data={
            "faculty_score": "9.0",
            "faculty_feedback": "Revised score after checking extra derivation step.",
        },
        follow_redirects=True,
    )
    assert resp_mod.status_code == 200
    assert b"updated" in resp_mod.data.lower()


# =========================================================================
# 7. Gemini Evaluator Construction & Missing Key Safety
# =========================================================================

def test_gemini_evaluator_missing_key_graceful_handling():
    """Test that GeminiSemanticEvaluator without API key returns structured review-required result with None score."""
    evaluator = GeminiSemanticEvaluator(api_key="")
    res = evaluator.evaluate_answer(
        question_text="Explain QuickSort",
        supportive_answer="Pivot-based partition algorithm",
        student_answer="QuickSort partitions elements around a chosen pivot",
        maximum_marks=10.0,
    )
    assert res.score is None  # Technical failure: score is None, NOT 0.0
    assert res.max_score == 10.0
    assert res.needs_faculty_review is True
    assert "GEMINI_API_KEY is missing" in res.error_message


# =========================================================================
# 8. Focused Consistency, Deterministic Scoring & Failure-Handling Tests
# =========================================================================

def test_deterministic_scoring_full_rubric_outcome(app, sample_faculty, sample_student, sample_assignment):
    """Test successful evaluation with Full rubric outcome calculates exactly 100% score deterministically."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(outcome="full")
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval,
    )
    assert error is None
    assert len(evaluations) == 2
    ev1 = evaluations[0]
    assert ev1.ai_score == pytest.approx(10.0)  # 100% of 10.0
    assert ev1.evaluation_status == "Evaluated"
    assert all(c["status"] == "full" for c in ev1.criteria_scores)


def test_deterministic_scoring_partial_rubric_outcome(app, sample_faculty, sample_student, sample_assignment):
    """Test successful evaluation with Partial rubric outcome calculates exactly 50% score deterministically."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(outcome="partial")
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval,
    )
    assert error is None
    assert len(evaluations) == 2
    ev1 = evaluations[0]
    assert ev1.ai_score == pytest.approx(5.0)  # 50% of 10.0
    assert ev1.evaluation_status == "Evaluated"
    assert all(c["status"] == "partial" for c in ev1.criteria_scores)


def test_deterministic_scoring_none_rubric_outcome(app, sample_faculty, sample_student, sample_assignment):
    """Test successful evaluation with None rubric outcome receives legitimate academic 0.0 with status Evaluated."""
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    mock_eval = MockSemanticEvaluator(outcome="none")
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=mock_eval,
    )
    assert error is None
    assert len(evaluations) == 2
    ev1 = evaluations[0]
    assert ev1.ai_score == pytest.approx(0.0)
    assert ev1.evaluation_status == "Evaluated"
    assert all(c["status"] == "none" for c in ev1.criteria_scores)


def test_gemini_429_resource_exhausted_failure_handling(app, sample_faculty, sample_student, sample_assignment):
    """Test Gemini 429 RESOURCE_EXHAUSTED sets score to None and status to Failed (never 0.0)."""
    from unittest.mock import MagicMock
    sub = _setup_test_submission(app, sample_assignment, sample_student)
    evaluator = GeminiSemanticEvaluator(api_key="fake-key-for-test")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota, please retry in 58s"
    )
    evaluator._client = mock_client

    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=sub.submission_id,
        faculty_id=sample_faculty.faculty_id,
        evaluator=evaluator,
    )
    assert error is None
    assert len(evaluations) == 2
    for ev in evaluations:
        assert ev.ai_score is None
        assert ev.final_score is None
        assert ev.evaluation_status == "Failed"
        assert ev.needs_faculty_review is True
        assert "429 RESOURCE_EXHAUSTED" in ev.error_message


def test_gemini_generic_network_failure_handling():
    """Test generic network/timeout failure sets score to None and status to Failed."""
    from unittest.mock import MagicMock
    evaluator = GeminiSemanticEvaluator(api_key="fake-key-for-test")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("Network connection reset by peer")
    evaluator._client = mock_client

    res = evaluator.evaluate_answer(
        question_text="Explain Dijkstra",
        supportive_answer="Shortest path algorithm",
        student_answer="Student answer text...",
        maximum_marks=10.0,
    )
    assert res.score is None
    assert res.needs_faculty_review is True
    assert "Network connection reset" in res.error_message


def test_gemini_malformed_response_failure_handling():
    """Test malformed response from Gemini sets score to None and status to Failed."""
    from unittest.mock import MagicMock
    evaluator = GeminiSemanticEvaluator(api_key="fake-key-for-test")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "NOT_VALID_JSON_<<<MALFORMED>>>"
    mock_client.models.generate_content.return_value = mock_response
    evaluator._client = mock_client

    res = evaluator.evaluate_answer(
        question_text="Explain Dijkstra",
        supportive_answer="Shortest path algorithm",
        student_answer="Student answer text...",
        maximum_marks=10.0,
    )
    assert res.score is None
    assert res.needs_faculty_review is True
    assert "Evaluation error" in res.error_message


def test_repeated_evaluation_score_stability():
    """Test that repeated evaluations of identical input produce strictly identical numerical scores."""
    import json
    from unittest.mock import MagicMock
    evaluator = GeminiSemanticEvaluator(api_key="fake-key-for-test")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "criteria": [
            {"name": "Core Concept", "max_marks": 5.0, "status": "full", "reason": "Accurate"},
            {"name": "Detail", "max_marks": 3.0, "status": "partial", "reason": "Missing one step"},
            {"name": "Clarity", "max_marks": 2.0, "status": "full", "reason": "Clear formatting"}
        ],
        "strengths": ["Clear definition"],
        "missing_elements": ["Minor detail"],
        "feedback": "Good job",
        "confidence": 0.95,
        "needs_faculty_review": False
    })
    mock_client.models.generate_content.return_value = mock_response
    evaluator._client = mock_client

    scores = []
    for _ in range(5):
        res = evaluator.evaluate_answer(
            question_text="Explain Dijkstra",
            supportive_answer="Shortest path algorithm",
            student_answer="Student answer text...",
            maximum_marks=10.0,
            rubric="5 pts Core Concept; 3 pts Detail; 2 pts Clarity"
        )
        # Expected calculation: 5.0*1.0 + 3.0*0.5 + 2.0*1.0 = 5.0 + 1.5 + 2.0 = 8.5
        scores.append(res.score)

    assert len(scores) == 5
    assert all(s == 8.5 for s in scores)
