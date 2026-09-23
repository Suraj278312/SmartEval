"""
Comprehensive test suite for Phase 3A: Question/Answer Segmentation and Question Matching.
Covers boundary detection, multi-page segmentation, hierarchical matching (explicit, textual, semantic),
confidence thresholds, manual review fallbacks, database persistence, and authorization isolation.
"""

import os
import fitz  # PyMuPDF
import pytest
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.document_page import DocumentPage
from app.processing.qa_matcher import (
    HierarchicalQAMatcher,
    CandidateAnswerSegment,
    MatchedQuestionAnswer,
)
from app.processing.pipeline import SubmissionProcessingPipeline
from tests.conftest import login_session
from tests.test_processing import MockHTRRecognizer, _create_test_pdf_file


# =========================================================================
# 1. Unit Tests: Boundary Detection & Numbering Formats
# =========================================================================

def test_detect_question_header_formats():
    """Test boundary detection across diverse numbering formats (Q1, 1., 1), Question 1, Ans 1)."""
    matcher = HierarchicalQAMatcher()

    test_cases = [
        ("Q1: Master Theorem", True, "Q1", 1, "Master Theorem"),
        ("Q.1 Derivation of Time Complexity", True, "Q1", 1, "Derivation of Time Complexity"),
        ("Question 2: QuickSelect analysis", True, "Q2", 2, "QuickSelect analysis"),
        ("Ans 3. Median of two arrays", True, "Ans 3", 3, "Median of two arrays"),
        ("Answer 1: Inheritance in Java", True, "Ans 1", 1, "Inheritance in Java"),
        ("1. Explain polymorphism", True, "1.", 1, "Explain polymorphism"),
        ("2) Recurrence relation", True, "2.", 2, "Recurrence relation"),
        ("(3) Bellman Ford proof", True, "3.", 3, "Bellman Ford proof"),
        ("[4] Dynamic programming memoization", True, "4.", 4, "Dynamic programming memoization"),
        ("Problem 5: Graph traversal", True, "Q5", 5, "Graph traversal"),
        ("Task 6: Sorting comparison", True, "Q6", 6, "Sorting comparison"),
        ("Regular answer line without header", False, None, None, "Regular answer line without header"),
    ]

    for line, expected_is_header, expected_label, expected_q_num, expected_body in test_cases:
        is_h, label, q_num, body = matcher.detect_question_header(line)
        assert is_h == expected_is_header, f"Failed on line: '{line}'"
        if expected_is_header:
            assert label == expected_label, f"Label mismatch for '{line}'"
            assert q_num == expected_q_num, f"Question number mismatch for '{line}'"


# =========================================================================
# 2. Unit Tests: Multi-Page Answer Segmentation
# =========================================================================

def test_segment_pages_multipage_answer():
    """Test that answer spanning across page 1 and page 2 preserves page_start=1 and page_end=2."""
    matcher = HierarchicalQAMatcher()

    pages = [
        {
            "page_number": 1,
            "extracted_text": "Q1: Master Theorem Derivation\nStep 1: Identify parameters.\nStep 2: Calculate log_b(a).",
            "confidence": 0.95,
        },
        {
            "page_number": 2,
            "extracted_text": "Step 3: Compare f(n) with n^(log_b a).\nConclusion: Case 3 applies.\n\nQ2: QuickSelect Analysis\nAverage time is O(n).",
            "confidence": 0.90,
        },
    ]

    candidates = matcher.segment_pages(pages)

    assert len(candidates) == 2
    
    # Candidate 1: Q1 spanning Page 1 to Page 2
    c1 = candidates[0]
    assert c1.detected_label == "Q1"
    assert c1.detected_question_num == 1
    assert c1.page_start == 1
    assert c1.page_end == 2
    assert "Step 1" in c1.text
    assert "Step 3" in c1.text

    # Candidate 2: Q2 on Page 2
    c2 = candidates[1]
    assert c2.detected_label == "Q2"
    assert c2.detected_question_num == 2
    assert c2.page_start == 2
    assert c2.page_end == 2
    assert "QuickSelect Analysis" in c2.text


# =========================================================================
# 3. Unit Tests: Hierarchical Question Matching Layers
# =========================================================================

def test_matching_explicit_numbering():
    """Test Layer 1: Explicit numbering maps directly with high confidence."""
    matcher = HierarchicalQAMatcher()

    questions = [
        {"question_id": 101, "question_number": 1, "question_text": "Explain Master Theorem.", "supportive_answer": "T(n) = aT(n/b) + f(n)"},
        {"question_id": 102, "question_number": 2, "question_text": "Describe QuickSelect.", "supportive_answer": "Partitioning algorithm for k-th element."},
    ]

    candidates = [
        CandidateAnswerSegment(
            text="Q1: Master Theorem states that T(n) = aT(n/b) + f(n).",
            page_start=1,
            page_end=1,
            detected_label="Q1",
            detected_question_num=1,
            average_ocr_confidence=0.92,
        ),
        CandidateAnswerSegment(
            text="Q2: QuickSelect uses Lomuto partitioning.",
            page_start=2,
            page_end=2,
            detected_label="Q2",
            detected_question_num=2,
            average_ocr_confidence=0.88,
        ),
    ]

    matches = matcher.match_candidates_to_questions(candidates, questions)

    assert len(matches) == 2
    assert matches[0].question_id == 101
    assert matches[0].match_method == "explicit_numbering"
    assert matches[0].match_confidence >= 0.85
    assert matches[0].status == "Matched"

    assert matches[1].question_id == 102
    assert matches[1].match_method == "explicit_numbering"
    assert matches[1].match_confidence >= 0.85
    assert matches[1].status == "Matched"


def test_matching_paraphrased_semantic_wording():
    """Test Layer 3: Unnumbered answer matching via semantic/textual similarity for varied wording."""
    matcher = HierarchicalQAMatcher()

    questions = [
        {
            "question_id": 201,
            "question_number": 1,
            "question_text": "Explain inheritance in Java with examples.",
            "supportive_answer": "Inheritance allows a subclass to inherit fields and methods from a superclass using extends.",
        }
    ]

    # Student did not write Q1, but reworded the question header
    candidates = [
        CandidateAnswerSegment(
            text="What do you understand by inheritance in Java? Discuss subclass and superclass with code examples.\nInheritance is an OOP mechanism where child class inherits methods.",
            page_start=1,
            page_end=1,
            detected_label=None,
            detected_question_num=None,
            average_ocr_confidence=0.90,
        )
    ]

    matches = matcher.match_candidates_to_questions(candidates, questions)

    assert len(matches) == 1
    assert matches[0].question_id == 201
    assert matches[0].match_method in ("text_similarity", "semantic_similarity")
    assert matches[0].match_confidence >= 0.60
    assert matches[0].page_start == 1


def test_matching_uncertain_low_confidence_triggers_review():
    """Test that borderline similarity produces Review Required status."""
    matcher = HierarchicalQAMatcher(high_confidence_threshold=0.75, review_required_threshold=0.50)

    questions = [
        {
            "question_id": 301,
            "question_number": 1,
            "question_text": "Describe asynchronous distributed consensus protocols.",
            "supportive_answer": "Paxos and Raft handle leader election and log replication.",
        }
    ]

    # Weak partial keyword overlap
    candidates = [
        CandidateAnswerSegment(
            text="Distributed networks protocol and packet routing header inspection.",
            page_start=1,
            page_end=1,
            detected_label=None,
            detected_question_num=None,
            average_ocr_confidence=0.80,
        )
    ]

    matches = matcher.match_candidates_to_questions(candidates, questions)

    assert len(matches) == 1
    # Should either match with Review Required or fall through to Unmatched
    assert matches[0].status == "Review Required"


def test_orphan_unmatched_content_fallback():
    """Test that completely unrelated content is preserved as unmatched with Review Required."""
    matcher = HierarchicalQAMatcher()

    questions = [
        {
            "question_id": 401,
            "question_number": 1,
            "question_text": "Derive time complexity of Dijkstra algorithm with Fibonacci heap.",
            "supportive_answer": "O(|V| log |V| + |E|)",
        }
    ]

    # Completely unrelated content (e.g. biology notes)
    candidates = [
        CandidateAnswerSegment(
            text="Photosynthesis is the process by which green plants convert sunlight into chemical energy in chloroplasts.",
            page_start=2,
            page_end=2,
            detected_label=None,
            detected_question_num=None,
            average_ocr_confidence=0.90,
        )
    ]

    matches = matcher.match_candidates_to_questions(candidates, questions)

    assert len(matches) == 1
    orphan = matches[0]
    assert orphan.question_id is None
    assert orphan.match_method == "unmatched"
    assert orphan.status == "Review Required"
    assert "Photosynthesis" in orphan.extracted_text
    assert orphan.page_start == 2


# =========================================================================
# 4. End-to-End Pipeline & DB Persistence Integration Tests
# =========================================================================

def test_pipeline_qa_segmentation_and_persistence(client, sample_student, sample_assignment, tmp_path):
    """Test full pipeline generates both DocumentPage and SubmissionAnswer DB records."""
    pdf_path = str(tmp_path / "submission_qa.pdf")
    _create_test_pdf_file(pdf_path, num_pages=2)
    pages_folder = str(tmp_path / "pages_qa")

    with client.application.app_context():
        # Ensure assignment has 2 questions
        q1 = Question.query.filter_by(assignment_id=sample_assignment.assignment_id, question_number=1).first()
        q2 = Question(
            assignment_id=sample_assignment.assignment_id,
            question_number=2,
            question_text="Describe QuickSelect algorithm average complexity.",
            supportive_answer="O(n) average case with Lomuto partitioning.",
            maximum_marks=10.0,
        )
        db.session.add(q2)
        db.session.commit()

        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="student_answers.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        # Mock OCR output containing explicit question headers across pages
        mock_ocr_text = (
            "Q1: Master Theorem Derivation\nT(n) = 3T(n/2) + O(n^2)\nCase 3 applies: Theta(n^2).\n\n"
            "Q2: QuickSelect Algorithm\nPartitioning gives O(n) average complexity."
        )
        mock_rec = MockHTRRecognizer(text=mock_ocr_text, confidence=0.92)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        result = pipeline.process_submission(sub_id, pages_folder=pages_folder)

        assert result.success is True
        assert result.status == SubmissionStatus.COMPLETED.value
        assert result.matched_answers_count >= 1

        # Verify DB SubmissionAnswer records
        refreshed_sub = db.session.get(Submission, sub_id)
        assert len(refreshed_sub.answers) >= 1

        ans1 = next((a for a in refreshed_sub.answers if a.question_id == q1.question_id), None)
        assert ans1 is not None
        assert ans1.match_method == "explicit_numbering"
        assert ans1.match_confidence >= 0.85
        assert ans1.status == "Matched"
        assert "Master Theorem" in ans1.extracted_text


def test_pipeline_reprocessing_idempotency(client, sample_student, sample_assignment, tmp_path):
    """Test re-processing cleans up prior SubmissionAnswer records without duplication."""
    pdf_path = str(tmp_path / "sub_idempotent.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_idemp")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="reprocess_test.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Q1: Master Theorem solution.\nResult is Theta(n^2).", confidence=0.90)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)

        # Run 1
        res1 = pipeline.process_submission(sub_id, pages_folder=pages_folder)
        count_run1 = SubmissionAnswer.query.filter_by(submission_id=sub_id).count()

        # Run 2 (Reprocess)
        res2 = pipeline.process_submission(sub_id, pages_folder=pages_folder)
        count_run2 = SubmissionAnswer.query.filter_by(submission_id=sub_id).count()

        assert count_run1 == count_run2
        assert count_run2 == 1


# =========================================================================
# 5. Faculty Review UI & Authorization Tests
# =========================================================================

def test_faculty_review_ui_displays_matched_answers(client, sample_faculty, sample_student, sample_assignment, tmp_path):
    """Test faculty review route renders Q&A segmentation section with question prompts and recognized answers."""
    pdf_path = str(tmp_path / "faculty_qa_test.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_fac_qa")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="faculty_view_doc.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Q1: Master Theorem Derivation step by step.", confidence=0.88)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        pipeline.process_submission(sub_id, pages_folder=pages_folder)

    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    resp = client.get(f"/faculty/submissions/{sub_id}/review")
    assert resp.status_code == 200
    assert b"Matched Question &amp; Answer Segments" in resp.data or b"Matched Question & Answer Segments" in resp.data
    assert b"Master Theorem Derivation" in resp.data
    assert b"Explicit Numbering" in resp.data
    assert b"Questions Matched" in resp.data


def test_student_isolation_cannot_access_faculty_qa_review(client, sample_faculty, sample_student, sample_assignment, tmp_path):
    """Test student cannot access faculty review UI or view teacher reference answers."""
    pdf_path = str(tmp_path / "student_iso.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_iso")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="iso_doc.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Q1: Secret answer.", confidence=0.95)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        pipeline.process_submission(sub_id, pages_folder=pages_folder)

    # Login as student
    login_session(client, sample_student.student_id, "student", sample_student.name)

    # Attempt faculty review access
    resp = client.get(f"/faculty/submissions/{sub_id}/review")
    # Redirected away or unauthorized (302 redirect to login/dashboard or 403)
    assert resp.status_code in (302, 403)
