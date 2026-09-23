"""
Unit and integration tests for Asynchronous Submission Processing and DocumentPage Collection Fix.
"""

import io
import os
import time
import pytest
from unittest.mock import MagicMock, patch
from app.extensions import db
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.models.submission_answer import SubmissionAnswer
from app.services.submission_service import SubmissionService, _run_async_pipeline
from app.services.evaluation_service import EvaluationService
from app.processing.pipeline import SubmissionProcessingPipeline
from app.processing.handwriting_recognizer import HTRRecognizerInterface, HTRResult
from tests.conftest import login_session


class MockQuickRecognizer(HTRRecognizerInterface):
    """Deterministic fast mock recognizer for pipeline tests."""
    def recognize(self, image_input) -> HTRResult:
        return HTRResult(
            text="Q1 Polymorphism is an OOP concept in Java.\nQ2 Stack follows LIFO and Queue follows FIFO.\nQ3 Binary search has O(log n) time complexity.",
            confidence=0.88,
            is_low_confidence=False,
        )


def test_async_upload_returns_immediately(client, sample_student, sample_assignment, monkeypatch):
    """Test that student scanner submission returns immediately without waiting for pipeline."""
    import base64
    from PIL import Image
    login_session(client, sample_student.student_id, "student", sample_student.name)

    thread_started = []

    def mock_start(self):
        thread_started.append(self.name)

    monkeypatch.setattr("threading.Thread.start", mock_start)

    # Make test image base64
    img = Image.new("RGB", (300, 400), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "session_id": "async_sess_1",
        "pages": [{"page_number": 1, "image_data": b64}],
    }

    t0 = time.time()
    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    elapsed = time.time() - t0

    # Response should be near-instantaneous (< 1.5 seconds)
    assert elapsed < 1.5
    assert resp.status_code == 200
    assert len(thread_started) == 1

    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()
        assert sub is not None
        assert sub.status == SubmissionStatus.PROCESSING.value
        assert os.path.exists(sub.file_path)


def test_background_worker_execution_updates_status(app, sample_student, sample_assignment, sample_pdf_bytes, tmp_path):
    """Test that _run_async_pipeline successfully processes a submission in background."""
    with app.app_context():
        upload_dir = str(tmp_path / "uploads")
        pages_dir = str(tmp_path / "pages")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(pages_dir, exist_ok=True)

        pdf_path = os.path.join(upload_dir, "test_bg.pdf")
        with open(pdf_path, "wb") as f:
            f.write(sample_pdf_bytes)

        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="test_bg.pdf",
            file_size_bytes=len(sample_pdf_bytes),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

    # Execute async worker directly (as the thread would)
    with patch("app.processing.pipeline.get_default_recognizer", return_value=MockQuickRecognizer()):
        _run_async_pipeline(app, sub_id, pages_dir)

    with app.app_context():
        updated_sub = db.session.get(Submission, sub_id)
        assert updated_sub.status in (SubmissionStatus.COMPLETED.value, SubmissionStatus.REVIEW_REQUIRED.value)
        assert len(updated_sub.pages) > 0
        assert len(updated_sub.answers) > 0


def test_pipeline_passes_doc_pages_to_matcher(app, sample_assignment, sample_student, sample_pdf_bytes, tmp_path):
    """Test that SubmissionProcessingPipeline passes in-memory doc_pages to qa_matcher."""
    upload_dir = str(tmp_path / "uploads")
    pages_dir = str(tmp_path / "pages")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    pdf_path = os.path.join(upload_dir, "test_matcher.pdf")
    with open(pdf_path, "wb") as f:
        f.write(sample_pdf_bytes)

    with app.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="test_matcher.pdf",
            file_size_bytes=len(sample_pdf_bytes),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        pipeline = SubmissionProcessingPipeline(recognizer=MockQuickRecognizer())
        res = pipeline.process_submission(sub_id, pages_folder=pages_dir)

        assert res.success is True
        assert res.total_pages > 0
        assert res.matched_answers_count > 0

        # Check DB records
        pages = DocumentPage.query.filter_by(submission_id=sub_id).all()
        assert len(pages) > 0
        for p in pages:
            assert p.extracted_text is not None
            assert len(p.extracted_text) > 0

        answers = SubmissionAnswer.query.filter_by(submission_id=sub_id).all()
        assert len(answers) > 0


def test_evaluation_blocked_when_processing(app, sample_faculty, sample_assignment, sample_student, sample_pdf_bytes):
    """Test that EvaluationService blocks evaluation if submission is still in Processing state."""
    with app.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path="dummy.pdf",
            original_filename="dummy.pdf",
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        evals, err = EvaluationService.evaluate_submission_answers(
            submission_id=sub_id,
            faculty_id=sample_faculty.faculty_id,
        )

        assert evals == []
        assert "currently being processed" in err


def test_evaluation_blocked_when_pages_empty(app, sample_faculty, sample_assignment, sample_student):
    """Test that EvaluationService blocks evaluation if submission has zero pages/answers."""
    with app.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path="dummy.pdf",
            original_filename="dummy.pdf",
            status=SubmissionStatus.COMPLETED.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        evals, err = EvaluationService.evaluate_submission_answers(
            submission_id=sub_id,
            faculty_id=sample_faculty.faculty_id,
        )

        assert evals == []
        assert "No extracted pages or recognized answers" in err


def test_background_worker_exception_marks_failed(app, sample_student, sample_assignment):
    """Test that unhandled exception in background worker marks submission as Failed without crash."""
    with app.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path="non_existent.pdf",
            original_filename="non_existent.pdf",
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

    # Running async worker on missing file
    _run_async_pipeline(app, sub_id, "dummy_pages_folder")

    with app.app_context():
        updated_sub = db.session.get(Submission, sub_id)
        assert updated_sub.status == SubmissionStatus.FAILED.value
        assert "failed" in updated_sub.evaluation_notes.lower()
