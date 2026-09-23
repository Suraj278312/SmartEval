"""
Unit and integration tests for PDF Submissions, File Validation, and Anti-Bypass Security.
"""

import io
import os
import pytest
from werkzeug.datastructures import FileStorage
from app.models.submission import Submission, SubmissionStatus
from app.services.submission_service import SubmissionService
from app.processing.handwriting_recognizer import HTRRecognizerInterface, HTRResult
from tests.conftest import login_session


class FastMockRecognizer(HTRRecognizerInterface):
    """Fast mock recognizer for submission testing."""
    def recognize(self, image_input) -> HTRResult:
        return HTRResult(
            text="Recognized handwritten text for testing.",
            confidence=0.92,
            is_low_confidence=False,
        )


@pytest.fixture(autouse=True)
def mock_htr_for_submission_tests(monkeypatch):
    """Automatically mock HTR during submission endpoint tests."""
    monkeypatch.setattr(
        "app.processing.pipeline.get_default_recognizer",
        lambda: FastMockRecognizer(),
    )


def test_submission_service_valid_pdf(app, sample_student, sample_assignment, sample_pdf_bytes):
    """Test SubmissionService directly saves valid PDF and runs pipeline."""
    file_storage = FileStorage(
        stream=io.BytesIO(sample_pdf_bytes),
        filename="service_test.pdf",
        content_type="application/pdf",
    )

    sub, error = SubmissionService.create_or_update_submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file=file_storage,
        upload_folder=app.config["UPLOAD_FOLDER"],
        pages_folder=app.config["PAGES_FOLDER"],
        async_processing=False,
    )

    assert error is None
    assert sub is not None
    assert sub.original_filename == "service_test.pdf"
    assert os.path.exists(sub.file_path)


def test_submission_service_non_pdf_rejected(app, sample_student, sample_assignment):
    """Test SubmissionService rejects non-PDF files."""
    text_file = FileStorage(
        stream=io.BytesIO(b"Plain text notes"),
        filename="notes.txt",
        content_type="text/plain",
    )

    sub, error = SubmissionService.create_or_update_submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file=text_file,
        upload_folder=app.config["UPLOAD_FOLDER"],
    )

    assert sub is None
    assert "Only PDF (.pdf) documents are accepted" in error


def test_submission_service_bad_magic_bytes_rejected(app, sample_student, sample_assignment):
    """Test SubmissionService rejects files with .pdf extension but invalid header bytes."""
    fake_pdf = FileStorage(
        stream=io.BytesIO(b"Bad binary header without pdf magic"),
        filename="fake.pdf",
        content_type="application/pdf",
    )

    sub, error = SubmissionService.create_or_update_submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        file=fake_pdf,
        upload_folder=app.config["UPLOAD_FOLDER"],
    )

    assert sub is None
    assert "file header does not match standard PDF signatures" in error


def test_student_direct_file_upload_endpoint_rejected(client, sample_student, sample_assignment, sample_pdf_bytes):
    """
    Test that students attempting direct file uploads via HTTP /submit are blocked,
    enforcing that submissions must go through the in-app camera scanner.
    """
    login_session(client, sample_student.student_id, "student", sample_student.name)

    data = {
        "submission_file": (io.BytesIO(sample_pdf_bytes), "my_handwritten_solution.pdf")
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert resp.status_code == 400 or b"Direct PDF file upload is disabled for students" in resp.data
    assert b"In-App Camera Scanner" in resp.data

    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()
        assert sub is None
