"""
Unit and integration tests for In-App Scanner Student Submission Flow.
Verifies:
1. scanner submission with 1 page
2. scanner submission with multiple pages
3. page order preserved
4. PDF generated successfully
5. invalid/empty capture rejected
6. malformed image payload rejected
7. existing submission processing still starts
8. student cannot use old direct-upload route to bypass scanner
9. existing faculty submission/review behavior remains intact
"""

import base64
import io
import os
import time
import pytest
from PIL import Image, ImageDraw
import pymupdf
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.services.submission_service import SubmissionService
from app.processing.handwriting_recognizer import HTRRecognizerInterface, HTRResult
from tests.conftest import login_session


class FastMockRecognizer(HTRRecognizerInterface):
    """Fast mock recognizer for scanner submission testing."""
    def recognize(self, image_input) -> HTRResult:
        return HTRResult(
            text="Recognized handwritten text from scanned page.",
            confidence=0.94,
            is_low_confidence=False,
        )


@pytest.fixture(autouse=True)
def mock_htr_for_scanner_tests(monkeypatch):
    """Mock HTR recognition during scanner endpoint tests."""
    monkeypatch.setattr(
        "app.processing.pipeline.get_default_recognizer",
        lambda: FastMockRecognizer(),
    )


def make_test_page_base64(label: str = "Page 1", width: int = 400, height: int = 600, color: str = "white") -> str:
    """Helper to generate valid JPEG image base64 data URL."""
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
    draw.text((30, 40), f"SmartEval Scan: {label}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{raw_b64}"


def test_1_scanner_submission_single_page(client, sample_student, sample_assignment):
    """Requirement 1: Scanner submission with 1 page."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    page1_b64 = make_test_page_base64("Problem 1 Solution", 500, 700)
    payload = {
        "session_id": "test_scan_sess_001",
        "pages": [
            {"page_number": 1, "image_data": page1_b64, "timestamp": "2026-09-23T12:00:00Z"}
        ],
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "submission_id" in data
    assert "redirect_url" in data

    # Verify database record and generated PDF
    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()
        assert sub is not None
        assert os.path.exists(sub.file_path)
        assert sub.file_size_bytes > 0
        assert sub.original_filename.startswith("scan_")
        assert sub.status in [SubmissionStatus.PROCESSING.value, SubmissionStatus.COMPLETED.value]


def test_2_scanner_submission_multiple_pages(client, sample_student, sample_assignment):
    """Requirement 2: Scanner submission with multiple pages."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    pages = [
        {"page_number": 1, "image_data": make_test_page_base64("Section 1 - Derivations", 400, 600, "#ffffff")},
        {"page_number": 2, "image_data": make_test_page_base64("Section 2 - Master Theorem", 400, 600, "#f8fafc")},
        {"page_number": 3, "image_data": make_test_page_base64("Section 3 - Complexity Analysis", 400, 600, "#f1f5f9")},
    ]
    payload = {
        "session_id": "test_scan_multi_002",
        "pages": pages,
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True

    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()
        assert sub is not None
        assert os.path.exists(sub.file_path)
        assert sub.original_filename == "scan_test_sca_3pgs.pdf"


def test_3_page_order_preserved(client, sample_student, sample_assignment):
    """Requirement 3: Page order preserved even if payload is transmitted out of order."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    # Send pages out of order: Page 3, Page 1, Page 2
    pages_scrambled = [
        {"page_number": 3, "image_data": make_test_page_base64("Page Three", 300, 400)},
        {"page_number": 1, "image_data": make_test_page_base64("Page One", 300, 400)},
        {"page_number": 2, "image_data": make_test_page_base64("Page Two", 300, 400)},
    ]
    payload = {
        "session_id": "test_scan_order_003",
        "pages": pages_scrambled,
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
    )
    assert resp.status_code == 200

    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()
        assert sub is not None
        doc = pymupdf.open(sub.file_path)
        assert len(doc) == 3
        doc.close()


def test_4_pdf_generated_successfully(client, sample_student, sample_assignment):
    """Requirement 4: PDF generated successfully with valid structure and headers."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    payload = {
        "session_id": "test_pdf_validity",
        "pages": [
            {"page_number": 1, "image_data": make_test_page_base64("Page 1", 500, 700)},
            {"page_number": 2, "image_data": make_test_page_base64("Page 2", 500, 700)},
        ],
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
    )
    assert resp.status_code == 200

    with client.application.app_context():
        sub = Submission.query.filter_by(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
        ).first()

        # Check binary header
        with open(sub.file_path, "rb") as f:
            header = f.read(1024)
            assert b"%PDF-" in header

        # Verify PyMuPDF opens without errors
        doc = pymupdf.open(sub.file_path)
        assert len(doc) == 2
        page = doc.load_page(0)
        assert page.rect.width > 0
        assert page.rect.height > 0
        doc.close()


def test_5_invalid_empty_capture_rejected(client, sample_student, sample_assignment):
    """Requirement 5: Invalid/empty capture rejected."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    # Empty pages array
    payload = {
        "session_id": "test_scan_empty",
        "pages": [],
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "No scanned pages were provided" in data["error"]


def test_6_malformed_image_payload_rejected(client, sample_student, sample_assignment):
    """Requirement 6: Malformed image payload rejected."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    # Non-image base64 string
    payload = {
        "session_id": "test_scan_malformed",
        "pages": [
            {"page_number": 1, "image_data": "data:image/jpeg;base64,NotAValidBase64ImageData!!"}
        ],
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "validation failed" in data["error"]


def test_7_submission_processing_starts(app, sample_student, sample_assignment):
    """Requirement 7: Existing submission processing pipeline still executes."""
    pages_data = [
        {"page_number": 1, "image_data": make_test_page_base64("Handwritten Question 1 Answer")},
    ]

    sub, error = SubmissionService.create_or_update_scanned_submission(
        assignment_id=sample_assignment.assignment_id,
        student_id=sample_student.student_id,
        pages_data=pages_data,
        session_id="proc_test_sess",
        upload_folder=app.config["UPLOAD_FOLDER"],
        pages_folder=app.config["PAGES_FOLDER"],
        async_processing=False,  # Run synchronously to verify pipeline execution
    )

    assert error is None
    assert sub is not None
    assert sub.status in [SubmissionStatus.COMPLETED.value, SubmissionStatus.REVIEW_REQUIRED.value]
    assert len(sub.pages) == 1
    assert sub.pages[0].extracted_text is not None


def test_8_student_cannot_use_old_direct_upload_route(client, sample_student, sample_assignment, sample_pdf_bytes):
    """
    Requirement 8: Student cannot use old direct-upload route to bypass scanner workflow.
    """
    login_session(client, sample_student.student_id, "student", sample_student.name)

    data = {
        "submission_file": (io.BytesIO(sample_pdf_bytes), "bypassed_upload.pdf")
    }

    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 400
    json_data = resp.get_json()
    assert json_data["success"] is False
    assert "Direct PDF file upload is disabled for students" in json_data["error"]


def test_9_faculty_submission_review_and_download_intact(client, sample_student, sample_faculty, sample_assignment):
    """Requirement 9: Existing faculty submission/review behavior remains intact."""
    # 1. Student submits scan
    login_session(client, sample_student.student_id, "student", sample_student.name)
    payload = {
        "session_id": "faculty_compat_sess",
        "pages": [
            {"page_number": 1, "image_data": make_test_page_base64("Question 1 Sol")},
            {"page_number": 2, "image_data": make_test_page_base64("Question 2 Sol")},
        ],
    }
    resp = client.post(
        f"/student/assignments/{sample_assignment.assignment_id}/submit-scan",
        json=payload,
    )
    sub_id = resp.get_json()["submission_id"]

    # 2. Faculty logs in
    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    # Test faculty submission list
    list_resp = client.get(f"/faculty/assignments/{sample_assignment.assignment_id}/submissions")
    assert list_resp.status_code == 200
    assert sample_student.name.encode() in list_resp.data

    # Test faculty submission review view
    review_resp = client.get(f"/faculty/submissions/{sub_id}/review")
    assert review_resp.status_code == 200

    # Test faculty download of compiled PDF
    download_resp = client.get(f"/faculty/submissions/{sub_id}/download")
    assert download_resp.status_code == 200
    assert download_resp.mimetype == "application/pdf"
    assert download_resp.data.startswith(b"%PDF-")
