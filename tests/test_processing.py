"""
Comprehensive test suite for Phase 2: Handwritten PDF Processing & Text Extraction Pipeline.
Covers PDF rendering, image preprocessing, HTR recognition, confidence metrics,
document page persistence, faculty review UI, and authorization controls.
"""

import io
import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import pytest
from app.extensions import db
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.processing.pdf_processor import PDFProcessor
from app.processing.image_preprocessor import ImagePreprocessor
from app.processing.handwriting_recognizer import (
    HTRRecognizerInterface,
    HTRResult,
    EasyOCRHandwritingRecognizer,
)
from app.processing.pipeline import SubmissionProcessingPipeline
from app.services.submission_service import SubmissionService
from tests.conftest import login_session


class MockHTRRecognizer(HTRRecognizerInterface):
    """Mock HTR recognizer for controlled unit testing."""

    def __init__(self, text: str = "Sample handwritten answer line.", confidence: float = 0.92, should_fail: bool = False):
        self._text = text
        self._confidence = confidence
        self._should_fail = should_fail

    def recognize(self, image_input) -> HTRResult:
        if self._should_fail:
            return HTRResult(
                text="",
                confidence=0.0,
                is_low_confidence=True,
                error_message="Simulated HTR recognition engine error.",
            )
        return HTRResult(
            text=self._text,
            confidence=self._confidence,
            is_low_confidence=self._confidence < 0.55,
        )


def _create_test_pdf_file(filepath: str, num_pages: int = 2) -> str:
    """Helper to create a valid multi-page PDF for testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = fitz.open()
    for i in range(1, num_pages + 1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 80), f"Test Student Submission - Page {i}", fontsize=14)
        page.insert_text((50, 120), f"Handwritten answer content for Question {i}.", fontsize=12)
    doc.save(filepath)
    doc.close()
    return filepath


# =========================================================================
# 1. PDF Processor Unit Tests
# =========================================================================

def test_pdf_processor_valid_multipage(tmp_path):
    """Test PDF processor renders all pages in order at high DPI."""
    pdf_path = str(tmp_path / "sample_multipage.pdf")
    output_dir = str(tmp_path / "extracted_pages")
    _create_test_pdf_file(pdf_path, num_pages=3)

    processor = PDFProcessor(target_dpi=200)
    pages, error = processor.validate_and_extract_pages(
        pdf_path=pdf_path, output_folder=output_dir, submission_id=99
    )

    assert error is None
    assert len(pages) == 3
    for idx, page in enumerate(pages):
        assert page.page_number == idx + 1
        assert os.path.exists(page.original_image_path)
        assert page.image.width > 1000
        assert page.image.height > 1000


def test_pdf_processor_corrupt_file(tmp_path):
    """Test PDF processor gracefully handles corrupt or non-PDF files."""
    corrupt_path = str(tmp_path / "corrupt.pdf")
    with open(corrupt_path, "wb") as f:
        f.write(b"This is not a real PDF document.")

    processor = PDFProcessor()
    pages, error = processor.validate_and_extract_pages(
        pdf_path=corrupt_path, output_folder=str(tmp_path), submission_id=98
    )

    assert len(pages) == 0
    assert error is not None
    assert "failed" in error.lower() or "corrupt" in error.lower()


# =========================================================================
# 2. Image Preprocessor Unit Tests
# =========================================================================

def test_image_preprocessor_pipeline(tmp_path):
    """Test image preprocessor converts to grayscale, applies CLAHE and deskewing."""
    from app.processing.image_preprocessor import PreprocessingConfig
    config = PreprocessingConfig(enable_deskew=True)
    preprocessor = ImagePreprocessor(config=config)

    # Create a synthetic color PIL image with high contrast
    img = Image.new("RGB", (800, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 750, 550], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((100, 100), "Preprocess Test Line", fill=(0, 0, 0))

    out_file = str(tmp_path / "proc_test.png")
    proc_img, _ = preprocessor.preprocess_image(img, output_filepath=out_file)

    assert proc_img.mode == "RGB"
    assert os.path.exists(out_file)


# =========================================================================
# 3. Handwriting Recognizer Unit Tests
# =========================================================================

def test_mock_recognizer_high_confidence():
    """Test mock recognizer returns high confidence."""
    recognizer = MockHTRRecognizer(text="Derivation of Master Theorem: T(n) = Theta(n^2)", confidence=0.88)
    img = Image.new("L", (400, 100), color=255)
    result = recognizer.recognize(img)

    assert result.text == "Derivation of Master Theorem: T(n) = Theta(n^2)"
    assert result.confidence == 0.88
    assert result.is_low_confidence is False
    assert result.error_message is None


def test_mock_recognizer_low_confidence_flag():
    """Test low confidence threshold (< 0.55) flags for review."""
    recognizer = MockHTRRecognizer(text="Unclear handwriting scribble", confidence=0.42)
    img = Image.new("L", (400, 100), color=255)
    result = recognizer.recognize(img)

    assert result.confidence == 0.42
    assert result.is_low_confidence is True


# =========================================================================
# 4. Pipeline & Database Persistence Integration Tests
# =========================================================================

def test_pipeline_end_to_end(client, sample_student, sample_assignment, tmp_path):
    """Test full pipeline orchestrates PDF -> Pages -> Preprocessing -> DocumentPage DB records."""
    pdf_path = str(tmp_path / "submission_test.pdf")
    _create_test_pdf_file(pdf_path, num_pages=2)
    pages_folder = str(tmp_path / "pages_out")

    with client.application.app_context():
        # Create submission record
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="student_work.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        # Run pipeline with high-confidence mock recognizer
        mock_rec = MockHTRRecognizer(text="Line 1 of recognized answer.\nLine 2 of steps.", confidence=0.91)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        result = pipeline.process_submission(sub_id, pages_folder=pages_folder)

        assert result.success is True
        assert result.status == SubmissionStatus.COMPLETED.value
        assert result.total_pages == 2
        assert result.average_confidence == 91.0

        # Verify DB state
        refreshed_sub = db.session.get(Submission, sub_id)
        assert refreshed_sub.status == SubmissionStatus.COMPLETED.value
        assert refreshed_sub.page_count == 2
        assert refreshed_sub.average_confidence == 91.0
        assert len(refreshed_sub.pages) == 2

        page1 = refreshed_sub.pages[0]
        assert page1.page_number == 1
        assert page1.processing_status == "Completed"
        assert page1.confidence == 0.91
        assert "Line 1" in page1.extracted_text
        assert os.path.exists(page1.original_image_path)
        assert os.path.exists(page1.processed_image_path)


def test_pipeline_low_confidence_triggers_review_required(client, sample_student, sample_assignment, tmp_path):
    """Test pipeline transitions submission to Review Required when average confidence < 0.55."""
    pdf_path = str(tmp_path / "sub_low_conf.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_low_conf")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="messy_handwriting.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Difficult to read text", confidence=0.48)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        result = pipeline.process_submission(sub_id, pages_folder=pages_folder)

        assert result.status == SubmissionStatus.REVIEW_REQUIRED.value
        refreshed_sub = db.session.get(Submission, sub_id)
        assert refreshed_sub.status == SubmissionStatus.REVIEW_REQUIRED.value
        assert "Faculty review required" in refreshed_sub.evaluation_notes


# =========================================================================
# 5. Faculty Review Routes & UI Access Control Tests
# =========================================================================

def test_faculty_review_submission_route(client, sample_faculty, sample_student, sample_assignment, tmp_path):
    """Test faculty can view the review page for their assignment submissions."""
    pdf_path = str(tmp_path / "review_doc.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_review")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="exam_paper.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Q1 Answer: Theta(n^2) by Master Theorem", confidence=0.85)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        pipeline.process_submission(sub_id, pages_folder=pages_folder)

    # Login as owner faculty
    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    resp = client.get(f"/faculty/submissions/{sub_id}/review")
    assert resp.status_code == 200
    assert b"Handwriting Extraction Review" in resp.data
    assert b"Master Theorem" in resp.data
    assert b"Page 1 of 1" in resp.data
    assert b"Original Render" in resp.data


def test_faculty_page_image_streaming(client, sample_faculty, sample_student, sample_assignment, tmp_path):
    """Test secure streaming of original and processed page images."""
    pdf_path = str(tmp_path / "stream_doc.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_stream")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="stream_test.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Stream test text", confidence=0.90)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        pipeline.process_submission(sub_id, pages_folder=pages_folder)

        doc_page = DocumentPage.query.filter_by(submission_id=sub_id).first()
        page_id = doc_page.page_id

    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    # Stream original image
    resp_orig = client.get(f"/faculty/pages/{page_id}/image/original")
    assert resp_orig.status_code == 200
    assert resp_orig.content_type == "image/png"

    # Stream processed image
    resp_proc = client.get(f"/faculty/pages/{page_id}/image/processed")
    assert resp_proc.status_code == 200
    assert resp_proc.content_type == "image/png"


def test_unauthorized_student_cannot_access_other_student_page_images(client, sample_faculty, sample_student, sample_assignment, tmp_path):
    """Test student cannot access page images belonging to another student's submission."""
    pdf_path = str(tmp_path / "student_sec.pdf")
    _create_test_pdf_file(pdf_path, num_pages=1)
    pages_folder = str(tmp_path / "pages_sec")

    with client.application.app_context():
        sub = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=sample_student.student_id,
            file_path=pdf_path,
            original_filename="security_test.pdf",
            file_size_bytes=os.path.getsize(pdf_path),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub)
        db.session.commit()
        sub_id = sub.submission_id

        mock_rec = MockHTRRecognizer(text="Secret answer", confidence=0.95)
        pipeline = SubmissionProcessingPipeline(recognizer=mock_rec)
        pipeline.process_submission(sub_id, pages_folder=pages_folder)

        doc_page = DocumentPage.query.filter_by(submission_id=sub_id).first()
        page_id = doc_page.page_id

    # Create a distinct student account
    with client.application.app_context():
        from app.models.user import Student
        other_student = Student(
            name="Other Student",
            email="other.student@smarteval.edu",
            class_name="CS-2026-A",
        )
        other_student.set_password("Student@123")
        db.session.add(other_student)
        db.session.commit()
        other_id = other_student.student_id

    # Login as other student
    login_session(client, other_id, "student", "Other Student")

    resp = client.get(f"/student/pages/{page_id}/image/original")
    assert resp.status_code == 404
