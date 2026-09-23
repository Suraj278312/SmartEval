import base64
import io
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Union, Dict, Any
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from flask import current_app
from app.extensions import db
from app.models.assignment import Assignment
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.models.user import Faculty, Student

logger = logging.getLogger(__name__)


def _run_async_pipeline(app, submission_id: int, pages_folder: str) -> None:
    """
    Execute document processing pipeline in a background thread.
    Uses an isolated application context and safe database session.
    """
    with app.app_context():
        logger.info("Starting background HTR processing for submission #%d...", submission_id)
        try:
            from app.processing.pipeline import SubmissionProcessingPipeline
            pipeline = SubmissionProcessingPipeline()
            result = pipeline.process_submission(submission_id, pages_folder=pages_folder)
            logger.info(
                "Background processing finished for submission #%d: status=%s, pages=%d, conf=%.1f%%, matched=%d",
                submission_id,
                result.status,
                result.total_pages,
                result.average_confidence,
                result.matched_answers_count,
            )
        except Exception as e:
            logger.exception("Background processing error for submission #%d: %s", submission_id, str(e))
            try:
                sub = db.session.get(Submission, submission_id)
                if sub:
                    sub.status = SubmissionStatus.FAILED.value
                    sub.evaluation_notes = f"Background processing failed: {str(e)}"
                    db.session.commit()
            except Exception as db_err:
                logger.error("Failed to record failure status for submission #%d: %s", submission_id, str(db_err))


class SubmissionService:
    """Service handling file validation, storage, submission tracking, and HTR processing."""

    PDF_MAGIC_BYTES = b"%PDF-"

    @classmethod
    def validate_pdf_file(
        cls, file: Optional[FileStorage], max_bytes: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform strict multi-layer PDF validation:
        1. File presence
        2. Filename extension (.pdf)
        3. Magic bytes / header inspection (%PDF-)
        4. File size limits
        """
        if not file or not file.filename or file.filename.strip() == "":
            return False, "No file was selected for upload."

        # 1. Extension check
        filename = file.filename.lower()
        if not filename.endswith(".pdf"):
            return False, "Invalid file format. Only PDF (.pdf) documents are accepted."

        # 2. Magic byte check
        file.seek(0)
        header = file.read(1024)
        if not header.startswith(cls.PDF_MAGIC_BYTES) and cls.PDF_MAGIC_BYTES not in header[:1024]:
            file.seek(0)
            return False, "Invalid PDF file. The file header does not match standard PDF signatures."

        # 3. Size check
        file.seek(0, os.SEEK_END)
        size_bytes = file.tell()
        file.seek(0)  # Reset pointer to start

        if size_bytes == 0:
            return False, "The uploaded file is empty (0 bytes)."

        if size_bytes > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            return (
                False,
                f"File size exceeds the allowable limit of {max_mb:.0f} MB.",
            )

        return True, None

    @classmethod
    def decode_and_validate_page_image(
        cls, raw_data: Union[str, bytes, FileStorage]
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Safely decode and validate an in-app scanner captured image:
        - Decodes base64 data URL string or raw bytes
        - Inspects magic bytes (PNG / JPEG / WebP)
        - Verifies valid dimensions and decodability with PIL
        - Returns (PIL.Image in RGB mode, error_message)
        """
        if raw_data is None:
            return None, "Empty page image data received."

        img_bytes: Optional[bytes] = None
        if isinstance(raw_data, FileStorage):
            img_bytes = raw_data.read()
        elif isinstance(raw_data, bytes):
            img_bytes = raw_data
        elif isinstance(raw_data, str):
            # Check for data URL header: data:image/jpeg;base64,...
            clean_str = raw_data.strip()
            if not clean_str:
                return None, "Empty page image data received."
            if "," in clean_str:
                _, encoded = clean_str.split(",", 1)
                try:
                    img_bytes = base64.b64decode(encoded)
                except Exception:
                    return None, "Malformed base64 image data payload."
            else:
                try:
                    img_bytes = base64.b64decode(clean_str)
                except Exception:
                    return None, "Malformed base64 image data payload."
        else:
            return None, "Unsupported image payload type."

        if not img_bytes or len(img_bytes) == 0:
            return None, "Page image payload contains 0 bytes."

        # Magic byte check for standard image signatures
        is_png = img_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = img_bytes.startswith(b"\xff\xd8\xff")
        is_webp = img_bytes.startswith(b"RIFF") and b"WEBP" in img_bytes[:16]

        if not (is_png or is_jpeg or is_webp):
            return None, "Invalid image format. Captured pages must be valid PNG or JPEG images."

        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.verify()  # Verify image integrity
            # Reopen because verify() closes the file pointer
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.load()

            # Dimension checks
            if pil_img.width < 50 or pil_img.height < 50:
                return None, f"Captured page image dimensions ({pil_img.width}x{pil_img.height}) are too small."
            if pil_img.width > 8000 or pil_img.height > 8000:
                return None, f"Captured page image dimensions ({pil_img.width}x{pil_img.height}) exceed maximum allowed size."

            # Ensure image is in RGB mode for standard PDF compilation
            if pil_img.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                if pil_img.mode == "P":
                    pil_img = pil_img.convert("RGBA")
                if "A" in pil_img.getbands():
                    bg.paste(pil_img, mask=pil_img.split()[-1])
                else:
                    bg.paste(pil_img)
                return bg, None
            elif pil_img.mode != "RGB":
                return pil_img.convert("RGB"), None
            return pil_img, None

        except Exception as e:
            return None, f"Corrupted or invalid image data: {str(e)}"

    @classmethod
    def create_or_update_scanned_submission(
        cls,
        assignment_id: int,
        student_id: int,
        pages_data: List[Union[Dict[str, Any], Any]],
        session_id: Optional[str] = None,
        upload_folder: Optional[str] = None,
        pages_folder: Optional[str] = None,
        max_bytes: int = 16 * 1024 * 1024,
        run_pipeline: bool = True,
        async_processing: bool = True,
    ) -> Tuple[Optional[Submission], Optional[str]]:
        """
        Validate and securely compile multiple in-app camera scanned pages into a single PDF submission.
        Preserves page ordering, records capture provenance, replaces old file on resubmission,
        and triggers the HTR processing pipeline.
        """
        # Ensure assignment exists and is published
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, published=True
        ).first()

        if not assignment:
            return None, "Assignment is not available or has been unpublished."

        if not pages_data or not isinstance(pages_data, list) or len(pages_data) == 0:
            return None, "No scanned pages were provided. Please scan at least one page."

        if len(pages_data) > 50:
            return None, f"Scanned page count ({len(pages_data)}) exceeds maximum limit of 50 pages."

        # Sort pages by page_number if dictionary contains it
        def _get_page_sort_key(item, idx):
            if isinstance(item, dict):
                return item.get("page_number", idx + 1)
            elif isinstance(item, (tuple, list)) and len(item) >= 1 and isinstance(item[0], int):
                return item[0]
            return idx + 1

        try:
            sorted_pages = sorted(
                list(enumerate(pages_data)),
                key=lambda pair: _get_page_sort_key(pair[1], pair[0])
            )
        except Exception:
            sorted_pages = list(enumerate(pages_data))

        validated_images: List[Image.Image] = []
        for seq_idx, (orig_idx, page_item) in enumerate(sorted_pages):
            raw_img = page_item
            if isinstance(page_item, dict):
                raw_img = (
                    page_item.get("image_data")
                    or page_item.get("image")
                    or page_item.get("data")
                    or page_item.get("file")
                )
            elif isinstance(page_item, (tuple, list)) and len(page_item) >= 2:
                raw_img = page_item[1]

            pil_img, err = cls.decode_and_validate_page_image(raw_img)
            if err or not pil_img:
                return None, f"Page {seq_idx + 1} validation failed: {err}"
            validated_images.append(pil_img)

        try:
            if not upload_folder:
                upload_folder = current_app.config["UPLOAD_FOLDER"]
            os.makedirs(upload_folder, exist_ok=True)
            if not pages_folder:
                pages_folder = current_app.config.get(
                    "PAGES_FOLDER",
                    os.path.join(os.path.dirname(upload_folder), "pages")
                )
            os.makedirs(pages_folder, exist_ok=True)

            scan_session = session_id or uuid.uuid4().hex[:16]
            unique_token = uuid.uuid4().hex[:12]
            stored_filename = f"sub_a{assignment_id}_s{student_id}_{unique_token}.pdf"
            stored_filepath = os.path.join(upload_folder, stored_filename)

            # Compile into standard multi-page PDF
            first_page = validated_images[0]
            if len(validated_images) == 1:
                first_page.save(stored_filepath, format="PDF", resolution=150.0)
            else:
                first_page.save(
                    stored_filepath,
                    format="PDF",
                    save_all=True,
                    append_images=validated_images[1:],
                    resolution=150.0,
                )

            # Validate generated PDF size
            size_bytes = os.path.getsize(stored_filepath)
            if size_bytes > max_bytes:
                try:
                    os.remove(stored_filepath)
                except OSError:
                    pass
                max_mb = max_bytes / (1024 * 1024)
                return None, f"Generated PDF size exceeds the limit of {max_mb:.0f} MB."

            # Provenance metadata
            capture_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            original_name = f"scan_{scan_session[:8]}_{len(validated_images)}pgs.pdf"
            provenance_note = (
                f"In-App Camera Scan | Session: {scan_session[:16]} | "
                f"{len(validated_images)} page(s) | Submitted: {capture_time_str}"
            )

            # Check for existing submission by this student for this assignment
            submission = Submission.query.filter_by(
                assignment_id=assignment_id, student_id=student_id
            ).first()

            old_file_to_remove = None
            if submission:
                old_file_to_remove = submission.file_path
                submission.file_path = stored_filepath
                submission.original_filename = original_name
                submission.file_size_bytes = size_bytes
                submission.submission_date = datetime.now(timezone.utc)
                submission.status = SubmissionStatus.PROCESSING.value
                submission.evaluation_notes = provenance_note
            else:
                submission = Submission(
                    assignment_id=assignment_id,
                    student_id=student_id,
                    file_path=stored_filepath,
                    original_filename=original_name,
                    file_size_bytes=size_bytes,
                    submission_date=datetime.now(timezone.utc),
                    status=SubmissionStatus.PROCESSING.value,
                    evaluation_notes=provenance_note,
                )
                db.session.add(submission)

            db.session.commit()

            # Clean up old file if resubmitted
            if old_file_to_remove and os.path.exists(old_file_to_remove) and old_file_to_remove != stored_filepath:
                try:
                    os.remove(old_file_to_remove)
                except OSError:
                    pass

            # Trigger HTR document processing pipeline
            if run_pipeline:
                if async_processing:
                    app_obj = current_app._get_current_object()
                    worker = threading.Thread(
                        target=_run_async_pipeline,
                        args=(app_obj, submission.submission_id, pages_folder),
                        daemon=True,
                        name=f"HTR-Worker-Sub-{submission.submission_id}",
                    )
                    worker.start()
                else:
                    from app.processing.pipeline import SubmissionProcessingPipeline
                    pipeline = SubmissionProcessingPipeline()
                    pipeline.process_submission(submission.submission_id, pages_folder=pages_folder)
                    submission = db.session.get(Submission, submission.submission_id)

            return submission, None

        except Exception as e:
            db.session.rollback()
            return None, f"Failed to save scanned submission: {str(e)}"

    @classmethod
    def create_or_update_submission(
        cls,
        assignment_id: int,
        student_id: int,
        file: FileStorage,
        upload_folder: str,
        pages_folder: Optional[str] = None,
        max_bytes: int = 16 * 1024 * 1024,
        run_pipeline: bool = True,
        async_processing: bool = True,
    ) -> Tuple[Optional[Submission], Optional[str]]:
        """
        Validate and securely save an uploaded handwritten assignment PDF.
        Supports resubmission by updating the existing record and replacing old file.
        Automatically triggers HTR document processing pipeline in a background thread.
        """
        # Ensure assignment exists and is published
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, published=True
        ).first()

        if not assignment:
            return None, "Assignment is not available or has been unpublished."

        # Validate PDF
        is_valid, error = cls.validate_pdf_file(file, max_bytes)
        if not is_valid:
            return None, error

        try:
            os.makedirs(upload_folder, exist_ok=True)
            if not pages_folder:
                pages_folder = os.path.join(os.path.dirname(upload_folder), "pages")
            os.makedirs(pages_folder, exist_ok=True)

            original_name = secure_filename(file.filename) or "submission.pdf"
            file.seek(0, os.SEEK_END)
            size_bytes = file.tell()
            file.seek(0)

            # Unique non-colliding storage filename
            unique_token = uuid.uuid4().hex[:12]
            stored_filename = f"sub_a{assignment_id}_s{student_id}_{unique_token}.pdf"
            stored_filepath = os.path.join(upload_folder, stored_filename)

            # Check for existing submission by this student for this assignment
            submission = Submission.query.filter_by(
                assignment_id=assignment_id, student_id=student_id
            ).first()

            old_file_to_remove = None
            if submission:
                old_file_to_remove = submission.file_path
                submission.file_path = stored_filepath
                submission.original_filename = original_name
                submission.file_size_bytes = size_bytes
                submission.submission_date = datetime.now(timezone.utc)
                submission.status = SubmissionStatus.PROCESSING.value
                submission.evaluation_notes = "Document processing and text extraction in progress..."
            else:
                submission = Submission(
                    assignment_id=assignment_id,
                    student_id=student_id,
                    file_path=stored_filepath,
                    original_filename=original_name,
                    file_size_bytes=size_bytes,
                    submission_date=datetime.now(timezone.utc),
                    status=SubmissionStatus.PROCESSING.value,
                    evaluation_notes="Document processing and text extraction in progress...",
                )
                db.session.add(submission)

            # Save new file to disk
            file.save(stored_filepath)

            # Commit DB changes
            db.session.commit()

            # Clean up old file if resubmitted
            if old_file_to_remove and os.path.exists(old_file_to_remove) and old_file_to_remove != stored_filepath:
                try:
                    os.remove(old_file_to_remove)
                except OSError:
                    pass

            # Trigger HTR document processing pipeline
            if run_pipeline:
                if async_processing:
                    app_obj = current_app._get_current_object()
                    worker = threading.Thread(
                        target=_run_async_pipeline,
                        args=(app_obj, submission.submission_id, pages_folder),
                        daemon=True,
                        name=f"HTR-Worker-Sub-{submission.submission_id}",
                    )
                    worker.start()
                else:
                    from app.processing.pipeline import SubmissionProcessingPipeline
                    pipeline = SubmissionProcessingPipeline()
                    pipeline.process_submission(submission.submission_id, pages_folder=pages_folder)
                    submission = db.session.get(Submission, submission.submission_id)

            return submission, None

        except Exception as e:
            db.session.rollback()
            return None, f"Failed to save submission: {str(e)}"

    @classmethod
    def retry_submission_processing(
        cls,
        submission_id: int,
        user: Union[Faculty, Student],
        pages_folder: Optional[str] = None,
        async_processing: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Reprocess an existing submission through the HTR pipeline.
        Enforces authorization (faculty owner or submitting student).
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return False, "Submission not found."

        # Authorization check
        if user.role == "student":
            if submission.student_id != user.student_id:
                return False, "Unauthorized: You can only reprocess your own submissions."
        elif user.role == "faculty":
            if submission.assignment.faculty_id != user.faculty_id:
                return False, "Unauthorized: You do not own this assignment."
        else:
            return False, "Unauthorized access."

        if not os.path.exists(submission.file_path):
            return False, "Original submission PDF file is missing from disk."

        if not pages_folder:
            pages_folder = os.path.join(os.path.dirname(submission.file_path), "..", "pages")
            pages_folder = os.path.abspath(pages_folder)

        submission.status = SubmissionStatus.PROCESSING.value
        submission.evaluation_notes = "Document processing and text extraction in progress..."
        db.session.commit()

        if async_processing:
            app_obj = current_app._get_current_object()
            worker = threading.Thread(
                target=_run_async_pipeline,
                args=(app_obj, submission.submission_id, pages_folder),
                daemon=True,
                name=f"HTR-Worker-Sub-{submission.submission_id}",
            )
            worker.start()
            return True, None
        else:
            from app.processing.pipeline import SubmissionProcessingPipeline
            pipeline = SubmissionProcessingPipeline()
            res = pipeline.process_submission(submission.submission_id, pages_folder=pages_folder)
            return res.success, res.error_message

    @staticmethod
    def get_student_submissions(student_id: int) -> List[Submission]:
        """Fetch all submissions made by a student."""
        return (
            Submission.query.filter_by(student_id=student_id)
            .order_by(Submission.submission_date.desc())
            .all()
        )

    @staticmethod
    def get_assignment_submissions_for_faculty(
        assignment_id: int, faculty_id: int
    ) -> Tuple[Optional[Assignment], List[Submission], Optional[str]]:
        """Fetch all student submissions for an assignment owned by faculty."""
        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id, faculty_id=faculty_id
        ).first()

        if not assignment:
            return None, [], "Assignment not found or unauthorized access."

        submissions = (
            Submission.query.filter_by(assignment_id=assignment_id)
            .order_by(Submission.submission_date.desc())
            .all()
        )
        return assignment, submissions, None

    @staticmethod
    def get_submission_for_review(
        submission_id: int, faculty_id: int
    ) -> Tuple[Optional[Submission], Optional[str]]:
        """
        Fetch submission with full page extraction details for faculty review.
        Enforces faculty assignment ownership.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission record not found."

        if submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this submission."

        return submission, None

    @staticmethod
    def get_submission_file_for_user(
        submission_id: int, user: Union[Faculty, Student]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Authorize and return filepath for submission PDF download/view.
        Ensures students only access their own submissions and faculty only access
        submissions for their assignments.
        Returns (file_path, original_filename, error_message).
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, None, "Submission record not found."

        # Authorization check
        if user.role == "student":
            if submission.student_id != user.student_id:
                return None, None, "Unauthorized: You may only access your own submissions."
        elif user.role == "faculty":
            if submission.assignment.faculty_id != user.faculty_id:
                return None, None, "Unauthorized: You do not own this assignment."
        else:
            return None, None, "Unauthorized access."

        if not os.path.exists(submission.file_path):
            return None, None, "Submission file not found on disk."

        return submission.file_path, submission.original_filename, None

    @staticmethod
    def get_page_image_for_user(
        page_id: int, image_type: str, user: Union[Faculty, Student]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Authorize and return the filepath to a DocumentPage rendered or preprocessed PNG image.
        image_type: 'original' or 'processed'.
        Enforces role-based ownership.
        Returns (file_path, error_message).
        """
        page = db.session.get(DocumentPage, page_id)
        if not page:
            return None, "Page record not found."

        submission = page.submission
        if not submission:
            return None, "Associated submission not found."

        if not user:
            return None, "Unauthorized access."

        # Authorization
        if user.role == "student":
            if submission.student_id != user.student_id:
                return None, "Unauthorized: You may only access images of your own submission."
        elif user.role == "faculty":
            if submission.assignment.faculty_id != user.faculty_id:
                return None, "Unauthorized: You do not own this assignment."
        else:
            return None, "Unauthorized access."

        if image_type == "original":
            target_path = page.original_image_path
        elif image_type == "processed":
            target_path = page.processed_image_path
        else:
            return None, "Invalid image type requested. Expected 'original' or 'processed'."

        if not target_path or not os.path.exists(target_path):
            return None, "Requested page image file does not exist on disk."

        return target_path, None
