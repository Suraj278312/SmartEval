"""
Submission Processing Pipeline for SmartEval Phase 2.
Orchestrates PDF validation, page conversion, image preprocessing, HTR extraction,
confidence evaluation, and database persistence.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from app.extensions import db
from app.models.submission import Submission, SubmissionStatus
from app.models.document_page import DocumentPage
from app.models.submission_answer import SubmissionAnswer
from app.processing.pdf_processor import PDFProcessor
from app.processing.image_preprocessor import ImagePreprocessor
from app.processing.handwriting_recognizer import (
    HTRRecognizerInterface,
    get_default_recognizer,
)
from app.processing.qa_matcher import HierarchicalQAMatcher


@dataclass
class PipelineResult:
    """Outcome of document processing pipeline execution."""
    submission_id: int
    success: bool
    status: str
    total_pages: int
    average_confidence: float
    matched_answers_count: int = 0
    review_required_answers_count: int = 0
    error_message: Optional[str] = None


class SubmissionProcessingPipeline:
    """
    End-to-end document processing pipeline:
    PDF -> Render Page Images -> Preprocess Pages -> HTR Recognition -> Save DocumentPage records ->
    Q&A Segmentation & Question Matching -> Save SubmissionAnswer records -> Update Submission.
    """

    def __init__(
        self,
        pdf_processor: Optional[PDFProcessor] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        recognizer: Optional[HTRRecognizerInterface] = None,
        qa_matcher: Optional[HierarchicalQAMatcher] = None,
    ):
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.recognizer = recognizer or get_default_recognizer()
        self.qa_matcher = qa_matcher or HierarchicalQAMatcher()

    def process_submission(
        self, submission_id: int, pages_folder: str
    ) -> PipelineResult:
        """
        Execute full document extraction pipeline on a submission record.
        Safely captures failures and updates the database record.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return PipelineResult(
                submission_id=submission_id,
                success=False,
                status=SubmissionStatus.FAILED.value,
                total_pages=0,
                average_confidence=0.0,
                error_message="Submission record not found.",
            )

        # 1. Update status to Processing
        submission.status = SubmissionStatus.PROCESSING.value
        submission.evaluation_notes = "Document processing and text extraction in progress..."
        db.session.commit()

        # Clean up any existing DocumentPage and SubmissionAnswer records and artifacts for this submission
        for existing_page in submission.pages:
            for path in [existing_page.original_image_path, existing_page.processed_image_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        SubmissionAnswer.query.filter_by(submission_id=submission_id).delete()
        DocumentPage.query.filter_by(submission_id=submission_id).delete()
        db.session.commit()

        # 2. Render PDF pages to PNG images
        os.makedirs(pages_folder, exist_ok=True)
        extracted_pages, pdf_error = self.pdf_processor.validate_and_extract_pages(
            pdf_path=submission.file_path,
            output_folder=pages_folder,
            submission_id=submission.submission_id,
        )

        if pdf_error or not extracted_pages:
            submission.status = SubmissionStatus.FAILED.value
            submission.evaluation_notes = f"PDF Processing Failed: {pdf_error or 'Unknown error'}"
            db.session.commit()
            return PipelineResult(
                submission_id=submission_id,
                success=False,
                status=SubmissionStatus.FAILED.value,
                total_pages=0,
                average_confidence=0.0,
                matched_answers_count=0,
                review_required_answers_count=0,
                error_message=pdf_error,
            )

        # 3. Preprocess and Run HTR on each page
        has_review_required = False
        has_failure = False
        confidences: List[float] = []
        doc_pages: List[DocumentPage] = []

        try:
            for page_data in extracted_pages:
                page_num = page_data.page_number
                proc_filename = f"proc_sub_{submission.submission_id}_p{page_num}.png"
                proc_filepath = os.path.join(pages_folder, proc_filename)

                # Preprocess image
                processed_pil, _ = self.image_preprocessor.preprocess_image(
                    image_input=page_data.image,
                    output_filepath=proc_filepath,
                )

                # Run Handwriting Recognition (HTR)
                htr_result = self.recognizer.recognize(processed_pil)

                # Determine page status
                if htr_result.error_message:
                    page_status = "Failed"
                    has_failure = True
                elif htr_result.is_low_confidence:
                    page_status = "Review Required"
                    has_review_required = True
                else:
                    page_status = "Completed"

                if htr_result.confidence is not None:
                    confidences.append(htr_result.confidence)

                # Create DocumentPage record
                doc_page = DocumentPage(
                    submission_id=submission.submission_id,
                    page_number=page_num,
                    original_image_path=page_data.original_image_path,
                    processed_image_path=proc_filepath,
                    extracted_text=htr_result.text,
                    confidence=htr_result.confidence,
                    processing_status=page_status,
                    error_message=htr_result.error_message,
                )
                doc_page.submission = submission
                db.session.add(doc_page)
                doc_pages.append(doc_page)

            db.session.flush()

            # 4. Phase 3A: Question/Answer Segmentation and Question Matching
            assignment_questions = submission.assignment.questions if submission.assignment else []
            matched_results = self.qa_matcher.process_and_match(
                pages=doc_pages,
                questions=assignment_questions,
            )

            matched_count = 0
            review_req_count = 0

            for match in matched_results:
                ans_record = SubmissionAnswer(
                    submission_id=submission.submission_id,
                    question_id=match.question_id,
                    extracted_text=match.extracted_text,
                    detected_label=match.detected_label,
                    page_start=match.page_start,
                    page_end=match.page_end,
                    match_method=match.match_method,
                    match_confidence=match.match_confidence,
                    status=match.status,
                    review_notes=match.review_notes,
                )
                db.session.add(ans_record)
                if match.status == "Matched":
                    matched_count += 1
                else:
                    review_req_count += 1

            db.session.flush()

            # 5. Final Submission Status Determination
            avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

            if has_failure:
                submission.status = SubmissionStatus.FAILED.value
                submission.evaluation_notes = "One or more pages encountered recognition errors."
            elif has_review_required or avg_conf < 0.55:
                submission.status = SubmissionStatus.REVIEW_REQUIRED.value
                submission.evaluation_notes = (
                    f"Text extracted across {len(extracted_pages)} page(s) with low confidence "
                    f"({round(avg_conf * 100, 1)}%). Faculty review required."
                )
            else:
                submission.status = SubmissionStatus.COMPLETED.value
                submission.evaluation_notes = (
                    f"Successfully extracted text from {len(extracted_pages)} page(s) "
                    f"({matched_count} question(s) matched) with average confidence {round(avg_conf * 100, 1)}%."
                )

            db.session.commit()

            return PipelineResult(
                submission_id=submission_id,
                success=not has_failure,
                status=submission.status,
                total_pages=len(extracted_pages),
                average_confidence=round(avg_conf * 100, 1),
                matched_answers_count=matched_count,
                review_required_answers_count=review_req_count,
                error_message=None if not has_failure else submission.evaluation_notes,
            )

        except Exception as e:
            db.session.rollback()
            submission.status = SubmissionStatus.FAILED.value
            submission.evaluation_notes = f"Pipeline execution failed unexpectedly: {str(e)}"
            db.session.commit()
            return PipelineResult(
                submission_id=submission_id,
                success=False,
                status=SubmissionStatus.FAILED.value,
                total_pages=len(extracted_pages),
                average_confidence=0.0,
                matched_answers_count=0,
                review_required_answers_count=0,
                error_message=str(e),
            )
