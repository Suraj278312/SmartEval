"""
Evaluation service for SmartEval Phase 3B.
Coordinates AI semantic evaluation of student submission answers against assignment questions,
supportive answers, and rubrics, and handles faculty approval and score overrides.
"""

import logging
from typing import Optional, Tuple, List, Union
from app.extensions import db
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.user import Faculty, Student
from app.processing.evaluator import (
    BaseSemanticEvaluator,
    GeminiSemanticEvaluator,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service orchestrating AI semantic evaluation and faculty grading workflows."""

    @classmethod
    def evaluate_submission_answers(
        cls,
        submission_id: int,
        faculty_id: int,
        evaluator: Optional[BaseSemanticEvaluator] = None,
    ) -> Tuple[List[AnswerEvaluation], Optional[str]]:
        """
        Evaluate all question answers for a submission using the semantic evaluator.
        Enforces faculty assignment ownership.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return [], "Submission record not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return [], "Unauthorized: You do not own the assignment for this submission."

        if not submission.assignment.questions:
            return [], "Assignment has no questions configured to evaluate."

        # Guard: Prevent evaluating while submission is still in processing state
        if submission.status == SubmissionStatus.PROCESSING.value:
            return [], "Submission is currently being processed. Please wait until text extraction is complete."

        # Guard: Must have extracted pages or segmented answers
        if not submission.pages and not submission.answers:
            return [], "No extracted pages or recognized answers are available for this submission."

        # If submission has pages but answers haven't been segmented yet, trigger segmentation
        if not submission.answers and submission.pages:
            try:
                from app.processing.qa_matcher import HierarchicalQAMatcher
                matcher = HierarchicalQAMatcher()
                matcher.segment_and_match_submission(submission)
                db.session.commit()
                # Refresh submission
                submission = db.session.get(Submission, submission_id)
            except Exception as e:
                logger.warning(f"Auto QA segmentation failed before evaluation: {e}")

        # Initialize evaluator (default to GeminiSemanticEvaluator)
        if evaluator is None:
            evaluator = GeminiSemanticEvaluator()

        evaluations: List[AnswerEvaluation] = []
        questions = sorted(submission.assignment.questions, key=lambda q: q.question_number)
        avg_htr_conf = (submission.average_confidence / 100.0) if submission.average_confidence else 1.0

        try:
            for question in questions:
                # Find matching SubmissionAnswer if present
                matched_answer = next(
                    (a for a in submission.answers if a.question_id == question.question_id),
                    None,
                )

                extracted_text = matched_answer.extracted_text if matched_answer else ""
                match_conf = (
                    matched_answer.match_confidence
                    if (matched_answer and matched_answer.match_confidence is not None)
                    else (1.0 if extracted_text else 0.0)
                )

                # Execute evaluation
                res: EvaluationResult = evaluator.evaluate_answer(
                    question_text=question.question_text,
                    supportive_answer=question.supportive_answer or "",
                    student_answer=extracted_text,
                    maximum_marks=float(question.maximum_marks),
                    rubric=question.rubric,
                    match_confidence=match_conf,
                    htr_confidence=avg_htr_conf,
                )

                # Check if evaluation record already exists
                existing_eval = AnswerEvaluation.query.filter_by(
                    submission_id=submission_id, question_id=question.question_id
                ).first()

                criteria_dict_list = [c.to_dict() for c in res.criteria]

                # Determine status
                if res.score is None or res.error_message:
                    eval_status = "Failed"
                    needs_review = True
                else:
                    eval_status = "Evaluated"
                    needs_review = res.needs_faculty_review

                if existing_eval:
                    existing_eval.answer_id = matched_answer.answer_id if matched_answer else None
                    existing_eval.extracted_answer = extracted_text
                    existing_eval.ai_score = res.score
                    existing_eval.maximum_marks = float(question.maximum_marks)
                    existing_eval.criteria_scores = criteria_dict_list
                    existing_eval.strengths = res.strengths
                    existing_eval.missing_elements = res.missing_elements
                    existing_eval.feedback = res.feedback
                    existing_eval.ai_confidence = res.confidence
                    existing_eval.semantic_similarity_score = res.semantic_similarity_score
                    existing_eval.needs_faculty_review = needs_review
                    existing_eval.error_message = res.error_message
                    if existing_eval.evaluation_status not in ("Approved", "Modified"):
                        existing_eval.evaluation_status = eval_status
                    eval_record = existing_eval
                else:
                    eval_record = AnswerEvaluation(
                        submission_id=submission_id,
                        question_id=question.question_id,
                        answer_id=matched_answer.answer_id if matched_answer else None,
                        extracted_answer=extracted_text,
                        ai_score=res.score,
                        maximum_marks=float(question.maximum_marks),
                        criteria_scores=criteria_dict_list,
                        strengths=res.strengths,
                        missing_elements=res.missing_elements,
                        feedback=res.feedback,
                        ai_confidence=res.confidence,
                        semantic_similarity_score=res.semantic_similarity_score,
                        evaluation_status=eval_status,
                        needs_faculty_review=needs_review,
                        error_message=res.error_message,
                    )
                    db.session.add(eval_record)

                evaluations.append(eval_record)

            # Update overall submission status
            if any(e.needs_faculty_review for e in evaluations):
                submission.status = SubmissionStatus.REVIEW_REQUIRED.value
            else:
                submission.status = SubmissionStatus.COMPLETED.value

            db.session.commit()

            # Phase 4: Sync preliminary result calculation
            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(submission_id=submission_id, faculty_id=faculty_id)

            return evaluations, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to evaluate submission {submission_id}: {str(e)}", exc_info=True)
            return [], f"Evaluation process failed: {str(e)}"

    @classmethod
    def approve_evaluation(
        cls, evaluation_id: int, faculty_id: int
    ) -> Tuple[Optional[AnswerEvaluation], Optional[str]]:
        """
        Faculty quick approval: confirms the AI score as the final score.
        """
        evaluation = db.session.get(AnswerEvaluation, evaluation_id)
        if not evaluation:
            return None, "Evaluation record not found."

        if not evaluation.submission or evaluation.submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this evaluation."

        try:
            evaluation.evaluation_status = "Approved"
            evaluation.needs_faculty_review = False
            evaluation.faculty_score = evaluation.ai_score
            db.session.commit()

            # Phase 4: Update finalized totals
            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(
                submission_id=evaluation.submission_id, faculty_id=faculty_id
            )

            return evaluation, None
        except Exception as e:
            db.session.rollback()
            return None, f"Failed to approve evaluation: {str(e)}"

    @classmethod
    def approve_all_evaluations(
        cls, submission_id: int, faculty_id: int
    ) -> Tuple[List[AnswerEvaluation], Optional[str]]:
        """
        Faculty bulk approval: confirms all AI scores across all questions as final scores.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return [], "Submission record not found."

        if not submission.assignment or submission.assignment.faculty_id != faculty_id:
            return [], "Unauthorized: You do not own the assignment for this submission."

        if not submission.evaluations:
            return [], "Submission has no evaluations to approve."

        try:
            for ev in submission.evaluations:
                ev.evaluation_status = "Approved"
                ev.needs_faculty_review = False
                ev.faculty_score = ev.ai_score

            submission.status = SubmissionStatus.FACULTY_REVIEWED.value
            db.session.commit()

            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(
                submission_id=submission_id, faculty_id=faculty_id
            )

            return list(submission.evaluations), None
        except Exception as e:
            db.session.rollback()
            return [], f"Failed to approve all evaluations: {str(e)}"

    @classmethod
    def flag_for_review(
        cls, evaluation_id: int, faculty_id: int
    ) -> Tuple[Optional[AnswerEvaluation], Optional[str]]:
        """
        Mark a question evaluation as requiring manual review.
        """
        evaluation = db.session.get(AnswerEvaluation, evaluation_id)
        if not evaluation:
            return None, "Evaluation record not found."

        if not evaluation.submission or evaluation.submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this evaluation."

        try:
            evaluation.evaluation_status = "Review Required"
            evaluation.needs_faculty_review = True
            evaluation.submission.status = SubmissionStatus.REVIEW_REQUIRED.value
            db.session.commit()

            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(
                submission_id=evaluation.submission_id, faculty_id=faculty_id
            )

            return evaluation, None
        except Exception as e:
            db.session.rollback()
            return None, f"Failed to flag for review: {str(e)}"

    @classmethod
    def modify_evaluation(
        cls,
        evaluation_id: int,
        faculty_score: float,
        faculty_feedback: Optional[str],
        faculty_id: int,
    ) -> Tuple[Optional[AnswerEvaluation], Optional[str]]:
        """
        Faculty override: sets custom score and custom feedback.
        Enforces 0.0 <= faculty_score <= maximum_marks.
        """
        evaluation = db.session.get(AnswerEvaluation, evaluation_id)
        if not evaluation:
            return None, "Evaluation record not found."

        if not evaluation.submission or evaluation.submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this evaluation."

        try:
            score_val = float(faculty_score)
        except (ValueError, TypeError):
            return None, "Invalid score value provided. Must be a numeric value."

        if score_val < 0.0 or score_val > evaluation.maximum_marks:
            return (
                None,
                f"Score must be between 0.0 and maximum marks ({evaluation.maximum_marks}).",
            )

        try:
            evaluation.faculty_score = round(score_val, 2)
            if faculty_feedback is not None:
                evaluation.faculty_feedback = faculty_feedback.strip()
            evaluation.evaluation_status = "Modified"
            evaluation.needs_faculty_review = False
            db.session.commit()

            # Phase 4: Sync updated totals
            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(
                submission_id=evaluation.submission_id, faculty_id=faculty_id
            )

            return evaluation, None
        except Exception as e:
            db.session.rollback()
            return None, f"Failed to modify evaluation: {str(e)}"

    @classmethod
    def re_evaluate_answer(
        cls,
        evaluation_id: int,
        faculty_id: int,
        evaluator: Optional[BaseSemanticEvaluator] = None,
    ) -> Tuple[Optional[AnswerEvaluation], Optional[str]]:
        """
        Re-evaluate a single question answer using the semantic evaluator.
        """
        evaluation = db.session.get(AnswerEvaluation, evaluation_id)
        if not evaluation:
            return None, "Evaluation record not found."

        if not evaluation.submission or evaluation.submission.assignment.faculty_id != faculty_id:
            return None, "Unauthorized: You do not own the assignment for this evaluation."

        question = evaluation.question
        if not question:
            return None, "Associated question not found."

        if evaluator is None:
            evaluator = GeminiSemanticEvaluator()

        avg_htr_conf = (
            evaluation.submission.average_confidence / 100.0
            if evaluation.submission.average_confidence
            else 1.0
        )
        matched_answer = evaluation.answer
        match_conf = (
            matched_answer.match_confidence
            if (matched_answer and matched_answer.match_confidence is not None)
            else 1.0
        )

        try:
            res: EvaluationResult = evaluator.evaluate_answer(
                question_text=question.question_text,
                supportive_answer=question.supportive_answer or "",
                student_answer=evaluation.extracted_answer,
                maximum_marks=float(question.maximum_marks),
                rubric=question.rubric,
                match_confidence=match_conf,
                htr_confidence=avg_htr_conf,
            )

            evaluation.ai_score = res.score
            evaluation.maximum_marks = float(question.maximum_marks)
            evaluation.criteria_scores = [c.to_dict() for c in res.criteria]
            evaluation.strengths = res.strengths
            evaluation.missing_elements = res.missing_elements
            evaluation.feedback = res.feedback
            evaluation.ai_confidence = res.confidence
            evaluation.semantic_similarity_score = res.semantic_similarity_score
            evaluation.error_message = res.error_message

            if res.score is None or res.error_message:
                evaluation.evaluation_status = "Failed"
                evaluation.needs_faculty_review = True
            else:
                evaluation.evaluation_status = "Evaluated"
                evaluation.needs_faculty_review = res.needs_faculty_review

            db.session.commit()

            # Phase 4: Sync preliminary result calculation
            from app.services.result_service import ResultService
            ResultService.calculate_submission_result(
                submission_id=evaluation.submission_id, faculty_id=faculty_id
            )

            return evaluation, None
        except Exception as e:
            db.session.rollback()
            return None, f"Re-evaluation failed: {str(e)}"

    @classmethod
    def get_submission_evaluations_for_user(
        cls,
        submission_id: int,
        user: Union[Faculty, Student],
    ) -> Tuple[Optional[List[AnswerEvaluation]], Optional[str]]:
        """
        Authorize and fetch evaluations for a submission.
        Students can only access their own submissions; faculty can only access their assignments.
        """
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return None, "Submission record not found."

        if user.role == "student":
            if submission.student_id != user.student_id:
                return None, "Unauthorized: You may only view evaluations for your own submissions."
        elif user.role == "faculty":
            if submission.assignment.faculty_id != user.faculty_id:
                return None, "Unauthorized: You do not own this assignment."
        else:
            return None, "Unauthorized access."

        evaluations = (
            AnswerEvaluation.query.filter_by(submission_id=submission_id)
            .join(Question, AnswerEvaluation.question_id == Question.question_id)
            .order_by(Question.question_number)
            .all()
        )
        return evaluations, None
