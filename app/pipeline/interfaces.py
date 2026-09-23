"""
Modular AI Evaluation Pipeline Interfaces.
Defines abstract contracts for future OCR/HTR, preprocessing, extraction,
question matching, semantic evaluation, rubric scoring, and feedback generation.

NOTE FOR PHASE 1:
These interfaces provide the structural abstraction so future AI modules can be plugged in
without modifying or rewriting core database models or user workflows.
No OCR/AI grading is executed in Phase 1.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExtractedAnswer:
    """Represents a segmented answer corresponding to a question."""
    question_number: int
    raw_text: str
    confidence_score: float = 0.0
    bounding_boxes: List[Dict[str, int]] = field(default_factory=list)


@dataclass
class QuestionEvaluationResult:
    """Represents the graded evaluation for an individual question."""
    question_id: int
    question_number: int
    awarded_marks: float
    maximum_marks: float
    semantic_similarity_score: float = 0.0
    rubric_criteria_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    flagged_for_review: bool = False


@dataclass
class PipelineEvaluationReport:
    """Complete evaluation report for a student submission."""
    submission_id: int
    total_awarded_marks: float
    total_maximum_marks: float
    percentage: float
    question_results: List[QuestionEvaluationResult] = field(default_factory=list)
    overall_feedback: str = ""
    requires_faculty_review: bool = False
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)


class BasePreprocessor(ABC):
    """Abstract interface for PDF normalization, page splitting, and de-skewing."""

    @abstractmethod
    def preprocess(self, pdf_path: str) -> List[Any]:
        """
        Process the raw PDF submission into cleaned image page arrays.
        :param pdf_path: Path to the uploaded PDF file.
        :return: List of preprocessed page images/data.
        """
        pass


class BaseHTREngine(ABC):
    """Abstract interface for Handwritten Text Recognition (HTR/OCR)."""

    @abstractmethod
    def recognize_handwriting(self, preprocessed_pages: List[Any]) -> str:
        """
        Transcribe handwritten text from processed page images.
        :param preprocessed_pages: Cleaned page image representations.
        :return: Transcribed text string.
        """
        pass


class BaseQAMatcher(ABC):
    """Abstract interface for segmenting and mapping student answers to assignment questions."""

    @abstractmethod
    def match_answers_to_questions(
        self, transcribed_text: str, questions: List[Dict[str, Any]]
    ) -> List[ExtractedAnswer]:
        """
        Extract question boundaries and map answers to target assignment questions.
        :param transcribed_text: Transcribed full submission text.
        :param questions: List of question definitions from assignment.
        :return: List of ExtractedAnswer objects mapped to question numbers.
        """
        pass


class BaseSemanticEvaluator(ABC):
    """Abstract interface for semantic AI grading against supportive answers and rubrics."""

    @abstractmethod
    def evaluate(
        self,
        extracted_answers: List[ExtractedAnswer],
        assignment_questions: List[Dict[str, Any]],
    ) -> PipelineEvaluationReport:
        """
        Compare student answers against supportive answers and rubrics to generate marks.
        :param extracted_answers: Extracted student answers.
        :param assignment_questions: Assignment questions with supportive answers and rubrics.
        :return: PipelineEvaluationReport with individual question scores.
        """
        pass
