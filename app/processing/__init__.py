"""
Document Processing package for SmartEval Phase 2.
"""

from app.processing.pdf_processor import PDFProcessor, ExtractedPageData
from app.processing.image_preprocessor import ImagePreprocessor, PreprocessingConfig
from app.processing.handwriting_recognizer import (
    HTRRecognizerInterface,
    HTRResult,
    LineExtraction,
    EasyOCRHandwritingRecognizer,
    get_default_recognizer,
)
from app.processing.pipeline import (
    SubmissionProcessingPipeline,
    PipelineResult,
)
from app.processing.ocr_postprocessor import OCRPostProcessor
from app.processing.qa_matcher import (
    HierarchicalQAMatcher,
    CandidateAnswerSegment,
    MatchedQuestionAnswer,
)

from app.processing.evaluator import (
    BaseSemanticEvaluator,
    GeminiSemanticEvaluator,
    MockSemanticEvaluator,
    RubricEngine,
    EvaluationResult,
    CriterionScore,
)

__all__ = [
    "PDFProcessor",
    "ExtractedPageData",
    "ImagePreprocessor",
    "PreprocessingConfig",
    "HTRRecognizerInterface",
    "HTRResult",
    "LineExtraction",
    "EasyOCRHandwritingRecognizer",
    "OCRPostProcessor",
    "get_default_recognizer",
    "SubmissionProcessingPipeline",
    "PipelineResult",
    "HierarchicalQAMatcher",
    "CandidateAnswerSegment",
    "MatchedQuestionAnswer",
    "BaseSemanticEvaluator",
    "GeminiSemanticEvaluator",
    "MockSemanticEvaluator",
    "RubricEngine",
    "EvaluationResult",
    "CriterionScore",
]

