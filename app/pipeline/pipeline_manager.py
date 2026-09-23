"""
Pipeline Manager for orchestrating submission processing.
In Phase 1, registers pipeline components and manages lifecycle state hooks.
"""

from typing import Optional
from app.pipeline.interfaces import (
    BasePreprocessor,
    BaseHTREngine,
    BaseQAMatcher,
    BaseSemanticEvaluator,
    PipelineEvaluationReport,
)


class EvaluationPipelineManager:
    """
    Orchestrator for the multi-stage evaluation pipeline.
    Connects Preprocessor -> HTR Engine -> Q&A Matcher -> Semantic Evaluator.
    """

    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        htr_engine: Optional[BaseHTREngine] = None,
        qa_matcher: Optional[BaseQAMatcher] = None,
        evaluator: Optional[BaseSemanticEvaluator] = None,
    ):
        self.preprocessor = preprocessor
        self.htr_engine = htr_engine
        self.qa_matcher = qa_matcher
        self.evaluator = evaluator

    @property
    def is_pipeline_configured(self) -> bool:
        """Check if all stages of the AI pipeline have been provided."""
        return all([
            self.preprocessor is not None,
            self.htr_engine is not None,
            self.qa_matcher is not None,
            self.evaluator is not None,
        ])

    def process_submission(self, submission_id: int) -> Optional[PipelineEvaluationReport]:
        """
        Execute evaluation pipeline on a submission.
        In Phase 1, returns None as AI engines are not yet attached.
        """
        if not self.is_pipeline_configured:
            # Phase 1: AI modules intentionally not connected yet
            return None
        
        # Future Phase 2+ Execution flow:
        # 1. preprocessed = self.preprocessor.preprocess(submission.file_path)
        # 2. transcribed = self.htr_engine.recognize_handwriting(preprocessed)
        # 3. matched = self.qa_matcher.match_answers_to_questions(transcribed, questions)
        # 4. report = self.evaluator.evaluate(matched, questions)
        # 5. update submission status to 'Completed' or 'Review Required'
        return None
