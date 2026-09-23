"""
AI Semantic Answer Evaluator for SmartEval Phase 3B.
Evaluates student handwritten answer transcripts against faculty questions,
reference supportive answers, and rubrics using Google Gemini API (google-genai SDK).
Provides structured criteria breakdown, missing-concept detection, and scoring safety clamping.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Structured Data Contracts & Pydantic Schemas
# =============================================================================

class CriterionEvaluationSchema(BaseModel):
    """Schema for individual rubric criterion evaluation."""
    name: str = Field(description="Name or title of the rubric criterion")
    max_marks: float = Field(description="Maximum marks allowable for this criterion")
    status: str = Field(description="Discrete evaluation outcome: 'full' (100%), 'partial' (50%), or 'none' (0%)")
    reason: str = Field(description="Specific justification for the discrete outcome awarded")


class GeminiEvaluationResponseSchema(BaseModel):
    """Structured JSON schema enforced on Gemini API output."""
    criteria: List[CriterionEvaluationSchema] = Field(description="Breakdown of criteria evaluations")
    strengths: List[str] = Field(description="List of accurate concepts correctly explained by the student")
    missing_elements: List[str] = Field(description="List of key concepts or steps missing from the student answer")
    feedback: str = Field(description="Concise, actionable feedback explaining the evaluation")
    confidence: float = Field(description="AI evaluation confidence score between 0.0 and 1.0")
    needs_faculty_review: bool = Field(description="Flag indicating if the answer has ambiguity requiring human review")


@dataclass
class CriterionScore:
    """Dataclass representing an evaluated rubric criterion."""
    name: str
    max_marks: float
    awarded_marks: float
    status: str  # 'full', 'partial', 'none'
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_marks": round(self.max_marks, 2),
            "awarded_marks": round(self.awarded_marks, 2),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class EvaluationResult:
    """Complete structured result returned by the semantic evaluation engine."""
    score: Optional[float]
    max_score: float
    criteria: List[CriterionScore] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    feedback: str = ""
    confidence: float = 0.85
    needs_faculty_review: bool = True
    semantic_similarity_score: float = 0.0
    raw_response: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2) if self.score is not None else None,
            "max_score": round(self.max_score, 2),
            "criteria": [c.to_dict() for c in self.criteria],
            "strengths": self.strengths,
            "missing_elements": self.missing_elements,
            "feedback": self.feedback,
            "confidence": round(self.confidence, 2),
            "needs_faculty_review": self.needs_faculty_review,
            "semantic_similarity_score": round(self.semantic_similarity_score, 2),
            "error_message": self.error_message,
        }


# =============================================================================
# 2. Rubric Parsing & Building Engine
# =============================================================================

class RubricEngine:
    """
    Parses and builds structured rubric criteria from question definitions.
    Enforces that criteria marks sum exactly to the question maximum marks.
    """

    @classmethod
    def parse_rubric(
        cls, rubric_text: Optional[str], maximum_marks: float
    ) -> List[Dict[str, Any]]:
        """
        Parse rubric text into structured criteria list.
        Supports patterns like:
        - '3 pts Definition, 4 pts Core concepts, 3 pts Examples'
        - '5 pts Dijkstra limitations; 5 pts Bellman-Ford mechanics; 5 pts cycle detection'
        - '10 pts recurrence relation, 10 pts base cases'
        - '1. Definition (2 marks)\n2. Derivation (5 marks)'
        If no rubric is supplied or parsing fails, generates a balanced default rubric.
        """
        if not rubric_text or not rubric_text.strip():
            return cls._generate_default_rubric(maximum_marks)

        text = rubric_text.strip()
        criteria = []

        # Pattern 1: e.g., '3 pts Definition' or '5 points Algorithm steps' or '4 marks Examples'
        pt_pattern = re.compile(
            r"(?:^|[;,|\n])\s*(?:(\d+(?:\.\d+)?)\s*(?:pts|points?|marks?)\s*[:\-]?\s*([^;,|\n]+)|"
            r"([^;,|\n]+)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:pts|points?|marks?))",
            re.IGNORECASE,
        )

        for match in pt_pattern.finditer(text):
            g = match.groups()
            if g[0] and g[1]:
                pts = float(g[0])
                name = g[1].strip()
            elif g[2] and g[3]:
                name = g[2].strip()
                pts = float(g[3])
            else:
                continue

            if name and pts > 0:
                criteria.append({"name": name, "max_marks": pts})

        # Pattern 2: Numbered list with marks in parentheses, e.g., '1. Concept (4 marks)'
        if not criteria:
            num_pattern = re.compile(
                r"(?:^|\n)\s*(?:\d+[\.\)]\s*)?([^(\n]+)\s*\(\s*(\d+(?:\.\d+)?)\s*(?:pts|points?|marks?)\s*\)",
                re.IGNORECASE,
            )
            for match in num_pattern.finditer(text):
                name = match.group(1).strip()
                pts = float(match.group(2))
                if name and pts > 0:
                    criteria.append({"name": name, "max_marks": pts})

        # If parsing found criteria, validate and normalize sum to maximum_marks
        if criteria:
            total_parsed = sum(c["max_marks"] for c in criteria)
            if total_parsed > 0 and abs(total_parsed - maximum_marks) > 0.01:
                # Scale criteria proportionally so sum matches maximum_marks
                scale = maximum_marks / total_parsed
                for c in criteria:
                    c["max_marks"] = round(c["max_marks"] * scale, 2)
                # Correct minor rounding difference on largest item
                diff = maximum_marks - sum(c["max_marks"] for c in criteria)
                if abs(diff) > 0.001 and criteria:
                    criteria[0]["max_marks"] = round(criteria[0]["max_marks"] + diff, 2)
            return criteria

        return cls._generate_default_rubric(maximum_marks)

    @classmethod
    def _generate_default_rubric(cls, maximum_marks: float) -> List[Dict[str, Any]]:
        """Generate a balanced 3-tier default academic rubric."""
        max_m = max(1.0, float(maximum_marks))
        c1 = round(max_m * 0.50, 2)  # 50% Core conceptual correctness
        c2 = round(max_m * 0.30, 2)  # 30% Completeness & Technical detail
        c3 = round(max_m - c1 - c2, 2)  # 20% Clarity & Accuracy

        return [
            {"name": "Core Conceptual Correctness", "max_marks": c1},
            {"name": "Completeness & Technical Detail", "max_marks": c2},
            {"name": "Clarity & Academic Accuracy", "max_marks": c3},
        ]


# =============================================================================
# 3. Base Evaluator Interface
# =============================================================================

class BaseSemanticEvaluator(ABC):
    """Abstract base class for semantic evaluation providers."""

    @abstractmethod
    def evaluate_answer(
        self,
        question_text: str,
        supportive_answer: str,
        student_answer: str,
        maximum_marks: float,
        rubric: Optional[str] = None,
        match_confidence: float = 1.0,
        htr_confidence: float = 1.0,
    ) -> EvaluationResult:
        """
        Evaluate student extracted answer against reference answer and rubric.
        Returns a structured EvaluationResult.
        """
        pass


# =============================================================================
# 4. Google Gemini GenAI Semantic Evaluator
# =============================================================================

class GeminiSemanticEvaluator(BaseSemanticEvaluator):
    """
    Production semantic evaluator using the Google GenAI SDK and Gemini Flash models.
    Enforces structured schema outputs, deterministic Python numerical scoring, and safety checks.
    """

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.model_name = model if model is not None else os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client = None

    @property
    def client(self):
        """Lazy loader for Google GenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def evaluate_answer(
        self,
        question_text: str,
        supportive_answer: str,
        student_answer: str,
        maximum_marks: float,
        rubric: Optional[str] = None,
        match_confidence: float = 1.0,
        htr_confidence: float = 1.0,
    ) -> EvaluationResult:
        """
        Execute semantic answer evaluation using Gemini with structured output.
        Deterministic scoring: Gemini assigns discrete status (full/partial/none),
        and Python computes exact numerical marks to guarantee reproducibility.
        """
        max_marks = float(maximum_marks)

        # 1. Edge Case: Empty or blank student answer
        if not student_answer or not student_answer.strip():
            parsed_rubric = RubricEngine.parse_rubric(rubric, max_marks)
            criteria = [
                CriterionScore(
                    name=c["name"],
                    max_marks=c["max_marks"],
                    awarded_marks=0.0,
                    status="none",
                    reason="No student answer was provided for this question.",
                )
                for c in parsed_rubric
            ]
            return EvaluationResult(
                score=0.0,  # Legitimate academic zero
                max_score=max_marks,
                criteria=criteria,
                strengths=[],
                missing_elements=["Entire answer is missing."],
                feedback="No answer content was submitted for this question.",
                confidence=1.0,
                needs_faculty_review=False,
                semantic_similarity_score=0.0,
            )

        # 2. Check for missing API Key (Technical failure -> score=None)
        if not self.api_key:
            return EvaluationResult(
                score=None,
                max_score=max_marks,
                criteria=[],
                strengths=[],
                missing_elements=[],
                feedback="AI evaluation could not run because GEMINI_API_KEY is not configured.",
                confidence=0.0,
                needs_faculty_review=True,
                error_message="GEMINI_API_KEY is missing. Please set your Gemini API key in .env.",
            )

        # 3. Calculate objective semantic cosine similarity support
        try:
            from app.processing.qa_matcher import HierarchicalQAMatcher
            matcher = HierarchicalQAMatcher()
            sem_sim = matcher.calculate_semantic_similarity(student_answer, supportive_answer or question_text)
        except Exception:
            sem_sim = 0.5

        # 4. Parse rubric into structured format
        rubric_criteria = RubricEngine.parse_rubric(rubric, max_marks)
        rubric_description = "\n".join(
            f"- {c['name']} (Maximum Marks: {c['max_marks']} pts)" for c in rubric_criteria
        )

        # 5. Build prompt
        system_instruction = (
            "You are an expert academic professor and objective coursework evaluator. "
            "Your task is to evaluate handwritten student answers transcribed from exam papers. "
            "\nEVALUATION PRINCIPLES:\n"
            "1. Evaluate meaning and conceptual understanding, NOT word-for-word string equality.\n"
            "2. For each rubric criterion, assess the outcome as one of three discrete levels:\n"
            "   - 'full': The concept or requirement is fully met and understood.\n"
            "   - 'partial': The answer is partially correct, incomplete, or contains minor errors.\n"
            "   - 'none': The concept is not addressed, entirely incorrect, or irrelevant.\n"
            "3. Award full credit for correct alternate wording, paraphrased definitions, and equivalent formulations.\n"
            "4. Do NOT reward unrelated content just because some isolated keywords overlap.\n"
            "5. Explicitly identify what the student got right (strengths) and what key concepts were missing (missing_elements).\n"
            "6. Keep feedback concise, actionable, and encouraging."
        )

        user_prompt = f"""
QUESTION:
{question_text}

REFERENCE / SUPPORTIVE ANSWER (Teacher's Solution):
{supportive_answer or 'Evaluate based on academic correctness for the question prompt.'}

MAXIMUM MARKS:
{max_marks}

CONFIGURED RUBRIC:
{rubric_description}

STUDENT'S SUBMITTED ANSWER (OCR Transcript):
{student_answer}

OCR Recognition Confidence: {round(htr_confidence * 100, 1)}%
Question Matching Confidence: {round(match_confidence * 100, 1)}%

Please evaluate the student's answer against the reference answer and rubric.
Return a structured JSON evaluation adhering to the schema.
"""

        try:
            from google.genai import types

            # Generate content using Gemini structured outputs with deterministic settings
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=GeminiEvaluationResponseSchema,
                    temperature=0.0,  # Zero temperature for deterministic academic grading
                    seed=42,          # Fixed seed for reproducibility
                ),
            )

            raw_text = response.text
            parsed_data = json.loads(raw_text)

            ai_conf = float(parsed_data.get("confidence", 0.85))
            feedback = str(parsed_data.get("feedback", "")).strip()
            strengths = list(parsed_data.get("strengths", []))
            missing = list(parsed_data.get("missing_elements", []))
            needs_review = bool(parsed_data.get("needs_faculty_review", False))

            # Deterministic Python numerical mark calculation from discrete rubric outcomes:
            # Full = 100% of criterion max_marks
            # Partial = 50% of criterion max_marks
            # None = 0% of criterion max_marks
            raw_criteria = parsed_data.get("criteria", [])
            criteria_scores: List[CriterionScore] = []

            for item in raw_criteria:
                c_name = str(item.get("name", "Criterion")).strip()
                c_max = float(item.get("max_marks", 1.0))
                c_status = str(item.get("status", "partial")).lower().strip()
                if c_status not in ("full", "partial", "none"):
                    c_status = "partial"

                if c_status == "full":
                    c_awarded = round(c_max * 1.0, 2)
                elif c_status == "partial":
                    c_awarded = round(c_max * 0.5, 2)
                else:  # none
                    c_awarded = 0.0

                c_reason = str(item.get("reason", "")).strip()
                criteria_scores.append(
                    CriterionScore(
                        name=c_name,
                        max_marks=c_max,
                        awarded_marks=c_awarded,
                        status=c_status,
                        reason=c_reason,
                    )
                )

            # Ensure criteria sum matches clamped score
            if criteria_scores:
                computed_score = sum(c.awarded_marks for c in criteria_scores)
                # Server-side safety clamp to [0.0, max_marks]
                final_score = max(0.0, min(max_marks, round(computed_score, 2)))
            else:
                final_score = 0.0

            # Automatically flag for faculty review if uncertainty detected
            if htr_confidence < 0.55 or match_confidence < 0.75 or ai_conf < 0.70:
                needs_review = True

            return EvaluationResult(
                score=final_score,
                max_score=max_marks,
                criteria=criteria_scores,
                strengths=strengths,
                missing_elements=missing,
                feedback=feedback,
                confidence=ai_conf,
                needs_faculty_review=needs_review,
                semantic_similarity_score=sem_sim,
                raw_response=raw_text,
            )

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {str(e)}", exc_info=True)
            return EvaluationResult(
                score=None,  # Technical failure: score is None/NULL
                max_score=max_marks,
                criteria=[],
                strengths=[],
                missing_elements=[],
                feedback="AI evaluation encountered an unexpected error and requires manual grading.",
                confidence=0.0,
                needs_faculty_review=True,
                error_message=f"Evaluation error: {str(e)}",
            )


# =============================================================================
# 5. Mock Semantic Evaluator for Deterministic Testing
# =============================================================================

class MockSemanticEvaluator(BaseSemanticEvaluator):
    """
    Mock evaluator for fast, offline, and deterministic unit/integration testing.
    Allows configuring preset scores, discrete outcomes ('full', 'partial', 'none'),
    partial marks, missing elements, or errors.
    """

    def __init__(
        self,
        score: Optional[float] = None,
        percentage: Optional[float] = None,
        outcome: Optional[str] = None,  # "full", "partial", "none"
        criteria: Optional[List[CriterionScore]] = None,
        strengths: Optional[List[str]] = None,
        missing_elements: Optional[List[str]] = None,
        feedback: str = "Well articulated conceptual explanation with accurate technical reasoning.",
        confidence: float = 0.92,
        needs_faculty_review: bool = False,
        should_fail: bool = False,
        error_message: str = "Simulated API timeout error.",
    ):
        self._score = score
        self._percentage = percentage
        self._outcome = outcome
        self._criteria = criteria
        self._strengths = strengths if strengths is not None else ["Accurate core definition", "Clear logical flow"]
        self._missing_elements = missing_elements if missing_elements is not None else []
        self._feedback = feedback
        self._confidence = confidence
        self._needs_faculty_review = needs_faculty_review
        self._should_fail = should_fail
        self._error_message = error_message

    def evaluate_answer(
        self,
        question_text: str,
        supportive_answer: str,
        student_answer: str,
        maximum_marks: float,
        rubric: Optional[str] = None,
        match_confidence: float = 1.0,
        htr_confidence: float = 1.0,
    ) -> EvaluationResult:
        """Return preset or computed evaluation result for unit testing."""
        max_marks = float(maximum_marks)

        if self._should_fail:
            return EvaluationResult(
                score=None,  # Technical failure: score is None/NULL
                max_score=max_marks,
                criteria=[],
                strengths=[],
                missing_elements=[],
                feedback="Evaluation failed.",
                confidence=0.0,
                needs_faculty_review=True,
                error_message=self._error_message,
            )

        if not student_answer or not student_answer.strip():
            parsed_rubric = RubricEngine.parse_rubric(rubric, max_marks)
            return EvaluationResult(
                score=0.0,
                max_score=max_marks,
                criteria=[
                    CriterionScore(
                        name=c["name"],
                        max_marks=c["max_marks"],
                        awarded_marks=0.0,
                        status="none",
                        reason="No student answer was provided for this question.",
                    )
                    for c in parsed_rubric
                ],
                strengths=[],
                missing_elements=["Entire answer is missing."],
                feedback="No answer content provided.",
                confidence=1.0,
                needs_faculty_review=False,
            )

        # Build criteria and compute marks deterministically
        parsed_rubric = RubricEngine.parse_rubric(rubric, max_marks)
        if self._criteria:
            criteria = self._criteria
            score = max(0.0, min(max_marks, round(sum(c.awarded_marks for c in criteria), 2)))
        elif self._outcome:
            status = self._outcome.lower().strip()
            criteria = []
            for c in parsed_rubric:
                c_max = c["max_marks"]
                if status == "full":
                    awarded = round(c_max * 1.0, 2)
                elif status == "partial":
                    awarded = round(c_max * 0.5, 2)
                else:
                    awarded = 0.0
                criteria.append(
                    CriterionScore(
                        name=c["name"],
                        max_marks=c_max,
                        awarded_marks=awarded,
                        status=status,
                        reason=f"Outcome {status} for {c['name']}.",
                    )
                )
            score = max(0.0, min(max_marks, round(sum(c.awarded_marks for c in criteria), 2)))
        elif self._score is not None:
            score = max(0.0, min(max_marks, float(self._score)))
            ratio = score / max_marks if max_marks > 0 else 0.0
            criteria = [
                CriterionScore(
                    name=c["name"],
                    max_marks=c["max_marks"],
                    awarded_marks=round(c["max_marks"] * ratio, 2),
                    status="full" if ratio >= 0.85 else ("partial" if ratio > 0.1 else "none"),
                    reason=f"Awarded based on {c['name']} evaluation.",
                )
                for c in parsed_rubric
            ]
        elif self._percentage is not None:
            score = max(0.0, min(max_marks, round(max_marks * float(self._percentage), 2)))
            ratio = score / max_marks if max_marks > 0 else 0.0
            criteria = [
                CriterionScore(
                    name=c["name"],
                    max_marks=c["max_marks"],
                    awarded_marks=round(c["max_marks"] * ratio, 2),
                    status="full" if ratio >= 0.85 else ("partial" if ratio > 0.1 else "none"),
                    reason=f"Awarded based on {c['name']} evaluation.",
                )
                for c in parsed_rubric
            ]
        else:
            # Default to full (100%)
            criteria = [
                CriterionScore(
                    name=c["name"],
                    max_marks=c["max_marks"],
                    awarded_marks=round(c["max_marks"] * 1.0, 2),
                    status="full",
                    reason=f"Fully satisfied {c['name']}.",
                )
                for c in parsed_rubric
            ]
            score = max(0.0, min(max_marks, round(sum(c.awarded_marks for c in criteria), 2)))

        from app.processing.qa_matcher import HierarchicalQAMatcher
        matcher = HierarchicalQAMatcher()
        sem_sim = matcher.calculate_semantic_similarity(student_answer, supportive_answer or question_text)

        return EvaluationResult(
            score=score,
            max_score=max_marks,
            criteria=criteria,
            strengths=self._strengths,
            missing_elements=self._missing_elements,
            feedback=self._feedback,
            confidence=self._confidence,
            needs_faculty_review=self._needs_faculty_review,
            semantic_similarity_score=sem_sim,
        )
