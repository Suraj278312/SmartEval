"""
Question/Answer Segmentation and Question Matching Engine for SmartEval Phase 3A.

Segments extracted student handwritten text across multi-page submissions,
detects question boundaries, and hierarchically matches segments to assignment questions
using explicit numbering, textual similarity, and semantic keyword similarity.
"""

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from difflib import SequenceMatcher

from app.models.assignment import Question
from app.models.document_page import DocumentPage


@dataclass
class CandidateAnswerSegment:
    """Represents a segmented block of handwritten student answer text."""
    text: str
    page_start: int
    page_end: int
    detected_label: Optional[str] = None  # e.g., 'Q1', '1.', 'Question 2'
    detected_question_num: Optional[int] = None  # e.g., 1, 2, 3
    header_line: Optional[str] = None
    average_ocr_confidence: float = 1.0
    line_count: int = 1


@dataclass
class MatchedQuestionAnswer:
    """Represents the mapped relationship between a question and a student answer segment."""
    question_id: Optional[int]
    question_number: Optional[int]
    question_text: Optional[str]
    extracted_text: str
    page_start: int
    page_end: int
    detected_label: Optional[str]
    match_method: str  # 'explicit_numbering', 'text_similarity', 'semantic_similarity', 'unmatched'
    match_confidence: float  # 0.0 to 1.0
    status: str  # 'Matched', 'Review Required', 'Unmatched'
    review_notes: Optional[str] = None


class HierarchicalQAMatcher:
    """
    Modular engine for segmenting handwritten document text and mapping to assignment questions.
    Layer 1: Explicit question numbering / labels
    Layer 2: Textual & lexical overlap similarity
    Layer 3: Semantic keyword / TF-IDF vector similarity
    Layer 4: Manual review fallback for low-confidence or orphan segments
    """

    # Common English and academic stopwords for semantic vector comparison
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
        "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        "explain", "describe", "discuss", "define", "write", "state", "briefly", "detail",
        "answer", "question", "following", "give", "example", "examples", "calculate",
    }

    # Regex patterns for detecting question boundaries
    QUESTION_PATTERNS = [
        # Pattern 1: Q1, Q.1, Q-1, Q 1, Q1:, Q1., Question 1, Question 1:, Question - 1
        re.compile(
            r"^[ \t]*(?:Q(?:uestion)?|Ans(?:wer)?|Problem|Task|Prompt)[ \t.]*#?[ \t\-]*(\d+)[ \t]*[:.)\-\]]?(?:[ \t]+(.*))?$",
            re.IGNORECASE,
        ),
        # Pattern 2: Explicit numbered headers: 1., 2., 3., 1), 2), 3), 1:, 2:
        re.compile(r"^[ \t]*(\d+)[\.\)\:\-][ \t]+(.*)$", re.IGNORECASE),
        # Pattern 3: Parenthesized or bracketed numbers: (1), (2), [1], [2]
        re.compile(r"^[ \t]*[\(\[](\d+)[\)\]][ \t]*(.*)$", re.IGNORECASE),
        # Pattern 4: Standalone Q1, Q2, 1., Ans1 with no trailing text
        re.compile(r"^[ \t]*(?:Q|Ans|Question|Answer)?[ \t.]*#?[ \t]*(\d+)[ \t]*[:.)\-\]]?$", re.IGNORECASE),
    ]

    def __init__(
        self,
        high_confidence_threshold: float = 0.75,
        review_required_threshold: float = 0.50,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.review_required_threshold = review_required_threshold

    # =========================================================================
    # 1. Boundary Detection & Multi-Page Answer Segmentation
    # =========================================================================

    def detect_question_header(self, line: str) -> Tuple[bool, Optional[str], Optional[int], str]:
        """
        Check if a text line represents a question boundary/header.
        Returns (is_header, detected_label, detected_question_num, remaining_body_text).
        """
        clean_line = line.strip()
        if not clean_line:
            return False, None, None, ""

        # Test against all question header regex patterns
        for pattern in self.QUESTION_PATTERNS:
            match = pattern.match(clean_line)
            if match:
                groups = match.groups()
                num_str = groups[0]
                try:
                    q_num = int(num_str)
                except (ValueError, TypeError):
                    continue

                body_text = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                
                # Determine label representation (e.g. Q1, Ans 1, 1.)
                clean_lower = clean_line.lower()
                if clean_lower.startswith("q") or "question" in clean_lower:
                    label = f"Q{q_num}"
                elif clean_lower.startswith("a") or "ans" in clean_lower:
                    label = f"Ans {q_num}"
                elif clean_lower.startswith("problem") or clean_lower.startswith("task") or clean_lower.startswith("prompt"):
                    label = f"Q{q_num}"
                else:
                    label = f"{q_num}."

                return True, label, q_num, body_text

        return False, None, None, clean_line

    def segment_pages(
        self, pages: List[Union[DocumentPage, Dict[str, Any]]]
    ) -> List[CandidateAnswerSegment]:
        """
        Segment extracted text across ordered pages into candidate answer blocks.
        Preserves page ranges for multi-page answers and retains all text content.
        """
        if not pages:
            return []

        # Sort pages by page number
        sorted_pages = sorted(
            pages,
            key=lambda p: p.page_number if hasattr(p, "page_number") else p.get("page_number", 1)
        )

        candidates: List[CandidateAnswerSegment] = []
        current_lines: List[str] = []
        current_page_start: Optional[int] = None
        current_page_end: Optional[int] = None
        current_label: Optional[str] = None
        current_q_num: Optional[int] = None
        current_header: Optional[str] = None
        current_ocr_confs: List[float] = []

        def flush_current_segment():
            nonlocal current_lines, current_page_start, current_page_end
            nonlocal current_label, current_q_num, current_header, current_ocr_confs

            content = "\n".join(current_lines).strip()
            if content and current_page_start is not None:
                avg_ocr = (
                    sum(current_ocr_confs) / len(current_ocr_confs)
                    if current_ocr_confs
                    else 1.0
                )
                candidates.append(
                    CandidateAnswerSegment(
                        text=content,
                        page_start=current_page_start,
                        page_end=current_page_end or current_page_start,
                        detected_label=current_label,
                        detected_question_num=current_q_num,
                        header_line=current_header,
                        average_ocr_confidence=avg_ocr,
                        line_count=len(current_lines),
                    )
                )

            current_lines = []
            current_page_start = None
            current_page_end = None
            current_label = None
            current_q_num = None
            current_header = None
            current_ocr_confs = []

        for page in sorted_pages:
            p_num = page.page_number if hasattr(page, "page_number") else page.get("page_number", 1)
            raw_text = (
                page.extracted_text if hasattr(page, "extracted_text") else page.get("extracted_text", "")
            ) or ""
            p_conf = (
                page.confidence if hasattr(page, "confidence") else page.get("confidence", 1.0)
            )
            if p_conf is None:
                p_conf = 1.0
            elif p_conf > 1.0:
                p_conf = p_conf / 100.0

            lines = raw_text.splitlines()

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if current_lines:
                        current_lines.append("")
                    continue

                is_header, label, q_num, body_text = self.detect_question_header(stripped)

                if is_header:
                    # Flush previous segment if one was active
                    if current_lines:
                        flush_current_segment()

                    # Start new segment
                    current_page_start = p_num
                    current_page_end = p_num
                    current_label = label
                    current_q_num = q_num
                    current_header = stripped
                    current_ocr_confs.append(p_conf)

                    if body_text:
                        current_lines.append(body_text)
                else:
                    # Regular answer line
                    if current_page_start is None:
                        current_page_start = p_num
                        current_page_end = p_num
                        current_ocr_confs.append(p_conf)
                    else:
                        current_page_end = p_num
                        current_ocr_confs.append(p_conf)

                    current_lines.append(stripped)

        # Flush any trailing segment
        if current_lines:
            flush_current_segment()

        # Fallback: If no headers were detected at all, treat whole text or paragraphs as candidate segments
        if not candidates and any(
            (p.extracted_text if hasattr(p, "extracted_text") else p.get("extracted_text", ""))
            for p in sorted_pages
        ):
            full_parts = []
            for p in sorted_pages:
                p_num = p.page_number if hasattr(p, "page_number") else p.get("page_number", 1)
                t = (p.extracted_text if hasattr(p, "extracted_text") else p.get("extracted_text", "")) or ""
                if t.strip():
                    full_parts.append((p_num, t.strip()))

            if full_parts:
                start_p = full_parts[0][0]
                end_p = full_parts[-1][0]
                combined_text = "\n\n".join(part[1] for part in full_parts)
                candidates.append(
                    CandidateAnswerSegment(
                        text=combined_text,
                        page_start=start_p,
                        page_end=end_p,
                        detected_label=None,
                        detected_question_num=None,
                        header_line=None,
                        average_ocr_confidence=1.0,
                        line_count=len(combined_text.splitlines()),
                    )
                )

        return candidates

    # =========================================================================
    # 2. Similarity Calculations
    # =========================================================================

    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Extract clean alphanumeric lowercase words from text."""
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        return words

    def _get_content_tokens(self, text: str) -> List[str]:
        """Extract meaningful non-stopword tokens."""
        tokens = self._tokenize_and_clean(text)
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculate string and token-level lexical similarity between two text snippets.
        Combines SequenceMatcher ratio and token Jaccard overlap.
        """
        if not text_a or not text_b:
            return 0.0

        clean_a = " ".join(self._tokenize_and_clean(text_a))
        clean_b = " ".join(self._tokenize_and_clean(text_b))

        if not clean_a or not clean_b:
            return 0.0

        # 1. SequenceMatcher ratio
        seq_ratio = SequenceMatcher(None, clean_a, clean_b).ratio()

        # 2. Token Jaccard similarity
        tokens_a = set(self._get_content_tokens(text_a))
        tokens_b = set(self._get_content_tokens(text_b))

        if not tokens_a or not tokens_b:
            jaccard = 0.0
            coverage = 0.0
        else:
            intersection = tokens_a.intersection(tokens_b)
            union = tokens_a.union(tokens_b)
            jaccard = len(intersection) / len(union) if union else 0.0
            coverage = max(
                len(intersection) / len(tokens_a) if tokens_a else 0.0,
                len(intersection) / len(tokens_b) if tokens_b else 0.0,
            )

        # Weighted combination: 50% Coverage + 30% Token Jaccard + 20% SequenceMatcher
        return round(0.5 * coverage + 0.3 * jaccard + 0.2 * seq_ratio, 4)

    def calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity over term-frequency vectors with n-gram subword shingles.
        Incorporates key-concept coverage for reworded prompts.
        """
        if not text_a or not text_b:
            return 0.0

        tokens_a = self._get_content_tokens(text_a)
        tokens_b = self._get_content_tokens(text_b)

        if not tokens_a or not tokens_b:
            return self.calculate_text_similarity(text_a, text_b)

        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = set_a.intersection(set_b)
        
        # Concept coverage (what fraction of question key terms appear in candidate)
        coverage = max(
            len(intersection) / len(set_a) if set_a else 0.0,
            len(intersection) / len(set_b) if set_b else 0.0,
        )

        # Build term frequency vectors including word bi-grams
        def get_features(tokens: List[str]) -> Counter:
            features = Counter(tokens)
            for i in range(len(tokens) - 1):
                features[f"{tokens[i]}_{tokens[i+1]}"] += 1.5
            return features

        vec_a = get_features(tokens_a)
        vec_b = get_features(tokens_b)

        all_features = set(vec_a.keys()).union(set(vec_b.keys()))
        dot_product = sum(vec_a.get(f, 0.0) * vec_b.get(f, 0.0) for f in all_features)
        norm_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))
        norm_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm_a * norm_b)

        # If key concept coverage is high (e.g. inheritance + java present), boost score
        combined_score = max(cosine_sim, 0.7 * coverage + 0.3 * cosine_sim)
        return min(1.0, max(0.0, round(combined_score, 4)))

    # =========================================================================
    # 3. Hierarchical Question Matching Algorithm
    # =========================================================================

    def match_candidates_to_questions(
        self,
        candidates: List[CandidateAnswerSegment],
        questions: List[Union[Question, Dict[str, Any]]],
    ) -> List[MatchedQuestionAnswer]:
        """
        Execute hierarchical layered matching:
        Priority 1: Explicit question numbering / labels
        Priority 2: Textual & lexical overlap
        Priority 3: Semantic keyword similarity
        Priority 4: Unmatched fallback for unresolved segments
        """
        if not candidates and not questions:
            return []

        # Normalize questions list into dicts
        norm_questions = []
        for q in questions:
            q_id = q.question_id if hasattr(q, "question_id") else q.get("question_id")
            q_num = q.question_number if hasattr(q, "question_number") else q.get("question_number", 1)
            q_text = q.question_text if hasattr(q, "question_text") else q.get("question_text", "")
            supp_ans = q.supportive_answer if hasattr(q, "supportive_answer") else q.get("supportive_answer", "")
            norm_questions.append({
                "question_id": q_id,
                "question_number": q_num,
                "question_text": q_text,
                "supportive_answer": supp_ans,
            })

        # Track assignment of candidate segments and questions
        matched_answers: List[MatchedQuestionAnswer] = []
        assigned_question_ids = set()
        unassigned_candidates = list(candidates)

        # ---------------------------------------------------------------------
        # Layer 1: Explicit Question Numbering (Priority 1)
        # ---------------------------------------------------------------------
        remaining_after_layer1 = []
        for cand in unassigned_candidates:
            matched = False
            if cand.detected_question_num is not None:
                # Find assignment question with matching question_number
                target_q = next(
                    (q for q in norm_questions if q["question_number"] == cand.detected_question_num),
                    None,
                )
                if target_q and target_q["question_id"] not in assigned_question_ids:
                    # High confidence explicit match (0.92 - 0.98)
                    base_conf = 0.95
                    # Slightly modulate with OCR confidence
                    adj_conf = round(min(0.98, max(0.85, base_conf * (0.9 + 0.1 * cand.average_ocr_confidence))), 2)

                    status = "Matched" if adj_conf >= self.high_confidence_threshold else "Review Required"

                    matched_answers.append(
                        MatchedQuestionAnswer(
                            question_id=target_q["question_id"],
                            question_number=target_q["question_number"],
                            question_text=target_q["question_text"],
                            extracted_text=cand.text,
                            page_start=cand.page_start,
                            page_end=cand.page_end,
                            detected_label=cand.detected_label,
                            match_method="explicit_numbering",
                            match_confidence=adj_conf,
                            status=status,
                            review_notes=(
                                f"Explicitly mapped via detected question number #{cand.detected_question_num}."
                            ),
                        )
                    )
                    assigned_question_ids.add(target_q["question_id"])
                    matched = True

            if not matched:
                remaining_after_layer1.append(cand)

        # ---------------------------------------------------------------------
        # Layer 2: Textual Similarity / Lexical Matching (Priority 2)
        # ---------------------------------------------------------------------
        remaining_after_layer2 = []
        unassigned_questions = [q for q in norm_questions if q["question_id"] not in assigned_question_ids]

        for cand in remaining_after_layer1:
            best_q = None
            best_score = 0.0

            # Compare candidate header or first lines against remaining questions
            cand_sample = cand.header_line or cand.text[:200]

            for q in unassigned_questions:
                # Score against question text
                score_q = self.calculate_text_similarity(cand_sample, q["question_text"])
                # Also check full candidate text against question
                score_full = self.calculate_text_similarity(cand.text[:500], q["question_text"])
                score = max(score_q, score_full)

                if score > best_score:
                    best_score = score
                    best_q = q

            if best_q and best_score >= 0.60:
                # Strong textual match (0.75 - 0.90)
                conf = round(min(0.90, max(0.75, best_score * 0.95)), 2)
                status = "Matched" if conf >= self.high_confidence_threshold else "Review Required"

                matched_answers.append(
                    MatchedQuestionAnswer(
                        question_id=best_q["question_id"],
                        question_number=best_q["question_number"],
                        question_text=best_q["question_text"],
                        extracted_text=cand.text,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                        detected_label=cand.detected_label,
                        match_method="text_similarity",
                        match_confidence=conf,
                        status=status,
                        review_notes=(
                            f"Mapped via text lexical overlap with Question #{best_q['question_number']} "
                            f"(score: {round(best_score * 100, 1)}%)."
                        ),
                    )
                )
                assigned_question_ids.add(best_q["question_id"])
                unassigned_questions = [q for q in unassigned_questions if q["question_id"] != best_q["question_id"]]
            else:
                remaining_after_layer2.append(cand)

        # ---------------------------------------------------------------------
        # Layer 3: Semantic Similarity / Keyword Cosine Similarity (Priority 3)
        # ---------------------------------------------------------------------
        remaining_after_layer3 = []

        for cand in remaining_after_layer2:
            best_q = None
            best_score = 0.0
            first_line = cand.text.splitlines()[0] if cand.text else ""
            cand_text_sample = cand.header_line or first_line

            for q in unassigned_questions:
                sem_score_header = self.calculate_semantic_similarity(cand_text_sample, q["question_text"])
                sem_score_full = self.calculate_semantic_similarity(cand.text[:800], q["question_text"])
                sem_score = max(sem_score_header, sem_score_full)

                # If question has supportive answer, check overlap with supportive answer keywords as fallback
                if q["supportive_answer"]:
                    supp_score = self.calculate_semantic_similarity(cand.text[:800], q["supportive_answer"][:400])
                    sem_score = max(sem_score, supp_score * 0.85)

                if sem_score > best_score:
                    best_score = sem_score
                    best_q = q

            if best_q and best_score >= 0.40:
                # Semantic similarity match (0.55 - 0.85)
                conf = round(min(0.85, max(0.55, best_score)), 2)
                status = "Matched" if conf >= self.high_confidence_threshold else "Review Required"

                matched_answers.append(
                    MatchedQuestionAnswer(
                        question_id=best_q["question_id"],
                        question_number=best_q["question_number"],
                        question_text=best_q["question_text"],
                        extracted_text=cand.text,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                        detected_label=cand.detected_label,
                        match_method="semantic_similarity",
                        match_confidence=conf,
                        status=status,
                        review_notes=(
                            f"Mapped via semantic cosine similarity with Question #{best_q['question_number']} "
                            f"(similarity: {round(best_score * 100, 1)}%)."
                        ),
                    )
                )
                assigned_question_ids.add(best_q["question_id"])
                unassigned_questions = [q for q in unassigned_questions if q["question_id"] != best_q["question_id"]]
            else:
                remaining_after_layer3.append(cand)

        # ---------------------------------------------------------------------
        # Layer 4: Unmatched / Orphan Content Fallback
        # ---------------------------------------------------------------------
        for orphan in remaining_after_layer3:
            matched_answers.append(
                MatchedQuestionAnswer(
                    question_id=None,
                    question_number=None,
                    question_text=None,
                    extracted_text=orphan.text,
                    page_start=orphan.page_start,
                    page_end=orphan.page_end,
                    detected_label=orphan.detected_label,
                    match_method="unmatched",
                    match_confidence=0.0,
                    status="Review Required",
                    review_notes=(
                        "Unresolved answer segment. Could not reliably match this text to an assignment question. "
                        "Faculty review required."
                    ),
                )
            )

        # Sort mapped answers by question_number (unmatched items at the end)
        matched_answers.sort(
            key=lambda a: (0 if a.question_number is not None else 1, a.question_number or 999, a.page_start)
        )

        return matched_answers

    # =========================================================================
    # 4. Pipeline Integration Helper
    # =========================================================================

    def process_and_match(
        self,
        pages: List[Union[DocumentPage, Dict[str, Any]]],
        questions: List[Union[Question, Dict[str, Any]]],
    ) -> List[MatchedQuestionAnswer]:
        """
        Execute full segmentation and hierarchical question matching workflow:
        Document Pages -> CandidateAnswerSegments -> MatchedQuestionAnswers.
        """
        candidate_segments = self.segment_pages(pages)
        return self.match_candidates_to_questions(candidate_segments, questions)
