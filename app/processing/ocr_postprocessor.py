"""
OCR Post-Processing and Text Reconstruction Engine for SmartEval.

Performs spatial line grouping of EasyOCR bounding-box fragments,
left-to-right fragment sorting, top-to-bottom line sequencing,
noise token filtering, and safe question label OCR substitution correction.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.processing.handwriting_recognizer import LineExtraction, HTRResult


class OCRPostProcessor:
    """
    Lightweight, deterministic post-processing layer applied to raw OCR extractions.
    Groups spatial fragments into coherent text lines, removes isolated punctuation noise,
    normalizes question headers (e.g. QI -> Q1, QZ -> Q2, 01 -> Q1), and preserves
    raw OCR text for complete faculty auditability.
    """

    # Non-alphanumeric noise patterns that represent OCR artifacts when isolated
    ISOLATED_NOISE_RE = re.compile(r"^[\^~&_\-|/\\@*#:;=.,`'\"<>(){}\[\]!]+$")

    # Question header normalization patterns (at the start of a line)
    HEADER_PATTERNS = [
        # QI, Ql, Q|, Q/ -> Q1
        (re.compile(r"^[ \t]*Q[Il|/][ \t.:\-\)]+", re.IGNORECASE), "Q1 "),
        (re.compile(r"^[ \t]*Q[Il|/]$", re.IGNORECASE), "Q1"),
        # QII, Qll, QZ -> Q2
        (re.compile(r"^[ \t]*Q(?:II|ll|Z)[ \t.:\-\)]+", re.IGNORECASE), "Q2 "),
        (re.compile(r"^[ \t]*Q(?:II|ll|Z)$", re.IGNORECASE), "Q2"),
        # QIII, Qlll -> Q3
        (re.compile(r"^[ \t]*Q(?:III|lll)[ \t.:\-\)]+", re.IGNORECASE), "Q3 "),
        (re.compile(r"^[ \t]*Q(?:III|lll)$", re.IGNORECASE), "Q3"),
        # Question I, Question II, Question III
        (re.compile(r"^[ \t]*Question[ \t.]+I[ \t.:\-\)]+", re.IGNORECASE), "Question 1 "),
        (re.compile(r"^[ \t]*Question[ \t.]+II[ \t.:\-\)]+", re.IGNORECASE), "Question 2 "),
        (re.compile(r"^[ \t]*Question[ \t.]+III[ \t.:\-\)]+", re.IGNORECASE), "Question 3 "),
        # Ans I, Ans II, Ans III
        (re.compile(r"^[ \t]*Ans(?:wer)?[ \t.]+I[ \t.:\-\)]+", re.IGNORECASE), "Ans 1 "),
        (re.compile(r"^[ \t]*Ans(?:wer)?[ \t.]+II[ \t.:\-\)]+", re.IGNORECASE), "Ans 2 "),
        (re.compile(r"^[ \t]*Ans(?:wer)?[ \t.]+III[ \t.:\-\)]+", re.IGNORECASE), "Ans 3 "),
        # Problem I / Task I
        (re.compile(r"^[ \t]*(?:Problem|Task)[ \t.]+I[ \t.:\-\)]+", re.IGNORECASE), "Problem 1 "),
        (re.compile(r"^[ \t]*(?:Problem|Task)[ \t.]+II[ \t.:\-\)]+", re.IGNORECASE), "Problem 2 "),
        (re.compile(r"^[ \t]*(?:Problem|Task)[ \t.]+III[ \t.:\-\)]+", re.IGNORECASE), "Problem 3 "),
        # OCR Confusions: 01, O1 -> Q1; G2, 02, O2, Qa -> Q2; Gg, 03, O3 -> Q3
        (re.compile(r"^[ \t]*[0O]1[ \t.:\-\)]+", re.IGNORECASE), "Q1 "),
        (re.compile(r"^[ \t]*[0O]1$", re.IGNORECASE), "Q1"),
        (re.compile(r"^[ \t]*(?:[0OG]2|Qa)[ \t.:\-\)]+", re.IGNORECASE), "Q2 "),
        (re.compile(r"^[ \t]*(?:[0OG]2|Qa)$", re.IGNORECASE), "Q2"),
        (re.compile(r"^[ \t]*(?:[0OGg]3|Gg)[ \t.:\-\)]+", re.IGNORECASE), "Q3 "),
        (re.compile(r"^[ \t]*(?:[0OGg]3|Gg)$", re.IGNORECASE), "Q3"),
        # Q.1, Q.2, Q.3, Q-1, Q-2, Q 1, Q 2, Q 3
        (re.compile(r"^[ \t]*Q[ \t.\-]*(\d+)[ \t.:\-\)]*", re.IGNORECASE), r"Q\1 "),
        # Ans.1, Ans.2, Ans 1, Ans-1
        (re.compile(r"^[ \t]*Ans(?:wer)?[ \t.\-]*(\d+)[ \t.:\-\)]*", re.IGNORECASE), r"Ans \1 "),
    ]

    def __init__(
        self,
        noise_confidence_threshold: float = 0.25,
        line_overlap_ratio: float = 0.35,
    ):
        self.noise_confidence_threshold = noise_confidence_threshold
        self.line_overlap_ratio = line_overlap_ratio

    def is_noise_fragment(self, text: str, confidence: float) -> bool:
        """Determines whether an individual raw OCR fragment is pure noise."""
        clean = text.strip()
        if not clean:
            return True
        if self.ISOLATED_NOISE_RE.match(clean):
            # If purely non-alphanumeric and low confidence or isolated artifact
            if confidence < 0.35:
                return True
            if clean in ("^", "~&", "} /", "}", "{", "~", "`", "@_", "\\", "|/"):
                return True
        return False

    def reconstruct_spatial_lines(
        self, extractions: List[LineExtraction]
    ) -> List[LineExtraction]:
        """
        Groups bounding-box fragments into coherent horizontal text lines.
        Fragments within the same vertical band are merged left-to-right.
        Lines are ordered top-to-bottom.
        """
        if not extractions:
            return []

        # Filter out isolated noise fragments early
        valid_extractions = [
            ext for ext in extractions
            if not self.is_noise_fragment(ext.text, ext.confidence)
        ]

        if not valid_extractions:
            valid_extractions = extractions

        # Check if bounding boxes exist
        has_bboxes = any(ext.bounding_box is not None for ext in valid_extractions)
        if not has_bboxes:
            # Fall back to sequential list
            return [ext for ext in valid_extractions if ext.text.strip()]

        # Compute bounding geometry for each fragment
        items = []
        heights = []
        for ext in valid_extractions:
            text = ext.text.strip()
            if not text:
                continue

            bbox = ext.bounding_box
            if bbox and len(bbox) == 4:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                y_mid = (y_min + y_max) / 2.0
                h = max(1.0, y_max - y_min)
                heights.append(h)
            else:
                x_min, x_max = 0, 0
                y_min, y_max = 0, 0
                y_mid = 0
                h = 20.0

            items.append({
                "text": text,
                "confidence": ext.confidence,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_mid": y_mid,
                "height": h,
                "bbox": bbox,
            })

        if not items:
            return []

        median_height = float(np.median(heights)) if heights else 20.0
        y_tolerance = max(14.0, median_height * 0.55)

        # Sort items by vertical center coordinate
        items.sort(key=lambda it: it["y_mid"])

        # Cluster items into horizontal text lines
        clusters: List[List[Dict[str, Any]]] = []

        for item in items:
            assigned = False
            for cluster in clusters:
                # Calculate cluster vertical bounds
                c_y_min = min(it["y_min"] for it in cluster)
                c_y_max = max(it["y_max"] for it in cluster)
                c_y_mid = (c_y_min + c_y_max) / 2.0

                # Check vertical distance and overlap
                v_dist = abs(item["y_mid"] - c_y_mid)
                overlap_min = max(item["y_min"], c_y_min)
                overlap_max = min(item["y_max"], c_y_max)
                overlap = max(0.0, overlap_max - overlap_min)
                min_h = min(item["height"], c_y_max - c_y_min)
                overlap_ratio = overlap / max(1.0, min_h)

                if v_dist <= y_tolerance or overlap_ratio >= self.line_overlap_ratio:
                    cluster.append(item)
                    assigned = True
                    break

            if not assigned:
                clusters.append([item])

        # Sort clusters top-to-bottom by average y_min
        clusters.sort(key=lambda cl: float(np.mean([it["y_min"] for it in cl])))

        # Reconstruct each line
        reconstructed_lines: List[LineExtraction] = []
        for cluster in clusters:
            # Sort fragments strictly left-to-right by x_min
            cluster.sort(key=lambda it: it["x_min"])

            # Combine fragment text
            line_parts = []
            total_chars = 0
            weighted_conf_sum = 0.0

            for it in cluster:
                txt = it["text"]
                line_parts.append(txt)
                w = len(txt)
                total_chars += w
                weighted_conf_sum += it["confidence"] * w

            line_text = " ".join(line_parts).strip()
            if not line_text:
                continue

            line_conf = (
                weighted_conf_sum / total_chars
                if total_chars > 0
                else float(np.mean([it["confidence"] for it in cluster]))
            )

            # Enclosing bounding box
            enclosing_x_min = min(it["x_min"] for it in cluster)
            enclosing_x_max = max(it["x_max"] for it in cluster)
            enclosing_y_min = min(it["y_min"] for it in cluster)
            enclosing_y_max = max(it["y_max"] for it in cluster)

            enclosing_bbox = [
                [int(enclosing_x_min), int(enclosing_y_min)],
                [int(enclosing_x_max), int(enclosing_y_min)],
                [int(enclosing_x_max), int(enclosing_y_max)],
                [int(enclosing_x_min), int(enclosing_y_max)],
            ]

            reconstructed_lines.append(
                LineExtraction(
                    text=line_text,
                    confidence=line_conf,
                    bounding_box=enclosing_bbox,
                )
            )

        return reconstructed_lines

    def filter_line_noise(self, text: str, confidence: float) -> str:
        """
        Cleans isolated low-confidence noise tokens while preserving academic syntax.
        """
        tokens = text.split()
        cleaned_tokens = []

        for tok in tokens:
            # Strip pure symbol artifacts if low confidence or isolated
            if self.ISOLATED_NOISE_RE.match(tok):
                # Preserve valid math/programming operators (+, -, =, *, ==, !=, <=, >=)
                if tok in ("+", "-", "=", "*", "==", "!=", "<=", ">="):
                    cleaned_tokens.append(tok)
                elif confidence >= 0.70 and len(tok) == 1 and tok not in ("^", "~", "`", "|", "\\", "}", "{"):
                    cleaned_tokens.append(tok)
                else:
                    # Filter out noise like ~&, ^, } /, @_, etc.
                    continue
            else:
                # Remove stray leading/trailing junk punctuation
                tok_clean = tok.strip("~^`|\\{}")
                if tok_clean:
                    cleaned_tokens.append(tok_clean)

        return " ".join(cleaned_tokens).strip()

    def normalize_question_header(self, line: str) -> Tuple[str, Optional[str]]:
        """
        Normalizes OCR glyph errors in question labels (e.g., QI -> Q1, QZ -> Q2, 01 -> Q1).
        Returns (normalized_line, detected_header).
        """
        cleaned_line = line.strip()
        detected_header = None

        for pattern, replacement in self.HEADER_PATTERNS:
            if pattern.search(cleaned_line):
                cleaned_line = pattern.sub(replacement, cleaned_line, count=1).strip()
                tokens = cleaned_line.split()
                if tokens:
                    first_word = tokens[0]
                    if re.match(r"^(?:Q\d+|\d+\.)$", first_word, re.IGNORECASE):
                        detected_header = first_word
                    elif first_word.lower() in ("ans", "answer", "question", "problem", "task") and len(tokens) > 1:
                        second_word = tokens[1].rstrip(".:-)")
                        if second_word.isdigit():
                            detected_header = f"{first_word.capitalize()} {second_word}"
                break

        if not detected_header:
            tokens = cleaned_line.split()
            if tokens:
                first_word = tokens[0]
                if re.match(r"^(?:Q\d+|\d+\.)$", first_word, re.IGNORECASE):
                    detected_header = first_word
                elif first_word.lower() in ("ans", "answer", "question", "problem", "task") and len(tokens) > 1:
                    second_word = tokens[1].rstrip(".:-)")
                    if second_word.isdigit():
                        detected_header = f"{first_word.capitalize()} {second_word}"

        return cleaned_line, detected_header

    def post_process(self, result: HTRResult) -> HTRResult:
        """
        Executes the full post-processing pipeline on an HTRResult:
        1. Preserves raw text in metadata['raw_text'].
        2. Spatially clusters bounding boxes into ordered text lines.
        3. Filters low-confidence noise tokens.
        4. Normalizes question headers.
        5. Updates text, lines, confidence, and metadata.
        """
        raw_text = result.text or ""
        raw_lines = result.lines or []

        if not raw_lines and not raw_text:
            return result

        # 1. Spatial line reconstruction
        if raw_lines:
            reconstructed = self.reconstruct_spatial_lines(raw_lines)
        else:
            # Reconstruct from raw text lines
            reconstructed = [
                LineExtraction(text=l.strip(), confidence=result.confidence, bounding_box=None)
                for l in raw_text.splitlines()
                if l.strip() and not self.is_noise_fragment(l.strip(), result.confidence)
            ]

        # 2. Filter noise and normalize headers
        cleaned_lines: List[LineExtraction] = []
        detected_headers: List[str] = []
        confidences: List[float] = []

        for line_ext in reconstructed:
            # Noise filtering
            filtered_text = self.filter_line_noise(line_ext.text, line_ext.confidence)
            if not filtered_text:
                continue

            # Question header normalization
            normalized_text, header_tag = self.normalize_question_header(filtered_text)
            if header_tag:
                detected_headers.append(header_tag)

            if normalized_text:
                cleaned_lines.append(
                    LineExtraction(
                        text=normalized_text,
                        confidence=line_ext.confidence,
                        bounding_box=line_ext.bounding_box,
                    )
                )
                confidences.append(line_ext.confidence)

        # 3. Assemble cleaned text
        cleaned_full_text = "\n".join(l.text for l in cleaned_lines)
        avg_conf = float(np.mean(confidences)) if confidences else result.confidence

        # 4. Update metadata
        metadata = dict(result.metadata or {})
        metadata["raw_text"] = raw_text
        metadata["post_processed"] = True
        metadata["detected_headers"] = detected_headers
        metadata["reconstructed_line_count"] = len(cleaned_lines)
        metadata["raw_segment_count"] = len(raw_lines)

        return HTRResult(
            text=cleaned_full_text,
            confidence=avg_conf,
            lines=cleaned_lines,
            is_low_confidence=(avg_conf < 0.40) or (len(cleaned_lines) == 0),
            error_message=result.error_message,
            metadata=metadata,
        )
