"""
Image Preprocessing Pipeline for Handwritten Assignment Pages.
Applies non-destructive computer vision enhancements to improve HTR legibility:
- 4-way document orientation detection (0°, 90°, 180°, 270°) and rotation correction
- Horizontal notebook ruled-line suppression with intersecting handwriting stroke preservation
- Sub-pixel deskew and safe margin cropping
- Candidate line region segmentation for downstream HTR engines
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


@dataclass
class OrientationResult:
    """Structured outcome of 4-way document orientation detection."""
    detected_rotation: int  # 0, 90, 180, 270 (degrees clockwise to rotate to upright)
    confidence: float       # 0.0 to 1.0 confidence score
    method: str             # Algorithm/method name
    needs_review: bool      # True if confidence is below threshold or ambiguous
    candidate_scores: Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"OrientationResult(detected_rotation={self.detected_rotation}°, "
            f"confidence={self.confidence:.2f}, needs_review={self.needs_review}, "
            f"method='{self.method}')"
        )


@dataclass
class LineRegion:
    """Candidate handwritten text line region for subsequent HTR line recognition."""
    line_index: int
    bbox: Tuple[int, int, int, int]      # (x, y, width, height) in page coordinates
    confidence: float                    # Quality/prominence metric (0.0 to 1.0)
    relative_top: float                  # y / page_height (0.0 to 1.0)
    relative_bottom: float               # (y + height) / page_height (0.0 to 1.0)
    image: Optional[np.ndarray] = None   # Cropped line patch array


@dataclass
class PreprocessingConfig:
    """Configuration options for handwritten page preprocessing."""
    enable_orientation_detection: bool = True
    enable_line_suppression: bool = True
    enable_grayscale: bool = True
    enable_denoise: bool = True
    enable_contrast_enhancement: bool = True
    enable_deskew: bool = True
    enable_margin_crop: bool = True
    enable_normalization: bool = True
    target_width: int = 1800
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    max_deskew_angle: float = 45.0
    orientation_min_confidence: float = 0.60
    line_suppression_kernel_ratio: float = 0.035
    line_suppression_inpaint_radius: int = 2


class OrientationDetector:
    """
    Dedicated 4-way document orientation detector (0°, 90°, 180°, 270°).
    Uses projection profile variance, ink distribution asymmetry, and stroke gradients.
    """

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence

    def detect_orientation(
        self, image_input: Union[Image.Image, np.ndarray]
    ) -> OrientationResult:
        """
        Evaluate candidate orientations (0°, 90°, 180°, 270°) and return
        a structured OrientationResult indicating degrees CW needed to make the image upright.
        """
        if isinstance(image_input, Image.Image):
            gray = np.array(image_input.convert("L"))
        elif len(image_input.shape) == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_input.copy()

        h_orig, w_orig = gray.shape[:2]

        # 1. Binarize to isolate foreground ink
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total_ink = np.sum(thresh > 0)

        # Check for empty or near-blank page (< 0.3% ink coverage)
        if total_ink < (h_orig * w_orig * 0.003):
            return OrientationResult(
                detected_rotation=0,
                confidence=0.0,
                method="insufficient_ink",
                needs_review=True,
                candidate_scores={0: 0.25, 90: 0.25, 180: 0.25, 270: 0.25},
            )

        # 2. Evaluate all 4 candidate rotations (0°, 90°, 180°, 270°)
        # Angle represents degrees clockwise to rotate the page to make it upright.
        raw_scores: Dict[int, float] = {}

        for angle in [0, 90, 180, 270]:
            # k is the counter-clockwise 90-deg rotations to rotate page by `angle` clockwise
            k = (360 - angle) // 90
            test_thresh = np.rot90(thresh, k % 4)
            h, _ = test_thresh.shape[:2]

            # Feature A: Horizontal projection profile variance vs Vertical projection profile variance
            # Horizontal text lines produce high peak-to-valley variance when upright
            h_proj = np.sum(test_thresh, axis=1) / 255.0
            v_proj = np.sum(test_thresh, axis=0) / 255.0
            h_var = float(np.var(h_proj))
            v_var = float(np.var(v_proj))
            axis_score = h_var / (v_var + 1e-5)

            # Feature B: Top margin ink vs Bottom margin ink (top 22% vs bottom 22%)
            # In upright academic pages, header ink and top-down writing flow make top margin > bottom margin
            top_ink = float(np.sum(test_thresh[: int(h * 0.22), :]) / 255.0)
            bot_ink = float(np.sum(test_thresh[int(h * 0.78) :, :]) / 255.0)
            margin_score = (top_ink + 20.0) / (bot_ink + 20.0)

            # Upright text lines have horizontal variance and top-margin > bottom-margin
            combined = axis_score * (margin_score ** 1.2)
            raw_scores[angle] = max(0.001, combined)

        # 3. Normalize scores
        total_score = sum(raw_scores.values()) + 1e-6
        norm_scores = {k: round(v / total_score, 3) for k, v in raw_scores.items()}

        sorted_candidates = sorted(
            norm_scores.items(), key=lambda item: item[1], reverse=True
        )
        best_angle, best_prob = sorted_candidates[0]
        _, second_prob = sorted_candidates[1]

        score_margin = best_prob - second_prob

        # Confidence metric derived from top candidate probability and margin over runner-up
        confidence = float(min(1.0, max(0.0, best_prob * (1.0 + score_margin * 0.5))))

        # Ambiguous check: low confidence or indistinguishable top two candidates
        needs_review = (confidence < self.min_confidence) or (score_margin < 0.15)

        return OrientationResult(
            detected_rotation=best_angle,
            confidence=round(confidence, 2),
            method="projection_and_margin_variance",
            needs_review=needs_review,
            candidate_scores=norm_scores,
        )


class ImagePreprocessor:
    """
    Modular image preprocessor for handwritten documents.
    Features:
    - 4-way orientation detection & upright rotation
    - Ruled-line suppression protecting handwriting strokes
    - CLAHE contrast enhancement & subtle Gaussian denoising
    - Sub-pixel deskew & safe margin cropping
    - Resolution normalization & line region segmentation
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        orientation_detector: Optional[OrientationDetector] = None,
    ):
        self.config = config or PreprocessingConfig()
        self.orientation_detector = (
            orientation_detector
            or OrientationDetector(min_confidence=self.config.orientation_min_confidence)
        )
        self.last_orientation_result: Optional[OrientationResult] = None

    def detect_orientation(
        self, image_input: Union[Image.Image, np.ndarray]
    ) -> OrientationResult:
        """Evaluate document orientation."""
        return self.orientation_detector.detect_orientation(image_input)

    def preprocess_image(
        self,
        image_input: Union[Image.Image, np.ndarray],
        output_filepath: Optional[str] = None,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Run the full preprocessing pipeline on a PIL Image or OpenCV array.
        Returns (processed_pil_image, processed_cv2_array).
        """
        # Convert PIL to OpenCV BGR numpy array
        if isinstance(image_input, Image.Image):
            cv_img = np.array(image_input.convert("RGB"))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        else:
            cv_img = image_input.copy()

        # 1. Grayscale Conversion
        if self.config.enable_grayscale and len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = (
                cv_img
                if len(cv_img.shape) == 2
                else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            )

        # 2. Orientation Detection & Correction
        if self.config.enable_orientation_detection:
            orient_res = self.orientation_detector.detect_orientation(gray)
            self.last_orientation_result = orient_res
            if orient_res.detected_rotation != 0 and not orient_res.needs_review:
                gray = self._rotate_image_degrees(gray, orient_res.detected_rotation)
        else:
            self.last_orientation_result = OrientationResult(
                detected_rotation=0,
                confidence=1.0,
                method="disabled",
                needs_review=False,
                candidate_scores={0: 1.0},
            )

        # 3. Noise Reduction (Subtle Gaussian Blur to preserve pen strokes)
        if self.config.enable_denoise:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Contrast Enhancement via CLAHE
        if self.config.enable_contrast_enhancement:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size,
            )
            gray = clahe.apply(gray)

        # 5. Notebook Ruled-Line Suppression
        if self.config.enable_line_suppression:
            gray = self._suppress_ruled_lines(gray)

        # 6. Deskew / Minor Rotation Correction (preserved separately from orientation)
        if self.config.enable_deskew:
            gray = self._deskew_image(gray)

        # 7. Safe Margin Cropping (detects page contents bounding box)
        if self.config.enable_margin_crop:
            gray = self._crop_safe_margins(gray)

        # 8. Resolution Normalization
        if self.config.enable_normalization and gray.shape[1] > 0:
            h, w = gray.shape[:2]
            if w != self.config.target_width and w > 0:
                aspect = h / float(w)
                new_w = self.config.target_width
                new_h = int(new_w * aspect)
                gray = cv2.resize(
                    gray,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA if w > new_w else cv2.INTER_CUBIC,
                )

        # Convert back to PIL Image (RGB)
        processed_pil = Image.fromarray(gray).convert("RGB")

        # Save to disk if requested
        if output_filepath:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            processed_pil.save(output_filepath, format="PNG")

        return processed_pil, gray

    def _rotate_image_degrees(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image clockwise by 90, 180, or 270 degrees."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _suppress_ruled_lines(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and attenuate horizontal notebook ruled lines using morphology,
        protecting intersecting vertical/diagonal handwriting strokes.
        """
        try:
            h, w = gray.shape[:2]
            # Foreground binarization using adaptive thresholding for sensitivity to ruled lines
            block_size = max(15, (w // 60) * 2 + 1)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 8
            )

            # 1. Detect horizontal continuous lines
            kernel_len = max(20, int(w * self.config.line_suppression_kernel_ratio))
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

            # Fast exit if no continuous horizontal lines detected
            if np.sum(horizontal_lines > 0) < (h * w * 0.0003):
                return gray

            # 2. Detect vertical/slanted handwriting strokes that intersect lines
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
            vertical_strokes = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

            # 3. Dilate vertical strokes around junctions to prevent clipping handwriting ink
            v_dilated = cv2.dilate(
                vertical_strokes,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            )

            # 4. Mask of ruled lines to remove (lines not part of intersecting handwriting)
            lines_to_remove = cv2.bitwise_and(
                horizontal_lines, cv2.bitwise_not(v_dilated)
            )

            # 5. Non-destructive inpainting
            if np.sum(lines_to_remove > 0) > 0:
                suppressed = cv2.inpaint(
                    gray,
                    lines_to_remove,
                    inpaintRadius=self.config.line_suppression_inpaint_radius,
                    flags=cv2.INPAINT_TELEA,
                )
                return suppressed

            return gray
        except Exception:
            return gray

    def _deskew_image(self, gray: np.ndarray) -> np.ndarray:
        """Detect and correct document minor skew using foreground contours."""
        try:
            # Invert and threshold to find text foreground pixels
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            coords = np.column_stack(np.where(thresh > 0))

            if len(coords) < 100:
                return gray  # Not enough text pixels to compute angle safely

            # Compute minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]

            # Adjust angle for OpenCV coordinate system
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Only correct if angle is reasonable and non-trivial
            if 0.5 <= abs(angle) <= self.config.max_deskew_angle:
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    gray,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return rotated

            return gray
        except Exception:
            return gray

    def _crop_safe_margins(self, gray: np.ndarray, padding: int = 30) -> np.ndarray:
        """Trim excessive whitespace around document with protective padding."""
        try:
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return gray

            # Find bounding box of all text contours
            x_min, y_min = float("inf"), float("inf")
            x_max, y_max = 0, 0

            for cnt in contours:
                if cv2.contourArea(cnt) > 20:  # Ignore small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            h_orig, w_orig = gray.shape[:2]

            # If valid bounding box found
            if x_min < x_max and y_min < y_max:
                x_start = max(0, int(x_min - padding))
                y_start = max(0, int(y_min - padding))
                x_end = min(w_orig, int(x_max + padding))
                y_end = min(h_orig, int(y_max + padding))

                # Ensure minimum crop area (at least 30% of original) to avoid over-cropping
                if (x_end - x_start) > (w_orig * 0.3) and (y_end - y_start) > (
                    h_orig * 0.3
                ):
                    return gray[y_start:y_end, x_start:x_end]

            return gray
        except Exception:
            return gray

    def extract_line_regions(
        self,
        image_input: Union[Image.Image, np.ndarray],
        min_line_height: int = 15,
        padding: int = 6,
        gaussian_sigma: float = 4.0,
    ) -> List[LineRegion]:
        """
        Identify likely handwritten text bands using horizontal projection profile
        and whitespace valleys. Returns candidate line regions sorted in top-to-bottom reading order.
        """
        if isinstance(image_input, Image.Image):
            gray = np.array(image_input.convert("L"))
        elif len(image_input.shape) == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_input.copy()

        h, w = gray.shape[:2]
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 1. Horizontal projection profile
        h_profile = np.sum(thresh, axis=1) / 255.0

        # 2. 1D Gaussian smoothing
        kernel_size = int(gaussian_sigma * 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_1d = cv2.getGaussianKernel(kernel_size, gaussian_sigma).ravel()
        smoothed = np.convolve(h_profile, kernel_1d, mode="same")

        # 3. Adaptive threshold for text band presence
        mean_val = np.mean(smoothed)
        max_val = np.max(smoothed) if len(smoothed) > 0 else 1.0
        if max_val <= 0:
            return []

        line_threshold = max(2.0, mean_val * 0.20)

        # 4. Identify contiguous text bands (valleys define line boundaries)
        in_line = False
        start_y = 0
        raw_bands: List[Tuple[int, int, float]] = []

        for y, val in enumerate(smoothed):
            if val > line_threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= line_threshold and in_line:
                in_line = False
                end_y = y
                if (end_y - start_y) >= min_line_height:
                    raw_bands.append(
                        (start_y, end_y, float(np.max(smoothed[start_y:end_y])))
                    )

        if in_line and (h - start_y) >= min_line_height:
            raw_bands.append((start_y, h, float(np.max(smoothed[start_y:h]))))

        line_regions: List[LineRegion] = []

        for idx, (sy, ey, peak_val) in enumerate(raw_bands):
            # Apply protective vertical padding
            padded_sy = max(0, sy - padding)
            padded_ey = min(h, ey + padding)
            band_h = padded_ey - padded_sy

            # Determine horizontal bounding box
            band_thresh = thresh[padded_sy:padded_ey, :]
            x_indices = np.where(np.sum(band_thresh, axis=0) > 0)[0]

            if len(x_indices) > 0:
                x_min = max(0, int(x_indices[0] - padding))
                x_max = min(w, int(x_indices[-1] + padding))
                band_w = x_max - x_min
            else:
                x_min = 0
                x_max = w
                band_w = w

            # Quality/confidence metric based on normalized peak prominence
            conf = float(min(1.0, max(0.2, peak_val / (max_val + 1e-5))))

            region = LineRegion(
                line_index=idx + 1,
                bbox=(x_min, padded_sy, band_w, band_h),
                confidence=round(conf, 2),
                relative_top=round(padded_sy / float(h), 4),
                relative_bottom=round(padded_ey / float(h), 4),
                image=gray[padded_sy:padded_ey, x_min:x_max].copy(),
            )
            line_regions.append(region)

        # Sort strictly in top-to-bottom reading order
        line_regions.sort(key=lambda r: r.bbox[1])
        return line_regions
