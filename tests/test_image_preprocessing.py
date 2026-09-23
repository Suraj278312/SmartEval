"""
Deterministic Unit Tests for Image Preprocessing Stage 1:
- 4-way Orientation Detection (0°, 90°, 180°, 270°)
- Ambiguous orientation & Review Required conditions
- Notebook Ruled-Line Suppression & handwriting stroke preservation
- Line Segmentation Preparation & reading-order verification
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pytest
from app.processing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
    OrientationDetector,
    OrientationResult,
    LineRegion,
)


def _generate_synthetic_document_page(
    lines_count: int = 10,
    with_ruled_lines: bool = True,
    width: int = 800,
    height: int = 1100,
) -> Image.Image:
    """
    Generate a deterministic synthetic document page with horizontal text lines
    and optional notebook ruled lines. Text is concentrated near the top (academic layout).
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 1. Draw notebook ruled lines across the page
    if with_ruled_lines:
        line_spacing = 40
        for y in range(120, height - 80, line_spacing):
            draw.line([(60, y), (width - 60, y)], fill=(190, 195, 210), width=1)

    # 2. Draw handwritten-like text lines (dark ink)
    # Header box at top
    draw.rectangle([60, 50, 260, 90], outline=(40, 40, 40), width=2)
    draw.text((70, 60), "STUDENT NAME: Alice", fill=(20, 20, 20))

    # Text lines
    for i in range(lines_count):
        y = 135 + (i * 40)
        if y > height - 100:
            break
        # Draw text content
        draw.text(
            (80, y),
            f"Step {i+1}: Applying derivation equation T(n) = 2T(n/2) + O(n).",
            fill=(15, 15, 15),
        )
        # Add some vertical strokes crossing lines (descenders / ascenders)
        draw.line([(150 + i * 20, y - 5), (150 + i * 20, y + 25)], fill=(10, 10, 10), width=2)

    return img


# =========================================================================
# 1. Orientation Detector Tests
# =========================================================================

def test_orientation_detection_upright():
    """Test upright page (0°) is detected with high confidence and needs_review=False."""
    detector = OrientationDetector()
    page = _generate_synthetic_document_page(lines_count=12, with_ruled_lines=True)

    result = detector.detect_orientation(page)

    assert isinstance(result, OrientationResult)
    assert result.detected_rotation == 0
    assert result.confidence >= 0.60
    assert result.needs_review is False
    assert 0 in result.candidate_scores
    assert result.candidate_scores[0] > result.candidate_scores[180]


def test_orientation_detection_180_inverted():
    """Test 180° upside-down page is detected as 180° rotation needed."""
    detector = OrientationDetector()
    page = _generate_synthetic_document_page(lines_count=12, with_ruled_lines=True)
    inverted_page = page.rotate(180)

    result = detector.detect_orientation(inverted_page)

    assert result.detected_rotation == 180
    assert result.confidence >= 0.60
    assert result.needs_review is False
    assert result.candidate_scores[180] > result.candidate_scores[0]


def test_orientation_detection_90_clockwise():
    """Test 90° clockwise rotated page is detected as requiring 270° CW rotation to upright."""
    detector = OrientationDetector()
    page = _generate_synthetic_document_page(lines_count=12, with_ruled_lines=True)
    # Rotate 90 CW (PIL rotate(-90, expand=True))
    rotated_90_cw = page.rotate(-90, expand=True)

    result = detector.detect_orientation(rotated_90_cw)

    assert result.detected_rotation == 270
    assert result.confidence >= 0.60
    assert result.needs_review is False


def test_orientation_detection_270_clockwise():
    """Test 270° clockwise rotated page is detected as requiring 90° CW rotation to upright."""
    detector = OrientationDetector()
    page = _generate_synthetic_document_page(lines_count=12, with_ruled_lines=True)
    # Rotate 270 CW (PIL rotate(-270, expand=True))
    rotated_270_cw = page.rotate(-270, expand=True)

    result = detector.detect_orientation(rotated_270_cw)

    assert result.detected_rotation == 90
    assert result.confidence >= 0.60
    assert result.needs_review is False


def test_orientation_detection_ambiguous_review_required():
    """Test blank or symmetrical pages trigger needs_review=True without fabricated confidence."""
    detector = OrientationDetector(min_confidence=0.60)

    # 1. Blank page
    blank_img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    blank_res = detector.detect_orientation(blank_img)
    assert blank_res.needs_review is True
    assert blank_res.confidence == 0.0
    assert blank_res.method == "insufficient_ink"

    # 2. Symmetrical circle with no text directionality
    sym_array = np.full((600, 600), 255, dtype=np.uint8)
    cv2.circle(sym_array, (300, 300), 100, 0, -1)
    sym_res = detector.detect_orientation(sym_array)
    assert sym_res.needs_review is True
    assert sym_res.confidence < 0.60


# =========================================================================
# 2. Ruled-Line Suppression Tests
# =========================================================================

def test_ruled_line_suppression_returns_valid_image():
    """Test ruled-line suppression produces clean, valid image arrays without corrupting dimensions."""
    config = PreprocessingConfig(enable_line_suppression=True)
    preprocessor = ImagePreprocessor(config=config)

    page = _generate_synthetic_document_page(lines_count=8, with_ruled_lines=True)
    proc_pil, proc_cv = preprocessor.preprocess_image(page)

    assert isinstance(proc_pil, Image.Image)
    assert isinstance(proc_cv, np.ndarray)
    assert proc_cv.shape[0] > 0
    assert proc_cv.shape[1] > 0
    assert proc_pil.mode == "RGB"


def test_ruled_line_suppression_preserves_handwriting_strokes():
    """
    Test that vertical and slanted handwriting strokes intersecting ruled lines
    retain their ink density after morphological line suppression.
    """
    preprocessor = ImagePreprocessor()

    # Create synthetic canvas with a ruled horizontal line and a crossing vertical pen stroke
    canvas = np.full((200, 400), 255, dtype=np.uint8)
    # Ruled line across row 100
    canvas[100, 50:350] = 180
    # Crossing vertical stroke (dark pen ink) at col 200 from row 70 to 130
    canvas[70:130, 199:202] = 20

    # Suppress ruled lines
    suppressed = preprocessor._suppress_ruled_lines(canvas)

    # Ruled line away from the intersection (e.g. at col 100) should be lightened/inpainted
    assert suppressed[100, 100] > 180 or suppressed[100, 100] == 255

    # Intersecting stroke at (100, 200) should preserve dark ink (< 100)
    assert suppressed[100, 200] < 100
    assert suppressed[80, 200] < 100
    assert suppressed[120, 200] < 100


# =========================================================================
# 3. Line Segmentation Preparation Tests
# =========================================================================

def test_extract_line_regions_reading_order():
    """Test line regions are extracted and sorted strictly in top-to-bottom reading order."""
    preprocessor = ImagePreprocessor()
    page = _generate_synthetic_document_page(lines_count=6, with_ruled_lines=True)
    _, proc_cv = preprocessor.preprocess_image(page)

    regions = preprocessor.extract_line_regions(proc_cv)

    assert len(regions) >= 4
    for idx, region in enumerate(regions):
        assert isinstance(region, LineRegion)
        assert region.line_index == idx + 1
        x, y, w, h = region.bbox
        assert w > 50
        assert h > 10
        assert 0.0 <= region.relative_top <= 1.0
        assert 0.0 <= region.relative_bottom <= 1.0
        assert region.relative_top < region.relative_bottom
        assert region.confidence > 0.0

        # Verify strict vertical reading order
        if idx > 0:
            prev_region = regions[idx - 1]
            assert region.bbox[1] >= prev_region.bbox[1]


# =========================================================================
# 4. Full Pipeline Integration & Backward Compatibility Tests
# =========================================================================

def test_full_pipeline_with_orientation_correction_and_line_suppression(tmp_path):
    """Test full preprocessing pipeline detects inverted page, rotates it upright, and suppresses lines."""
    config = PreprocessingConfig(
        enable_orientation_detection=True,
        enable_line_suppression=True,
        enable_deskew=True,
    )
    preprocessor = ImagePreprocessor(config=config)

    # Create upside-down document
    page = _generate_synthetic_document_page(lines_count=10, with_ruled_lines=True)
    inverted = page.rotate(180)

    out_file = str(tmp_path / "proc_inverted_test.png")
    proc_pil, proc_cv = preprocessor.preprocess_image(inverted, output_filepath=out_file)

    assert preprocessor.last_orientation_result is not None
    assert preprocessor.last_orientation_result.detected_rotation == 180
    assert os.path.exists(out_file)

    # Extracted line regions from the preprocessed image should now start from top
    regions = preprocessor.extract_line_regions(proc_cv)
    assert len(regions) > 0
    assert regions[0].relative_top < 0.30  # First line starts near the top of the corrected page


def test_preprocessing_config_toggles():
    """Test disabling orientation detection or line suppression respects config flags."""
    # Disabled orientation detection
    cfg_no_orient = PreprocessingConfig(enable_orientation_detection=False)
    prep_no_orient = ImagePreprocessor(config=cfg_no_orient)
    page = _generate_synthetic_document_page(lines_count=5)
    prep_no_orient.preprocess_image(page)
    assert prep_no_orient.last_orientation_result.method == "disabled"

    # Disabled line suppression
    cfg_no_lines = PreprocessingConfig(enable_line_suppression=False)
    prep_no_lines = ImagePreprocessor(config=cfg_no_lines)
    _, proc_cv = prep_no_lines.preprocess_image(page)
    assert proc_cv is not None
