"""
Unit Tests for TrOCR Handwriting Recognition Engine.
Covers:
- Recognizer initialization & lazy loading
- Line region extraction & padding
- Mocked inference & token probability confidence estimation
- Line ordering & multi-line text reconstruction
- Empty/blank page handling
- Error handling & EasyOCR fallback mechanism
- HTR engine selection via environment configuration
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from app.processing.handwriting_recognizer import (
    TrOCRHandwritingRecognizer,
    EasyOCRHandwritingRecognizer,
    HTRResult,
    LineExtraction,
    get_default_recognizer,
    set_default_recognizer,
)
from app.processing.image_preprocessor import ImagePreprocessor, LineRegion


def _create_mock_page_image(lines_count: int = 3) -> Image.Image:
    """Helper to create a synthetic document image with dark lines."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(lines_count):
        y = 100 + (i * 80)
        draw.rectangle([50, y, 550, y + 35], fill=(30, 30, 30))
    return img


def _create_mock_trocr_model(decoded_texts=None, high_confidence=True):
    """Create a mock TrOCR model and processor for deterministic testing."""
    if decoded_texts is None:
        decoded_texts = ["Recognized handwritten text line"]

    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.eos_token_id = 2
    mock_model.config.pad_token_id = 1
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    def mock_generate(pixel_values, **kwargs):
        batch_size = pixel_values.shape[0]
        # Simulate sequence [start_token(0), token1(10), token2(20), eos(2)]
        seq = torch.tensor([[0, 10, 20, 2]] * batch_size)
        scores = []
        target_tokens = [10, 20, 2]
        for step_idx in range(3):
            # Shape: (batch_size, 50)
            logits = torch.zeros((batch_size, 50))
            if high_confidence:
                tgt = target_tokens[step_idx]
                logits[:, tgt] = 20.0  # Softmax will be ~0.999
            else:
                logits[:] = 0.0  # Low confidence
            scores.append(logits)
        return MagicMock(sequences=seq, scores=scores)

    mock_model.generate.side_effect = mock_generate

    mock_processor = MagicMock()
    mock_processor.side_effect = lambda images, return_tensors: MagicMock(
        pixel_values=torch.zeros((len(images), 3, 384, 384))
    )
    mock_processor.batch_decode.side_effect = lambda sequences, **kwargs: [
        decoded_texts[i % len(decoded_texts)] for i in range(len(sequences))
    ]

    return mock_model, mock_processor


# =========================================================================
# 1. Initialization & Lazy Loading Tests
# =========================================================================

def test_trocr_initialization_defaults():
    """Test TrOCR recognizer initializes without loading the heavy model immediately."""
    recognizer = TrOCRHandwritingRecognizer()

    assert recognizer.model_name == "microsoft/trocr-base-handwritten"
    assert recognizer.device in ("cuda", "cpu")
    assert recognizer.fallback_to_easyocr is True
    assert recognizer._is_loaded is False
    assert recognizer._model is None
    assert recognizer._processor is None


def test_trocr_custom_configuration():
    """Test TrOCR honors custom model name, device, and batch size."""
    recognizer = TrOCRHandwritingRecognizer(
        model_name="custom/trocr-model",
        device="cpu",
        low_confidence_threshold=0.65,
        fallback_to_easyocr=False,
        batch_size=8,
    )

    assert recognizer.model_name == "custom/trocr-model"
    assert recognizer.device == "cpu"
    assert recognizer.low_confidence_threshold == 0.65
    assert recognizer.fallback_to_easyocr is False
    assert recognizer.batch_size == 8


# =========================================================================
# 2. Recognition with Mocked Model
# =========================================================================

def test_trocr_mock_successful_recognition():
    """Test TrOCR processes page lines and reconstructs text with confidence scores."""
    mock_model, mock_processor = _create_mock_trocr_model(
        decoded_texts=["Step 1: Base Case", "Step 2: Induction Step", "Step 3: Conclusion"],
        high_confidence=True,
    )

    recognizer = TrOCRHandwritingRecognizer(device="cpu")
    recognizer._model = mock_model
    recognizer._processor = mock_processor
    recognizer._is_loaded = True

    page_img = _create_mock_page_image(lines_count=3)
    result = recognizer.recognize(page_img)

    assert isinstance(result, HTRResult)
    assert result.error_message is None
    assert len(result.lines) >= 1
    assert result.confidence >= 0.85
    assert result.is_low_confidence is False
    assert "Step" in result.text
    assert result.metadata["provider"] == "TrOCR"
    assert result.metadata["model"] == "microsoft/trocr-base-handwritten"


def test_trocr_preserves_line_reading_order():
    """Test reconstructed text maintains top-to-bottom line sequence."""
    mock_texts = ["First Top Line", "Second Middle Line", "Third Bottom Line"]
    mock_model, mock_processor = _create_mock_trocr_model(decoded_texts=mock_texts, high_confidence=True)

    recognizer = TrOCRHandwritingRecognizer(device="cpu", batch_size=1)
    recognizer._model = mock_model
    recognizer._processor = mock_processor
    recognizer._is_loaded = True

    page_img = _create_mock_page_image(lines_count=3)
    result = recognizer.recognize(page_img)

    assert len(result.lines) >= 3
    # Check that lines contain line texts separated by newlines
    lines = result.text.split("\n")
    assert len(lines) >= 3
    # Check that bounding boxes are in vertical descending order
    for i in range(1, len(result.lines)):
        prev_y = result.lines[i - 1].bounding_box[0][1]
        curr_y = result.lines[i].bounding_box[0][1]
        assert curr_y >= prev_y


def test_trocr_empty_or_blank_page():
    """Test blank image with no line regions returns safe empty HTR result."""
    recognizer = TrOCRHandwritingRecognizer(device="cpu")
    blank_img = Image.new("RGB", (600, 800), color=(255, 255, 255))

    result = recognizer.recognize(blank_img)

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.lines == []
    assert result.is_low_confidence is True
    assert result.metadata["total_lines"] == 0


def test_trocr_low_confidence_triggers_review_flag():
    """Test low average line confidence flags is_low_confidence=True."""
    mock_model, mock_processor = _create_mock_trocr_model(
        decoded_texts=["Unclear scribble"],
        high_confidence=False,
    )

    recognizer = TrOCRHandwritingRecognizer(low_confidence_threshold=0.55)
    recognizer._model = mock_model
    recognizer._processor = mock_processor
    recognizer._is_loaded = True

    page_img = _create_mock_page_image(lines_count=2)
    result = recognizer.recognize(page_img)

    assert result.confidence < 0.55
    assert result.is_low_confidence is True


# =========================================================================
# 3. Error Handling & Fallback to EasyOCR
# =========================================================================

def test_trocr_failure_with_easyocr_fallback():
    """Test that when TrOCR encounters a model failure, it seamlessly executes EasyOCR fallback."""
    # Create mock EasyOCR recognizer that succeeds
    mock_easyocr = MagicMock(spec=EasyOCRHandwritingRecognizer)
    mock_easyocr.recognize.return_value = HTRResult(
        text="Fallback EasyOCR text",
        confidence=0.88,
        lines=[LineExtraction(text="Fallback EasyOCR text", confidence=0.88)],
        is_low_confidence=False,
        metadata={"provider": "EasyOCR"},
    )

    recognizer = TrOCRHandwritingRecognizer(
        fallback_to_easyocr=True,
        easyocr_fallback=mock_easyocr,
    )

    # Force model load error
    recognizer._load_error = "Simulated CUDA Out Of Memory or download failure."

    page_img = _create_mock_page_image(lines_count=2)
    result = recognizer.recognize(page_img)

    mock_easyocr.recognize.assert_called_once()
    assert result.text == "Fallback EasyOCR text"
    assert result.confidence == 0.88
    assert result.metadata["fallback_from"] == "TrOCR"
    assert "Simulated CUDA" in result.metadata["primary_error"]


def test_trocr_failure_without_fallback():
    """Test that when fallback is disabled, TrOCR returns an error HTRResult."""
    recognizer = TrOCRHandwritingRecognizer(fallback_to_easyocr=False)
    recognizer._load_error = "Model weights not found."

    page_img = _create_mock_page_image(lines_count=2)
    result = recognizer.recognize(page_img)

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.is_low_confidence is True
    assert "Model weights not found" in result.error_message
    assert result.metadata["failed"] is True


# =========================================================================
# 4. Factory & Engine Selection Tests
# =========================================================================

def test_get_default_recognizer_easyocr(monkeypatch):
    """Test get_default_recognizer returns EasyOCR by default."""
    monkeypatch.delenv("HTR_ENGINE", raising=False)
    set_default_recognizer(None)

    recognizer = get_default_recognizer()
    assert isinstance(recognizer, EasyOCRHandwritingRecognizer)


def test_get_default_recognizer_trocr(monkeypatch):
    """Test get_default_recognizer returns TrOCR when HTR_ENGINE=trocr."""
    monkeypatch.setenv("HTR_ENGINE", "trocr")
    set_default_recognizer(None)

    recognizer = get_default_recognizer()
    assert isinstance(recognizer, TrOCRHandwritingRecognizer)
    assert recognizer.model_name == "microsoft/trocr-base-handwritten"

    # Reset
    set_default_recognizer(None)
