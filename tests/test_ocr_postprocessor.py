"""
Unit tests for OCRPostProcessor: Spatial Line Grouping, Noise Filtering, and Question Header Recovery.
"""

from PIL import Image
import pytest
from app.processing.handwriting_recognizer import LineExtraction, HTRResult, EasyOCRHandwritingRecognizer
from app.processing.ocr_postprocessor import OCRPostProcessor


def test_spatial_line_reconstruction_and_reading_order():
    """Test fragments on the same horizontal line are merged left-to-right, and lines ordered top-to-bottom."""
    processor = OCRPostProcessor()

    # Create fragments scattered across 2 lines
    extractions = [
        # Line 2 second word (x=200..300, y=190..210)
        LineExtraction(
            text="Queue",
            confidence=0.85,
            bounding_box=[[200, 190], [300, 190], [300, 210], [200, 210]],
        ),
        # Line 1 first word (x=50..150, y=90..110)
        LineExtraction(
            text="Explain",
            confidence=0.90,
            bounding_box=[[50, 90], [150, 90], [150, 110], [50, 110]],
        ),
        # Line 2 first word (x=50..180, y=190..210)
        LineExtraction(
            text="Stack vs",
            confidence=0.88,
            bounding_box=[[50, 190], [180, 190], [180, 210], [50, 210]],
        ),
        # Line 1 second word (x=160..320, y=90..110)
        LineExtraction(
            text="Polymorphism",
            confidence=0.92,
            bounding_box=[[160, 90], [320, 90], [320, 110], [160, 110]],
        ),
    ]

    reconstructed = processor.reconstruct_spatial_lines(extractions)

    assert len(reconstructed) == 2
    assert reconstructed[0].text == "Explain Polymorphism"
    assert reconstructed[1].text == "Stack vs Queue"
    assert reconstructed[0].bounding_box[0] == [50, 90]
    assert reconstructed[0].bounding_box[1] == [320, 90]


def test_question_header_normalization():
    """Test OCR substitutions in question labels are normalized safely."""
    processor = OCRPostProcessor()

    cases = [
        ("QI. Explain polymorphism in Java.", "Q1 Explain polymorphism in Java.", "Q1"),
        ("Ql Explain polymorphism in Java.", "Q1 Explain polymorphism in Java.", "Q1"),
        ("Q| Polymorphism concept", "Q1 Polymorphism concept", "Q1"),
        ("QII. Difference between Stack and Queue", "Q2 Difference between Stack and Queue", "Q2"),
        ("QZ Difference between Stack and Queue", "Q2 Difference between Stack and Queue", "Q2"),
        ("QIII. Binary Search algorithm", "Q3 Binary Search algorithm", "Q3"),
        ("01 Explain OOP concepts", "Q1 Explain OOP concepts", "Q1"),
        ("G2 Stack is a LIFO structure", "Q2 Stack is a LIFO structure", "Q2"),
        ("Gg Binary Search complexity is O(log n)", "Q3 Binary Search complexity is O(log n)", "Q3"),
        ("Ans I. Polymorphism allows multiple forms", "Ans 1 Polymorphism allows multiple forms", "Ans 1"),
        ("Ans II: Stack uses LIFO", "Ans 2 Stack uses LIFO", "Ans 2"),
        ("Question I. Define inheritance", "Question 1 Define inheritance", "Question 1"),
        ("1. Polymorphism is key in Java", "1. Polymorphism is key in Java", "1."),
    ]

    for raw_input, expected_output, expected_header in cases:
        norm_line, header_tag = processor.normalize_question_header(raw_input)
        assert norm_line == expected_output, f"Failed normalizing '{raw_input}': got '{norm_line}'"
        assert header_tag == expected_header, f"Failed header detection for '{raw_input}': got '{header_tag}'"


def test_noise_token_filtering():
    """Test isolated low-confidence noise tokens are stripped without losing valid words."""
    processor = OCRPostProcessor()

    noisy_line = "Q1 Explain polymorphism ~& in Java ^ } /"
    cleaned = processor.filter_line_noise(noisy_line, confidence=0.20)
    assert cleaned == "Q1 Explain polymorphism in Java"

    code_line = "Time complexity is O(log n) and space is O(1) + 2"
    cleaned_code = processor.filter_line_noise(code_line, confidence=0.85)
    assert "O(log n)" in cleaned_code
    assert "O(1)" in cleaned_code
    assert "+" in cleaned_code


def test_post_process_preserves_raw_text_and_sets_metadata():
    """Test full post_process pipeline saves raw text in metadata and cleans text."""
    processor = OCRPostProcessor()

    raw_extractions = [
        LineExtraction("01", 0.70, [[50, 100], [80, 100], [80, 120], [50, 120]]),
        LineExtraction("Explain", 0.90, [[90, 100], [170, 100], [170, 120], [90, 120]]),
        LineExtraction("polymorphism", 0.85, [[180, 100], [300, 100], [300, 120], [180, 120]]),
        LineExtraction("^", 0.10, [[310, 100], [320, 100], [320, 120], [310, 120]]),
        LineExtraction("G2", 0.65, [[50, 200], [80, 200], [80, 220], [50, 220]]),
        LineExtraction("Stack vs Queue", 0.88, [[90, 200], [250, 200], [250, 220], [90, 220]]),
    ]

    initial_result = HTRResult(
        text="01\nExplain\npolymorphism\n^\nG2\nStack vs Queue",
        confidence=0.68,
        lines=raw_extractions,
        metadata={"provider": "EasyOCR"},
    )

    cleaned_result = processor.post_process(initial_result)

    assert cleaned_result.metadata["post_processed"] is True
    assert cleaned_result.metadata["raw_text"] == "01\nExplain\npolymorphism\n^\nG2\nStack vs Queue"
    assert "Q1 Explain polymorphism" in cleaned_result.text
    assert "Q2 Stack vs Queue" in cleaned_result.text
    assert "^" not in cleaned_result.text
    assert "Q1" in cleaned_result.metadata["detected_headers"]
    assert "Q2" in cleaned_result.metadata["detected_headers"]


def test_post_process_empty_result():
    """Test handling of empty or blank OCR results."""
    processor = OCRPostProcessor()
    empty_res = HTRResult(text="", confidence=0.0, lines=[], is_low_confidence=True)
    out = processor.post_process(empty_res)
    assert out.text == ""
    assert out.is_low_confidence is True


def test_easyocr_recognizer_post_processing_toggle(monkeypatch):
    """Test EasyOCR recognizer honors enable_post_processing flag."""
    rec_with_post = EasyOCRHandwritingRecognizer(enable_post_processing=True)
    rec_without_post = EasyOCRHandwritingRecognizer(enable_post_processing=False)

    fake_results = [
        ([[50, 100], [80, 100], [80, 120], [50, 120]], "QI", 0.85),
        ([[90, 100], [180, 100], [180, 120], [90, 120]], "Polymorphism", 0.90),
    ]

    class FakeReader:
        def readtext(self, *args, **kwargs):
            return fake_results

    monkeypatch.setattr(rec_with_post, "_reader", FakeReader())
    monkeypatch.setattr(rec_without_post, "_reader", FakeReader())

    dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    res_post = rec_with_post.recognize(dummy_img)
    res_no_post = rec_without_post.recognize(dummy_img)

    # Post-processed: grouped into single line and normalized "Q1 Polymorphism"
    assert "Q1 Polymorphism" in res_post.text
    assert res_post.metadata.get("post_processed") is True
    assert res_post.metadata.get("raw_text") == "QI\nPolymorphism"

    # Without post-processing: raw split lines "QI\nPolymorphism"
    assert res_no_post.text == "QI\nPolymorphism"
    assert "post_processed" not in res_no_post.metadata
