"""
Handwriting-Capable Text Recognition (HTR) Module for SmartEval.
Provides a pluggable recognizer interface supporting EasyOCR and TrOCR (Transformer OCR).
Features:
- EasyOCR fallback implementation (PyTorch CRNN/ResNet)
- TrOCR implementation (microsoft/trocr-base-handwritten VisionEncoderDecoder)
- Lazy model loading on first inference request
- Automatic device selection (CUDA / CPU)
- Line-level generation score confidence estimation
- Resilient fallback mechanism
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from app.processing.image_preprocessor import ImagePreprocessor, LineRegion

logger = logging.getLogger(__name__)


@dataclass
class LineExtraction:
    """Detailed transcription and confidence for an individual line or text block."""
    text: str
    confidence: float
    bounding_box: Optional[List[List[int]]] = None


@dataclass
class HTRResult:
    """Comprehensive handwriting recognition result for a page image."""
    text: str
    confidence: float  # 0.0 to 1.0
    lines: List[LineExtraction] = field(default_factory=list)
    is_low_confidence: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class HTRRecognizerInterface(ABC):
    """Abstract Base Class for handwriting recognition providers."""

    @abstractmethod
    def recognize(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> HTRResult:
        """
        Recognize handwritten text from an image.
        :param image_input: File path, numpy array, or PIL Image.
        :return: HTRResult with text, confidence, and line details.
        """
        pass


class EasyOCRHandwritingRecognizer(HTRRecognizerInterface):
    """
    Handwriting recognition implementation using EasyOCR (PyTorch CRNN/ResNet).
    Supports GPU acceleration if available, with CPU fallback.
    Applies OCRPostProcessor for spatial line reconstruction and question header recovery.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        low_confidence_threshold: float = 0.55,
        enable_post_processing: bool = True,
    ):
        self.languages = languages or ["en"]
        self.low_confidence_threshold = low_confidence_threshold
        self.enable_post_processing = enable_post_processing
        self._reader = None
        self._post_processor = None

    @property
    def reader(self):
        """Lazy loader for EasyOCR reader model."""
        if self._reader is None:
            import easyocr
            use_gpu = torch.cuda.is_available()
            logger.info("Initializing EasyOCR reader (gpu=%s)...", use_gpu)
            self._reader = easyocr.Reader(self.languages, gpu=use_gpu)
        return self._reader

    @property
    def post_processor(self):
        """Lazy loader for OCRPostProcessor."""
        if self._post_processor is None:
            from app.processing.ocr_postprocessor import OCRPostProcessor
            self._post_processor = OCRPostProcessor()
        return self._post_processor

    def recognize(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> HTRResult:
        """Execute handwriting recognition on the provided image."""
        try:
            # Convert input to numpy array if PIL Image or string path
            if isinstance(image_input, Image.Image):
                img_array = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, str):
                pil_img = Image.open(image_input).convert("RGB")
                img_array = np.array(pil_img)
            else:
                img_array = image_input

            # Run EasyOCR readtext: returns list of (bbox, text, confidence)
            results = self.reader.readtext(img_array, paragraph=False)

            if not results:
                return HTRResult(
                    text="",
                    confidence=0.0,
                    lines=[],
                    is_low_confidence=True,
                    metadata={"provider": "EasyOCR", "total_segments": 0},
                    error_message=None,
                )

            extracted_lines: List[LineExtraction] = []
            confidences: List[float] = []
            text_parts: List[str] = []

            for bbox, text, conf in results:
                text_clean = text.strip()
                if text_clean:
                    conf_val = float(conf)
                    confidences.append(conf_val)
                    text_parts.append(text_clean)
                    extracted_lines.append(
                        LineExtraction(
                            text=text_clean,
                            confidence=conf_val,
                            bounding_box=(
                                [[int(pt[0]), int(pt[1])] for pt in bbox]
                                if bbox
                                else None
                            ),
                        )
                    )

            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            full_text = "\n".join(text_parts)
            is_low_conf = (avg_confidence < self.low_confidence_threshold) or (
                len(text_parts) == 0
            )

            raw_result = HTRResult(
                text=full_text,
                confidence=avg_confidence,
                lines=extracted_lines,
                is_low_confidence=is_low_conf,
                metadata={
                    "provider": "EasyOCR",
                    "total_segments": len(extracted_lines),
                    "min_confidence": (
                        float(np.min(confidences)) if confidences else 0.0
                    ),
                    "max_confidence": (
                        float(np.max(confidences)) if confidences else 0.0
                    ),
                    "gpu_used": torch.cuda.is_available(),
                },
            )

            if self.enable_post_processing:
                return self.post_processor.post_process(raw_result)
            return raw_result

        except Exception as e:
            logger.exception("EasyOCR recognition error: %s", str(e))
            return HTRResult(
                text="",
                confidence=0.0,
                lines=[],
                is_low_confidence=True,
                error_message=f"HTR recognition error: {str(e)}",
                metadata={"provider": "EasyOCR", "failed": True},
            )


class TrOCRHandwritingRecognizer(HTRRecognizerInterface):
    """
    Advanced Transformer-based handwriting recognition using Microsoft's TrOCR.
    Processes segmented line crops sequentially or in batches, estimating token-probability confidence.
    Supports lazy loading and graceful fallback to EasyOCR upon initialization or runtime failure.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        low_confidence_threshold: float = 0.55,
        fallback_to_easyocr: bool = True,
        batch_size: int = 4,
        preprocessor: Optional[ImagePreprocessor] = None,
        easyocr_fallback: Optional[EasyOCRHandwritingRecognizer] = None,
    ):
        self.model_name = model_name or os.environ.get(
            "HTR_TROCR_MODEL", "microsoft/trocr-base-handwritten"
        )
        # Device selection: explicit override or auto-detect CUDA -> CPU
        if device:
            self.device = device
        else:
            env_dev = os.environ.get("HTR_DEVICE")
            if env_dev:
                self.device = env_dev
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.low_confidence_threshold = low_confidence_threshold
        self.fallback_to_easyocr = fallback_to_easyocr
        self.batch_size = max(1, batch_size)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._easyocr_fallback = easyocr_fallback

        # Lazy-loaded attributes
        self._processor = None
        self._model = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

    @property
    def easyocr_fallback(self) -> EasyOCRHandwritingRecognizer:
        """Lazy loader for EasyOCR fallback instance."""
        if self._easyocr_fallback is None:
            self._easyocr_fallback = EasyOCRHandwritingRecognizer(
                low_confidence_threshold=self.low_confidence_threshold
            )
        return self._easyocr_fallback

    def _load_model(self) -> None:
        """
        Lazily load the TrOCR processor and model on first recognition request.
        Safely captures missing dependencies, download errors, and OOM.
        """
        if self._is_loaded:
            return

        if self._load_error:
            raise RuntimeError(
                f"TrOCR initialization previously failed: {self._load_error}"
            )

        try:
            logger.info(
                "Loading TrOCR model '%s' onto device '%s'...",
                self.model_name,
                self.device,
            )
            from transformers.models.trocr import TrOCRProcessor
            from transformers.models.roberta import RobertaTokenizer
            from transformers.models.vit import ViTImageProcessor
            from transformers.models.vision_encoder_decoder import (
                VisionEncoderDecoderModel,
            )

            try:
                # Try local cache first for fast offline loading
                tokenizer = RobertaTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                image_processor = ViTImageProcessor.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self._processor = TrOCRProcessor(
                    image_processor=image_processor, tokenizer=tokenizer
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception:
                # Fall back to online fetch if not in local cache
                tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
                image_processor = ViTImageProcessor.from_pretrained(self.model_name)
                self._processor = TrOCRProcessor(
                    image_processor=image_processor, tokenizer=tokenizer
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
            logger.info("TrOCR model '%s' successfully initialized.", self.model_name)

        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load TrOCR model '%s': %s", self.model_name, str(e))
            raise RuntimeError(f"Failed to load TrOCR model: {str(e)}") from e

    def _recognize_line_crops(
        self, line_crops: List[Image.Image]
    ) -> List[Tuple[str, float]]:
        """
        Run TrOCR inference on a list of PIL line crop images with token-probability confidence.
        Returns list of (recognized_text, line_confidence).
        """
        if not line_crops:
            return []

        self._load_model()
        results: List[Tuple[str, float]] = []

        for i in range(0, len(line_crops), self.batch_size):
            batch_imgs = line_crops[i : i + self.batch_size]

            # Process images into model input pixel tensors
            inputs = self._processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            with torch.inference_mode():
                outputs = self._model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128,
                )

            decoded_texts = self._processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )

            # Calculate token-probability confidence for each line in batch
            for b in range(len(batch_imgs)):
                text = decoded_texts[b].strip()
                token_probs = []

                if outputs.scores and len(outputs.scores) > 0:
                    for step_idx, step_logits in enumerate(outputs.scores):
                        if step_idx + 1 < outputs.sequences.shape[1]:
                            token_id = outputs.sequences[b, step_idx + 1].item()
                            # Stop calculating on EOS or PAD
                            if hasattr(self._model.config, "eos_token_id") and token_id == self._model.config.eos_token_id:
                                break
                            if hasattr(self._model.config, "pad_token_id") and token_id == self._model.config.pad_token_id:
                                break

                            step_softmax = torch.softmax(step_logits[b], dim=-1)
                            prob = float(step_softmax[token_id].item())
                            token_probs.append(prob)

                if token_probs:
                    line_conf = float(np.mean(token_probs))
                else:
                    line_conf = 0.5 if text else 0.0

                results.append((text, line_conf))

        return results

    def recognize(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> HTRResult:
        """
        Execute handwriting recognition on a page image using line segmentation and TrOCR.
        Falls back to EasyOCR if TrOCR fails and fallback is enabled.
        """
        # Convert input to PIL Image and OpenCV numpy array
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            cv_img = cv2_from_pil(pil_img)
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
            cv_img = cv2_from_pil(pil_img)
        else:
            cv_img = image_input
            pil_img = Image.fromarray(cv_img).convert("RGB")

        # 1. Extract line regions from the page
        line_regions = self.preprocessor.extract_line_regions(cv_img)

        # 2. Filter valid line regions (ignore noise or zero-ink bands)
        valid_regions: List[LineRegion] = []
        for r in line_regions:
            x, y, w, h = r.bbox
            if w >= 20 and h >= 8:
                valid_regions.append(r)

        if not valid_regions:
            logger.info("TrOCR: No text line regions detected on page.")
            return HTRResult(
                text="",
                confidence=0.0,
                lines=[],
                is_low_confidence=True,
                metadata={
                    "provider": "TrOCR",
                    "model": self.model_name,
                    "device": self.device,
                    "total_lines": 0,
                },
                error_message=None,
            )

        # 3. Crop line images with safe padding
        line_crops: List[Image.Image] = []
        h_page, w_page = cv_img.shape[:2]

        for r in valid_regions:
            x, y, w, h = r.bbox
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_page, x + w + pad)
            y2 = min(h_page, y + h + pad)

            crop_arr = cv_img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop_arr).convert("RGB")
            line_crops.append(crop_pil)

        # 4. Attempt TrOCR recognition with fallback handling
        try:
            logger.info(
                "Using HTR engine: TrOCR | Model: %s | Device: %s | Lines detected: %d",
                self.model_name,
                self.device,
                len(line_crops),
            )

            rec_outputs = self._recognize_line_crops(line_crops)

            extracted_lines: List[LineExtraction] = []
            line_texts: List[str] = []
            confidences: List[float] = []

            for idx, (text, conf) in enumerate(rec_outputs):
                if text:  # Ignore completely blank lines
                    r = valid_regions[idx]
                    x, y, w, h = r.bbox
                    bbox_poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                    extracted_lines.append(
                        LineExtraction(
                            text=text,
                            confidence=conf,
                            bounding_box=bbox_poly,
                        )
                    )
                    line_texts.append(text)
                    confidences.append(conf)

            full_text = "\n".join(line_texts)
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            is_low_conf = (avg_confidence < self.low_confidence_threshold) or (
                len(line_texts) == 0
            )

            logger.info(
                "TrOCR completed: %d lines recognized | Page confidence: %.2f%%",
                len(extracted_lines),
                avg_confidence * 100,
            )

            return HTRResult(
                text=full_text,
                confidence=avg_confidence,
                lines=extracted_lines,
                is_low_confidence=is_low_conf,
                metadata={
                    "provider": "TrOCR",
                    "model": self.model_name,
                    "device": self.device,
                    "total_lines_detected": len(valid_regions),
                    "total_lines_recognized": len(extracted_lines),
                    "min_confidence": (
                        float(np.min(confidences)) if confidences else 0.0
                    ),
                    "max_confidence": (
                        float(np.max(confidences)) if confidences else 0.0
                    ),
                },
            )

        except Exception as e:
            logger.warning(
                "TrOCR recognition failed: %s. Fallback enabled: %s",
                str(e),
                self.fallback_to_easyocr,
            )

            if self.fallback_to_easyocr:
                logger.info("Executing EasyOCR fallback...")
                fallback_result = self.easyocr_fallback.recognize(image_input)
                fallback_result.metadata["fallback_from"] = "TrOCR"
                fallback_result.metadata["primary_error"] = str(e)
                return fallback_result

            return HTRResult(
                text="",
                confidence=0.0,
                lines=[],
                is_low_confidence=True,
                error_message=f"TrOCR recognition error: {str(e)}",
                metadata={"provider": "TrOCR", "failed": True},
            )


def cv2_from_pil(pil_img: Image.Image) -> np.ndarray:
    """Helper to convert PIL Image to OpenCV numpy array."""
    import cv2
    arr = np.array(pil_img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


# Default recognizer singleton
_default_recognizer: Optional[HTRRecognizerInterface] = None


def get_default_recognizer() -> HTRRecognizerInterface:
    """
    Retrieve or initialize the system's active HTR recognizer based on HTR_ENGINE configuration.
    Defaults to EasyOCR for safe baseline operation.
    """
    global _default_recognizer
    if _default_recognizer is None:
        engine_type = os.environ.get("HTR_ENGINE", "easyocr").strip().lower()
        if engine_type == "trocr":
            logger.info("Initializing default HTR engine: TrOCR")
            _default_recognizer = TrOCRHandwritingRecognizer()
        else:
            logger.info("Initializing default HTR engine: EasyOCR")
            _default_recognizer = EasyOCRHandwritingRecognizer()
    return _default_recognizer


def set_default_recognizer(recognizer: Optional[HTRRecognizerInterface]) -> None:
    """Set or reset the global default recognizer singleton (useful for testing)."""
    global _default_recognizer
    _default_recognizer = recognizer
