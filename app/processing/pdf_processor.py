"""
PDF Processing Module for SmartEval Phase 2.
Renders PDF pages into high-resolution images with robust error handling.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import pypdfium2 as pdfium
import pymupdf


@dataclass
class ExtractedPageData:
    """Represents a page rendered from an uploaded PDF document."""
    page_number: int
    original_image_path: str
    image: Image.Image
    width: int
    height: int


class PDFProcessorError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """
    Renders PDF documents to high-resolution page images.
    Supports pypdfium2 with PyMuPDF fallback.
    """

    def __init__(self, target_dpi: int = 200, max_pages: int = 50):
        self.target_dpi = target_dpi
        self.max_pages = max_pages

    def validate_and_extract_pages(
        self,
        pdf_path: str,
        output_folder: str,
        submission_id: int
    ) -> Tuple[List[ExtractedPageData], Optional[str]]:
        """
        Open and render each page of the PDF to a PNG image.
        Returns (list_of_pages, error_message).
        """
        if not os.path.exists(pdf_path):
            return [], f"PDF file not found at path: {pdf_path}"

        if os.path.getsize(pdf_path) == 0:
            return [], "The uploaded PDF file is empty (0 bytes)."

        os.makedirs(output_folder, exist_ok=True)
        pages_data: List[ExtractedPageData] = []

        # 1. Attempt rendering with pypdfium2
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)

            if total_pages == 0:
                return [], "The PDF document contains 0 pages."

            if total_pages > self.max_pages:
                return [], f"PDF page count ({total_pages}) exceeds the allowable limit of {self.max_pages} pages."

            # Calculate render scale for target DPI (72 standard points/inch)
            render_scale = self.target_dpi / 72.0

            for i in range(total_pages):
                page_num = i + 1
                page = pdf[i]
                
                # Render to PIL Image
                pil_image = page.render(
                    scale=render_scale,
                    rotation=0,
                ).to_pil()

                filename = f"orig_sub_{submission_id}_p{page_num}.png"
                filepath = os.path.join(output_folder, filename)
                pil_image.save(filepath, format="PNG")

                pages_data.append(
                    ExtractedPageData(
                        page_number=page_num,
                        original_image_path=filepath,
                        image=pil_image,
                        width=pil_image.width,
                        height=pil_image.height,
                    )
                )

            pdf.close()
            return pages_data, None

        except Exception as e_pdfium:
            # Fallback to PyMuPDF (fitz)
            try:
                doc = pymupdf.open(pdf_path)
                if doc.is_encrypted:
                    doc.close()
                    return [], "The uploaded PDF is password-protected or encrypted."

                total_pages = len(doc)
                if total_pages == 0:
                    doc.close()
                    return [], "The PDF document contains 0 pages."

                if total_pages > self.max_pages:
                    doc.close()
                    return [], f"PDF page count ({total_pages}) exceeds limit of {self.max_pages}."

                zoom = self.target_dpi / 72.0
                mat = pymupdf.Matrix(zoom, zoom)

                for i in range(total_pages):
                    page_num = i + 1
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=mat)
                    
                    filename = f"orig_sub_{submission_id}_p{page_num}.png"
                    filepath = os.path.join(output_folder, filename)
                    pix.save(filepath)

                    pil_img = Image.open(filepath)
                    pages_data.append(
                        ExtractedPageData(
                            page_number=page_num,
                            original_image_path=filepath,
                            image=pil_img,
                            width=pil_img.width,
                            height=pil_img.height,
                        )
                    )

                doc.close()
                return pages_data, None

            except Exception as e_mupdf:
                return [], f"Failed to parse and render PDF: {str(e_pdfium)} (Fallback error: {str(e_mupdf)})"
