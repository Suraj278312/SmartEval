"""
DocumentPage model for SmartEval.
Stores per-page image artifacts, extracted handwritten text, and confidence scores.
"""

from datetime import datetime, timezone
from app.extensions import db


class DocumentPage(db.Model):
    """Represents a single processed page of a student submission."""
    __tablename__ = "document_page"

    page_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    submission_id = db.Column(
        db.Integer,
        db.ForeignKey("submission.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    page_number = db.Column(db.Integer, nullable=False, default=1)
    original_image_path = db.Column(db.String(255), nullable=True)
    processed_image_path = db.Column(db.String(255), nullable=True)
    extracted_text = db.Column(db.Text, nullable=True)
    confidence = db.Column(db.Float, nullable=True)  # 0.0 to 1.0 (or 0.0 to 100.0)
    processing_status = db.Column(
        db.String(50), default="Completed", nullable=False, index=True
    )  # 'Pending', 'Completed', 'Review Required', 'Failed'
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    @property
    def confidence_percentage(self) -> float:
        """Return confidence as a percentage (0.0 to 100.0)."""
        if self.confidence is None:
            return 0.0
        if self.confidence <= 1.0:
            return round(self.confidence * 100, 1)
        return round(self.confidence, 1)

    @property
    def is_low_confidence(self) -> bool:
        """Flag pages with confidence below 55% as low confidence."""
        if self.confidence is None:
            return True
        val = self.confidence if self.confidence > 1.0 else self.confidence * 100
        return val < 55.0

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "submission_id": self.submission_id,
            "page_number": self.page_number,
            "extracted_text": self.extracted_text,
            "confidence": self.confidence,
            "confidence_percentage": self.confidence_percentage,
            "is_low_confidence": self.is_low_confidence,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<DocumentPage {self.page_id}: Sub {self.submission_id}, "
            f"Page #{self.page_number}, Conf={self.confidence_percentage}%>"
        )
