"""
Application configuration for SmartEval.
Supports Development, Testing, and Production environments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "smarteval-dev-insecure-secret-key-998822")
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'smarteval.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads & Document Processing Storage
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", str(BASE_DIR / "uploads" / "submissions")
    )
    PAGES_FOLDER = os.environ.get(
        "PAGES_FOLDER", str(BASE_DIR / "uploads" / "pages")
    )
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB limit
    MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 50))  # Max pages per submission
    ALLOWED_EXTENSIONS = {"pdf"}
    
    # AI Semantic Evaluation (Google Gemini)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    # Handwriting Text Recognition (HTR)
    HTR_ENGINE = os.environ.get("HTR_ENGINE", "easyocr")  # "easyocr" or "trocr"
    HTR_TROCR_MODEL = os.environ.get("HTR_TROCR_MODEL", "microsoft/trocr-base-handwritten")
    HTR_DEVICE = os.environ.get("HTR_DEVICE", None)  # None for auto-detect (CUDA if available, else CPU), or "cuda", "cpu"
    HTR_FALLBACK_TO_EASYOCR = os.environ.get("HTR_FALLBACK_TO_EASYOCR", "true").lower() in ("true", "1", "yes")
    HTR_LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("HTR_LOW_CONFIDENCE_THRESHOLD", 0.55))

    # Session
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    UPLOAD_FOLDER = str(BASE_DIR / "tests" / "test_uploads" / "submissions")
    PAGES_FOLDER = str(BASE_DIR / "tests" / "test_uploads" / "pages")
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
