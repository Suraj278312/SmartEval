"""
Pytest fixtures and test helpers for SmartEval test suite.
"""

import io
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
import pytest

from app import create_app
from app.extensions import db as _db
from app.models.user import Faculty, Student
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus


@pytest.fixture(scope="session")
def app_instance():
    """Create test application configured for testing."""
    test_upload_dir = tempfile.mkdtemp(prefix="smarteval_test_uploads_")
    test_pages_dir = os.path.join(test_upload_dir, "pages")
    os.makedirs(test_pages_dir, exist_ok=True)
    
    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = test_upload_dir
    app.config["PAGES_FOLDER"] = test_pages_dir
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key-12345"

    yield app

    # Clean up temp folder
    shutil.rmtree(test_upload_dir, ignore_errors=True)


@pytest.fixture
def app(app_instance):
    """Provide app with an active application context for each test."""
    with app_instance.app_context():
        _db.create_all()
        yield app_instance
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app):
    """Test HTTP client."""
    return app.test_client()


@pytest.fixture
def sample_faculty(app):
    """Create and persist a sample faculty account."""
    f = Faculty(
        name="Prof. Alan Turing",
        email="faculty@smarteval.edu",
        department="Computer Science",
    )
    f.set_password("Faculty@123")
    _db.session.add(f)
    _db.session.commit()
    # Refresh to ensure attributes remain loaded
    _db.session.refresh(f)
    return f


@pytest.fixture
def sample_student(app):
    """Create and persist a sample student account."""
    s = Student(
        name="Ada Lovelace",
        email="student1@smarteval.edu",
        class_name="CS-2026-A",
    )
    s.set_password("Student@123")
    _db.session.add(s)
    _db.session.commit()
    # Refresh to ensure attributes remain loaded
    _db.session.refresh(s)
    return s


@pytest.fixture
def sample_assignment(app, sample_faculty):
    """Create a sample published assignment with 2 questions and supportive answers."""
    now = datetime.now(timezone.utc)
    a = Assignment(
        faculty_id=sample_faculty.faculty_id,
        title="Algorithms Midterm Problem Set",
        subject="CS301",
        description="Complete the derivations.",
        deadline=now + timedelta(days=5),
        published=True,
    )
    _db.session.add(a)
    _db.session.flush()

    q1 = Question(
        assignment_id=a.assignment_id,
        question_number=1,
        question_text="Derive Master Theorem for T(n) = 2T(n/2) + O(n).",
        supportive_answer="Case 2 applies: a=2, b=2, log_2(2)=1, f(n)=O(n). Result: T(n)=Theta(n log n).",
        maximum_marks=10.0,
        rubric="5 pts formulation, 5 pts final bound",
    )

    q2 = Question(
        assignment_id=a.assignment_id,
        question_number=2,
        question_text="Describe QuickSort worst case.",
        supportive_answer="Worst case is O(n^2) when array is sorted and first element chosen as pivot.",
        maximum_marks=15.0,
        rubric="8 pts pivot explanation, 7 pts recursion depth",
    )

    _db.session.add_all([q1, q2])
    _db.session.commit()
    _db.session.refresh(a)
    return a


@pytest.fixture
def sample_pdf_bytes():
    """Generate minimal valid PDF byte stream."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 80), "Test Submission", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def login_session(client, user_id: int, role: str, name: str = "Test User"):
    """Helper to inject authenticated user session directly."""
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["role"] = role
        sess["user_name"] = name
        sess["user_email"] = f"{role}@test.edu"
