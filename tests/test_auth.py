"""
Unit and integration tests for Authentication & User Registration.
"""

from app.models.user import Faculty, Student
from app.services.auth_service import AuthService
from tests.conftest import login_session


def test_faculty_registration_and_password_hashing(app):
    """Test faculty registration stores hashed password rather than plaintext."""
    with app.app_context():
        faculty, error = AuthService.register_faculty(
            name="Dr. Hopper",
            email="hopper@navy.mil",
            password="SecurePassword@123",
            department="Computer Engineering",
        )
        assert error is None
        assert faculty is not None
        assert faculty.email == "hopper@navy.mil"
        assert faculty.password_hash != "SecurePassword@123"
        assert faculty.check_password("SecurePassword@123") is True
        assert faculty.check_password("WrongPassword") is False
        assert faculty.role == "faculty"


def test_student_registration_and_attributes(app):
    """Test student registration and class attribute mapping."""
    with app.app_context():
        student, error = AuthService.register_student(
            name="Claude Shannon",
            email="shannon@mit.edu",
            password="InfoTheory@123",
            class_name="CS-2026-A",
        )
        assert error is None
        assert student is not None
        assert student.class_name == "CS-2026-A"
        assert student.role == "student"
        assert student.check_password("InfoTheory@123") is True


def test_duplicate_email_registration_rejected(app, sample_faculty):
    """Test duplicate email registration is rejected."""
    with app.app_context():
        # Attempt to register student with existing faculty email
        student, error = AuthService.register_student(
            name="Imposter",
            email="faculty@smarteval.edu",
            password="Password@123",
            class_name="CS-2026",
        )
        assert student is None
        assert "already exists" in error


def test_auth_login_endpoint(client, sample_faculty, sample_student):
    """Test login form submissions for both roles."""
    # 1. Faculty login success
    resp = client.post(
        "/auth/login",
        data={
            "email": "faculty@smarteval.edu",
            "password": "Faculty@123",
            "role": "faculty",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Faculty Overview" in resp.data

    # 2. Invalid password rejected
    client.get("/auth/logout")
    resp_bad = client.post(
        "/auth/login",
        data={
            "email": "faculty@smarteval.edu",
            "password": "WrongPassword",
            "role": "faculty",
        },
        follow_redirects=True,
    )
    assert b"Invalid email or password" in resp_bad.data

    # 3. Student login success
    resp_student = client.post(
        "/auth/login",
        data={
            "email": "student1@smarteval.edu",
            "password": "Student@123",
            "role": "student",
        },
        follow_redirects=True,
    )
    assert resp_student.status_code == 200
    assert b"Student Dashboard" in resp_student.data


def test_logout(client, sample_student):
    """Test session clearance on logout."""
    login_session(client, sample_student.student_id, "student", sample_student.name)
    resp = client.get("/auth/logout", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Sign In to SmartEval" in resp.data
