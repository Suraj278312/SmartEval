"""
Security and Role-Based Authorization Tests for SmartEval.
Verifies data isolation, reference answer privacy, and access control.
"""

import io
from app.extensions import db
from app.models.submission import Submission, SubmissionStatus
from app.models.user import Faculty, Student
from tests.conftest import login_session


def test_supportive_answers_never_leaked_to_student_html(client, sample_student, sample_assignment):
    """Ensure faculty reference answers are strictly omitted from student assignment views."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    resp = client.get(f"/student/assignments/{sample_assignment.assignment_id}")
    assert resp.status_code == 200

    # Question text should be visible
    assert b"Derive Master Theorem" in resp.data

    # Supportive/reference answer MUST NOT be in response data
    assert b"Case 2 applies: a=2" not in resp.data
    assert b"Theta(n log n)" not in resp.data
    assert b"Worst case is O(n^2)" not in resp.data


def test_supportive_answers_never_leaked_to_student_api(client, sample_student, sample_assignment):
    """Ensure API endpoint omits supportive_answer when accessed without faculty privileges."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    resp = client.get(f"/api/assignments/{sample_assignment.assignment_id}")
    assert resp.status_code == 200
    json_data = resp.get_json()

    assert "questions" in json_data
    for q in json_data["questions"]:
        assert "supportive_answer" not in q


def test_student_cannot_access_faculty_routes(client, sample_student):
    """Test student is blocked from faculty endpoints."""
    login_session(client, sample_student.student_id, "student", sample_student.name)

    resp = client.get("/faculty/dashboard", follow_redirects=True)
    assert b"Access restricted: This section requires Faculty privileges" in resp.data
    assert b"Student Dashboard" in resp.data

    resp2 = client.get("/faculty/assignments/new", follow_redirects=True)
    assert b"Access restricted: This section requires Faculty privileges" in resp2.data


def test_faculty_cannot_access_student_routes(client, sample_faculty):
    """Test faculty is redirected from student endpoints."""
    login_session(client, sample_faculty.faculty_id, "faculty", sample_faculty.name)

    resp = client.get("/student/dashboard", follow_redirects=True)
    assert b"Access restricted: This section requires Student privileges" in resp.data
    assert b"Faculty Overview" in resp.data


def test_student_isolation_cannot_download_other_student_submission(
    client, app, sample_student, sample_assignment, sample_pdf_bytes
):
    """Test Student A cannot download or view Student B's submission PDF."""
    with app.app_context():
        # Create Student B
        student_b = Student(
            name="Bob Smith",
            email="bob@university.edu",
            class_name="CS-2026-A",
        )
        student_b.set_password("Student@123")
        db.session.add(student_b)
        db.session.commit()

        # Create submission for Student B
        upload_folder = app.config["UPLOAD_FOLDER"]
        sub_b_path = f"{upload_folder}/sub_b.pdf"
        with open(sub_b_path, "wb") as f:
            f.write(sample_pdf_bytes)

        sub_b = Submission(
            assignment_id=sample_assignment.assignment_id,
            student_id=student_b.student_id,
            file_path=sub_b_path,
            original_filename="bob_submission.pdf",
            file_size_bytes=len(sample_pdf_bytes),
            status=SubmissionStatus.UPLOADED.value,
        )
        db.session.add(sub_b)
        db.session.commit()
        sub_b_id = sub_b.submission_id

    # Log in as Student A (Ada Lovelace) and try to download Student B's submission
    login_session(client, sample_student.student_id, "student", sample_student.name)
    resp = client.get(f"/student/submissions/{sub_b_id}/download", follow_redirects=True)

    assert b"Unauthorized: You may only access your own submissions" in resp.data


def test_unauthenticated_requests_redirect_to_login(client):
    """Test unauthenticated users cannot access dashboard."""
    resp = client.get("/faculty/dashboard", follow_redirects=True)
    assert b"Please sign in to access this page" in resp.data

    resp_student = client.get("/student/dashboard", follow_redirects=True)
    assert b"Please sign in to access this page" in resp_student.data
