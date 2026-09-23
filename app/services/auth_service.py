"""
Authentication and session authorization service.
Handles registration, password validation, role checks, and session management.
"""

from functools import wraps
from typing import Optional, Tuple, Union
from flask import session, redirect, url_for, flash, abort, request
from app.extensions import db
from app.models.user import Faculty, Student


class AuthService:
    """Service class for user authentication and authorization."""

    @staticmethod
    def register_faculty(
        name: str, email: str, password: str, department: str
    ) -> Tuple[Optional[Faculty], Optional[str]]:
        """Register a new faculty account."""
        email = email.strip().lower()
        if not name or not email or not password or not department:
            return None, "All fields are required."

        if Faculty.query.filter_by(email=email).first() or Student.query.filter_by(email=email).first():
            return None, "An account with this email address already exists."

        if len(password) < 6:
            return None, "Password must be at least 6 characters long."

        faculty = Faculty(
            name=name.strip(),
            email=email,
            department=department.strip(),
        )
        faculty.set_password(password)

        db.session.add(faculty)
        db.session.commit()
        return faculty, None

    @staticmethod
    def register_student(
        name: str, email: str, password: str, class_name: str
    ) -> Tuple[Optional[Student], Optional[str]]:
        """Register a new student account."""
        email = email.strip().lower()
        if not name or not email or not password or not class_name:
            return None, "All fields are required."

        if Student.query.filter_by(email=email).first() or Faculty.query.filter_by(email=email).first():
            return None, "An account with this email address already exists."

        if len(password) < 6:
            return None, "Password must be at least 6 characters long."

        student = Student(
            name=name.strip(),
            email=email,
            class_name=class_name.strip(),
        )
        student.set_password(password)

        db.session.add(student)
        db.session.commit()
        return student, None

    @staticmethod
    def authenticate_user(
        email: str, password: str, role: str
    ) -> Tuple[Optional[Union[Faculty, Student]], Optional[str]]:
        """Verify user credentials against specified role."""
        email = email.strip().lower()
        if not email or not password:
            return None, "Email and password are required."

        if role == "faculty":
            user = Faculty.query.filter_by(email=email).first()
        elif role == "student":
            user = Student.query.filter_by(email=email).first()
        else:
            return None, "Invalid role specified."

        if not user or not user.check_password(password):
            return None, "Invalid email or password."

        return user, None

    @staticmethod
    def login_user(user: Union[Faculty, Student]) -> None:
        """Store authenticated user state in secure session."""
        session.clear()
        session["user_id"] = user.faculty_id if user.role == "faculty" else user.student_id
        session["role"] = user.role
        session["user_name"] = user.name
        session["user_email"] = user.email

    @staticmethod
    def logout_user() -> None:
        """Clear the current user session."""
        session.clear()

    @staticmethod
    def get_current_user() -> Optional[Union[Faculty, Student]]:
        """Retrieve the currently authenticated user from the database."""
        user_id = session.get("user_id")
        role = session.get("role")

        if not user_id or not role:
            return None

        if role == "faculty":
            return db.session.get(Faculty, user_id)
        elif role == "student":
            return db.session.get(Student, user_id)
        return None


def login_required(f):
    """Decorator ensuring a user is authenticated."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def role_required(required_role: str):
    """Decorator ensuring the authenticated user matches the required role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = session.get("user_id")
            role = session.get("role")

            if not user_id:
                flash("Please sign in to access this page.", "warning")
                return redirect(url_for("auth.login", next=request.path))

            if role != required_role:
                flash(
                    f"Access restricted: This section requires {required_role.capitalize()} privileges.",
                    "danger",
                )
                if role == "faculty":
                    return redirect(url_for("faculty.dashboard"))
                elif role == "student":
                    return redirect(url_for("student.dashboard"))
                return redirect(url_for("auth.login"))

            return f(*args, **kwargs)
        return decorated_function
    return decorator
