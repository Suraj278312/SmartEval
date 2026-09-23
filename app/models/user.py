"""
User models: Faculty and Student entities for SmartEval.
Includes secure password hashing, verification, and role discrimination.
"""

from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db


class Faculty(db.Model):
    """Faculty entity representing academic instructors/graders."""
    __tablename__ = "faculty"

    faculty_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    assignments = db.relationship(
        "Assignment",
        backref="faculty",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="desc(Assignment.created_at)"
    )

    def set_password(self, password: str) -> None:
        """Hash and securely store faculty password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify faculty password against stored hash."""
        return check_password_hash(self.password_hash, password)

    @property
    def role(self) -> str:
        return "faculty"

    def to_dict(self) -> dict:
        return {
            "faculty_id": self.faculty_id,
            "name": self.name,
            "email": self.email,
            "department": self.department,
            "role": "faculty",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<Faculty {self.faculty_id}: {self.email}>"


class Student(db.Model):
    """Student entity representing assignment submitters."""
    __tablename__ = "student"

    student_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    class_name = db.Column("class", db.String(50), nullable=False)  # mapped to DB column 'class'
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    submissions = db.relationship(
        "Submission",
        backref="student",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="desc(Submission.submission_date)"
    )

    def set_password(self, password: str) -> None:
        """Hash and securely store student password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify student password against stored hash."""
        return check_password_hash(self.password_hash, password)

    @property
    def role(self) -> str:
        return "student"

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "class": self.class_name,
            "role": "student",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<Student {self.student_id}: {self.email}>"
