"""
Authentication routes for Faculty and Student users.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from app.services.auth_service import AuthService

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login endpoint supporting both Faculty and Student roles."""
    if session.get("user_id"):
        role = session.get("role")
        if role == "faculty":
            return redirect(url_for("faculty.dashboard"))
        elif role == "student":
            return redirect(url_for("student.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "student").strip().lower()
        next_url = request.args.get("next")

        user, error = AuthService.authenticate_user(email, password, role)
        if error:
            flash(error, "danger")
            return render_template("auth/login.html", email=email, role=role)

        AuthService.login_user(user)
        flash(f"Welcome back, {user.name}!", "success")

        if next_url and next_url.startswith("/"):
            return redirect(next_url)

        if role == "faculty":
            return redirect(url_for("faculty.dashboard"))
        return redirect(url_for("student.dashboard"))

    return render_template("auth/login.html", role=request.args.get("role", "student"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration endpoint for Faculty or Student accounts."""
    if session.get("user_id"):
        role = session.get("role")
        if role == "faculty":
            return redirect(url_for("faculty.dashboard"))
        elif role == "student":
            return redirect(url_for("student.dashboard"))

    if request.method == "POST":
        role = request.form.get("role", "student").strip().lower()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template(
                "auth/register.html",
                name=name,
                email=email,
                role=role,
                department=request.form.get("department", ""),
                class_name=request.form.get("class_name", ""),
            )

        if role == "faculty":
            department = request.form.get("department", "").strip()
            user, error = AuthService.register_faculty(name, email, password, department)
        elif role == "student":
            class_name = request.form.get("class_name", "").strip()
            user, error = AuthService.register_student(name, email, password, class_name)
        else:
            flash("Invalid role selected.", "danger")
            return render_template("auth/register.html")

        if error:
            flash(error, "danger")
            return render_template(
                "auth/register.html",
                name=name,
                email=email,
                role=role,
                department=request.form.get("department", ""),
                class_name=request.form.get("class_name", ""),
            )

        flash("Registration successful! Please sign in with your credentials.", "success")
        return redirect(url_for("auth.login", role=role))

    return render_template("auth/register.html", role=request.args.get("role", "student"))


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """Clear session and log out user."""
    AuthService.logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
