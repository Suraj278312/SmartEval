"""
SmartEval Flask Application Factory.
"""

import os
from datetime import datetime, timezone
from flask import Flask, redirect, url_for, session, render_template
from app.config import config_by_name
from app.extensions import db
from app.models.submission import SubmissionStatus
from app.services.auth_service import AuthService


def create_app(config_name: str = "default") -> Flask:
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration
    config_obj = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_obj)

    # Ensure required runtime directories exist
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config.get("PAGES_FOLDER", os.path.join(app.config["UPLOAD_FOLDER"], "..", "pages")), exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.blueprints.auth import auth_bp
    from app.blueprints.faculty import faculty_bp
    from app.blueprints.student import student_bp
    from app.blueprints.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(faculty_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(api_bp)

    # Root route redirector
    @app.route("/")
    def index():
        role = session.get("role")
        if role == "faculty":
            return redirect(url_for("faculty.dashboard"))
        elif role == "student":
            return redirect(url_for("student.dashboard"))
        return redirect(url_for("auth.login"))

    # Global context processors
    @app.context_processor
    def inject_global_context():
        user = AuthService.get_current_user()
        return {
            "current_user": user,
            "current_role": session.get("role"),
            "current_year": datetime.now(timezone.utc).year,
            "SubmissionStatus": SubmissionStatus,
        }

    # Error handlers
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def file_too_large_error(error):
        return render_template(
            "errors/error.html",
            error_title="File Exceeds Size Limit",
            error_message="The uploaded document exceeds the allowable upload limit (16 MB). Please compress the file and retry.",
            status_code=413
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template("errors/500.html"), 500

    # Ensure tables exist on startup for development
    with app.app_context():
        db.create_all()

    return app
