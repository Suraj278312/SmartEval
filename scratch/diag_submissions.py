import sys
sys.path.insert(0, ".")

from app import create_app
from app.extensions import db
from app.models.submission import Submission
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.document_page import DocumentPage
from app.models.assignment import Assignment, Question

app = create_app("development")
with app.app_context():
    submissions = Submission.query.all()
    print(f"Total Submissions in DB: {len(submissions)}")
    for s in submissions:
        print(f"\n--- Submission ID: {s.submission_id} | Status: {s.status} | Filename: {s.original_filename} | Student ID: {s.student_id} ---")
        pages = DocumentPage.query.filter_by(submission_id=s.submission_id).all()
        answers = SubmissionAnswer.query.filter_by(submission_id=s.submission_id).all()
        evals = AnswerEvaluation.query.filter_by(submission_id=s.submission_id).all()
        print(f"  Pages count: {len(pages)}")
        for p in pages:
            print(f"    Page {p.page_number}: len={len(p.extracted_text or '')}, status={p.processing_status}, conf={p.confidence}")
        print(f"  Answers count: {len(answers)}")
        for a in answers:
            print(f"    Ans ID {a.answer_id} (Q{a.question_id}): len={len(a.extracted_text or '')}, label={a.detected_label}, match_method={a.match_method}, status={a.status}")
        print(f"  Evaluations count: {len(evals)}")
        for e in evals:
            print(f"    Eval ID {e.evaluation_id} (Q{e.question_id}): ai_score={e.ai_score}/{e.maximum_marks}, status={e.evaluation_status}, error={e.error_message}")
            print(f"      Feedback: {e.feedback}")
            print(f"      Strengths: {e.strengths}")
            print(f"      Missing: {e.missing_elements}")
