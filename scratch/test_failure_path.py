import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models.submission import Submission, SubmissionStatus
from app.models.evaluation import AnswerEvaluation
from app.processing.evaluator import MockSemanticEvaluator
from app.services.evaluation_service import EvaluationService

app = create_app()
with app.app_context():
    submission_id = 5
    faculty_id = 1
    
    sub = db.session.get(Submission, submission_id)
    print(f"Testing Submission {submission_id} (Initial Status: {sub.status})")

    # Simulate 429 failure using MockSemanticEvaluator
    mock_fail_evaluator = MockSemanticEvaluator(should_fail=True)
    
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=submission_id,
        faculty_id=faculty_id,
        evaluator=mock_fail_evaluator,
    )

    print(f"Evaluation service returned {len(evaluations)} evaluation(s), error: {error}")
    
    # Reload from DB directly to ensure committed state
    db.session.expire_all()
    sub_reloaded = db.session.get(Submission, submission_id)
    print(f"Submission status after failed evaluation: {sub_reloaded.status}")
    print(f"Submission is_fully_evaluated: {sub_reloaded.is_fully_evaluated}")
    print(f"Submission total_ai_score: {sub_reloaded.total_ai_score}")
    print(f"Submission total_final_score: {sub_reloaded.total_final_score}")

    for ev in sub_reloaded.evaluations:
        print(f"  Eval ID {ev.evaluation_id} (Q{ev.question_id}):")
        print(f"    ai_score = {ev.ai_score}")
        print(f"    final_score = {ev.final_score}")
        print(f"    status = {ev.evaluation_status}")
        print(f"    is_failed = {ev.is_failed}")
        print(f"    error_message = {ev.error_message}")
        
        assert ev.ai_score is None, f"ai_score is not None: {ev.ai_score}"
        assert ev.final_score is None, f"final_score is not None: {ev.final_score}"
        assert ev.evaluation_status == "Failed", f"status is not Failed: {ev.evaluation_status}"
        assert ev.is_failed is True
        assert ev.error_message is not None and len(ev.error_message) > 0

    assert sub_reloaded.is_fully_evaluated is False
    print("\nSimulated 429 failure test PASSED completely without exceptions or rollbacks!")
