import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models.submission import Submission
from app.models.evaluation import AnswerEvaluation
from app.processing.evaluator import MockSemanticEvaluator
from app.services.evaluation_service import EvaluationService

app = create_app()
with app.app_context():
    submission_id = 5
    faculty_id = 1
    
    # Simulate legitimate academic zero outcome
    mock_zero_evaluator = MockSemanticEvaluator(score=0.0, outcome="none", percentage=0.0)
    
    evaluations, error = EvaluationService.evaluate_submission_answers(
        submission_id=submission_id,
        faculty_id=faculty_id,
        evaluator=mock_zero_evaluator,
    )

    print(f"Evaluation service returned {len(evaluations)} evaluation(s), error: {error}")
    
    # Reload from DB directly to ensure committed state
    db.session.expire_all()
    sub_reloaded = db.session.get(Submission, submission_id)
    print(f"Submission status after legitimate zero evaluation: {sub_reloaded.status}")
    print(f"Submission is_fully_evaluated: {sub_reloaded.is_fully_evaluated}")

    for ev in sub_reloaded.evaluations:
        print(f"  Eval ID {ev.evaluation_id} (Q{ev.question_id}):")
        print(f"    ai_score = {ev.ai_score}")
        print(f"    final_score = {ev.final_score}")
        print(f"    status = {ev.evaluation_status}")
        print(f"    is_failed = {ev.is_failed}")
        print(f"    error_message = {ev.error_message}")
        
        assert ev.ai_score == 0.0, f"ai_score is not 0.0: {ev.ai_score}"
        assert ev.final_score == 0.0, f"final_score is not 0.0: {ev.final_score}"
        assert ev.evaluation_status == "Evaluated", f"status is not Evaluated: {ev.evaluation_status}"
        assert ev.is_failed is False
        assert ev.error_message is None

    assert sub_reloaded.is_fully_evaluated is True
    print("\nLegitimate academic zero test PASSED completely!")
