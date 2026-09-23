import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.models.submission import Submission
from app.models.submission_answer import SubmissionAnswer
from app.models.assignment import Question
from app.processing.evaluator import GeminiSemanticEvaluator

app = create_app("development")
with app.app_context():
    sub = Submission.query.get(5)
    ans = SubmissionAnswer.query.filter_by(submission_id=5, question_id=7).first()
    q = Question.query.get(7)

    print(f"Question: {q.question_text}")
    print(f"Student Answer ({len(ans.extracted_text)} chars): {ans.extracted_text}")
    print(f"Rubric: {q.rubric}")
    print(f"Max Marks: {q.maximum_marks}")

    evaluator = GeminiSemanticEvaluator()
    print(f"Model: {evaluator.model_name}")

    results = []
    for i in range(3):
        print(f"\n--- RUN {i+1} ---")
        try:
            res = evaluator.evaluate_answer(
                question_text=q.question_text,
                supportive_answer=q.supportive_answer or "",
                student_answer=ans.extracted_text,
                maximum_marks=float(q.maximum_marks),
                rubric=q.rubric,
                match_confidence=ans.match_confidence or 1.0,
                htr_confidence=0.85
            )
            print(f"Run {i+1} score: {res.score}/{res.max_score}")
            print(f"Confidence: {res.confidence}")
            print(f"Feedback: {res.feedback}")
            print(f"Criteria: {[(c.name, c.awarded_marks) for c in res.criteria]}")
            results.append(res)
        except Exception as e:
            print(f"Run {i+1} error: {e}")
            results.append(None)
