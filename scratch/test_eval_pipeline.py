import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from app.processing.evaluator import GeminiSemanticEvaluator

evaluator = GeminiSemanticEvaluator()
print(f"Evaluator model: {evaluator.model_name}")
print(f"API key loaded: {evaluator.api_key[:6]}...{evaluator.api_key[-4:] if evaluator.api_key else 'None'}")

question_text = "Explain the concept of Overfitting in Machine Learning and mention two ways to prevent it."
supportive_answer = "Overfitting occurs when a model learns the training data too well, capturing noise instead of general patterns. As a result, it performs well on training data but poorly on unseen test data. Prevention methods: 1. Regularization (L1/L2), 2. Cross-validation, 3. Dropout, 4. Early stopping."
student_answer = "Overfitting is when the model memorizes the training data including noise and does not generalize to new test data. It can be prevented using regularization techniques like L1 or L2, and by using cross-validation."
rubric = "4 pts Concept Definition; 2 pts Consequences; 4 pts Prevention Methods"
max_marks = 10.0

print("\nRunning evaluation with Gemini...")
result = evaluator.evaluate_answer(
    question_text=question_text,
    supportive_answer=supportive_answer,
    student_answer=student_answer,
    maximum_marks=max_marks,
    rubric=rubric
)

print(f"\nEvaluation Result:")
print(f"Score: {result.score}/{result.max_score}")
print(f"Confidence: {result.confidence}")
print(f"Feedback: {result.feedback}")
print(f"Strengths: {result.strengths}")
print(f"Missing elements: {result.missing_elements}")
print(f"Criteria Breakdown:")
for c in result.criteria:
    print(f" - {c.name}: {c.awarded_marks}/{c.max_marks} ({c.status}) -> {c.reason}")
