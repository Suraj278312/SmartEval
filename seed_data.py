"""
Seed data script for SmartEval.
Populates demo Faculty, Students, multi-question Assignments with supportive answers,
and sample valid PDF submissions for testing.
"""

import os
from datetime import datetime, timedelta, timezone
from app import create_app
from app.extensions import db
from app.models.user import Faculty, Student
from app.models.assignment import Assignment, Question
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_answer import SubmissionAnswer
from app.models.evaluation import AnswerEvaluation
from app.models.document_page import DocumentPage



import pymupdf
from app.processing.pipeline import SubmissionProcessingPipeline


def create_sample_multi_page_pdf(output_path: str) -> int:
    """Generate a clean multi-page academic PDF document and return byte size."""
    doc = pymupdf.open()
    
    # Page 1: Student info & Question 1 Master Theorem
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((50, 60), "STUDENT SUBMISSION RECEIPT", fontsize=11, color=(0.4, 0.4, 0.4))
    page1.insert_text((50, 85), "Ada Lovelace  |  Class: CS-2026-A  |  Roll: CS301-01", fontsize=13, color=(0.1, 0.1, 0.1))
    page1.insert_text((50, 120), "Assignment 1: Divide and Conquer Algorithms", fontsize=15, color=(0, 0.2, 0.6))
    
    page1.insert_text((50, 170), "Question 1: Master Theorem Derivation", fontsize=13, color=(0.1, 0.1, 0.1))
    page1.insert_text((50, 200), "Given recurrence: T(n) = 3T(n/2) + O(n^2)", fontsize=11)
    page1.insert_text((50, 230), "Step 1: Identify parameters: a = 3, b = 2, f(n) = n^2.", fontsize=11)
    page1.insert_text((50, 260), "Step 2: Calculate log_b(a) = log_2(3) = 1.585.", fontsize=11)
    page1.insert_text((50, 290), "Step 3: Compare f(n) with n^(log_b a):", fontsize=11)
    page1.insert_text((70, 315), "Since n^2 = Omega(n^(1.585 + epsilon)) with epsilon approx 0.415,", fontsize=11)
    page1.insert_text((70, 340), "the driver function f(n) grows polynomially faster.", fontsize=11)
    page1.insert_text((50, 375), "Step 4: Regularity condition check:", fontsize=11)
    page1.insert_text((70, 400), "a * f(n/b) = 3 * (n/2)^2 = (3/4) * n^2 <= c * n^2 for c = 3/4 < 1.", fontsize=11)
    page1.insert_text((50, 435), "Conclusion: Case 3 of Master Theorem applies. Hence, T(n) = Theta(n^2).", fontsize=11)
    
    # Page 2: Question 2 QuickSelect & Question 3 Median
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((50, 60), "Ada Lovelace  |  CS301 Assignment 1 (Page 2)", fontsize=11, color=(0.4, 0.4, 0.4))
    page2.insert_text((50, 100), "Question 2: QuickSelect Partitioning Analysis", fontsize=13, color=(0.1, 0.1, 0.1))
    page2.insert_text((50, 130), "QuickSelect uses Lomuto or Hoare partitioning to position a pivot element.", fontsize=11)
    page2.insert_text((50, 160), "If pivot index == k, the k-th smallest element is found.", fontsize=11)
    page2.insert_text((50, 190), "Recurrence: T(n) = T(n/2) + O(n) on average.", fontsize=11)
    page2.insert_text((50, 220), "Average case time complexity: O(n). Worst case complexity: O(n^2).", fontsize=11)
    
    page2.insert_text((50, 275), "Question 3: Median of Two Sorted Arrays in O(log(min(m, n)))", fontsize=13, color=(0.1, 0.1, 0.1))
    page2.insert_text((50, 305), "Perform binary search partition on smaller array A at index i and array B at j.", fontsize=11)
    page2.insert_text((50, 335), "Condition: maxLeftA <= minRightB and maxLeftB <= minRightA.", fontsize=11)
    page2.insert_text((50, 365), "If matched, calculate median from boundary values in O(1) time.", fontsize=11)
    
    doc.save(output_path)
    doc.close()
    return os.path.getsize(output_path)


def seed_database():
    """Populate database with clean, realistic academic test entities."""
    app = create_app("development")

    with app.app_context():
        print("Clearing and rebuilding database schema...")
        db.drop_all()
        db.create_all()

        print("Creating demo Faculty accounts...")
        f1 = Faculty(
            name="Prof. Alan Turing",
            email="faculty@smarteval.edu",
            department="Computer Science & Artificial Intelligence",
        )
        f1.set_password("Faculty@123")

        f2 = Faculty(
            name="Dr. Grace Hopper",
            email="dr.hopper@smarteval.edu",
            department="Software Systems Engineering",
        )
        f2.set_password("Faculty@123")

        db.session.add_all([f1, f2])
        db.session.flush()

        print("Creating demo Student accounts...")
        s1 = Student(
            name="Ada Lovelace",
            email="student1@smarteval.edu",
            class_name="CS-2026-A",
        )
        s1.set_password("Student@123")

        s2 = Student(
            name="Claude Shannon",
            email="student2@smarteval.edu",
            class_name="CS-2026-A",
        )
        s2.set_password("Student@123")

        s3 = Student(
            name="Katherine Johnson",
            email="student3@smarteval.edu",
            class_name="CS-2026-B",
        )
        s3.set_password("Student@123")

        db.session.add_all([s1, s2, s3])
        db.session.flush()

        print("Creating sample Assignments with questions & supportive reference answers...")
        now = datetime.now(timezone.utc)

        # 1. Published Assignment: Divide and Conquer Algorithms
        a1 = Assignment(
            faculty_id=f1.faculty_id,
            title="Assignment 1: Divide and Conquer Algorithms",
            subject="CS301 - Design & Analysis of Algorithms",
            description=(
                "Solve the following analytical problems on divide-and-conquer recurrences. "
                "Write your complete derivation steps clearly on white paper, scan the pages "
                "into a single clear PDF document, and upload before the deadline."
            ),
            deadline=now + timedelta(days=7),
            published=True,
        )
        db.session.add(a1)
        db.session.flush()

        q1_1 = Question(
            assignment_id=a1.assignment_id,
            question_number=1,
            question_text=(
                "Derive the tight asymptotic runtime bound for the recurrence T(n) = 3T(n/2) + O(n^2) "
                "using the Master Theorem. State the values of a, b, f(n), and the applicable case."
            ),
            supportive_answer=(
                "Master Theorem Formulation: a = 3, b = 2, f(n) = n^2.\n"
                "Compute log_b(a) = log_2(3) ≈ 1.585.\n"
                "Compare f(n) with n^(log_b a): Since 2 > 1.585, f(n) = Ω(n^(log_2(3) + ε)) where ε ≈ 0.415.\n"
                "Regularity Condition: a*f(n/b) = 3(n/2)^2 = (3/4)n^2 <= c*f(n) for c = 3/4 < 1.\n"
                "Hence, Case 3 of Master Theorem applies: T(n) = Θ(n^2)."
            ),
            maximum_marks=10.0,
            rubric="3 pts correct parameters (a,b,log_b a), 3 pts case identification, 4 pts regularity condition & final bound",
        )

        q1_2 = Question(
            assignment_id=a1.assignment_id,
            question_number=2,
            question_text=(
                "Explain the partitioning step of the QuickSelect algorithm for finding the k-th smallest element. "
                "What is the average-case and worst-case time complexity?"
            ),
            supportive_answer=(
                "Lomuto or Hoare partitioning selects a pivot element and rearranges elements such that "
                "elements < pivot are on the left, and elements > pivot are on the right.\n"
                "If pivot index equals k, return pivot. If index > k, recurse on left subarray; else recurse on right.\n"
                "Average-case complexity: O(n) due to expected half-reduction.\n"
                "Worst-case complexity: O(n^2) if bad pivots are chosen repeatedly."
            ),
            maximum_marks=10.0,
            rubric="4 pts partition mechanism explanation, 3 pts average complexity derivation, 3 pts worst case scenario",
        )

        q1_3 = Question(
            assignment_id=a1.assignment_id,
            question_number=3,
            question_text=(
                "Given two sorted arrays of size m and n respectively, describe a divide-and-conquer approach "
                "to find the median of the combined array in O(log(min(m, n))) time."
            ),
            supportive_answer=(
                "Binary search partition on smaller array: Partition array A at i and array B at j such that "
                "i + j = (m + n + 1) // 2.\n"
                "Condition: maxLeftA <= minRightB and maxLeftB <= minRightA.\n"
                "If matched: Median is max(maxLeftA, maxLeftB) (if odd) or average of maxLeft and minRight (if even).\n"
                "If maxLeftA > minRightB, move binary search left in A. Else move right."
            ),
            maximum_marks=10.0,
            rubric="4 pts binary search partition formulation, 3 pts boundary condition checks, 3 pts time complexity proof",
        )

        db.session.add_all([q1_1, q1_2, q1_3])

        # 2. Published Assignment: Graph Theory & Shortest Paths
        a2 = Assignment(
            faculty_id=f2.faculty_id,
            title="Problem Set 2: Graph Theory & Network Routing",
            subject="CS402 - Advanced Computer Networks",
            description="Complete the routing algorithm trace and Dijkstra vs Bellman-Ford analysis.",
            deadline=now + timedelta(days=10),
            published=True,
        )
        db.session.add(a2)
        db.session.flush()

        q2_1 = Question(
            assignment_id=a2.assignment_id,
            question_number=1,
            question_text="Compare Dijkstra's and Bellman-Ford algorithms regarding negative edge weights and cycle detection.",
            supportive_answer=(
                "Dijkstra assumes non-negative edge weights because greedy selection does not revisit finalized nodes.\n"
                "Bellman-Ford relaxes all edges |V|-1 times and can handle negative weights.\n"
                "Negative cycle detection: A |V|-th relaxation iteration that improves distance indicates a negative cycle."
            ),
            maximum_marks=15.0,
            rubric="5 pts Dijkstra limitations, 5 pts Bellman-Ford mechanics, 5 pts cycle detection proof",
        )

        q2_2 = Question(
            assignment_id=a2.assignment_id,
            question_number=2,
            question_text="Provide pseudo-code for finding strongly connected components using Tarjan's or Kosaraju's algorithm.",
            supportive_answer=(
                "Kosaraju: 1. Run DFS on G and push vertices to stack in order of completion.\n"
                "2. Transpose graph G -> G_T.\n"
                "3. Pop vertices from stack and run DFS on G_T to identify connected components."
            ),
            maximum_marks=10.0,
            rubric="5 pts algorithm steps, 3 pts transpose logic, 2 pts correctness",
        )

        db.session.add_all([q2_1, q2_2])

        # 3. Draft (Unpublished) Assignment: Midterm Lab Exam
        a3 = Assignment(
            faculty_id=f1.faculty_id,
            title="Draft: Dynamic Programming Comprehensive Exam",
            subject="CS301 - Design & Analysis of Algorithms",
            description="Upcoming midterm exam draft (Unpublished).",
            deadline=now + timedelta(days=14),
            published=False,  # DRAFT
        )
        db.session.add(a3)
        db.session.flush()

        q3_1 = Question(
            assignment_id=a3.assignment_id,
            question_number=1,
            question_text="Formulate the state transition equation for the 0/1 Knapsack Problem with capacity W.",
            supportive_answer="DP[i][w] = max(DP[i-1][w], DP[i-1][w - wt[i]] + val[i]) if wt[i] <= w else DP[i-1][w]",
            maximum_marks=20.0,
            rubric="10 pts recurrence relation, 10 pts base cases",
        )
        db.session.add(q3_1)

        # Create a sample multi-page uploaded submission for student1 on Assignment 1
        upload_folder = app.config["UPLOAD_FOLDER"]
        pages_folder = app.config["PAGES_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(pages_folder, exist_ok=True)

        demo_pdf_filename = f"sub_a{a1.assignment_id}_s{s1.student_id}_demo.pdf"
        demo_pdf_path = os.path.join(upload_folder, demo_pdf_filename)

        file_size = create_sample_multi_page_pdf(demo_pdf_path)

        sub1 = Submission(
            assignment_id=a1.assignment_id,
            student_id=s1.student_id,
            file_path=demo_pdf_path,
            original_filename="Ada_Lovelace_CS301_Assignment1.pdf",
            file_size_bytes=file_size,
            submission_date=now - timedelta(hours=2),
            status=SubmissionStatus.PROCESSING.value,
        )
        db.session.add(sub1)
        db.session.commit()

        print("Executing Phase 2 & 3A Document Processing & Q/A Segmentation Pipeline on demo submission...")
        pipeline = SubmissionProcessingPipeline()
        res = pipeline.process_submission(sub1.submission_id, pages_folder=pages_folder)
        print(f"Pipeline finished: status={res.status}, pages={res.total_pages}, confidence={res.average_confidence}%, matched_answers={res.matched_answers_count}")

        # Ensure realistic extracted text exists on pages for rich demo experience
        from app.models.document_page import DocumentPage
        from app.processing.qa_matcher import HierarchicalQAMatcher
        pages = DocumentPage.query.filter_by(submission_id=sub1.submission_id).order_by(DocumentPage.page_number).all()
        if len(pages) >= 2 and (not pages[0].extracted_text or len(pages[0].extracted_text.strip()) == 0):
            pages[0].extracted_text = (
                "Q1: Master Theorem Derivation\n"
                "Given recurrence: T(n) = 3T(n/2) + O(n^2)\n"
                "Step 1: Parameters are a = 3, b = 2, f(n) = n^2.\n"
                "Step 2: log_b(a) = log_2(3) approx 1.585.\n"
                "Step 3: Since f(n) = n^2 = Omega(n^(1.585 + 0.415)), f(n) grows polynomially faster.\n"
                "Step 4: Regularity check: 3*(n/2)^2 = (3/4)*n^2 <= c*n^2 for c = 3/4 < 1.\n"
                "Conclusion: Case 3 applies, so T(n) = Theta(n^2)."
            )
            pages[0].confidence = 0.94
            pages[1].extracted_text = (
                "Question 2: QuickSelect Partitioning Analysis\n"
                "QuickSelect uses Lomuto or Hoare partition around a pivot element.\n"
                "If pivot index == k, return pivot. Otherwise recurse on the appropriate side.\n"
                "Average time complexity is O(n), worst case complexity is O(n^2) when bad pivots occur.\n\n"
                "Question 3: Median of Two Sorted Arrays\n"
                "Use binary search partition on smaller array A at i and array B at j.\n"
                "Ensure maxLeftA <= minRightB and maxLeftB <= minRightA.\n"
                "Find median from boundary elements in O(log(min(m, n))) time."
            )
            pages[1].confidence = 0.91
            db.session.commit()

            matcher = HierarchicalQAMatcher()
            matched_results = matcher.process_and_match(pages=pages, questions=a1.questions)
            # Remove any empty/unmatched previous answers if any
            SubmissionAnswer.query.filter_by(submission_id=sub1.submission_id).delete()
            for match in matched_results:
                ans_record = SubmissionAnswer(
                    submission_id=sub1.submission_id,
                    question_id=match.question_id,
                    extracted_text=match.extracted_text,
                    detected_label=match.detected_label,
                    page_start=match.page_start,
                    page_end=match.page_end,
                    match_method=match.match_method,
                    match_confidence=match.match_confidence,
                    status=match.status,
                    review_notes=match.review_notes,
                )
                db.session.add(ans_record)
            db.session.commit()


        print("Executing Phase 3B Semantic Answer Evaluation on demo submission...")
        from app.services.evaluation_service import EvaluationService
        from app.services.result_service import ResultService
        from app.processing.evaluator import MockSemanticEvaluator
        eval_engine = MockSemanticEvaluator(percentage=0.90, feedback="Demonstrates clear analytical understanding of asymptotic notation, divide-and-conquer recurrences, and binary search boundary checks.")
        evals, eval_err = EvaluationService.evaluate_submission_answers(
            sub1.submission_id, faculty_id=f1.faculty_id, evaluator=eval_engine
        )
        print(f"Phase 3B evaluation generated for {len(evals)} questions (Total AI Score: {sub1.total_ai_score}/{a1.total_marks})")

        print("Executing Phase 4 Faculty Review & Publication Workflow...")
        # Quick approve all questions and publish result
        EvaluationService.approve_all_evaluations(sub1.submission_id, faculty_id=f1.faculty_id)
        ResultService.update_faculty_summary_feedback(
            submission_id=sub1.submission_id,
            faculty_id=f1.faculty_id,
            summary_feedback="Outstanding work on the algorithm derivations. Clean logical flow and rigorous proofs throughout.",
        )
        pub_res, pub_err = ResultService.publish_result(sub1.submission_id, faculty_id=f1.faculty_id)
        if pub_res:
            print(f"Phase 4 Result Published! Total: {pub_res.total_obtained_marks}/{pub_res.total_maximum_marks} ({pub_res.formatted_percentage}, Grade: {pub_res.grade_letter})")


        print("\n=======================================================")
        print(" SmartEval Demo Database Initialized Successfully!")
        print("=======================================================")

        print("\n[FACULTY TEST ACCOUNTS]")
        print("  Email: faculty@smarteval.edu   | Password: Faculty@123 (Prof. Alan Turing)")
        print("  Email: dr.hopper@smarteval.edu | Password: Faculty@123 (Dr. Grace Hopper)")
        print("\n[STUDENT TEST ACCOUNTS]")
        print("  Email: student1@smarteval.edu  | Password: Student@123 (Ada Lovelace, CS-2026-A)")
        print("  Email: student2@smarteval.edu  | Password: Student@123 (Claude Shannon, CS-2026-A)")
        print("  Email: student3@smarteval.edu  | Password: Student@123 (Katherine Johnson, CS-2026-B)")
        print("\n[SAMPLE ASSIGNMENTS & RESULTS]")
        print(f"  1. CS301 Assignment 1 (Published - 30 Marks, 3 Questions) [1 Submission - Published: 27.0/30.0 (90.00%)]")
        print("  2. CS402 Problem Set 2 (Published - 25 Marks, 2 Questions) [0 Submissions]")
        print("  3. CS301 Midterm Draft (Unpublished Draft - Hidden from Students)")
        print("=======================================================\n")


if __name__ == "__main__":
    seed_database()

