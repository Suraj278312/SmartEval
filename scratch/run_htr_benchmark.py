import os
import sys
import time
import json
import numpy as np
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processing.pdf_processor import PDFProcessor
from app.processing.image_preprocessor import ImagePreprocessor
from app.processing.handwriting_recognizer import EasyOCRHandwritingRecognizer, TrOCRHandwritingRecognizer
from app.processing.ocr_postprocessor import OCRPostProcessor
from app.processing.qa_matcher import HierarchicalQAMatcher, CandidateAnswerSegment

PDF_PATH = os.path.abspath('uploads/submissions/sub_a4_s1_8a5c2be2300b.pdf')
BENCH_DIR = os.path.abspath('scratch/benchmark_htr_output')
os.makedirs(BENCH_DIR, exist_ok=True)

print(f"==================================================")
print(f"HTR BENCHMARK EXPERIMENT")
print(f"Input PDF: {PDF_PATH}")
print(f"Output Directory: {BENCH_DIR}")
print(f"==================================================")

# 1. Render PDF to images using production PDFProcessor
pdf_proc = PDFProcessor(target_dpi=200)
t0 = time.time()
extracted_pages, render_err = pdf_proc.validate_and_extract_pages(PDF_PATH, BENCH_DIR, submission_id=999)
render_time = time.time() - t0
print(f"PDF Rendered: {len(extracted_pages)} pages in {render_time:.2f}s (Error: {render_err})")

# Questions for Assignment 4
questions = [
    {
        "question_id": 7,
        "question_number": 1,
        "question_text": "Explain polymorphism in Java.",
        "supportive_answer": "Polymorphism is an OOP concept in which one interface or method can have different forms. In Java, compile-time polymorphism is achieved through method overloading and runtime polymorphism through method overriding.",
        "keywords": ["polymorphism", "overloading", "overriding", "compile-time", "runtime", "interface", "method", "java", "forms"]
    },
    {
        "question_id": 8,
        "question_number": 2,
        "question_text": "Explain the difference between a stack and a queue. Give one practical example of each.",
        "supportive_answer": "A stack follows LIFO, while a queue follows FIFO. A stack can be used for function-call management or undo operations, while a queue can be used for printer jobs or scheduling.",
        "keywords": ["stack", "queue", "lifo", "fifo", "push", "pop", "undo", "printer", "linear", "data structure"]
    },
    {
        "question_id": 9,
        "question_number": 3,
        "question_text": "Explain binary search and its time complexity.",
        "supportive_answer": "Binary search works on a sorted collection by checking the middle element and repeatedly reducing the search interval by half. Its time complexity is O(log n).",
        "keywords": ["binary search", "sorted", "middle", "half", "time complexity", "O(log n)", "divide", "conquer"]
    }
]

class MockDocPage:
    def __init__(self, page_num, text, conf):
        self.page_number = page_num
        self.extracted_text = text
        self.confidence = conf

matcher = HierarchicalQAMatcher()
preprocessor = ImagePreprocessor()

# =============================================================================
# Candidate 1: Current Production Pipeline (EasyOCR + Preprocessing + Postprocessing)
# =============================================================================
print("\n>>> CANDIDATE 1: Current Production Pipeline (EasyOCR + Preproc + Postproc)")
recognizer_c1 = EasyOCRHandwritingRecognizer(enable_post_processing=True)
t_c1_start = time.time()
c1_pages = []
c1_doc_pages = []

for page_data in extracted_pages:
    p_num = page_data.page_number
    t_p0 = time.time()
    # 1. Preprocessing
    proc_img_path = os.path.join(BENCH_DIR, f"c1_proc_p{p_num}.png")
    proc_pil, _ = preprocessor.preprocess_image(page_data.image, output_filepath=proc_img_path)
    # 2. HTR with post-processing enabled
    htr_res = recognizer_c1.recognize(proc_pil)
    p_time = time.time() - t_p0
    
    c1_pages.append({
        "page_number": p_num,
        "time_seconds": round(p_time, 2),
        "confidence": round(htr_res.confidence, 4),
        "char_count": len(htr_res.text),
        "line_count": len(htr_res.lines),
        "text": htr_res.text
    })
    c1_doc_pages.append(MockDocPage(p_num, htr_res.text, htr_res.confidence))

c1_total_time = time.time() - t_c1_start
c1_avg_conf = float(np.mean([p["confidence"] for p in c1_pages]))
c1_candidates = matcher.segment_pages(c1_doc_pages)
c1_matched = matcher.match_candidates_to_questions(c1_candidates, questions)
c1_matched_count = len([m for m in c1_matched if m.question_id is not None])

print(f"  • Total Time: {c1_total_time:.2f}s ({c1_total_time/len(extracted_pages):.2f}s/page)")
print(f"  • Average Confidence: {c1_avg_conf:.2%}")
print(f"  • Questions Matched: {c1_matched_count}/{len(questions)}")
for m in c1_matched:
    status_label = f"Q{m.question_number}" if m.question_id else "UNMATCHED"
    print(f"    - {status_label} (Conf: {m.match_confidence:.2f}, Method: {m.match_method}): {m.extracted_text[:75]}...")

# =============================================================================
# Candidate 2: Raw EasyOCR (Ablation: Preprocessing + Raw EasyOCR without Postprocessor)
# =============================================================================
print("\n>>> CANDIDATE 2: Raw EasyOCR without Postprocessing (Ablation)")
recognizer_c2 = EasyOCRHandwritingRecognizer(enable_post_processing=False)
t_c2_start = time.time()
c2_pages = []
c2_doc_pages = []

for page_data in extracted_pages:
    p_num = page_data.page_number
    t_p0 = time.time()
    proc_img_path = os.path.join(BENCH_DIR, f"c1_proc_p{p_num}.png")
    proc_pil = Image.open(proc_img_path) if os.path.exists(proc_img_path) else page_data.image
    htr_res = recognizer_c2.recognize(proc_pil)
    p_time = time.time() - t_p0
    c2_pages.append({
        "page_number": p_num,
        "time_seconds": round(p_time, 2),
        "confidence": round(htr_res.confidence, 4),
        "char_count": len(htr_res.text),
        "text": htr_res.text
    })
    c2_doc_pages.append(MockDocPage(p_num, htr_res.text, htr_res.confidence))

c2_total_time = time.time() - t_c2_start
c2_avg_conf = float(np.mean([p["confidence"] for p in c2_pages]))
c2_candidates = matcher.segment_pages(c2_doc_pages)
c2_matched = matcher.match_candidates_to_questions(c2_candidates, questions)
c2_matched_count = len([m for m in c2_matched if m.question_id is not None])

print(f"  • Total Time: {c2_total_time:.2f}s ({c2_total_time/len(extracted_pages):.2f}s/page)")
print(f"  • Average Confidence: {c2_avg_conf:.2%}")
print(f"  • Questions Matched: {c2_matched_count}/{len(questions)}")
for m in c2_matched:
    status_label = f"Q{m.question_number}" if m.question_id else "UNMATCHED"
    print(f"    - {status_label} (Conf: {m.match_confidence:.2f}, Method: {m.match_method}): {m.extracted_text[:75]}...")

# =============================================================================
# Candidate 3: TrOCR (Transformer VisionEncoderDecoder)
# =============================================================================
print("\n>>> CANDIDATE 3: TrOCR (microsoft/trocr-base-handwritten)")
recognizer_c3 = TrOCRHandwritingRecognizer(fallback_to_easyocr=False)
c3_available = recognizer_c3.is_available
c3_pages = []
c3_total_time = 0.0
c3_avg_conf = 0.0
c3_matched_count = 0
c3_matched = []

if c3_available:
    t_c3_start = time.time()
    c3_doc_pages = []
    for page_data in extracted_pages:
        p_num = page_data.page_number
        t_p0 = time.time()
        proc_img_path = os.path.join(BENCH_DIR, f"c1_proc_p{p_num}.png")
        proc_pil = Image.open(proc_img_path) if os.path.exists(proc_img_path) else page_data.image
        htr_res = recognizer_c3.recognize(proc_pil)
        p_time = time.time() - t_p0
        c3_pages.append({
            "page_number": p_num,
            "time_seconds": round(p_time, 2),
            "confidence": round(htr_res.confidence, 4),
            "char_count": len(htr_res.text),
            "text": htr_res.text
        })
        c3_doc_pages.append(MockDocPage(p_num, htr_res.text, htr_res.confidence))
    c3_total_time = time.time() - t_c3_start
    c3_avg_conf = float(np.mean([p["confidence"] for p in c3_pages]))
    c3_candidates = matcher.segment_pages(c3_doc_pages)
    c3_matched = matcher.match_candidates_to_questions(c3_candidates, questions)
    c3_matched_count = len([m for m in c3_matched if m.question_id is not None])
    print(f"  • Total Time: {c3_total_time:.2f}s ({c3_total_time/len(extracted_pages):.2f}s/page)")
    print(f"  • Average Confidence: {c3_avg_conf:.2%}")
    print(f"  • Questions Matched: {c3_matched_count}/{len(questions)}")
    for m in c3_matched:
        status_label = f"Q{m.question_number}" if m.question_id else "UNMATCHED"
        print(f"    - {status_label} (Conf: {m.match_confidence:.2f}, Method: {m.match_method}): {m.extracted_text[:75]}...")
else:
    print("  • TrOCR weights not active or locally cached. Model is disabled in current runtime.")

# =============================================================================
# Semantic / Keyword Compensation Analysis
# =============================================================================
print("\n>>> RESEARCH EVALUATION: Keyword / Reference-Answer Semantic Compensation")
keyword_analysis = []
for q in questions:
    q_num = q["question_number"]
    matched = next((m for m in c1_matched if m.question_number == q_num), None)
    text = matched.extracted_text if matched else ""
    found = [kw for kw in q["keywords"] if kw.lower() in text.lower()]
    missing = [kw for kw in q["keywords"] if kw.lower() not in text.lower()]
    coverage = len(found) / len(q["keywords"])
    
    # Calculate character error patterns
    keyword_analysis.append({
        "question_number": q_num,
        "question_text": q["question_text"],
        "extracted_chars": len(text),
        "total_keywords": len(q["keywords"]),
        "found_keywords": found,
        "missing_keywords": missing,
        "coverage": round(coverage, 3),
        "extracted_preview": text[:150]
    })
    print(f"  • Question {q_num} ('{q['question_text'][:35]}...'):")
    print(f"    - Keyword Coverage: {coverage:.1%} ({len(found)}/{len(q['keywords'])})")
    print(f"    - Present Terms: {', '.join(found)}")
    print(f"    - Absent Terms: {', '.join(missing) if missing else 'None'}")

# Qualitative Error Analysis
print("\n>>> QUALITATIVE TRANSCRIPTION ERROR OBSERVATIONS")
for p in c1_pages:
    print(f"\n--- Page {p['page_number']} Text (Confidence: {p['confidence']:.1%}) ---")
    print(p["text"])

# Save detailed JSON artifact
report_data = {
    "benchmark_pdf": PDF_PATH,
    "pages_count": len(extracted_pages),
    "candidate_1_production_easyocr": {
        "model_name": "EasyOCR + Preprocessing + Postprocessing",
        "total_time_seconds": round(c1_total_time, 2),
        "average_confidence": round(c1_avg_conf, 4),
        "questions_matched": f"{c1_matched_count}/{len(questions)}",
        "per_page": c1_pages,
    },
    "candidate_2_raw_easyocr": {
        "model_name": "Raw EasyOCR (No Postprocessing)",
        "total_time_seconds": round(c2_total_time, 2),
        "average_confidence": round(c2_avg_conf, 4),
        "questions_matched": f"{c2_matched_count}/{len(questions)}",
        "per_page": c2_pages,
    },
    "candidate_3_trocr": {
        "model_name": "TrOCR (microsoft/trocr-base-handwritten)",
        "available": c3_available,
        "total_time_seconds": round(c3_total_time, 2) if c3_available else None,
        "average_confidence": round(c3_avg_conf, 4) if c3_available else None,
        "questions_matched": f"{c3_matched_count}/{len(questions)}" if c3_available else "N/A",
    },
    "keyword_compensation_analysis": keyword_analysis
}

with open(os.path.join(BENCH_DIR, "detailed_htr_benchmark.json"), "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\n==================================================")
print(f"Benchmark run complete! Results saved to detailed_htr_benchmark.json")
print(f"==================================================")
