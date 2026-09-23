import os
import sys
import time
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processing.handwriting_recognizer import TrOCRHandwritingRecognizer
from app.processing.qa_matcher import HierarchicalQAMatcher, CandidateAnswerSegment

BENCH_DIR = os.path.abspath('scratch/benchmark_htr_output')

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

print("Initializing TrOCRRecognizer...")
recognizer = TrOCRHandwritingRecognizer(fallback_to_easyocr=False)
matcher = HierarchicalQAMatcher()

pages_results = []
doc_pages = []

t_total_start = time.time()

for p_num in [1, 2, 3]:
    img_path = os.path.join(BENCH_DIR, f"c1_proc_p{p_num}.png")
    pil_img = Image.open(img_path)
    print(f"\nProcessing Page {p_num} with TrOCR...")
    t0 = time.time()
    htr_res = recognizer.recognize(pil_img)
    dt = time.time() - t0
    print(f"Page {p_num} complete in {dt:.2f}s | Conf: {htr_res.confidence:.2%} | Lines: {len(htr_res.lines)} | Chars: {len(htr_res.text)}")
    print(f"Text Preview:\n{htr_res.text[:200]}...")
    
    pages_results.append({
        "page_number": p_num,
        "time_seconds": round(dt, 2),
        "confidence": round(htr_res.confidence, 4),
        "char_count": len(htr_res.text),
        "line_count": len(htr_res.lines),
        "text": htr_res.text
    })
    doc_pages.append(MockDocPage(p_num, htr_res.text, htr_res.confidence))

total_time = time.time() - t_total_start
avg_conf = float(np.mean([p["confidence"] for p in pages_results]))

# Downstream Q&A segmentation and matching
candidates = matcher.segment_pages(doc_pages)
matched = matcher.match_candidates_to_questions(candidates, questions)
matched_count = len([m for m in matched if m.question_id is not None])

print("\n" + "="*50)
print(f"TrOCR Benchmark Results:")
print(f"Total Time: {total_time:.2f}s ({total_time/3:.2f}s/page)")
print(f"Avg Confidence: {avg_conf:.2%}")
print(f"Questions Matched: {matched_count}/{len(questions)}")
for m in matched:
    status_label = f"Q{m.question_number}" if m.question_id else "UNMATCHED"
    print(f"  - {status_label} (Conf: {m.match_confidence:.2f}, Method: {m.match_method}): {m.extracted_text[:75]}...")

out_data = {
    "model_name": "TrOCR (microsoft/trocr-base-handwritten)",
    "total_time_seconds": round(total_time, 2),
    "average_confidence": round(avg_conf, 4),
    "questions_matched": f"{matched_count}/{len(questions)}",
    "pages": pages_results,
    "matched_segments": [
        {
            "question_number": m.question_number,
            "question_id": m.question_id,
            "confidence": round(m.match_confidence, 4),
            "method": m.match_method,
            "text": m.extracted_text
        }
        for m in matched
    ]
}

with open(os.path.join(BENCH_DIR, "trocr_benchmark.json"), "w") as f:
    json.dump(out_data, f, indent=2)

print("Saved to trocr_benchmark.json")
