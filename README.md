# SmartEval – AI-Based Handwritten Assignment Evaluation System
## Final Production Release (Phase 5: Full Validation & Presentation Ready)

SmartEval is an end-to-end academic evaluation platform designed to manage coursework assignments, process handwritten student PDF submissions, extract recognized text with confidence scoring, segment/map student answers to assignment questions, perform preliminary **semantic conceptual evaluation** with rubric-based scoring, and enable instructor **grade moderation, final score calculation, result publication, grounded feedback synthesis, and formal printable grade reports**.

---

## 1. End-to-End System Architecture & Workflow

```
Student Uploads Handwritten PDF
            │
            ▼
┌───────────────────────────────┐
│     Multi-Layer Validator     │ (Magic bytes %PDF-, Size limit, Extension)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     PDF Page Renderer         │ (pypdfium2 / PyMuPDF: 200-300 DPI Rendering)
└──────────────┬────────────────┘
               │ Outputs ordered PNGs (uploads/pages/orig_sub_{id}_p{n}.png)
               ▼
┌───────────────────────────────┐
│     Image Preprocessor        │ (4-Way Orientation Correction [0/90/180/270°],
└──────────────┬────────────────┘  Ruled-Line Suppression, CLAHE, Line Segmentation)
               │ Outputs preprocessed PNGs & segmented line crops
               ▼
┌───────────────────────────────┐
│  HTR Engine (TrOCR / EasyOCR) │ (TrOCR: microsoft/trocr-base-handwritten / EasyOCR CRNN)
└──────────────┬────────────────┘  Line-level generation confidence & multi-line reconstruction
               │ Outputs DocumentPage records
               │ Outputs DocumentPage records
               ▼
┌───────────────────────────────┐
│   Q&A Segmentation Engine     │ (Detects boundaries: Q1, 1., Ans 1, Problem 1;
└──────────────┬────────────────┘  Tracks multi-page answer continuity)
               │
               ▼
┌───────────────────────────────┐
│ Hierarchical Question Matcher │ Priority 1: Explicit numbering (0.90–0.98)
└──────────────┬────────────────┘ Priority 2: Textual lexical overlap (0.75–0.90)
               │                  Priority 3: Semantic keyword similarity (0.55–0.85)
               │                  Priority 4: Unmatched orphan fallback
               ▼
┌───────────────────────────────┐
│  AI Semantic Evaluator (3B)   │ (Google Gemini GenAI SDK + Structured JSON Schemas)
└──────────────┬────────────────┘  - Conceptual correctness & partial credit rubric criteria
               │                  - Key strengths & missing concept identification
               │                  - AI evaluations remain preliminary / unconfirmed
               ▼
┌───────────────────────────────┐
│   Faculty Review Interface    │ (Faculty moderation and approval)
└──────────────┬────────────────┘  - Quick Approve AI Score / Manual Override Score & Feedback
               │                  - Flag for Manual Review / Re-evaluate individual question
               │                  - Clearly distinguishes AI Score from Final Faculty Score
               ▼
┌───────────────────────────────┐
│ Final Score Calculation Layer │ - Total Maximum Marks = sum(question max marks)
└──────────────┬────────────────┘ - Total Obtained Marks = sum(final approved scores)
               │                  - Percentage = (total obtained / total max) * 100
               │                  - Grounded Overall Feedback (Strengths, Improvements, Summary)
               ▼
┌───────────────────────────────┐
│  Publication & Gradebook (4)  │ - Faculty publishes verified result to student
└──────────────┬────────────────┘ - Results Dashboard with filters, search, and bulk actions
               │
               ├───────────────────────────────────────────────┐
               ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│   Student Result Scorecard    │               │  Formal Printable Report      │
│  - Total Marks & Percentage   │               │  - Official Academic Layout   │
│  - Question-wise Final Marks  │               │  - Institution Header & Meta  │
│  - Grounded Overall Feedback  │               │  - Question Score Table       │
│  - Zero Reference Answer Leak │               │  - Print & PDF-Ready Styling  │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 2. Project Folder Structure

```
SmartEval/
├── app/
│   ├── __init__.py                 # Application factory, blueprint registration, context processors
│   ├── config.py                   # Environment configuration (Dev, Test, Prod with GEMINI_API_KEY)
│   ├── extensions.py               # SQLAlchemy database extension
│   ├── models/
│   │   ├── __init__.py             # Model exports
│   │   ├── user.py                 # Faculty & Student models with secure password hashing
│   │   ├── assignment.py           # Assignment & Question models with privacy filtering
│   │   ├── submission.py           # Submission model with lifecycle states and relationships
│   │   ├── document_page.py        # [PHASE 2] DocumentPage model (page_num, paths, text, confidence)
│   │   ├── submission_answer.py    # [PHASE 3A] SubmissionAnswer model (question_id, text, pages, method)
│   │   ├── evaluation.py           # [PHASE 3B] AnswerEvaluation model (ai_score, criteria, strengths, feedback)
│   │   └── submission_result.py    # [PHASE 4] SubmissionResult model (totals, percentage, publication, feedback)
│   ├── services/
│   │   ├── __init__.py             # Service exports
│   │   ├── auth_service.py         # Auth, session management, RBAC decorators
│   │   ├── assignment_service.py   # Assignment CRUD, question builder, publish toggle
│   │   ├── submission_service.py   # PDF upload, pipeline triggering, retry, secure image streaming
│   │   ├── evaluation_service.py   # [PHASE 3B] AI evaluation orchestration, faculty approval & override
│   │   └── result_service.py       # [PHASE 4] Score calculation, publication, grounded feedback, reports
│   ├── processing/                 # Dedicated HTR & Evaluation Pipeline
│   │   ├── __init__.py             # Processing module exports
│   │   ├── pdf_processor.py        # Multi-page PDF renderer (pypdfium2 / PyMuPDF fallback)
│   │   ├── image_preprocessor.py   # OpenCV pipeline (grayscale, CLAHE, deskew, crop, normalize)
│   │   ├── handwriting_recognizer.py# HTRRecognizerInterface + EasyOCR PyTorch engine
│   │   ├── qa_matcher.py           # [PHASE 3A] HierarchicalQAMatcher & answer segmenter
│   │   └── evaluator.py            # [PHASE 3B] RubricEngine, GeminiSemanticEvaluator, MockSemanticEvaluator
│   ├── blueprints/
│   │   ├── auth/                   # Registration, login, logout
│   │   ├── faculty/                # Dashboard, Assignments, Submissions, Results & Gradebook, Reports
│   │   ├── student/                # Dashboard, Published assignments, Question viewer, Upload, Scorecards
│   │   └── api/                    # Health check & sanitized endpoints
│   ├── static/
│   │   ├── css/
│   │   │   └── main.css            # Custom CSS design system tokens and @media print stylesheet
│   │   └── js/
│   │       ├── app.js              # UI alerts, clipboard copy
│   │       ├── assignment_editor.js# Dynamic question builder with rubrics
│   │       └── submission_uploader.js # PDF dropzone uploader
│   └── templates/
│       ├── base.html               # Master layout with responsive navbar & flash container
│       ├── faculty/
│       │   ├── dashboard.html      # Metrics overview, active assignments
│       │   ├── assignment_list.html# Assignment registry with status toggles
│       │   ├── assignment_form.html# Question builder with reference answers & rubrics
│       │   ├── submissions.html    # Submissions table with score overview
│       │   ├── review_submission.html # AI evaluation breakdown, faculty override & publication
│       │   ├── results_dashboard.html # [PHASE 4] Results dashboard with filtering, search, sorting
│       │   └── submission_report.html # [PHASE 4] Formal printable academic grade report
│       └── student/
│           ├── dashboard.html      # Enrolled coursework, submission receipts, and published results
│           ├── assignment_list.html# Published assignments browse view
│           ├── assignment_view.html# Question view (NO reference answers) & upload dropzone
│           ├── submission_view.html# Submission status receipt & lifecycle stepper
│           └── result_view.html    # [PHASE 4] Student published result scorecard & feedback
├── instance/
│   └── smarteval.db                # SQLite database
├── uploads/
│   ├── submissions/                # Stored original student PDF submissions
│   └── pages/                      # Rendered original & preprocessed PNG page images
├── tests/
│   ├── conftest.py                 # Pytest fixtures for in-memory DB and test sessions
│   ├── test_auth.py                # Authentication, password hashing, RBAC tests
│   ├── test_assignments.py         # CRUD, question builder, publish toggle tests
│   ├── test_security.py            # Reference answer privacy, isolation tests
│   ├── test_submissions.py         # File validation, upload, resubmission tests
│   ├── test_processing.py          # PDF rendering, preprocessing, HTR, confidence tests
│   ├── test_qa_matching.py         # Boundary detection, segmentation, hierarchical matching tests
│   ├── test_evaluation.py          # Semantic evaluation, rubric parsing, scoring bounds tests
│   ├── test_phase4_results.py      # [PHASE 4] Approval, overrides, calculation, publication, reports
│   └── test_phase5_validation.py   # [PHASE 5] Multi-scenario AI grading, bounds safety, image isolation
├── seed_data.py                    # Populates demo accounts, multi-page PDFs, HTR, evaluations & results
├── run.py                          # Development server entry point
├── requirements.txt                # Full Python dependencies
├── .env.example                    # Environment template with GEMINI_API_KEY & GEMINI_MODEL
└── README.md
```

---

## 3. Technology Stack & Key Libraries

- **Web Framework**: Flask 3.1.2 with Jinja2 Templating
- **ORM & Database**: SQLAlchemy 2.0.52 & Flask-SQLAlchemy 3.1.1 (SQLite in-memory for testing, file-backed for development)
- **PDF Extraction**: `pypdfium2` (primary) and `PyMuPDF` / `pymupdf` (fallback)
- **Computer Vision & Preprocessing**: `OpenCV` (`cv2`), `Pillow` (`PIL`), and `NumPy` (4-way orientation detector, ruled-line suppressor, and line segmenter)
- **Handwriting Recognition (HTR)**:
  - **TrOCR (Primary / Next-Gen)**: Hugging Face `transformers` ViT + RoBERTa model (`microsoft/trocr-base-handwritten`) with PyTorch `torch.inference_mode()`, line-region batching, and generation score confidence.
  - **EasyOCR (Fallback / Baseline)**: PyTorch CRNN/ResNet recognition with CRAFT detection.
- **Semantic Text Matching**: `scikit-learn` (TF-IDF vectorization and cosine similarity scoring)
- **AI Semantic Evaluator**: Google GenAI SDK (`google-genai`) with Gemini Flash (`gemini-2.5-flash`)
- **Automated Testing**: `pytest` 9.1.1 and `pytest-mock`

---

## 4. Real Runtime Components vs Test Mocks

| Component | Test Suite (`tests/`) | Production / Live Runtime (`app/`) |
| :--- | :--- | :--- |
| **PDF Rendering** | Real `pypdfium2` / `PyMuPDF` rendering synthetic in-memory & file PDFs | Real `pypdfium2` / `PyMuPDF` rendering high-DPI page images |
| **Image Preprocessor**| Real OpenCV pipeline (4-way orientation, ruled-line suppression, CLAHE, deskew, line crop) | Real OpenCV pipeline with PNG caching in `uploads/pages/` |
| **HTR Engine** | `MockHTRRecognizer` / Mocked `TrOCRHandwritingRecognizer` (fast deterministic execution without downloading model weights) | Real `TrOCRHandwritingRecognizer` or `EasyOCRHandwritingRecognizer` with lazy model loading and automatic CUDA/CPU device selection |
| **Q&A Matching** | Real `HierarchicalQAMatcher` with regex header detection & TF-IDF similarity | Real `HierarchicalQAMatcher` mapping segmented student answers to questions |
| **AI Evaluation** | `MockSemanticEvaluator` (deterministic score presets, error simulations, edge cases) | `GeminiSemanticEvaluator` (`google-genai` calling Gemini Flash with structured Pydantic schemas) |
| **Result Calculation**| Real `ResultService` (strict mathematical summation, percentage, letter grade, grounded feedback) | Real `ResultService` calculating verified results upon faculty moderation |

---

## 5. Capabilities & Known Limitations

### A. Handwriting Recognition (HTR) & Preprocessing Pipeline
- **Step 1 Preprocessing**:
  - **4-Way Automatic Orientation Detection**: Tests 0°, 90°, 180°, and 270° rotations against horizontal stroke variance and morphological line structure, automatically uprighting upside-down or sideways pages before recognition.
  - **Ruled-Line Suppression**: Uses dual-scale morphological horizontal kernels to detect and subtract notebook ruled lines while preserving intersecting handwriting strokes.
  - **Handwritten Line Segmentation**: Extracts individual handwritten text lines with safe aspect-ratio padding for line-by-line recognizer consumption.
- **Step 2 Recognition Engines (TrOCR & EasyOCR)**:
  - **TrOCR Engine (`microsoft/trocr-base-handwritten`)**: Vision-Encoder-Decoder model designed specifically for handwritten text. Receives preprocessed line crops, generates text tokens, and derives token-probability generation confidence.
  - **Lazy Model Loading**: Model weights and processors are NOT loaded during application boot; they load on-demand upon the first recognition request.
  - **Safe Fallback**: If TrOCR fails to initialize (e.g. out-of-memory or missing weights) and `HTR_FALLBACK_TO_EASYOCR=True`, the pipeline seamlessly falls back to EasyOCR without crashing.
  - **Device Selection**: Automatically selects CUDA GPU if available; defaults gracefully to CPU. Can be overridden with `HTR_DEVICE`.
- **Limitations**:
  - First-time TrOCR model download requires ~1.27 GB (`pytorch_model.bin`) cached to `~/.cache/huggingface/hub/`.
  - CPU execution for full-page line batches takes ~30–60s per page on CPU vs 2–5s on modern CUDA GPUs.

### B. AI Semantic Evaluation (Gemini)
- **Strengths**:
  - **Conceptual Correctness**: Understands student reasoning and awards full credit for correct paraphrased answers without penalizing alternative phrasing.
  - **Partial Marks**: Maps understanding to individual rubric criteria with detailed justifications.
  - **Constructive Feedback**: Identifies both accurate concepts demonstrated (`strengths`) and missing concepts (`missing_elements`).
  - **Scoring Safety**: Server-side clamping guarantees scores remain strictly within $0.0 \le \text{score} \le \text{question.maximum\_marks}$.
- **Limitations**:
  - Requires a valid `GEMINI_API_KEY` in `.env` for live evaluation.
  - When `GEMINI_API_KEY` is not set or network errors occur, the system gracefully flags the submission for manual human review without crashing or corrupting data.

### C. Faculty Review & Human-in-the-Loop Policy
- AI evaluations are strictly preliminary recommendations. Final student grades and reports are only published after explicit instructor approval or score override.
- Original AI suggestions and modified faculty scores are preserved in distinct database fields for complete auditability.

---

## 6. Security & Isolation Architecture

1. **Role-Based Access Control (RBAC)**:
   - `@role_required("faculty")` and `@role_required("student")` protect all administrative and submission routes.
2. **Student Submission & Result Isolation**:
   - Students can only view their own submissions and published results. Accessing another student's submission or result yields an immediate unauthorized redirect.
   - Unpublished results and draft assignments are strictly hidden from students until published by the instructor.
3. **Reference Answer & AI Prompt Privacy**:
   - Faculty supportive answers (`supportive_answer`) and raw AI system prompts are filtered out at the service layer and are never transmitted to student views or JSON payloads.
4. **File Streaming Security**:
   - Rendered page images (`/faculty/submissions/<id>/pages/<n>/image`) verify user session authorization before streaming file bytes.

---

## 7. How to Run Locally

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Configure Environment (.env)
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

**Key Configuration Options**:
```ini
# --- HTR Engine Configuration ---
# Options: 'easyocr' (default safe baseline) or 'trocr' (vision-encoder-decoder)
HTR_ENGINE=easyocr
HTR_TROCR_MODEL=microsoft/trocr-base-handwritten
HTR_DEVICE=auto                    # 'auto', 'cuda', or 'cpu'
HTR_FALLBACK_TO_EASYOCR=True       # Fall back to EasyOCR if TrOCR initialization fails
HTR_LOW_CONFIDENCE_THRESHOLD=0.40  # Flag for review below this confidence

# --- AI Evaluation (Gemini) ---
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### Step 3: Seed Demo Data
```powershell
python seed_data.py
```

### Step 4: Launch Web Server
```powershell
python run.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 8. Automated Test Suite & Benchmarks

### Running Automated Unit Tests
Run the complete unit test suite (102 passing tests, isolated with mock HTR):
```powershell
python -m pytest tests/ -v
```

### Running the Manual HTR Benchmark (Real PDF & Real Models)
To benchmark EasyOCR vs TrOCR on actual student handwritten PDFs without modifying production records:
```powershell
python scratch/benchmark_htr.py
```

### Test Suite Summary:
- **Total Tests**: **102 passing tests** (0 failed, 0 warnings)
- **Module Coverage**:
  - `tests/test_auth.py`: Faculty/student registration, password hashing, login/logout, duplicate email rejection (5 tests).
  - `tests/test_assignments.py`: Multi-question creation, editing, cascading deletion, publish toggling, draft protection (5 tests).
  - `tests/test_submissions.py`: PDF magic byte validation, file size limits, upload storage, replacement handling (4 tests).
  - `tests/test_processing.py`: Multi-page PDF extraction, corrupt file rejection, OpenCV preprocessing, HTR confidence, page streaming (10 tests).
  - `tests/test_trocr_recognizer.py`: TrOCR initialization, lazy loading, device selection, line ordering, token confidence, review state, and EasyOCR fallback (10 tests).
  - `tests/test_qa_matching.py`: Question header regex parsing, multi-page answer continuation, hierarchical semantic matching, orphan fallback (10 tests).
  - `tests/test_evaluation.py`: Rubric parsing engine, Gemini response schema, score clamping, missing key handling, re-evaluation (16 tests).
  - `tests/test_phase4_results.py`: Faculty review, score modification, aggregate calculation, publication lifecycle, student scorecard, printable report (16 tests).
  - `tests/test_phase5_validation.py`: Correct/partial/incorrect/paraphrased evaluation scenarios, strict bounds checking, student image isolation, reference answer redaction (10 tests).
  - `tests/test_security.py`: RBAC enforcement, session isolation, cross-student submission protection, unauthenticated redirects (6 tests).

---

## 9. Demo Credentials

| Role | Email | Password | Details |
| :--- | :--- | :--- | :--- |
| **Faculty** | `faculty@smarteval.edu` | `Faculty@123` | Prof. Alan Turing (Owner of CS301 Assignment 1) |
| **Faculty** | `dr.hopper@smarteval.edu` | `Faculty@123` | Dr. Grace Hopper |
| **Student** | `student1@smarteval.edu` | `Student@123` | Ada Lovelace (Has demo submission with published results) |
| **Student** | `student2@smarteval.edu` | `Student@123` | Claude Shannon |
| **Student** | `student3@smarteval.edu` | `Student@123` | Katherine Johnson |

---

## 10. End-to-End Walkthrough Demonstration

1. **Student Submission**:
   - Log in as `student1@smarteval.edu`.
   - Browse published assignments and select *CS301: Divide and Conquer Algorithms*.
   - Upload a handwritten exam PDF via the drag-and-drop uploader.
2. **Automated Processing**:
   - The system validates magic bytes `%PDF-`, renders pages to 200 DPI PNGs, executes OpenCV preprocessing, and performs EasyOCR HTR recognition.
   - `HierarchicalQAMatcher` segments answers and matches them to questions with confidence scoring.
3. **Faculty Review & Moderation**:
   - Log in as `faculty@smarteval.edu`.
   - Open the submission review screen (`/faculty/submissions/1/review`).
   - Compare AI-suggested scores with question rubrics and student answer transcripts.
   - Use **Approve All** or edit individual scores and qualitative feedback.
   - Add summary feedback in the **Overall Feedback & Remarks** card.
4. **Publishing & Scorecards**:
   - Click **Publish Result**.
   - Student navigates to their dashboard to see the live status badge updated to `Published` and clicks **View Scorecard**.
   - Inspect total score, percentage ring, letter grade, strengths, areas for improvement, and question-by-question breakdown.
5. **Printable Academic Grade Report**:
   - Click **Print Grade Report** on the scorecard or faculty review view.
   - Browser displays a clean, institutional `@media print` formatted scorecard ready for physical printing or PDF download.
