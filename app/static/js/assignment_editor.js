/**
 * Dynamic Question Editor for Faculty Assignment Management.
 * Manages question blocks, renumbering, and live total marks calculation.
 */

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("questions-container");
  const addBtn = document.getElementById("add-question-btn");
  const totalMarksDisplay = document.getElementById("total-marks-badge");

  if (!container || !addBtn) return;

  function updateQuestionNumbersAndMarks() {
    const items = container.querySelectorAll(".question-item");
    let totalMarks = 0;

    items.forEach((item, index) => {
      const numSpan = item.querySelector(".q-number");
      if (numSpan) {
        numSpan.textContent = index + 1;
      }

      const marksInput = item.querySelector(".q-marks");
      if (marksInput) {
        const val = parseFloat(marksInput.value) || 0;
        totalMarks += val;
      }

      // Hide remove button if only 1 question remains
      const removeBtn = item.querySelector(".remove-question-btn");
      if (removeBtn) {
        removeBtn.style.display = items.length > 1 ? "inline-flex" : "none";
      }
    });

    if (totalMarksDisplay) {
      totalMarksDisplay.textContent = `${totalMarks} Marks`;
    }
  }

  function createQuestionElement(index) {
    const div = document.createElement("div");
    div.className = "question-item";
    div.innerHTML = `
      <div class="question-header">
        <div class="question-title">Question #<span class="q-number">${index}</span></div>
        <button type="button" class="btn btn-danger btn-sm remove-question-btn">
          ✕ Remove
        </button>
      </div>

      <div class="form-group">
        <label class="form-label">Question Text <span style="color:red;">*</span></label>
        <textarea name="question_text[]" class="form-control" rows="3" placeholder="Enter problem statement or prompt..." required></textarea>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">Maximum Marks <span style="color:red;">*</span></label>
          <input type="number" step="0.5" min="0.5" name="maximum_marks[]" class="form-control q-marks" value="10.0" required>
        </div>
        <div class="form-group">
          <label class="form-label">Grading Rubric / Criteria (Optional)</label>
          <input type="text" name="rubric[]" class="form-control" placeholder="e.g. 4 pts approach, 4 pts calculation, 2 pts final answer">
        </div>
      </div>

      <div class="faculty-only-box">
        <span class="faculty-only-badge">🔒 Supportive / Reference Answer (Faculty & AI Only)</span>
        <textarea name="supportive_answer[]" class="form-control" rows="3" placeholder="Provide reference solution, key derivation steps, formulas, or expected keywords... (Never visible to students)"></textarea>
        <div class="form-help">This supportive answer will guide future semantic grading and is strictly hidden from student views.</div>
      </div>
    `;

    // Attach remove event
    div.querySelector(".remove-question-btn").addEventListener("click", () => {
      div.remove();
      updateQuestionNumbersAndMarks();
    });

    // Attach marks change listener
    div.querySelector(".q-marks").addEventListener("input", updateQuestionNumbersAndMarks);

    return div;
  }

  // Add question click handler
  addBtn.addEventListener("click", () => {
    const currentCount = container.querySelectorAll(".question-item").length;
    const newQuestion = createQuestionElement(currentCount + 1);
    container.appendChild(newQuestion);
    updateQuestionNumbersAndMarks();
    newQuestion.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  // Attach event listeners to initial existing questions
  container.querySelectorAll(".question-item").forEach((item) => {
    const removeBtn = item.querySelector(".remove-question-btn");
    if (removeBtn) {
      removeBtn.addEventListener("click", () => {
        item.remove();
        updateQuestionNumbersAndMarks();
      });
    }

    const marksInput = item.querySelector(".q-marks");
    if (marksInput) {
      marksInput.addEventListener("input", updateQuestionNumbersAndMarks);
    }
  });

  // Initial calculation
  updateQuestionNumbersAndMarks();
});
