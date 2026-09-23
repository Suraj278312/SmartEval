/**
 * Student PDF Submission Uploader.
 * Handles drag-and-drop, instant client-side validation, and preview.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("pdf-dropzone");
  const fileInput = document.getElementById("submission_file");
  const filePreview = document.getElementById("file-preview");
  const fileNameDisplay = document.getElementById("preview-filename");
  const fileSizeDisplay = document.getElementById("preview-filesize");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const submitBtn = document.getElementById("submit-upload-btn");
  const errorMessage = document.getElementById("upload-client-error");

  if (!dropzone || !fileInput) return;

  const MAX_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function showError(msg) {
    if (errorMessage) {
      errorMessage.textContent = msg;
      errorMessage.style.display = "block";
    }
    if (submitBtn) submitBtn.disabled = true;
  }

  function clearError() {
    if (errorMessage) {
      errorMessage.textContent = "";
      errorMessage.style.display = "none";
    }
  }

  function handleFile(file) {
    clearError();

    if (!file) return;

    // Check extension / type
    const isPdf = file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
    if (!isPdf) {
      showError("Invalid file type: Please select a valid PDF document (.pdf).");
      fileInput.value = "";
      filePreview.style.display = "none";
      dropzone.style.display = "block";
      return;
    }

    // Check file size
    if (file.size > MAX_SIZE_BYTES) {
      showError(`File size exceeds 16 MB limit (${formatBytes(file.size)}). Please compress your PDF.`);
      fileInput.value = "";
      filePreview.style.display = "none";
      dropzone.style.display = "block";
      return;
    }

    // Show preview
    if (fileNameDisplay) fileNameDisplay.textContent = file.name;
    if (fileSizeDisplay) fileSizeDisplay.textContent = formatBytes(file.size);

    filePreview.style.display = "flex";
    dropzone.style.display = "none";
    if (submitBtn) submitBtn.disabled = false;
  }

  // Click on dropzone triggers hidden file input
  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      fileInput.files = dt.files;
      handleFile(dt.files[0]);
    }
  });

  // Remove file button
  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", (e) => {
      e.preventDefault();
      fileInput.value = "";
      filePreview.style.display = "none";
      dropzone.style.display = "block";
      clearError();
      if (submitBtn) submitBtn.disabled = true;
    });
  }
});
