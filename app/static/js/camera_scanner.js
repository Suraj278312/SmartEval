/**
 * SmartEval In-App Camera Scanner
 * Comprehensive client-side multi-page document scanner with live camera preview,
 * interactive 4-corner perspective correction, rotation, contrast enhancement,
 * multi-page gallery reordering, and secure payload compilation.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const startCameraBtn = document.getElementById("start-camera-btn");
  const stopCameraBtn = document.getElementById("stop-camera-btn");
  const switchCameraBtn = document.getElementById("switch-camera-btn");
  const capturePageBtn = document.getElementById("capture-page-btn");
  const videoElem = document.getElementById("scanner-video");
  const viewfinderWrapper = document.getElementById("scanner-viewfinder-wrapper");
  const cameraLiveIndicator = document.getElementById("camera-live-indicator");
  const cameraErrorBanner = document.getElementById("camera-error-banner");
  const cameraErrorMsg = document.getElementById("camera-error-message");

  // Editor / Crop Workspace Elements
  const cropWorkspace = document.getElementById("scanner-crop-workspace");
  const cropCanvas = document.getElementById("crop-canvas");
  const acceptPageBtn = document.getElementById("accept-page-btn");
  const retakePageBtn = document.getElementById("retake-page-btn");
  const rotateLeftBtn = document.getElementById("rotate-left-btn");
  const rotateRightBtn = document.getElementById("rotate-right-btn");
  const resetCropBtn = document.getElementById("reset-crop-btn");
  const filterSelect = document.getElementById("filter-select");

  // Gallery & Multi-Page Elements
  const pagesGallerySection = document.getElementById("scanned-pages-gallery");
  const pagesGrid = document.getElementById("scanned-pages-grid");
  const pageCountBadge = document.getElementById("scanned-page-count");
  const scanAnotherPageBtn = document.getElementById("scan-another-page-btn");
  const submitScanBtn = document.getElementById("submit-scan-btn");
  const submitProgressBanner = document.getElementById("submit-progress-banner");
  const submitErrorBanner = document.getElementById("submit-error-banner");
  const submitErrorMsg = document.getElementById("submit-error-message");

  // Lightbox Modal
  const previewModal = document.getElementById("scanner-preview-modal");
  const modalImg = document.getElementById("scanner-modal-img");
  const modalTitle = document.getElementById("scanner-modal-title");
  const closeModalBtn = document.getElementById("close-scanner-modal-btn");

  // Configuration from data attributes
  const scannerContainer = document.getElementById("scanner-root");
  if (!scannerContainer) return;

  const submitEndpoint = scannerContainer.dataset.submitUrl || window.location.pathname + "/submit-scan";
  const sessionId = "scan_sess_" + Date.now() + "_" + Math.random().toString(36).substring(2, 10);

  // State
  let currentStream = null;
  let videoDevices = [];
  let currentDeviceIndex = 0;
  let facingMode = "environment"; // Prioritize rear camera for documents

  // Scanned pages collection: [{ id, pageNumber, dataUrl, thumbnail, timestamp, originalImage, corners, rotation, filter }]
  let scannedPages = [];
  let editingPageIndex = -1; // -1 when adding a new capture, >= 0 when re-editing an existing page

  // Current capture editor state
  let rawCaptureCanvas = document.createElement("canvas");
  let currentRotation = 0; // 0, 90, 180, 270
  let currentFilter = "enhanced"; // "original", "enhanced", "grayscale"
  let cornerPoints = []; // [{x, y}, {x, y}, {x, y}, {x, y}] -> TL, TR, BR, BL
  let activeDraggingCorner = -1;
  let canvasScale = 1;

  // =========================================================================
  // Camera Management
  // =========================================================================

  async function initCameraDevices() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      videoDevices = devices.filter((d) => d.kind === "videoinput");
      if (switchCameraBtn) {
        switchCameraBtn.style.display = videoDevices.length > 1 ? "inline-flex" : "none";
      }
    } catch (e) {
      console.warn("Could not enumerate video devices:", e);
    }
  }

  async function startCamera() {
    hideCameraError();
    stopCamera();

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1920, min: 640 },
        height: { ideal: 1080, min: 480 },
      },
    };

    if (videoDevices.length > 1 && videoDevices[currentDeviceIndex]) {
      constraints.video.deviceId = { exact: videoDevices[currentDeviceIndex].deviceId };
    }

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      videoElem.srcObject = currentStream;
      await videoElem.play();

      if (viewfinderWrapper) viewfinderWrapper.style.display = "flex";
      if (cameraLiveIndicator) cameraLiveIndicator.style.display = "inline-flex";
      if (startCameraBtn) startCameraBtn.style.display = "none";
      if (stopCameraBtn) stopCameraBtn.style.display = "inline-flex";
      if (capturePageBtn) capturePageBtn.disabled = false;

      // Check mirror effect for user/front facing camera
      const isFront = facingMode === "user";
      videoElem.classList.toggle("mirror", isFront);

      await initCameraDevices();
    } catch (err) {
      console.error("Camera access error:", err);
      showCameraError(getCameraErrorMessage(err));
    }
  }

  function stopCamera() {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => track.stop());
      currentStream = null;
    }
    if (videoElem) videoElem.srcObject = null;
    if (cameraLiveIndicator) cameraLiveIndicator.style.display = "none";
    if (startCameraBtn) startCameraBtn.style.display = "inline-flex";
    if (stopCameraBtn) stopCameraBtn.style.display = "none";
    if (capturePageBtn) capturePageBtn.disabled = true;
  }

  async function switchCamera() {
    if (videoDevices.length > 1) {
      currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
    } else {
      facingMode = facingMode === "environment" ? "user" : "environment";
    }
    await startCamera();
  }

  function getCameraErrorMessage(err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return "Camera access was denied. Please allow camera permissions in your browser address bar settings to scan your assignment.";
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return "No camera device was detected on your system. Please connect a webcam or use a mobile device with a camera.";
    }
    if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      return "Camera is currently occupied by another application. Please close other camera apps and retry.";
    }
    return `Unable to start camera: ${err.message || err.name}`;
  }

  function showCameraError(msg) {
    if (cameraErrorBanner && cameraErrorMsg) {
      cameraErrorMsg.textContent = msg;
      cameraErrorBanner.style.display = "flex";
    }
  }

  function hideCameraError() {
    if (cameraErrorBanner) cameraErrorBanner.style.display = "none";
  }

  // =========================================================================
  // Frame Capture & Editor Setup
  // =========================================================================

  function captureCurrentFrame() {
    if (!videoElem || !currentStream) return;

    const vWidth = videoElem.videoWidth || 1280;
    const vHeight = videoElem.videoHeight || 720;

    rawCaptureCanvas.width = vWidth;
    rawCaptureCanvas.height = vHeight;
    const ctx = rawCaptureCanvas.getContext("2d");

    // Account for mirror if front camera
    if (facingMode === "user") {
      ctx.translate(vWidth, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(videoElem, 0, 0, vWidth, vHeight);

    // Default corners with 6% inset margin
    const marginX = vWidth * 0.06;
    const marginY = vHeight * 0.06;
    cornerPoints = [
      { x: marginX, y: marginY }, // TL
      { x: vWidth - marginX, y: marginY }, // TR
      { x: vWidth - marginX, y: vHeight - marginY }, // BR
      { x: marginX, y: vHeight - marginY }, // BL
    ];

    currentRotation = 0;
    currentFilter = filterSelect ? filterSelect.value : "enhanced";
    editingPageIndex = -1;

    openCropWorkspace();
  }

  function openCropWorkspace() {
    if (viewfinderWrapper) viewfinderWrapper.style.display = "none";
    if (cropWorkspace) cropWorkspace.style.display = "block";
    renderCropEditor();
  }

  function closeCropWorkspace() {
    if (cropWorkspace) cropWorkspace.style.display = "none";
    if (viewfinderWrapper) viewfinderWrapper.style.display = "flex";
  }

  // =========================================================================
  // Interactive Crop & 4-Corner Perspective Transform Engine
  // =========================================================================

  function renderCropEditor() {
    if (!cropCanvas || !rawCaptureCanvas) return;

    const ctx = cropCanvas.getContext("2d");
    const srcW = rawCaptureCanvas.width;
    const srcH = rawCaptureCanvas.height;

    // Responsive display sizing (max 640px width display)
    const containerW = Math.min(window.innerWidth - 60, 600);
    canvasScale = containerW / srcW;
    cropCanvas.width = srcW * canvasScale;
    cropCanvas.height = srcH * canvasScale;

    // 1. Draw base capture frame
    ctx.save();
    ctx.scale(canvasScale, canvasScale);
    ctx.drawImage(rawCaptureCanvas, 0, 0);
    ctx.restore();

    // 2. Draw darkened overlay outside quadrilateral
    drawDimmedOverlay(ctx, cornerPoints, canvasScale, cropCanvas.width, cropCanvas.height);

    // 3. Draw quadrilateral polygon boundary
    drawPolygonBoundary(ctx, cornerPoints, canvasScale);

    // 4. Draw 4 draggable corner handles
    drawCornerHandles(ctx, cornerPoints, canvasScale);
  }

  function drawDimmedOverlay(ctx, pts, scale, width, height) {
    ctx.save();
    ctx.fillStyle = "rgba(10, 15, 30, 0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, width, height);

    // Counter-clockwise cutout for quadrilateral
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    ctx.lineTo(pts[3].x * scale, pts[3].y * scale);
    ctx.lineTo(pts[2].x * scale, pts[2].y * scale);
    ctx.lineTo(pts[1].x * scale, pts[1].y * scale);
    ctx.closePath();
    ctx.fill("evenodd");
    ctx.restore();
  }

  function drawPolygonBoundary(ctx, pts, scale) {
    ctx.save();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    ctx.lineTo(pts[1].x * scale, pts[1].y * scale);
    ctx.lineTo(pts[2].x * scale, pts[2].y * scale);
    ctx.lineTo(pts[3].x * scale, pts[3].y * scale);
    ctx.closePath();
    ctx.stroke();

    // Draw grid lines inside
    ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    // Midpoints
    const midTop = { x: (pts[0].x + pts[1].x) / 2 * scale, y: (pts[0].y + pts[1].y) / 2 * scale };
    const midBottom = { x: (pts[3].x + pts[2].x) / 2 * scale, y: (pts[3].y + pts[2].y) / 2 * scale };
    const midLeft = { x: (pts[0].x + pts[3].x) / 2 * scale, y: (pts[0].y + pts[3].y) / 2 * scale };
    const midRight = { x: (pts[1].x + pts[2].x) / 2 * scale, y: (pts[1].y + pts[2].y) / 2 * scale };

    ctx.beginPath();
    ctx.moveTo(midTop.x, midTop.y);
    ctx.lineTo(midBottom.x, midBottom.y);
    ctx.moveTo(midLeft.x, midLeft.y);
    ctx.lineTo(midRight.x, midRight.y);
    ctx.stroke();
    ctx.restore();
  }

  function drawCornerHandles(ctx, pts, scale) {
    const labels = ["TL", "TR", "BR", "BL"];
    pts.forEach((p, idx) => {
      const cx = p.x * scale;
      const cy = p.y * scale;

      ctx.save();
      // Outer glow
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 8;

      // Outer circle
      ctx.fillStyle = idx === activeDraggingCorner ? "#0284c7" : "#0ea5e9";
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
      ctx.fill();

      // Inner white pin
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fill();

      // Border
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.restore();
    });
  }

  // Pointer & Touch Events for Dragging Corners
  function getPointerPos(e) {
    const rect = cropCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) / canvasScale,
      y: (clientY - rect.top) / canvasScale,
    };
  }

  if (cropCanvas) {
    const handleStart = (e) => {
      e.preventDefault();
      const pos = getPointerPos(e);
      const hitThreshold = 30 / canvasScale; // 30px hit radius

      let closestIdx = -1;
      let minDistance = Infinity;

      cornerPoints.forEach((p, i) => {
        const dist = Math.hypot(p.x - pos.x, p.y - pos.y);
        if (dist < hitThreshold && dist < minDistance) {
          minDistance = dist;
          closestIdx = i;
        }
      });

      activeDraggingCorner = closestIdx;
      if (activeDraggingCorner !== -1) {
        renderCropEditor();
      }
    };

    const handleMove = (e) => {
      if (activeDraggingCorner === -1) return;
      e.preventDefault();
      const pos = getPointerPos(e);

      // Clamp within image bounds
      const clampedX = Math.max(0, Math.min(rawCaptureCanvas.width, pos.x));
      const clampedY = Math.max(0, Math.min(rawCaptureCanvas.height, pos.y));

      cornerPoints[activeDraggingCorner] = { x: clampedX, y: clampedY };
      renderCropEditor();
    };

    const handleEnd = (e) => {
      if (activeDraggingCorner !== -1) {
        activeDraggingCorner = -1;
        renderCropEditor();
      }
    };

    cropCanvas.addEventListener("mousedown", handleStart);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleEnd);

    cropCanvas.addEventListener("touchstart", handleStart, { passive: false });
    window.addEventListener("touchmove", handleMove, { passive: false });
    window.addEventListener("touchend", handleEnd);
  }

  // =========================================================================
  // Perspective Homography & Filter Warp Computation
  // =========================================================================

  /**
   * Solve 3x3 Homography Matrix mapping unit square / destination rectangle to source quad.
   */
  function solveHomography(srcPts, dstW, dstH) {
    // Destination corners: (0,0), (dstW, 0), (dstW, dstH), (0, dstH)
    const dstPts = [
      { x: 0, y: 0 },
      { x: dstW, y: 0 },
      { x: dstW, y: dstH },
      { x: 0, y: dstH },
    ];

    // Compute coefficients mapping (u,v) in Dest to (x,y) in Source
    // Using standard projective 8-parameter system
    const A = [];
    const B = [];

    for (let i = 0; i < 4; i++) {
      const u = dstPts[i].x;
      const v = dstPts[i].y;
      const x = srcPts[i].x;
      const y = srcPts[i].y;

      A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]);
      B.push(x);
      A.push([0, 0, 0, u, v, 1, -u * y, -v * y]);
      B.push(y);
    }

    // Solve 8x8 linear equation system using Gaussian elimination
    const h = gaussianElimination(A, B);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function gaussianElimination(A, B) {
    const n = B.length;
    for (let i = 0; i < n; i++) {
      let maxEl = Math.abs(A[i][i]);
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(A[k][i]) > maxEl) {
          maxEl = Math.abs(A[k][i]);
          maxRow = k;
        }
      }

      for (let k = i; k < n; k++) {
        const tmp = A[maxRow][k];
        A[maxRow][k] = A[i][k];
        A[i][k] = tmp;
      }
      const tmpB = B[maxRow];
      B[maxRow] = B[i];
      B[i] = tmpB;

      for (let k = i + 1; k < n; k++) {
        const c = -A[k][i] / (A[i][i] || 1e-10);
        for (let j = i; j < n; j++) {
          if (i === j) {
            A[k][j] = 0;
          } else {
            A[k][j] += c * A[i][j];
          }
        }
        B[k] += c * B[i];
      }
    }

    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = B[i] / (A[i][i] || 1e-10);
      for (let k = i - 1; k >= 0; k--) {
        B[k] -= A[k][i] * x[i];
      }
    }
    return x;
  }

  /**
   * Perspective unwarp of quadrilateral region into a crisp rectangular canvas.
   */
  function warpPerspective(srcCanvas, pts) {
    const p0 = pts[0], p1 = pts[1], p2 = pts[2], p3 = pts[3];

    // Calculate natural target width and height based on quad side lengths
    const topW = Math.hypot(p1.x - p0.x, p1.y - p0.y);
    const botW = Math.hypot(p2.x - p3.x, p2.y - p3.y);
    const leftH = Math.hypot(p3.x - p0.x, p3.y - p0.y);
    const rightH = Math.hypot(p2.x - p1.x, p2.y - p1.y);

    const dstW = Math.round(Math.max(topW, botW));
    const dstH = Math.round(Math.max(leftH, rightH));

    if (dstW <= 10 || dstH <= 10) return srcCanvas;

    const outCanvas = document.createElement("canvas");
    outCanvas.width = dstW;
    outCanvas.height = dstH;
    const outCtx = outCanvas.getContext("2d");

    const srcCtx = srcCanvas.getContext("2d");
    const srcData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
    const srcBuf = srcData.data;
    const srcW = srcData.width;
    const srcH = srcData.height;

    const outData = outCtx.createImageData(dstW, dstH);
    const outBuf = outData.data;

    const H = solveHomography(pts, dstW, dstH);

    for (let v = 0; v < dstH; v++) {
      for (let u = 0; u < dstW; u++) {
        // Projective mapping (u, v) -> (x, y)
        const denom = H[6] * u + H[7] * v + H[8];
        const sx = (H[0] * u + H[1] * v + H[2]) / denom;
        const sy = (H[3] * u + H[4] * v + H[5]) / denom;

        const outIdx = (v * dstW + u) * 4;

        if (sx >= 0 && sx < srcW - 1 && sy >= 0 && sy < srcH - 1) {
          // Bilinear interpolation
          const x0 = Math.floor(sx);
          const x1 = x0 + 1;
          const y0 = Math.floor(sy);
          const y1 = y0 + 1;

          const dx = sx - x0;
          const dy = sy - y0;

          const idx00 = (y0 * srcW + x0) * 4;
          const idx10 = (y0 * srcW + x1) * 4;
          const idx01 = (y1 * srcW + x0) * 4;
          const idx11 = (y1 * srcW + x1) * 4;

          for (let c = 0; c < 3; c++) {
            const top = srcBuf[idx00 + c] * (1 - dx) + srcBuf[idx10 + c] * dx;
            const bot = srcBuf[idx01 + c] * (1 - dx) + srcBuf[idx11 + c] * dx;
            outBuf[outIdx + c] = Math.round(top * (1 - dy) + bot * dy);
          }
          outBuf[outIdx + 3] = 255;
        } else {
          // White background padding
          outBuf[outIdx] = 255;
          outBuf[outIdx + 1] = 255;
          outBuf[outIdx + 2] = 255;
          outBuf[outIdx + 3] = 255;
        }
      }
    }

    outCtx.putImageData(outData, 0, 0);
    return outCanvas;
  }

  /**
   * Apply document enhancement filters and rotation.
   */
  function applyFiltersAndRotation(canvas, rotationAngle, filterType) {
    let current = canvas;

    // 1. Rotation (90, 180, 270)
    if (rotationAngle % 360 !== 0) {
      const rad = (rotationAngle % 360) * (Math.PI / 180);
      const rotCanvas = document.createElement("canvas");
      if (rotationAngle === 90 || rotationAngle === 270) {
        rotCanvas.width = current.height;
        rotCanvas.height = current.width;
      } else {
        rotCanvas.width = current.width;
        rotCanvas.height = current.height;
      }
      const rCtx = rotCanvas.getContext("2d");
      rCtx.translate(rotCanvas.width / 2, rotCanvas.height / 2);
      rCtx.rotate(rad);
      rCtx.drawImage(current, -current.width / 2, -current.height / 2);
      current = rotCanvas;
    }

    // 2. Enhancement Filters
    if (filterType === "enhanced" || filterType === "grayscale") {
      const filtCanvas = document.createElement("canvas");
      filtCanvas.width = current.width;
      filtCanvas.height = current.height;
      const fCtx = filtCanvas.getContext("2d");
      fCtx.drawImage(current, 0, 0);

      const imgData = fCtx.getImageData(0, 0, filtCanvas.width, filtCanvas.height);
      const buf = imgData.data;
      const len = buf.length;

      for (let i = 0; i < len; i += 4) {
        // Luminance
        const gray = 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];

        if (filterType === "enhanced") {
          // Adaptive contrast enhancement for clean handwriting
          let boosted = (gray - 128) * 1.35 + 128;
          if (boosted > 175) boosted = Math.min(255, boosted * 1.12);
          if (boosted < 80) boosted = Math.max(0, boosted * 0.85);
          boosted = Math.max(0, Math.min(255, boosted));
          buf[i] = boosted;
          buf[i + 1] = boosted;
          buf[i + 2] = boosted;
        } else {
          // Grayscale
          buf[i] = gray;
          buf[i + 1] = gray;
          buf[i + 2] = gray;
        }
      }
      fCtx.putImageData(imgData, 0, 0);
      current = filtCanvas;
    }

    return current;
  }

  // =========================================================================
  // Page Commit & Gallery Management
  // =========================================================================

  function acceptCurrentPage() {
    // 1. Warp perspective based on user adjusted corner quad
    const warpedCanvas = warpPerspective(rawCaptureCanvas, cornerPoints);

    // 2. Apply chosen rotation and enhancement filter
    const finalCanvas = applyFiltersAndRotation(warpedCanvas, currentRotation, currentFilter);

    // 3. High quality JPEG representation
    const finalDataUrl = finalCanvas.toDataURL("image/jpeg", 0.92);

    // 4. Compact thumbnail for UI gallery
    const thumbCanvas = document.createElement("canvas");
    const thumbScale = Math.min(180 / finalCanvas.width, 240 / finalCanvas.height);
    thumbCanvas.width = finalCanvas.width * thumbScale;
    thumbCanvas.height = finalCanvas.height * thumbScale;
    const tCtx = thumbCanvas.getContext("2d");
    tCtx.drawImage(finalCanvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const thumbDataUrl = thumbCanvas.toDataURL("image/jpeg", 0.75);

    const timestamp = new Date().toISOString();

    if (editingPageIndex >= 0 && editingPageIndex < scannedPages.length) {
      // Updating existing page
      scannedPages[editingPageIndex].dataUrl = finalDataUrl;
      scannedPages[editingPageIndex].thumbnail = thumbDataUrl;
      scannedPages[editingPageIndex].timestamp = timestamp;
      scannedPages[editingPageIndex].corners = JSON.parse(JSON.stringify(cornerPoints));
      scannedPages[editingPageIndex].rotation = currentRotation;
      scannedPages[editingPageIndex].filter = currentFilter;
    } else {
      // Adding newly scanned page
      scannedPages.push({
        id: "page_" + (scannedPages.length + 1) + "_" + Date.now(),
        pageNumber: scannedPages.length + 1,
        dataUrl: finalDataUrl,
        thumbnail: thumbDataUrl,
        timestamp: timestamp,
        originalImage: rawCaptureCanvas.toDataURL("image/jpeg", 0.9),
        corners: JSON.parse(JSON.stringify(cornerPoints)),
        rotation: currentRotation,
        filter: currentFilter,
      });
    }

    editingPageIndex = -1;
    closeCropWorkspace();
    updateGalleryView();
  }

  function updateGalleryView() {
    if (!pagesGallerySection || !pagesGrid) return;

    pagesGrid.innerHTML = "";

    if (scannedPages.length === 0) {
      pagesGallerySection.style.display = "none";
      if (submitScanBtn) submitScanBtn.disabled = true;
      return;
    }

    pagesGallerySection.style.display = "block";
    if (pageCountBadge) {
      pageCountBadge.textContent = `${scannedPages.length} Page${scannedPages.length > 1 ? "s" : ""}`;
    }
    if (submitScanBtn) {
      submitScanBtn.disabled = false;
      submitScanBtn.innerHTML = `Submit Scanned Assignment (${scannedPages.length} Page${scannedPages.length > 1 ? "s" : ""}) &rarr;`;
    }

    scannedPages.forEach((page, idx) => {
      page.pageNumber = idx + 1; // Re-index

      const card = document.createElement("div");
      card.className = "scanner-page-card";
      card.innerHTML = `
        <div class="scanner-thumb-wrapper">
          <span class="scanner-page-badge">Page ${page.pageNumber}</span>
          <img src="${page.thumbnail}" alt="Page ${page.pageNumber}" class="scanner-thumb-img" />
        </div>
        <div class="scanner-card-actions">
          <button type="button" class="btn-icon-sm preview-page-btn" title="View Full Page" data-index="${idx}">🔍</button>
          <button type="button" class="btn-icon-sm edit-page-btn" title="Re-crop / Adjust" data-index="${idx}">✏️</button>
          <button type="button" class="btn-icon-sm move-left-btn" title="Move earlier" data-index="${idx}" ${idx === 0 ? "disabled" : ""}>&larr;</button>
          <button type="button" class="btn-icon-sm move-right-btn" title="Move later" data-index="${idx}" ${idx === scannedPages.length - 1 ? "disabled" : ""}>&rarr;</button>
          <button type="button" class="btn-icon-sm danger delete-page-btn" title="Delete page" data-index="${idx}">✕</button>
        </div>
      `;

      // Event handlers
      card.querySelector(".preview-page-btn").addEventListener("click", () => openLightbox(idx));
      card.querySelector(".edit-page-btn").addEventListener("click", () => editExistingPage(idx));
      card.querySelector(".move-left-btn").addEventListener("click", () => movePage(idx, -1));
      card.querySelector(".move-right-btn").addEventListener("click", () => movePage(idx, 1));
      card.querySelector(".delete-page-btn").addEventListener("click", () => deletePage(idx));

      pagesGrid.appendChild(card);
    });
  }

  function movePage(index, delta) {
    const newIdx = index + delta;
    if (newIdx < 0 || newIdx >= scannedPages.length) return;
    const temp = scannedPages[index];
    scannedPages[index] = scannedPages[newIdx];
    scannedPages[newIdx] = temp;
    updateGalleryView();
  }

  function deletePage(index) {
    if (confirm(`Remove Page ${index + 1} from your scanned submission?`)) {
      scannedPages.splice(index, 1);
      updateGalleryView();
    }
  }

  function openLightbox(index) {
    if (!previewModal || !modalImg || !scannedPages[index]) return;
    modalImg.src = scannedPages[index].dataUrl;
    if (modalTitle) modalTitle.textContent = `Page ${index + 1} of ${scannedPages.length}`;
    previewModal.style.display = "flex";
  }

  function closeLightbox() {
    if (previewModal) previewModal.style.display = "none";
  }

  function editExistingPage(index) {
    const page = scannedPages[index];
    if (!page) return;

    editingPageIndex = index;
    currentRotation = page.rotation || 0;
    currentFilter = page.filter || "enhanced";
    if (filterSelect) filterSelect.value = currentFilter;

    // Load original image into canvas
    const img = new Image();
    img.onload = () => {
      rawCaptureCanvas.width = img.width;
      rawCaptureCanvas.height = img.height;
      const ctx = rawCaptureCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      cornerPoints = page.corners ? JSON.parse(JSON.stringify(page.corners)) : [
        { x: img.width * 0.06, y: img.height * 0.06 },
        { x: img.width * 0.94, y: img.height * 0.06 },
        { x: img.width * 0.94, y: img.height * 0.94 },
        { x: img.width * 0.06, y: img.height * 0.94 },
      ];

      openCropWorkspace();
    };
    img.src = page.originalImage || page.dataUrl;
  }

  // =========================================================================
  // Submission & Pipeline Trigger
  // =========================================================================

  async function submitScannedAssignment() {
    if (scannedPages.length === 0) {
      alert("Please capture at least one page before submitting.");
      return;
    }

    if (!confirm(`Submit your ${scannedPages.length}-page handwritten assignment for evaluation?`)) {
      return;
    }

    // Hide previous errors & show progress state
    if (submitErrorBanner) submitErrorBanner.style.display = "none";
    if (submitProgressBanner) submitProgressBanner.style.display = "flex";
    if (submitScanBtn) submitScanBtn.disabled = true;

    // Stop camera to release hardware
    stopCamera();

    const payload = {
      session_id: sessionId,
      pages: scannedPages.map((p, idx) => ({
        page_number: idx + 1,
        image_data: p.dataUrl,
        timestamp: p.timestamp,
      })),
    };

    try {
      const response = await fetch(submitEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Redirect to submission receipt view
        window.location.href = data.redirect_url;
      } else {
        throw new Error(data.error || "Server failed to process scanned document.");
      }
    } catch (err) {
      console.error("Submission failed:", err);
      if (submitProgressBanner) submitProgressBanner.style.display = "none";
      if (submitErrorBanner && submitErrorMsg) {
        submitErrorMsg.textContent = err.message || "Failed to submit scanned pages. Please check your network and retry.";
        submitErrorBanner.style.display = "flex";
      }
      if (submitScanBtn) submitScanBtn.disabled = false;
    }
  }

  // =========================================================================
  // Event Bindings
  // =========================================================================

  if (startCameraBtn) startCameraBtn.addEventListener("click", startCamera);
  if (stopCameraBtn) stopCameraBtn.addEventListener("click", stopCamera);
  if (switchCameraBtn) switchCameraBtn.addEventListener("click", switchCamera);
  if (capturePageBtn) capturePageBtn.addEventListener("click", captureCurrentFrame);

  if (acceptPageBtn) acceptPageBtn.addEventListener("click", acceptCurrentPage);
  if (retakePageBtn) retakePageBtn.addEventListener("click", () => {
    closeCropWorkspace();
    startCamera();
  });

  if (rotateLeftBtn) rotateLeftBtn.addEventListener("click", () => {
    currentRotation = (currentRotation - 90 + 360) % 360;
  });

  if (rotateRightBtn) rotateRightBtn.addEventListener("click", () => {
    currentRotation = (currentRotation + 90) % 360;
  });

  if (resetCropBtn) resetCropBtn.addEventListener("click", () => {
    if (!rawCaptureCanvas) return;
    const w = rawCaptureCanvas.width;
    const h = rawCaptureCanvas.height;
    cornerPoints = [
      { x: 0, y: 0 },
      { x: w, y: 0 },
      { x: w, y: h },
      { x: 0, y: h },
    ];
    renderCropEditor();
  });

  if (filterSelect) filterSelect.addEventListener("change", (e) => {
    currentFilter = e.target.value;
  });

  if (scanAnotherPageBtn) scanAnotherPageBtn.addEventListener("click", () => {
    startCamera();
    window.scrollTo({ top: viewfinderWrapper.offsetTop - 80, behavior: "smooth" });
  });

  if (submitScanBtn) submitScanBtn.addEventListener("click", submitScannedAssignment);

  if (closeModalBtn) closeModalBtn.addEventListener("click", closeLightbox);
  if (previewModal) {
    previewModal.addEventListener("click", (e) => {
      if (e.target === previewModal) closeLightbox();
    });
  }

  // Auto start camera if assignment is pending submission
  const autoStart = scannerContainer.dataset.autoStart === "true";
  if (autoStart) {
    startCamera();
  }
});
