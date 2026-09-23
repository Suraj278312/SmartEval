/**
 * SmartEval - Global Interactive UI Engine
 * Handles navigation drawer, modals, flash alerts, and interactive states.
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Mobile Navigation Off-Canvas Drawer Toggle
  const navToggle = document.getElementById("mobileNavToggle");
  const mobileDrawer = document.getElementById("mobileDrawer");
  const drawerOverlay = document.getElementById("mobileDrawerOverlay");
  const drawerCloseBtn = document.getElementById("mobileDrawerClose");

  function openDrawer() {
    if (mobileDrawer && drawerOverlay) {
      mobileDrawer.classList.add("active");
      drawerOverlay.classList.add("active");
      document.body.style.overflow = "hidden";
    }
  }

  function closeDrawer() {
    if (mobileDrawer && drawerOverlay) {
      mobileDrawer.classList.remove("active");
      drawerOverlay.classList.remove("active");
      document.body.style.overflow = "";
    }
  }

  if (navToggle) {
    navToggle.addEventListener("click", openDrawer);
  }

  if (drawerCloseBtn) {
    drawerCloseBtn.addEventListener("click", closeDrawer);
  }

  if (drawerOverlay) {
    drawerOverlay.addEventListener("click", closeDrawer);
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mobileDrawer && mobileDrawer.classList.contains("active")) {
      closeDrawer();
    }
  });

  // 2. Alert Dismissal with Smooth Animation
  const alertCloseButtons = document.querySelectorAll(".alert-close");
  alertCloseButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const alert = e.target.closest(".alert");
      if (alert) {
        alert.style.transition = "opacity 0.2s ease, transform 0.2s ease";
        alert.style.opacity = "0";
        alert.style.transform = "translateY(-6px)";
        setTimeout(() => alert.remove(), 200);
      }
    });
  });

  // 3. Modals (Accessible Open/Close via Data Attributes)
  document.querySelectorAll("[data-open-modal]").forEach((trigger) => {
    trigger.addEventListener("click", (e) => {
      e.preventDefault();
      const modalId = trigger.getAttribute("data-open-modal");
      const modal = document.getElementById(modalId);
      if (modal) {
        modal.style.display = "flex";
        document.body.style.overflow = "hidden";
      }
    });
  });

  document.querySelectorAll("[data-close-modal]").forEach((trigger) => {
    trigger.addEventListener("click", () => {
      const modal = trigger.closest(".modal-backdrop");
      if (modal) {
        modal.style.display = "none";
        document.body.style.overflow = "";
      }
    });
  });

  // Close modal when clicking backdrop outside dialog
  document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) {
        backdrop.style.display = "none";
        document.body.style.overflow = "";
      }
    });
  });

  // 4. Confirm Dangerous Forms (e.g. Delete Assignment)
  const dangerForms = document.querySelectorAll("form.confirm-action, form.confirm-delete");
  dangerForms.forEach((form) => {
    form.addEventListener("submit", (e) => {
      const confirmMsg = form.getAttribute("data-confirm-message") || "Are you sure you want to proceed?";
      if (!confirm(confirmMsg)) {
        e.preventDefault();
      }
    });
  });
});

// Keyframes for interactive spinner
const style = document.createElement("style");
style.textContent = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);
