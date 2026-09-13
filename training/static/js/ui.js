// Ripples, composer autosize and the sidebar drawer.

function installRipples() {
  const selector = ".new-chat-btn, .icon-btn, .send-btn, .mic-btn, .session-item, " +
                   ".empty-chip, .reset-btn, .voice-actions button, .suggestion, " +
                   ".row-actions button, .msg-actions button, " +
                   ".toggle-chip, #jump-latest, .code-block-head button, .brand";
  document.addEventListener("pointerdown", (event) => {
    const host = event.target.closest(selector);
    if (!host) return;
    host.classList.add("ripple-host");
    const box = host.getBoundingClientRect();
    const size = Math.max(box.width, box.height);
    const ink = document.createElement("span");
    ink.className = "ripple-ink";
    ink.style.width = ink.style.height = size + "px";
    ink.style.left = (event.clientX - box.left - size / 2) + "px";
    ink.style.top = (event.clientY - box.top - size / 2) + "px";
    host.appendChild(ink);
    setTimeout(() => ink.remove(), 520);
  });
}

const COMPOSER_CAP = 200;

function composerExpanded() {
  const box = document.querySelector(".composer-box");
  return !!box && box.classList.contains("expanded");
}

/* Collapsed, the box grows with the text up to a cap. Expanded, it holds a tall panel
   whatever the text does, which is what the button is for: room to write before there is
   anything to fit. */
function autosize(textarea) {
  const expanded = composerExpanded();
  const limit = expanded ? Math.round(window.innerHeight * 0.5) : COMPOSER_CAP;
  textarea.style.height = "auto";
  textarea.style.height = (expanded ? limit : Math.min(textarea.scrollHeight, limit)) + "px";
}

/* The sidebar is a column on a wide screen and a drawer on a narrow one. Only the drawer
   needs dismissing, so the backdrop is shown from the same place that opens it and the
   width test lives in one function rather than being repeated at every call site. */
const DRAWER_WIDTH = 820;

function isDrawer() {
  return window.innerWidth <= DRAWER_WIDTH;
}

function setSidebar(open) {
  $("sidebar").classList.toggle("collapsed", !open);
  $("drawer-backdrop").classList.toggle("show", open && isDrawer());
  // only worth storing on a wide screen: as a drawer it always opens shut, so persisting
  // that state would mean it never came back
  if (!isDrawer()) writeStored(SIDEBAR_STORAGE_KEY, open);
}

// picking something from the drawer means you are done with it
function closeDrawer() {
  if (isDrawer()) setSidebar(false);
}

function syncSidebarToWidth() {
  if (isDrawer()) {
    // a drawer that is open when the screen shrinks would cover the chat with no way back
    setSidebar(false);
  } else {
    $("drawer-backdrop").classList.remove("show");
  }
}

/* =====================================================================
   Wiring
   ===================================================================== */
