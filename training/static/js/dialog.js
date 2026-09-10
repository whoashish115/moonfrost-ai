/* ---------------------------------------------------------------------
   One dialog, used for every question the interface needs to ask.
   ------------------------------------------------------------------- */
let dialogResolve = null;

function closeDialog(value) {
  const backdrop = $("dialog-backdrop");
  backdrop.classList.remove("open");
  backdrop.setAttribute("aria-hidden", "true");
  const resolve = dialogResolve;
  dialogResolve = null;
  if (resolve) resolve(value);
}

/* Returns a promise: false or null when dismissed, true or the typed text when accepted.
   Everything is optional except the title, so a simple question stays a one-liner. */
function openDialog({ title, body = "", confirmLabel = "OK", cancelLabel = "Cancel",
                      danger = false, input = null } = {}) {
  return new Promise((resolve) => {
    // a second dialog while one is open would strand the first promise for ever
    if (dialogResolve) closeDialog(input ? null : false);
    dialogResolve = resolve;

    $("dialog-title").textContent = title;
    $("dialog-body").textContent = body;
    $("dialog-body").hidden = !body;

    const field = $("dialog-input");
    field.hidden = input === null;
    if (input !== null) {
      field.value = input.value || "";
      field.placeholder = input.placeholder || "";
      field.maxLength = input.maxLength || 120;
    }

    const confirmButton = $("dialog-confirm");
    confirmButton.textContent = confirmLabel;
    confirmButton.className = danger ? "danger-action" : "primary";
    $("dialog-cancel").textContent = cancelLabel;

    const backdrop = $("dialog-backdrop");
    backdrop.classList.add("open");
    backdrop.setAttribute("aria-hidden", "false");
    setTimeout(() => (input !== null ? field : confirmButton).focus(), 20);
  });
}

const confirmDialog = (options) => openDialog(options);
const promptDialog = (options) =>
  openDialog({ confirmLabel: "Save", ...options, input: { ...(options.input || {}) } });

function bindDialog() {
  const field = $("dialog-input");

  $("dialog-confirm").addEventListener("click", () =>
    closeDialog(field.hidden ? true : field.value.trim()));
  $("dialog-cancel").addEventListener("click", () => closeDialog(field.hidden ? false : null));
  $("dialog-backdrop").addEventListener("mousedown", (event) => {
    if (event.target === $("dialog-backdrop")) closeDialog(field.hidden ? false : null);
  });

  document.addEventListener("keydown", (event) => {
    if (!dialogResolve) return;
    if (event.key === "Escape") { event.preventDefault(); closeDialog(field.hidden ? false : null); }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      closeDialog(field.hidden ? true : field.value.trim());
    }
  });
}
