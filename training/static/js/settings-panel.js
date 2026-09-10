/* =====================================================================
   Settings
   ===================================================================== */
function applySettingsToControls() {
  for (const key of SLIDERS) {
    $(`set-${key}`).value = settings[key];
    $(`val-${key}`).textContent = key === "top_k" || key === "max_new_tokens"
      ? Math.round(settings[key]) : Number(settings[key]).toFixed(2);
  }
  $("set-seed").value = settings.seed ?? "";
  $("set-system").value = settings.system || "";
}

function bindSettings() {
  for (const key of SLIDERS) {
    $(`set-${key}`).addEventListener("input", (event) => {
      settings[key] = Number(event.target.value);
      $(`val-${key}`).textContent = key === "top_k" || key === "max_new_tokens"
        ? Math.round(settings[key]) : settings[key].toFixed(2);
      writeStored(SETTINGS_STORAGE_KEY, settings);
    });
  }
  $("set-seed").addEventListener("change", (event) => {
    const raw = event.target.value.trim();
    settings.seed = raw === "" ? null : Number(raw);
    writeStored(SETTINGS_STORAGE_KEY, settings);
  });
  $("set-system").addEventListener("change", (event) => {
    settings.system = event.target.value;
    writeStored(SETTINGS_STORAGE_KEY, settings);
  });
  $("btn-reset-settings").onclick = () => {
    settings = { ...DEFAULTS };
    applySettingsToControls();
    writeStored(SETTINGS_STORAGE_KEY, settings);
    toast("Settings reset");
  };
}

function toggleSettings(open) {
  $("settings").classList.toggle("open", open);
  $("scrim").classList.toggle("show", open);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  // the theme control lives in the sidebar account menu now, as a label rather than an icon
  // the icon shows what pressing it will switch TO, so it changes on every toggle
  const themeButton = $("btn-theme");
  if (themeButton) {
    themeButton.querySelector("use").setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
    themeButton.title = theme === "dark" ? "Switch to light" : "Switch to dark";
  }
  writeStored("llm-theme", theme);
  // The accent, its hover state and the surface tint are all per-theme values written
  // onto the root element as inline custom properties. Switching the theme without
  // recomputing them left light mode holding dark mode's near-black hover wash, which
  // painted the selected chat row dark navy with dark text on it.
  if (typeof applyAppearance === "function") applyAppearance();
  if (typeof renderColourGrid === "function") renderColourGrid();
}

/* Spawns the ink circle at the pointer for any button that opts in. Delegated from the
   document so buttons rendered later (chat rows) need no extra wiring. */
/* Anything that throws where nobody is catching it says so, rather than failing silently
   and leaving the interface looking merely stuck. */
function installErrorReporting() {
  let last = "";
  const report = (message) => {
    const text = String(message || "Something went wrong").slice(0, 160);
    if (text === last) return;        // one toast per distinct problem, not one per retry
    last = text;
    setTimeout(() => { last = ""; }, 4000);
    toast(text);
  };
  window.addEventListener("error", (event) => report(event.message));
  window.addEventListener("unhandledrejection", (event) =>
    report(event.reason && event.reason.message ? event.reason.message : event.reason));
}
