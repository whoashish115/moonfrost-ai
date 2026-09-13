// Every event binding, and the startup sequence.

function bindEvents() {
  $("btn-new").onclick = newSession;
  $("btn-sidebar").onclick = () => setSidebar($("sidebar").classList.contains("collapsed"));
  $("drawer-backdrop").onclick = () => setSidebar(false);

  $("btn-use-instruct").addEventListener("click", async () => {
    const name = $("btn-use-instruct").dataset.model;
    if (!name) return;
    const select = $("model-select");
    select.value = name;
    select.dispatchEvent(new Event("change"));
  });
  $("btn-close-settings").onclick = () => toggleSettings(false);
  $("scrim").onclick = () => toggleSettings(false);


  $("session-search").addEventListener("input", renderSessions);
  $("btn-archive-view").addEventListener("click", toggleArchiveView);

  $("btn-send").onclick = () => (isGenerating ? stopGenerating() : sendMessage());

  const prompt = $("prompt");
  prompt.addEventListener("input", () => autosize(prompt));
  prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });

  $("model-select").addEventListener("change", async (event) => {
    const name = event.target.value;
    // reloading the checkpoint that is already loaded costs a multi-gigabyte disk read and a
    // GPU re-upload for no benefit, so ignore a "change" that didn't actually change anything
    if (name === statusInfo.checkpoint) return;
    const badge = $("status-badge");
    badge.className = "badge warn";
    badge.textContent = "loading model…";
    try {
      const result = await api("/api/load_model", { method: "POST", body: JSON.stringify({ model_name: name }) });
      toast(`Loaded ${result.model}`);
    } catch (error) {
      toast(`Could not load: ${error.message}`);
    }
    await refreshStatus();
    await loadModels();
  });

  // one delegated listener covers every message action and code-copy button, including
  // ones added later by streaming
  document.addEventListener("click", (event) => {
    const suggestion = event.target.closest("[data-suggest]");
    if (suggestion) {
      $("prompt").value = suggestion.dataset.suggest;
      autosize($("prompt"));
      sendMessage();
      return;
    }
    const copyCode = event.target.closest("[data-copy-code]");
    if (copyCode) {
      const codeElement = document.getElementById(copyCode.dataset.copyCode);
      if (codeElement) copyText(codeElement.textContent);
      return;
    }
    const action = event.target.closest(".msg-actions button");
    if (!action) return;
    const message = action.closest(".msg");
    if (action.dataset.action === "copy") copyText(message._raw ?? message._body.textContent);
    else if (action.dataset.action === "regenerate") regenerate();
    else if (action.dataset.action === "delete") deleteMessage(Number(message.dataset.messageId));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      toggleSettings(false);
      closeDrawer();
      if (isGenerating) stopGenerating();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault(); $("session-search").focus();
    }
  });
}

async function init() {
  // a drawer always starts shut; a column comes back the way it was left
  $("sidebar").classList.toggle(
    "collapsed", isDrawer() ? true : readStored(SIDEBAR_STORAGE_KEY, true) === false);
  // ?theme=dark or ?theme=light forces one for this load without changing what is stored,
  // which makes a screenshot or a shared link reproducible
  const requestedTheme = new URLSearchParams(location.search).get("theme");
  setTheme(requestedTheme === "dark" || requestedTheme === "light"
    ? requestedTheme
    : readStored("llm-theme", "light"));
  settings = { ...DEFAULTS, ...readStored(SETTINGS_STORAGE_KEY, {}) };
  appearance = { ...APPEARANCE_DEFAULTS, ...readStored(APPEARANCE_STORAGE_KEY, {}) };
  profile = { ...PROFILE_DEFAULTS, ...readStored(PROFILE_STORAGE_KEY, {}) };
  workspace = readStored(WORKSPACE_STORAGE_KEY, "default") || "default";
  applyAppearance();

  applySettingsToControls();
  bindSettings();
  bindAppearance();
  bindProfile();
  bindAccountRow();
  bindSettingsDialog();
  bindAttachments();
  bindSuggestions();
  bindComposerExpand();
  loadPromptLibrary().then(renderSuggestions);
  bindDialog();
  installRipples();
  installErrorReporting();
  setupSpeechInput();
  bindEvents();

  $("chat-scroll").addEventListener("scroll", updateJumpButton, { passive: true });
  $("jump-latest").addEventListener("click", () => {
    $("chat-scroll").scrollTo({ top: $("chat-scroll").scrollHeight, behavior: "smooth" });
  });

  // #about is a real address, so the page can be linked to and reloaded on it
  const openFromHash = () => showAbout(location.hash === "#about");
  window.addEventListener("hashchange", openFromHash);
  openFromHash();

  await Promise.all([refreshStatus(), loadModels(), loadSessions()]);
  // reopen whatever was last read, as long as it still exists
  const remembered = readStored(LAST_SESSION_STORAGE_KEY, null);
  const stillThere = sessions.some((session) => session.id === remembered);
  if (stillThere) await openSession(remembered);
  else if (sessions.length) await openSession(sessions[0].id);
  else await newSession();

  setInterval(refreshStatus, 8000);
  $("prompt").focus();
}

init();
