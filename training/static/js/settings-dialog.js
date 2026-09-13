/* ---------------------------------------------------------------------
   Settings dialog: tabs, data controls, About.
   ------------------------------------------------------------------- */
function showSettingsTab(name) {
  for (const button of document.querySelectorAll("#settings-tabs button")) {
    button.classList.toggle("on", button.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".settings-panel")) {
    panel.hidden = panel.dataset.panel !== name;
  }
  if (name === "data") refreshDataStats();
}

async function refreshDataStats() {
  try {
    const stats = await api(`/api/stats?workspace=${encodeURIComponent(workspace)}`);
    $("stat-active").textContent = stats.active;
    $("stat-archived").textContent = stats.archived;
    $("stat-messages").textContent = stats.messages;
  } catch (error) {
    toast("Could not read the chat statistics: " + error.message);
  }
}

function bindSettingsDialog() {
  // #settings is the full-screen backdrop; the dialog is its child, so a click that
  // lands on the backdrop itself is a click outside the dialog
  $("settings").addEventListener("mousedown", (event) => {
    if (event.target === $("settings")) toggleSettings(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && $("settings").classList.contains("open")) toggleSettings(false);
  });

  $("settings-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]");
    if (button) showSettingsTab(button.dataset.tab);
  });

  $("btn-export-chats").addEventListener("click", async () => {
    let data;
    try { data = await api("/api/sessions/export"); }
    catch (error) { toast("Export failed: " + error.message); return; }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "moonfrost-chats-" + new Date().toISOString().slice(0, 10) + ".json";
    link.click();
    URL.revokeObjectURL(link.href);
    toast("Exported " + data.chats + " chats, " + data.messages + " messages");
  });

  $("btn-archive-all").addEventListener("click", async () => {
    if (!await confirmDialog({
      title: "Archive every chat?",
      body: "They move to the archive and can be restored at any time. Nothing is deleted.",
      confirmLabel: "Archive all",
    })) return;
    try {
      const result = await api(`/api/sessions/archive_all?workspace=${encodeURIComponent(workspace)}`,
                               { method: "POST" });
      toast("Archived " + result.archived + " chats");
    } catch (error) { toast("Could not archive: " + error.message); return; }
    currentSessionId = null;
    await loadSessions();
    await refreshDataStats();
    if (!showingArchive) await newSession();
  });

  // both destructive buttons ask twice, because neither is recoverable
  $("btn-delete-archived").addEventListener("click", async () => {
    if (!await confirmDialog({
      title: "Delete archived chats?",
      body: "This permanently removes everything in the archive. It cannot be undone.",
      confirmLabel: "Delete archived", danger: true,
    })) return;
    try {
      const result = await api(`/api/sessions?archived_only=true&workspace=${encodeURIComponent(workspace)}`,
                               { method: "DELETE" });
      toast("Deleted " + result.deleted + " archived chats");
    } catch (error) { toast("Delete failed: " + error.message); return; }
    await loadSessions();
    await refreshDataStats();
  });

  $("btn-reset-browser").addEventListener("click", async () => {
    if (!await confirmDialog({
      title: "Reset this browser?",
      body: "Clears the theme, profile and sampling values saved here. Your chats are kept: "
          + "they live in the database, not in the browser.",
      confirmLabel: "Reset",
    })) return;
    for (const key of [APPEARANCE_STORAGE_KEY, PROFILE_STORAGE_KEY, SETTINGS_STORAGE_KEY,
                       WORKSPACE_STORAGE_KEY, "llm-theme"]) {
      try { localStorage.removeItem(key); } catch (error) { /* private browsing */ }
    }
    location.reload();
  });

  $("btn-delete-all").addEventListener("click", async () => {
    if (!await confirmDialog({
      title: "Delete every chat in this space?",
      body: "Every chat and message in this space, archived ones included, is removed "
          + "permanently. Other spaces are untouched. This cannot be undone.",
      confirmLabel: "Delete everything", danger: true,
    })) return;
    try {
      const result = await api(`/api/sessions?workspace=${encodeURIComponent(workspace)}`,
                               { method: "DELETE" });
      toast("Deleted " + result.deleted + " chats");
    } catch (error) { toast("Delete failed: " + error.message); return; }
    currentSessionId = null;
    await loadSessions();
    await refreshDataStats();
    await newSession();
  });
}
