/* Archived chats live in the same table with a flag, so nothing is copied or moved and
   restoring one is a single update. The list shows one set or the other. */
let showingArchive = false;

async function archiveSession(id, archived) {
  try {
    await api(`/api/sessions/${id}/archive`, { method: "POST", body: JSON.stringify({ archived }) });
  } catch (error) {
    toast("Could not " + (archived ? "archive" : "restore") + ": " + error.message);
    return;
  }
  toast(archived ? "Chat archived" : "Chat restored");
  if (id === currentSessionId && archived) currentSessionId = null;
  await loadSessions();
  if (!currentSessionId && !showingArchive) {
    if (sessions.length) await openSession(sessions[0].id); else await newSession();
  }
}

async function pinSession(id, pinned) {
  try {
    await api(`/api/sessions/${id}/pin`, { method: "POST", body: JSON.stringify({ pinned }) });
  } catch (error) {
    toast("Could not " + (pinned ? "pin" : "unpin") + ": " + error.message);
    return;
  }
  toast(pinned ? "Pinned to top" : "Unpinned");
  await loadSessions();
}

function toggleArchiveView() {
  showingArchive = !showingArchive;
  $("btn-archive-view").classList.toggle("on", showingArchive);
  $("btn-archive-view").title = showingArchive ? "Show active chats" : "Show archived chats";
  loadSessions();
}

function renderSessions() {
  const query = $("session-search").value.trim().toLowerCase();
  const list = $("session-list");
  list.innerHTML = "";
  const visible = sessions.filter((s) => !query || s.title.toLowerCase().includes(query));
  if (!visible.length) {
    const empty = query ? "No matches" : (showingArchive ? "Nothing archived" : "No chats yet");
    list.innerHTML = `<div style="padding:10px 12px;font-size:13px;color:var(--text-faint)">${empty}</div>`;
    return;
  }
  // the server already sorts pinned first; this only splits them into two labelled blocks
  const pinned = visible.filter((s) => s.pinned);
  const rest = visible.filter((s) => !s.pinned);

  const addLabel = (label, withIcon) => {
    const node = document.createElement("div");
    node.className = "list-label";
    node.innerHTML = (withIcon ? `<svg class="pin-mark"><use href="#i-pin"/></svg>` : "") +
                     `<span>${label}</span>`;
    list.appendChild(node);
  };

  const addDivider = () => {
    const node = document.createElement("div");
    node.className = "list-divider";
    list.appendChild(node);
  };

  const addRow = (session) => {
    const item = document.createElement("div");
    item.className = "session-item" + (session.id === currentSessionId ? " active" : "");
    item.innerHTML =
      (session.pinned ? `<svg class="pin-mark"><use href="#i-pin"/></svg>` : "") +
      `<span class="session-title"></span>` +
      `<button class="row-menu-btn" data-menu title="More"><svg class="icon icon-sm"><use href="#i-dots"/></svg></button>`;
    item.querySelector(".session-title").textContent = session.title;
    item.onclick = (event) => { if (!event.target.closest("button")) openSession(session.id); };
    item.querySelector("[data-menu]").onclick = (event) => {
      event.stopPropagation();
      openRowMenu(event.currentTarget, item, session);
    };
    list.appendChild(item);
  };

  /* The row above the list already says "Chats", so when the list grows its own headings
     that row would repeat one of them. It keeps the archive toggle either way, and gives
     up its text to the headings below. */
  $("section-label").textContent = pinned.length ? ""
    : (showingArchive ? "Archived" : "Chats");

  // once anything is pinned the list is always two sections, so the boundary does not
  // move around as chats are pinned and unpinned
  if (pinned.length) {
    addLabel("Pinned", false);
    pinned.forEach(addRow);
    addDivider();
    addLabel(showingArchive ? "Archived" : "Chats", false);
  }
  rest.forEach(addRow);
  if (pinned.length && !rest.length) {
    const empty = document.createElement("div");
    empty.style.cssText = "padding:6px 12px;font-size:12.5px;color:var(--text-faint)";
    empty.textContent = showingArchive ? "Nothing else archived" : "Nothing else here";
    list.appendChild(empty);
  }
}

/* One popup, moved to whichever row asked for it. Three buttons per row squeezed the
   title until it ellipsised, which looked like the row shrinking when the pointer
   entered it. */
let openRowMenuElement = null;

function closeRowMenu() {
  if (openRowMenuElement) { openRowMenuElement.remove(); openRowMenuElement = null; }
}

function openRowMenu(button, item, session) {
  const wasOpen = openRowMenuElement && openRowMenuElement.dataset.session === session.id;
  closeRowMenu();
  if (wasOpen) return;

  const menu = document.createElement("div");
  menu.className = "row-menu";
  menu.dataset.session = session.id;
  menu.innerHTML =
    `<button data-act="pin"><svg class="icon icon-sm"><use href="#i-pin"/></svg>` +
      `${session.pinned ? "Unpin" : "Pin to top"}</button>` +
    `<button data-act="rename"><svg class="icon icon-sm"><use href="#i-pencil"/></svg>Rename</button>` +
    `<button data-act="archive"><svg class="icon icon-sm"><use href="#i-archive"/></svg>` +
      `${showingArchive ? "Restore" : "Archive"}</button>` +
    `<button data-act="delete" class="danger"><svg class="icon icon-sm"><use href="#i-trash"/></svg>Delete</button>`;

  const box = button.getBoundingClientRect();
  menu.style.top = `${Math.min(box.bottom + 4, window.innerHeight - 180)}px`;
  menu.style.left = `${Math.max(8, box.right - 150)}px`;
  document.body.appendChild(menu);
  openRowMenuElement = menu;

  menu.addEventListener("click", (event) => {
    const action = event.target.closest("[data-act]");
    if (!action) return;
    closeRowMenu();
    if (action.dataset.act === "pin") pinSession(session.id, !session.pinned);
    if (action.dataset.act === "rename") startRename(item, session);
    if (action.dataset.act === "archive") archiveSession(session.id, !showingArchive);
    if (action.dataset.act === "delete") deleteSession(session.id);
  });
}

document.addEventListener("click", (event) => {
  if (openRowMenuElement && !event.target.closest(".row-menu, [data-menu]")) closeRowMenu();
});
window.addEventListener("resize", closeRowMenu);

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(syncSidebarToWidth, 120);
});

function startRename(item, session) {
  const titleSpan = item.querySelector(".session-title");
  const input = document.createElement("input");
  input.className = "session-rename-input";
  input.value = session.title;
  titleSpan.replaceWith(input);
  input.focus();
  input.select();
  const commit = async () => {
    const title = input.value.trim();
    if (title && title !== session.title) {
      try { await api(`/api/sessions/${session.id}`, { method: "PATCH", body: JSON.stringify({ title }) }); }
      catch (e) { toast("Rename failed"); }
    }
    await loadSessions();
  };
  input.onblur = commit;
  input.onkeydown = (event) => {
    if (event.key === "Enter") { event.preventDefault(); input.blur(); }
    if (event.key === "Escape") { input.onblur = null; renderSessions(); }
  };
}

async function newSession() {
  try {
    const session = await api(
      `/api/sessions?workspace=${encodeURIComponent(workspace)}`, { method: "POST" });
    currentSessionId = session.id;
    closeDrawer();
    $("context-usage").hidden = true;
    showWelcome();
    await loadSessions();
    collapseSidebarOnNarrowScreen();
    $("prompt").focus();
  } catch (e) { toast("Could not create a chat"); }
}

const NARROW_SCREEN = () => window.matchMedia("(max-width: 820px)").matches;
// Below this width the sidebar genuinely covers the conversation, so picking a chat has to
// close it. Between 640 and 820 it only overlaps a little, and closing it there meant the
// list collapsed every single time you clicked a chat.
const TINY_SCREEN = () => isDrawer();

/* Deliberately does nothing. The sidebar used to close itself every time a chat was
   opened, which made the whole list shift under the pointer mid-click. The toggle in the
   corner is now the only thing that moves it. */
function collapseSidebarOnNarrowScreen() { closeDrawer(); }

async function openSession(id) {
  currentSessionId = id;
  writeStored(LAST_SESSION_STORAGE_KEY, id);
  renderSessions();
  collapseSidebarOnNarrowScreen();
  let data;
  try { data = await api(`/api/sessions/${id}`); } catch (e) { toast("Could not open that chat"); return; }
  const chat = $("chat");
  chat.innerHTML = "";
  if (!data.messages.length) { showWelcome(); return; }
  for (const message of data.messages) {
    chat.appendChild(messageElement(message.role, message.content, {
      id: message.id, timestamp: message.timestamp,
      stats: { token_count: message.token_count, tokens_per_second: message.tokens_per_second, model: message.model }
    }));
  }
  scrollToBottom(true);
}

async function deleteSession(id) {
  if (!await confirmDialog({
    title: "Delete this chat?",
    body: "The chat and all of its messages are removed permanently.",
    confirmLabel: "Delete", danger: true,
  })) return;
  try { await api(`/api/sessions/${id}`, { method: "DELETE" }); } catch (e) { toast("Delete failed"); return; }
  if (id === currentSessionId) { currentSessionId = null; $("chat").innerHTML = ""; }
  await loadSessions();
  if (!currentSessionId) {
    if (sessions.length) await openSession(sessions[0].id); else await newSession();
  }
}
