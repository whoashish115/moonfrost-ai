// Profile, avatar resizing and switching between spaces.

function applyProfile() {
  const preview = $("profile-preview");
  if (preview) preview.innerHTML = userAvatarMarkup();
  const accountAvatar = $("account-avatar"), accountName = $("account-name");
  if (accountAvatar) accountAvatar.innerHTML = userAvatarMarkup();
  if (accountName) accountName.textContent = profile.name || "You";
  for (const avatar of document.querySelectorAll(".msg.user .avatar")) {
    avatar.innerHTML = userAvatarMarkup();
  }
  for (const label of document.querySelectorAll(".msg.user .msg-head span")) {
    label.textContent = profile.name || "You";
  }
  writeStored(PROFILE_STORAGE_KEY, profile);
}

/* Draws the chosen file into a 96px canvas and keeps the result as a data URL.
   Storing the original would blow past the localStorage quota on any modern photo. */
function shrinkToDataUrl(file, size = 96) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("could not read the file"));
    reader.onload = () => {
      const image = new Image();
      image.onerror = () => reject(new Error("not an image this browser can open"));
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = size;
        const context = canvas.getContext("2d");
        // centre-crop to a square so a wide photo is not squashed into the circle
        const edge = Math.min(image.width, image.height);
        context.drawImage(image, (image.width - edge) / 2, (image.height - edge) / 2,
                          edge, edge, 0, 0, size, size);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      };
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

/* Settings and theme are plain buttons on the row; only the profile, which is rarely
   touched, sits behind the dots. */
async function refreshWorkspaces() {
  const holder = $("workspace-list");
  if (!holder) return;
  try { workspaces = (await api("/api/workspaces")).workspaces || []; }
  catch (error) { workspaces = [{ name: "default", chats: 0 }]; }

  holder.innerHTML = workspaces.map((entry) =>
    `<button class="space${entry.name === workspace ? " on" : ""}" data-space="${escapeHtml(entry.name)}">` +
      `<svg class="icon icon-sm"><use href="#i-${entry.name === workspace ? "check" : "chevron"}"/></svg>` +
      `${escapeHtml(entry.name)}<span class="count">${entry.chats}</span></button>`).join("");
}

async function switchWorkspace(name) {
  workspace = name;
  writeStored(WORKSPACE_STORAGE_KEY, name);
  currentSessionId = null;
  showingArchive = false;
  await loadSessions();
  await refreshWorkspaces();
  if (sessions.length) await openSession(sessions[0].id); else await newSession();
  toast(`Switched to ${name}`);
}

function bindAccountRow() {
  const menu = $("account-menu");

  $("btn-settings").addEventListener("click", () => { toggleSettings(true); showSettingsTab("general"); });
  $("btn-theme").addEventListener("click", () =>
    setTheme(currentTheme() === "dark" ? "light" : "dark"));
  $("btn-about").addEventListener("click", () => showAbout(true));
  $("btn-about-back").addEventListener("click", () => showAbout(false));

  $("btn-account").addEventListener("click", (event) => {
    event.stopPropagation();
    menu.hidden = !menu.hidden;
    if (!menu.hidden) refreshWorkspaces();
  });

  // picking a space out of the list, rather than one of the fixed actions
  $("workspace-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-space]");
    if (!button) return;
    menu.hidden = true;
    if (button.dataset.space !== workspace) switchWorkspace(button.dataset.space);
  });

  // async: creating a space awaits the switch, and a plain arrow function makes that
  // await a top-level one, which is a parse error that takes the whole script with it
  menu.addEventListener("click", async (event) => {
    const action = event.target.closest("[data-account]");
    if (!action) return;
    menu.hidden = true;
    if (action.dataset.account === "new-space") {
      const name = ((await promptDialog({
        title: "New space",
        body: "A space keeps its own list of chats. The model and your settings are shared.",
        input: { placeholder: "work, personal, testing...", maxLength: 32 },
      })) || "").trim().slice(0, 32);
      if (!name) return;
      if (workspaces.some((entry) => entry.name === name)) { switchWorkspace(name); return; }
      await switchWorkspace(name);
    }
  });

  document.addEventListener("click", (event) => {
    if (!menu.hidden && !event.target.closest(".sidebar-foot")) menu.hidden = true;
  });
}
