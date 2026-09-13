/* ---------------------------------------------------------------------
   Attachments. Text files only: the model reads text, and anything else
   would have to be described rather than read, which is worse than saying no.
   ------------------------------------------------------------------- */
const MAX_ATTACHMENT_BYTES = 256 * 1024;
// A rough characters-per-token figure for this tokenizer, used to decide how much of a
// file can be included. The context window is 1,024 tokens in total, so a file is capped
// well below that to leave room for the question and the reply.
const CHARACTERS_PER_TOKEN = 3.6;
const ATTACHMENT_CHARACTER_BUDGET = Math.floor(520 * CHARACTERS_PER_TOKEN);

let attachments = [];

function formatBytes(count) {
  if (count < 1024) return count + " B";
  if (count < 1024 * 1024) return (count / 1024).toFixed(1) + " KB";
  return (count / (1024 * 1024)).toFixed(1) + " MB";
}

function renderAttachments() {
  const holder = $("attachments");
  holder.hidden = attachments.length === 0;
  holder.innerHTML = attachments.map((file, index) =>
    `<span class="file-chip">` +
      `<svg class="icon icon-sm"><use href="#i-file"/></svg>` +
      `<span class="name">${escapeHtml(file.name)}</span>` +
      `<span class="size">${formatBytes(file.bytes)}</span>` +
      `<button data-remove="${index}" title="Remove">` +
        `<svg class="icon icon-sm"><use href="#i-x"/></svg></button>` +
    `</span>`).join("");
}

function bindAttachments() {
  const picker = $("file-input");
  // a menu first, so the button can grow other kinds of attachment later without the
  // click silently meaning something different
  $("btn-attach").addEventListener("click", (event) => {
    event.stopPropagation();
    const existing = document.querySelector(".attach-menu");
    if (existing) { existing.remove(); return; }

    const menu = document.createElement("div");
    menu.className = "attach-menu";
    menu.innerHTML =
      `<button data-kind="text"><svg class="icon icon-sm"><use href="#i-file"/></svg>` +
      `<span><b>Text file</b><i>.txt, .md, .csv, .json, code</i></span></button>`;
    const box = $("btn-attach").getBoundingClientRect();
    menu.style.left = `${box.left}px`;
    menu.style.top = `${box.top - 8}px`;
    document.body.appendChild(menu);

    menu.addEventListener("click", (clicked) => {
      if (clicked.target.closest("[data-kind]")) { menu.remove(); picker.click(); }
    });
    setTimeout(() => {
      document.addEventListener("click", function away(event) {
        if (!event.target.closest(".attach-menu")) { menu.remove(); document.removeEventListener("click", away); }
      });
    }, 0);
  });

  $("attachments").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove]");
    if (!button) return;
    attachments.splice(Number(button.dataset.remove), 1);
    renderAttachments();
  });

  picker.addEventListener("change", async () => {
    for (const file of Array.from(picker.files || [])) {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        toast(`${file.name} is larger than ${formatBytes(MAX_ATTACHMENT_BYTES)}`);
        continue;
      }
      let content;
      try { content = await file.text(); }
      catch (error) { toast(`Could not read ${file.name}`); continue; }
      // a binary file read as text comes back full of replacement characters
      if ((content.match(/\uFFFD/g) || []).length > content.length * 0.01) {
        toast(`${file.name} does not look like a text file`);
        continue;
      }
      attachments.push({ name: file.name, bytes: file.size, content });
    }
    picker.value = "";
    renderAttachments();
  });

  // dropping a file onto the composer does the same thing
  const box = document.querySelector(".composer-box");
  box.addEventListener("dragover", (event) => { event.preventDefault(); });
  box.addEventListener("drop", (event) => {
    event.preventDefault();
    if (!event.dataTransfer.files.length) return;
    picker.files = event.dataTransfer.files;
    picker.dispatchEvent(new Event("change"));
  });
}

/* What the model receives: the question plus the file contents inline, truncated to fit
   the context window. The transcript shows chips instead, so the conversation stays
   readable while the model still sees the text. */
function buildMessageWithAttachments(question) {
  if (!attachments.length) return question;
  const budget = Math.floor(ATTACHMENT_CHARACTER_BUDGET / attachments.length);
  const parts = attachments.map((file) => {
    let body = file.content.trim();
    let note = "";
    if (body.length > budget) {
      body = body.slice(0, budget);
      note = `\n[truncated: ${formatBytes(file.bytes)} file, first ${budget} characters shown]`;
    }
    return `File: ${file.name}\n"""\n${body}\n"""${note}`;
  });
  return parts.join("\n\n") + (question ? `\n\n${question}` : "");
}

function attachmentChipsMarkup() {
  if (!attachments.length) return "";
  return `<div class="msg-files">` + attachments.map((file) =>
    `<span class="file-chip">` +
      `<svg class="icon icon-sm"><use href="#i-file"/></svg>` +
      `<span class="name">${escapeHtml(file.name)}</span>` +
      `<span class="size">${formatBytes(file.bytes)}</span>` +
    `</span>`).join("") + `</div>`;
}
