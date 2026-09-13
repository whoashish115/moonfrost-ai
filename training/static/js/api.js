/* =====================================================================
   API
   ===================================================================== */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* keep statusText */ }
    throw new Error(detail);
  }
  return response.json();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 2600);
}

function formatParameters(count) {
  if (!count) return "";
  return count >= 1e9 ? `${(count / 1e9).toFixed(1)}B` : `${(count / 1e6).toFixed(0)}M`;
}

async function refreshStatus() {
  try {
    statusInfo = await api("/api/status");
  } catch (e) {
    $("status-badge").className = "badge err";
    $("status-badge").textContent = "server unreachable";
    return;
  }
  updateBaseWarning();
  const badge = $("status-badge");
  if (statusInfo.loading) { badge.className = "badge warn"; badge.textContent = "loading model…"; }
  else if (statusInfo.loaded) {
    badge.className = "badge";
    badge.textContent = statusInfo.device === "cpu" ? "running on CPU" : "ready";
  } else { badge.className = "badge err"; badge.textContent = statusInfo.error || "no model loaded"; }

  // fade the tag out once it has been reporting ready for a moment; any other state, and
  // any change of state, brings it straight back
  clearTimeout(badgeFadeTimer);
  if (badge.textContent === lastBadgeText && badge.classList.contains("badge") &&
      !badge.classList.contains("warn") && !badge.classList.contains("err")) {
    badgeFadeTimer = setTimeout(() => badge.classList.add("settled"), 1800);
  } else {
    badge.classList.remove("settled");
  }
  lastBadgeText = badge.textContent;

  renderModelDetails();

  // The device name, VRAM figure, training step and validation loss used to sit in a
  // footer here. They are training diagnostics, not things a person chatting needs, and
  // they made the sidebar look like a monitoring panel.

  $("btn-send").disabled = !statusInfo.loaded;
}

/* Turns a checkpoint filename into something worth showing in a menu. The file on disk
   keeps its real name -- that is what the server loads and what belongs in a backup -- but
   "moonfrost-777m-instruct-v1.0.pt · 3.0GB · val 1.25" is a build artefact, not a label a
   person picks from. */
/* Turns "moonfrost-777m-instruct-v2.pt" into "Moonfrost Instruct v2". The size token is
   dropped because every checkpoint here is the same size, but the stage and the version
   are what actually tell two of them apart, so both are kept. */
function displayModelName(filename) {
  const base = filename.replace(/\.pt$/i, "");
  const version = base.match(/v(\d+(?:\.\d+)?)/i);
  const stage = base.match(/(?:^|[-_ ])(instruct|chat|base|sft)(?:$|[-_ ])/i);
  const words = base.split(/[-_\s]+/).filter((word) =>
    word && !/^v\d/i.test(word) && !/^\d+[mb]$/i.test(word) &&
    !/^(instruct|chat|base|final|best|last|sft)$/i.test(word));
  const parts = words.length
    ? [words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ")]
    : ["Checkpoint"];
  if (stage) {
    const word = stage[1].toLowerCase();
    parts.push(word === "sft" || word === "chat" ? "Instruct"
             : word.charAt(0).toUpperCase() + word.slice(1));
  }
  if (version) parts.push("v" + version[1]);
  return parts.join(" ");
}

let modelEntries = [];
let badgeFadeTimer = null;
let lastBadgeText = "";

async function loadModels() {
  let data;
  try { data = await api("/api/models"); } catch (e) { return; }
  modelEntries = data.models || [];
  const select = $("model-select");
  select.innerHTML = "";
  if (!data.models.length) {
    select.innerHTML = '<option>no checkpoints found</option>';
    return;
  }
  // only chat-tuned checkpoints are offered: a base model does not hold a conversation
  const chatModels = data.models.filter((entry) => entry.error || entry.is_chat_tuned);
  for (const entry of (chatModels.length ? chatModels : data.models)) {
    const option = document.createElement("option");
    option.value = entry.name;
    if (entry.error) {
      option.textContent = `${entry.name} - unreadable`;
      option.disabled = true;
    } else {
      option.textContent = displayModelName(entry.name);
      // the real filename stays available on hover, where it belongs: it matters when
      // you are looking for the file on disk, not when you are picking a model to chat with
      option.title = `${entry.name} · val ${entry.best_val.toFixed(3)} · ` +
        `${(entry.size_bytes / 1073741824).toFixed(1)}GB · ctx ${entry.max_sequence_length}`;
    }
    option.selected = entry.name === data.current;
    select.appendChild(option);
  }
  updateBaseWarning();
}

/* Shown whenever the loaded checkpoint has no chat tuning. The check is on what the
   server actually has loaded, not on what the picker shows, because those came apart
   once already. */
function updateBaseWarning() {
  const banner = $("base-warning");
  if (!banner) return;
  const current = statusInfo && statusInfo.checkpoint;
  const entry = modelEntries.find((model) => model.name === current);
  const isBase = !!entry && !entry.error && !entry.is_chat_tuned;
  banner.hidden = !isBase;

  const instruct = modelEntries.find((model) => !model.error && model.is_chat_tuned);
  $("btn-use-instruct").hidden = !instruct;
  $("btn-use-instruct").dataset.model = instruct ? instruct.name : "";
}

async function loadSessions() {
  try {
    sessions = await api(`/api/sessions?archived=${showingArchive ? "true" : "false"}` +
                         `&workspace=${encodeURIComponent(workspace)}`);
  } catch (error) {
    toast("Could not load chats: " + error.message);
    return;
  }
  renderSessions();
}

/* Everything about the loaded checkpoint, in one table. It used to be spread across the
   picker label, a line in the top bar and a footer, which meant the bar was mostly build
   metadata and the name itself was hard to read. */
function renderModelDetails() {
  const body = document.querySelector("#model-details tbody");
  if (!body) return;
  if (!statusInfo || !statusInfo.loaded) {
    body.innerHTML = '<tr><td>Status</td><td>no model loaded</td></tr>';
    return;
  }
  const entry = (modelEntries || []).find((item) => item.name === statusInfo.checkpoint) || {};
  const rows = [
    ["Name", displayModelName(statusInfo.checkpoint)],
    ["File", statusInfo.checkpoint],
    ["Total parameters", formatParameters(statusInfo.total_parameters)],
    ["Active per token", formatParameters(statusInfo.active_parameters)],
    ["Context length", `${statusInfo.context_length} tokens`],
    ["Validation loss", Number(statusInfo.best_val).toFixed(4)],
    ["Training step", String(statusInfo.step)],
  ];
  if (entry.num_routed_experts) rows.push(["Experts", `${entry.num_routed_experts} routed`]);
  if (entry.vocabulary_size) rows.push(["Vocabulary", entry.vocabulary_size.toLocaleString()]);
  if (entry.size_bytes) rows.push(["File size", `${(entry.size_bytes / 1073741824).toFixed(2)} GB`]);
  if (statusInfo.device_name) rows.push(["Running on", statusInfo.device_name]);

  body.innerHTML = rows.map(([label, value]) =>
    `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(String(value))}</td></tr>`).join("");
}
