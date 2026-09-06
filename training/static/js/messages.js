/* =====================================================================
   Chat rendering
   ===================================================================== */
/* A short clock time, and for a reply how long it took. Both come from the message's own
   record, so reopening a conversation shows the same values it showed live. */
function formatClock(value) {
  const when = value ? new Date(value) : new Date();
  if (isNaN(when)) return "";
  return when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function messageElement(role, content, { id = null, stats = null, timestamp = null } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  if (id !== null) wrapper.dataset.messageId = id;

  const head = document.createElement("div");
  head.className = "msg-head";
  head.innerHTML =
    // one letter in the badge, the full word beside it -- the badge used to repeat the
    // same word, so every message read "You You" / "AI Assistant"
    (role === "user" ? `<div class="avatar user">${userAvatarMarkup()}</div>`
                     : `<div class="avatar">${LOGO_MARK}</div>`) +
    `<span>${role === "user" ? escapeHtml(profile.name || "You") : ASSISTANT_NAME}</span>` +
    `<div class="msg-actions">` +
      `<button data-action="copy" title="Copy"><svg class="icon icon-sm"><use href="#i-copy"/></svg></button>` +
      (role === "assistant" ? `<button data-action="regenerate" title="Regenerate"><svg class="icon icon-sm"><use href="#i-refresh"/></svg></button>` : "") +
      (id !== null ? `<button data-action="delete" class="danger" title="Delete this and everything after"><svg class="icon icon-sm"><use href="#i-trash"/></svg></button>` : "") +
    `</div>`;

  const body = document.createElement("div");
  body.className = "msg-body" + (role === "assistant" ? " md" : "");
  if (role === "assistant") body.innerHTML = renderAssistantContent(content);
  else body.textContent = content;

  wrapper.append(head, body);
  wrapper._body = body;
  wrapper._raw = content;

  const statsLine = document.createElement("div");
  statsLine.className = "msg-stats";
  const pieces = [`<span class="msg-time">${escapeHtml(formatClock(timestamp))}</span>`];
  if (stats && stats.token_count) {
    pieces.push(`${stats.token_count} tokens`);
    if (stats.tokens_per_second) {
      pieces.push(`${stats.tokens_per_second.toFixed(1)} tok/s`);
      // how long the reply took to arrive, derived rather than stored separately
      pieces.push(`took ${(stats.token_count / stats.tokens_per_second).toFixed(1)}s`);
    }
    if (stats.model) pieces.push(escapeHtml(stats.model));
  }
  statsLine.innerHTML = pieces.join(" · ");
  wrapper.append(statsLine);
  wrapper._stats = statsLine;
  return wrapper;
}

/* How much of the model's context window this conversation is filling. It matters here
   because the window is only 1024 tokens: once the prompt approaches that, the server drops
   the OLDEST turns to make room, and the model silently forgets the start of the chat. */
function updateContextUsage(promptTokens) {
  const element = $("context-usage");
  const limit = statusInfo.context_length;
  if (!promptTokens || !limit) { element.hidden = true; return; }
  const share = promptTokens / limit;
  element.hidden = false;
  element.className = "context-usage" + (share > 0.9 ? " full" : share > 0.7 ? " warn" : "");
  element.textContent = `context ${promptTokens}/${limit}` +
    (share > 0.9 ? " · dropping old turns" : "");
  element.title = share > 0.9
    ? "The conversation no longer fits; the oldest turns are being dropped from the prompt."
    : `This conversation uses ${Math.round(share * 100)}% of the model's context window.`;
}

function isScrolledToBottom() {
  const el = $("chat-scroll");
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}
function updateJumpButton() {
  const scroller = $("chat-scroll"), button = $("jump-latest");
  if (!scroller || !button) return;
  const distance = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
  button.classList.toggle("show", distance > 240);
}

function scrollToBottom(force) {
  const el = $("chat-scroll");
  if (force || isScrolledToBottom()) el.scrollTop = el.scrollHeight;
}

// Openers chosen from what this model measurably handles: it knows its own identity, it
// holds a fact across turns, and it explains topics that appear in educational web text.
// "What is bitcoin?" used to sit here and was a poor first impression -- cryptocurrency is
// nearly absent from FineWeb-Edu, so the model has never really read about it.
// Every one of these was run through the model first and kept only if the answer was
// correct and ended on its own. Openers that read well but produce nonsense were dropped:
// "tell me a fun fact about space" gave "space is one of the five senses", and a recipe
// request invented six ingredients including two kinds of egg.
const WELCOME_SUGGESTIONS = [
  "Who are you?",
  "Who made you?",
  "Why is the sky blue?",
  "What is gravity?",
  "What is the capital of France?",
  "Explain photosynthesis simply.",
  "Write a haiku about winter.",
  "Give me three tips for studying.",
];

function showWelcome() {
  $("chat").innerHTML =
    `<div id="empty-state">` +
      `<div class="empty-mark">${LOGO_MARK}</div>` +
      `<div class="empty-title">${escapeHtml(ASSISTANT_NAME)}</div>` +
      `<div class="empty-sub">777M-parameter Mixture-of-Experts transformer, ` +
        `161M active per token. Multi-head Latent Attention with a 320-wide compressed ` +
        `KV cache, 32 routed experts with top-3 routing per layer. Pretrained from ` +
        `scratch on 6B tokens, 1,024-token context. Inference runs locally.</div>` +
      `<div class="empty-chips">` +
        WELCOME_SUGGESTIONS.map((suggestion) =>
          `<button class="empty-chip" data-suggest="${escapeHtml(suggestion)}">${escapeHtml(suggestion)}</button>`
        ).join("") +
      `</div>` +
      `<div class="empty-note">It chats, remembers what you tell it within a conversation ` +
        `and knows its own story. It is small, so it invents facts, so check anything ` +
        `that matters.</div>` +
    `</div>`;
}
