/* =====================================================================
   Streaming
   ===================================================================== */
function currentSettingsPayload() {
  return {
    temperature: settings.temperature,
    top_k: Math.round(settings.top_k),
    top_p: settings.top_p,
    min_p: settings.min_p,
    repetition_penalty: settings.repetition_penalty,
    max_new_tokens: Math.round(settings.max_new_tokens),
    seed: settings.seed === null || settings.seed === "" ? null : Number(settings.seed)
  };
}

function setGenerating(on) {
  isGenerating = on;
  if (!on) setTimeout(flushQueue, 0);   // after the current call unwinds
  // the mark doubles as the spinner: the newest assistant avatar pulses while it writes
  const avatars = document.querySelectorAll(".msg.assistant .avatar");
  const newest = avatars[avatars.length - 1];
  for (const avatar of avatars) avatar.classList.remove("loading");
  if (on && newest) newest.classList.add("loading");
  const button = $("btn-send");
  button.classList.toggle("stop", on);
  button.disabled = false;
  button.title = on ? "Stop generating" : "Send (Enter)";
  button.querySelector("use").setAttribute("href", on ? "#i-stop" : "#i-send");
  $("prompt").disabled = false;
}

/* SSE has to be parsed against a persistent buffer: a network chunk can end
   in the middle of a line, and an event's data can be split across two
   chunks. The previous UI ran chunk.split('\n') on each chunk in isolation,
   which silently dropped or corrupted text whenever that happened. */
async function consumeEventStream(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separator;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);

      let eventName = "message";
      const dataLines = [];
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      if (!dataLines.length) continue;
      let payload;
      try { payload = JSON.parse(dataLines.join("\n")); } catch (e) { continue; }
      const handler = handlers[eventName];
      if (handler) handler(payload);
    }
  }
}

async function streamInto(endpoint, body, bubble) {
  const contentDiv = bubble._body;
  let text = "";
  // a spinner until the first token arrives, then the streaming caret takes over
  contentDiv.innerHTML = '<span class="thinking"><span class="spinner"></span>Thinking…</span>';

  // live counters: the server emits one delta per token, so counting deltas tracks
  // generation speed closely enough to be useful while waiting
  const liveStats = document.createElement("div");
  liveStats.className = "msg-stats";
  liveStats.textContent = "generating…";
  bubble.append(liveStats);
  const startedAt = performance.now();
  let streamedTokens = 0;

  abortController = new AbortController();
  setGenerating(true);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal
    });
    if (!response.ok || !response.body) throw new Error(`server returned ${response.status}`);

    let pendingRender = false;
    const render = () => {
      pendingRender = false;
      const stick = isScrolledToBottom();
      contentDiv.innerHTML = renderAssistantContent(text, { streaming: true });
      if (stick) scrollToBottom(true);
    };

    await consumeEventStream(response, {
      start: (payload) => {
        activeRequestId = payload.request_id;
        updateContextUsage(payload.prompt_tokens);
      },
      delta: (payload) => {
        text += payload.text;
        bubble._raw = text;
        streamedTokens += 1;
        const seconds = (performance.now() - startedAt) / 1000;
        liveStats.textContent = `${streamedTokens} tokens · ${(streamedTokens / Math.max(seconds, 0.001)).toFixed(1)} tok/s`;
        // repaint on the next animation frame rather than on every token, so a fast
        // model doesn't spend all its time in the markdown renderer
        if (!pendingRender) { pendingRender = true; requestAnimationFrame(render); }
      },
      done: (payload) => {
        text = text.trimStart();
        contentDiv.innerHTML = renderAssistantContent(text);
        liveStats.textContent = payload.token_count
          ? `${payload.token_count} tokens · ${payload.tokens_per_second} tok/s` + (payload.stopped ? " · stopped" : "")
          : "";
      },
      error: (payload) => {
        contentDiv.innerHTML = `<p style="color:var(--danger)">${escapeHtml(payload.message)}</p>`;
      }
    });
  } catch (error) {
    if (error.name !== "AbortError") {
      contentDiv.innerHTML = `<p style="color:var(--danger)">Connection to the server failed: ${escapeHtml(error.message)}</p>`;
    } else if (!text) {
      contentDiv.innerHTML = `<p style="color:var(--text-faint)">Stopped.</p>`;
    }
  } finally {
    setGenerating(false);
    activeRequestId = null;
    abortController = null;
    contentDiv.querySelector(".caret")?.remove();
    if (liveStats.textContent === "generating…") liveStats.remove();
    await loadSessions();
    await refreshStatus();
    // reload so the new messages carry their database ids (needed for delete/regenerate)
    if (currentSessionId) await openSession(currentSessionId);
  }
}

/* Messages typed while the model is answering. They are held here rather than dropped,
   and flushed as one combined message when the current reply finishes. */
let messageQueue = [];

function queueMessage(input) {
  const text = input.value.trim();
  const outgoing = buildMessageWithAttachments(text);
  const chips = attachmentChipsMarkup();
  attachments = [];
  renderAttachments();
  input.value = "";
  setComposerExpanded(false);

  const bubble = messageElement("user", text || "(file only)");
  bubble.classList.add("queued");
  if (chips) bubble._body.insertAdjacentHTML("beforebegin", chips);
  const tag = document.createElement("div");
  tag.className = "queued-tag";
  tag.textContent = "queued";
  bubble.append(tag);
  $("chat").appendChild(bubble);
  scrollToBottom(true);

  messageQueue.push({ outgoing, bubble });
  toast(messageQueue.length === 1 ? "Queued until the reply finishes"
                                  : `${messageQueue.length} messages queued`);
}

/* Called once a generation ends. Everything queued goes as a single message, because the
   context window is small enough that three separate turns on one topic would waste it. */
async function flushQueue() {
  if (!messageQueue.length || isGenerating) return;
  const pending = messageQueue;
  messageQueue = [];
  for (const entry of pending) entry.bubble.remove();
  const combined = pending.map((entry) => entry.outgoing).join("\n\n");

  const chat = $("chat");
  chat.appendChild(messageElement("user", combined));
  const bubble = messageElement("assistant", "");
  chat.appendChild(bubble);
  scrollToBottom(true);

  await streamInto("/api/chat", {
    session_id: currentSessionId,
    message: combined,
    system_prompt: settings.system || null,
    settings: currentSettingsPayload()
  }, bubble);
}

async function sendMessage() {
  const input = $("prompt");
  const text = input.value.trim();
  // a message carrying only attachments is still worth sending
  if (!text && !attachments.length) return;
  // typing while the model is answering queues the message instead of discarding it
  if (isGenerating) { queueMessage(input); return; }
  if (!currentSessionId) await newSession();

  // the model reads the file contents inline; the transcript shows a chip instead, so a
  // long file does not bury the conversation it belongs to
  const outgoing = buildMessageWithAttachments(text);
  const chips = attachmentChipsMarkup();
  attachments = [];
  renderAttachments();

  input.value = "";
  setComposerExpanded(false);

  // The empty state is #empty-state; this used to look for #welcome, an id that has not
  // existed for a while, so it stayed on screen until the reply finished and something
  // else redrew the list.
  const empty = $("empty-state");
  if (empty) empty.remove();

  const chat = $("chat");
  const userMessage = messageElement("user", text || "(file only)");
  if (chips) userMessage._body.insertAdjacentHTML("beforebegin", chips);
  chat.appendChild(userMessage);
  const bubble = messageElement("assistant", "");
  chat.appendChild(bubble);
  scrollToBottom(true);

  await streamInto("/api/chat", {
    session_id: currentSessionId,
    message: outgoing,
    system_prompt: settings.system || null,
    settings: currentSettingsPayload()
  }, bubble);
}

async function regenerate() {
  if (isGenerating || !currentSessionId) return;
  const chat = $("chat");
  const messages = [...chat.querySelectorAll(".msg")];
  const last = messages[messages.length - 1];
  if (last && last.classList.contains("assistant")) last.remove();

  const bubble = messageElement("assistant", "");
  chat.appendChild(bubble);
  scrollToBottom(true);

  await streamInto("/api/regenerate", {
    session_id: currentSessionId,
    system_prompt: settings.system || null,
    settings: currentSettingsPayload()
  }, bubble);
}

async function stopGenerating() {
  if (activeRequestId) {
    try { await fetch(`/api/stop/${activeRequestId}`, { method: "POST" }); } catch (e) { /* fall through to abort */ }
  }
  if (abortController) abortController.abort();
}

async function deleteMessage(messageId) {
  if (!await confirmDialog({
    title: "Delete this message?",
    body: "This message and every message after it in the chat are removed.",
    confirmLabel: "Delete", danger: true,
  })) return;
  try { await api(`/api/messages/${messageId}`, { method: "DELETE" }); }
  catch (e) { toast("Delete failed"); return; }
  await openSession(currentSessionId);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied");
  } catch (e) {
    // clipboard API needs a secure context; http://localhost qualifies, a LAN IP does not
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed"; area.style.opacity = "0";
    document.body.appendChild(area); area.select();
    try { document.execCommand("copy"); toast("Copied"); } catch (e2) { toast("Copy failed"); }
    area.remove();
  }
}
