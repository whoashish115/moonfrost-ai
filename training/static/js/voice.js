/* ---------------------------------------------------------------------
   Voice input. SpeechRecognition is the browser's own engine -- in Chrome
   and Edge the audio goes to Google's service, nothing touches this server,
   and no key is needed. Firefox has no implementation, so the button stays
   hidden there rather than appearing and doing nothing.
   --------------------------------------------------------------------- */
let recognition = null;
let recognitionActive = false;
let dictated = "";        // everything the recogniser has settled on this session

function renderDictation(pending) {
  const box = $("voice-text");
  if (!box) return;
  box.innerHTML = escapeHtml(dictated) +
    (pending ? `<span class="pending"> ${escapeHtml(pending)}</span>` : "");
  box.scrollTop = box.scrollHeight;
}

function setupSpeechInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const button = $("btn-mic");
  if (!SpeechRecognition) return;          // stays hidden
  button.hidden = false;

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = navigator.language || "en-US";

  const prompt = $("prompt"), interim = $("mic-interim");

  recognition.addEventListener("result", (event) => {
    let settled = "", pending = "";
    for (let index = event.resultIndex; index < event.results.length; index++) {
      const result = event.results[index];
      if (result.isFinal) settled += result[0].transcript;
      else pending += result[0].transcript;
    }
    if (settled) {
      const separator = dictated && !/\s$/.test(dictated) ? " " : "";
      dictated += separator + settled.trim();
    }
    renderDictation(pending.trim());
  });

  recognition.addEventListener("error", (event) => {
    const status = $("voice-status");
    if (status) status.textContent = "Problem: " + event.error;
    stopDictation();
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      toast("Microphone permission denied");
    } else if (event.error !== "aborted" && event.error !== "no-speech") {
      toast("Dictation error: " + event.error);
    }
  });

  // Chrome ends the session on its own after a silence; restart so "continuous" is continuous
  recognition.addEventListener("end", () => {
    if (recognitionActive) {
      try { recognition.start(); } catch (e) { stopDictation(); }
    }
  });

  button.addEventListener("click", () => {
    if (recognitionActive) stopDictation(); else startDictation();
  });

  $("voice-cancel").addEventListener("click", () => { dictated = ""; stopDictation(); });
  $("voice-insert").addEventListener("click", commitDictation);
  document.addEventListener("keydown", (event) => {
    if (!recognitionActive) return;
    if (event.key === "Escape") { dictated = ""; stopDictation(); }
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); commitDictation(); }
  });
}

/* Moves what was heard into the composer and closes the panel. */
function commitDictation() {
  const prompt = $("prompt");
  const text = dictated.trim();
  if (text) {
    const separator = prompt.value && !/\s$/.test(prompt.value) ? " " : "";
    prompt.value = prompt.value + separator + text;
    prompt.dispatchEvent(new Event("input"));
  }
  dictated = "";
  stopDictation();
  prompt.focus();
}

function startDictation() {
  if (!recognition || recognitionActive) return;
  try { recognition.start(); } catch (e) { return; }
  recognitionActive = true;
  dictated = "";
  renderDictation("");
  $("voice-panel").classList.add("show");
  $("btn-mic").classList.add("listening");
  $("btn-mic").title = "Stop dictating";
}

function stopDictation() {
  if (!recognition) return;
  recognitionActive = false;
  try { recognition.stop(); } catch (e) { /* already stopped */ }
  $("voice-panel").classList.remove("show");
  $("btn-mic").classList.remove("listening");
  $("btn-mic").title = "Dictate";
  $("mic-interim").textContent = "";
}
