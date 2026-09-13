/* ---------------------------------------------------------------------
   Prompt suggestions. The list is whatever audition_many.py verified against
   this checkpoint, so it only offers questions the model was actually seen to
   answer, and it says so rather than implying general competence.
   ------------------------------------------------------------------- */
let promptLibrary = [];
let suggestIndex = -1;
// advanced each time the composer is focused, so the opening six vary between visits
let suggestionRotation = Math.floor(Math.random() * 7);

async function loadPromptLibrary() {
  try {
    const data = await (await fetch("/static/prompts.json", { cache: "no-cache" })).json();
    promptLibrary = (data.kept || []).map((entry) => ({
      text: entry.prompt, category: entry.category,
    }));
  } catch (error) {
    promptLibrary = [];        // the composer simply offers nothing
  }
}

/* Scores a suggestion against what has been typed. The lower tiers matter as much as the
   top ones: an exact-match-only filter leaves the panel empty the moment someone types a
   word the library does not contain, which is worse than offering something related. */
function scoreSuggestion(entry, query) {
  const text = entry.text.toLowerCase();
  if (!query) return 1;
  if (text.startsWith(query)) return 100;
  if (text.includes(query)) return 80;

  const words = query.split(/\s+/).filter(Boolean);
  const category = entry.category.toLowerCase();
  if (words.every((word) => text.includes(word) || category.includes(word))) return 60;

  // partial credit: how many typed words appear anywhere, counting word stems so that
  // "planets" still finds "a planet" and "explaining" still finds "explain"
  const stem = (word) => word.replace(/(ing|ed|es|s)$/, "");
  const hits = words.filter((word) => {
    const root = stem(word);
    return root.length > 2 && (text.includes(root) || category.includes(root));
  }).length;
  if (hits) return 20 + hits * 10;

  // last resort: share the first few letters with any word in the prompt, which catches
  // typing partway through a word
  const head = words[words.length - 1] || "";
  if (head.length >= 3 && text.split(/\s+/).some((piece) => piece.startsWith(head))) return 10;
  return 0;
}

function renderSuggestions() {
  const box = $("suggest-box");
  const query = $("prompt").value.trim().toLowerCase();

  if (!promptLibrary.length || isGenerating || appearance.suggestions === false) {
    box.hidden = true;
    return;
  }

  let ranked = promptLibrary
    .map((entry) => ({ entry, score: scoreSuggestion(entry, query) }))
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score);

  if (!query) {
    // With nothing typed the list is in file order, which put six identity questions in a
    // row. Take one per category first so the opening set shows what the model covers, and
    // rotate the starting point so it is not the same six every time.
    const byCategory = new Map();
    for (const row of ranked) {
      const bucket = byCategory.get(row.entry.category) || [];
      bucket.push(row);
      byCategory.set(row.entry.category, bucket);
    }
    const buckets = [...byCategory.values()];
    const spread = [];
    for (let depth = 0; spread.length < 6 && depth < 12; depth++) {
      for (const bucket of buckets) {
        const offset = (depth + suggestionRotation) % bucket.length;
        const pick = bucket[offset];
        if (pick && !spread.includes(pick)) spread.push(pick);
        if (spread.length === 6) break;
      }
    }
    ranked = spread;
  }
  ranked = ranked.slice(0, 6);

  // nothing scored: fall back to a spread rather than an empty panel, and say so
  const noMatches = !ranked.length;
  if (noMatches) {
    ranked = promptLibrary.slice(0, 40)
      .map((entry) => ({ entry, score: 1 }))
      .sort(() => Math.random() - 0.5)
      .slice(0, 6);
  }
  if (!ranked.length) { box.hidden = true; return; }
  suggestIndex = -1;
  box.innerHTML =
    `<div class="suggest-head"><span>${noMatches ? "Nothing matched, try one of these"
        : (query ? "Matching suggestions" : "Suggestions")}</span>` +
    `<span><span class="suggest-count">${promptLibrary.length} checked</span>` +
    `<button class="suggest-hide" id="btn-hide-suggestions">Hide</button></span></div>` +
    `<div class="suggest-list">` +
    ranked.map((row, index) =>
      `<button class="suggest-item" data-index="${index}" data-text="${escapeHtml(row.entry.text)}">` +
        `<span class="label">${escapeHtml(row.entry.text)}</span>` +
        `<span class="tag">${escapeHtml(row.entry.category)}</span>` +
      `</button>`).join("") + `</div>`;
  box.hidden = false;
}

/* The composer can be enlarged for a long message. Shift+Enter still inserts a newline,
   and once the text is long enough to need the room the box expands on its own. */
const CAN_EXPAND = () => window.matchMedia("(min-width: 561px)").matches;

function setComposerExpanded(expanded) {
  const box = document.querySelector(".composer-box");
  box.classList.toggle("expanded", expanded);
  const icon = $("btn-expand").querySelector("use");
  icon.setAttribute("href", expanded ? "#i-collapse" : "#i-expand");
  $("btn-expand").title = expanded ? "Shrink the box" : "Expand the box";
  autosize($("prompt"));
}

function bindComposerExpand() {
  const input = $("prompt");
  $("btn-expand").addEventListener("click", () =>
    setComposerExpanded(!document.querySelector(".composer-box").classList.contains("expanded")));

  input.addEventListener("keydown", (event) => {
    // a newline is the moment someone is writing something long, so make room for it
    if (event.key === "Enter" && event.shiftKey) {
      const box = document.querySelector(".composer-box");
      if (!box.classList.contains("expanded") && input.value.includes("\n")) {
        setComposerExpanded(true);
      }
    }
  });

  // the button is a mode, not a reaction to the text, so typing does not undo it
  window.addEventListener("resize", () => {
    if (composerExpanded() && !CAN_EXPAND()) setComposerExpanded(false);
    else if (composerExpanded()) autosize(input);
  });
}

function bindSuggestions() {
  const input = $("prompt"), box = $("suggest-box");

  input.addEventListener("input", renderSuggestions);

  // the Hide link turns the panel off; Settings turns it back on
  box.addEventListener("click", (event) => {
    if (!event.target.closest("#btn-hide-suggestions")) return;
    appearance.suggestions = false;
    writeStored(APPEARANCE_STORAGE_KEY, appearance);
    box.hidden = true;
    toast("Suggestions hidden. Turn them back on in Settings, Appearance.");
  });
  input.addEventListener("focus", () => { suggestionRotation++; renderSuggestions(); });
  input.addEventListener("blur", () => setTimeout(() => { box.hidden = true; }, 150));

  input.addEventListener("keydown", (event) => {
    if (box.hidden) return;
    const items = [...box.querySelectorAll(".suggest-item")];
    if (!items.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      suggestIndex = (suggestIndex + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((item, index) => item.classList.toggle("on", index === suggestIndex));
    } else if (event.key === "Tab" && suggestIndex >= 0) {
      event.preventDefault();
      input.value = items[suggestIndex].dataset.text;
      autosize(input);
      box.hidden = true;
    } else if (event.key === "Enter" && suggestIndex >= 0 && !event.shiftKey) {
      event.preventDefault();
      input.value = items[suggestIndex].dataset.text;
      box.hidden = true;
      sendMessage();
    } else if (event.key === "Escape") {
      box.hidden = true;
    }
  });

  box.addEventListener("mousedown", (event) => {
    const item = event.target.closest(".suggest-item");
    if (!item) return;
    event.preventDefault();          // keep focus so blur does not race the click
    input.value = item.dataset.text;
    autosize(input);
    box.hidden = true;
    input.focus();
  });
}
