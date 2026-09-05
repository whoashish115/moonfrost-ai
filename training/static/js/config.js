// Names, storage keys, defaults and the mutable state everything else reads.

"use strict";

/* =====================================================================
   State
   ===================================================================== */
/* Kept in step with DEFAULT_SAMPLING in chat_format.py; see the note there for why the
   repetition penalty sits at 1.15 rather than the more usual 1.05-1.1. */
// The model's own name, used for the message labels and inside the default system prompt,
// so renaming it is a one-line change here.
const ASSISTANT_NAME = "Moonfrost";
/* One mark for the whole interface. It is the artwork rather than a drawn shape, so it
   is not tinted by the theme; the surface behind it changes instead. */
const LOGO_MARK = '<img class="logo-cat" src="/static/cats/logo.png" alt="" draggable="false">';

// Your display name and picture, kept in this browser only. The picture is scaled down
// to 96px before it is stored, so it costs a few kilobytes rather than megabytes and
// comfortably fits the localStorage budget.
const PROFILE_STORAGE_KEY = "llm-profile-v1";
/* A space is a named group of chats. Everything else -- the model, the settings, the
   database file -- is shared; only the chat list is separated, which is the part worth
   keeping apart when one machine is used for unrelated things. */
const WORKSPACE_STORAGE_KEY = "llm-workspace-v1";
const SIDEBAR_STORAGE_KEY = "llm-sidebar-v1";
const LAST_SESSION_STORAGE_KEY = "llm-last-session-v1";
let workspace = "default";
let workspaces = [];
const PROFILE_DEFAULTS = { name: "You", avatar: "" };
let profile = { ...PROFILE_DEFAULTS };

function userAvatarMarkup() {
  if (profile.avatar) return `<img src="${profile.avatar}" alt="">`;
  return escapeHtml((profile.name || "You").trim().charAt(0).toUpperCase() || "Y");
}
/* What the model was fine-tuned to say when asked who made it. The project is by
   Ashish; "Magician" is the pen name that went into the identity data, so it is what
   the weights answer and what the system prompt has to agree with. */
const CREATOR_NAME = "Magician";

const DEFAULT_SYSTEM_PROMPT =
  `You are ${ASSISTANT_NAME}, a small AI assistant created by ${CREATOR_NAME}. ` +
  "Answer clearly and briefly. If you do not know something, say so.";

const DEFAULTS = {
  temperature: 0.7, top_p: 0.92, top_k: 40, min_p: 0.05,
  repetition_penalty: 1.15, max_new_tokens: 400, seed: null,
  // Sent as a <|system|> turn ahead of the conversation, which is the format the model was
  // fine-tuned on. A 161M-active model follows system prompts loosely, so keep it short --
  // a long one eats context and confuses more than it steers.
  system: DEFAULT_SYSTEM_PROMPT
};
const SLIDERS = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens"];
/* Bump this whenever DEFAULTS changes meaningfully. Stored settings are keyed by it, so a
   new default actually reaches people who already have values saved in this browser
   instead of being silently shadowed by a stale copy. */
const SETTINGS_STORAGE_KEY = "llm-settings-v3";

let settings = { ...DEFAULTS };
let sessions = [];
let currentSessionId = null;
let activeRequestId = null;
let abortController = null;
let isGenerating = false;
let statusInfo = {};

const APPEARANCE_STORAGE_KEY = "llm-appearance-v3";
