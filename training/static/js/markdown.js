/* =====================================================================
   Safe rendering: escape first, then apply markdown to the escaped text.
   The previous UI assigned model output straight to innerHTML, which meant
   any '<' in a reply silently ate the rest of the line: the browser treated
   whatever followed as an unknown element rather than as text.
   ===================================================================== */
function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function renderInline(escaped) {
  return escaped
    .replace(/`([^`\n]+)`/g, (_, code) => `<code>${code}</code>`)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

let codeBlockCounter = 0;
function renderMarkdown(rawText) {
  const escaped = escapeHtml(rawText);
  const lines = escaped.split("\n");
  const out = [];
  let index = 0;

  const flushList = (tag, items) =>
    `<${tag}>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${tag}>`;

  while (index < lines.length) {
    const line = lines[index];

    // fenced code block
    const fence = line.match(/^\s*```([\w+-]*)\s*$/);
    if (fence) {
      const language = fence[1] || "";
      const body = [];
      index++;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) { body.push(lines[index]); index++; }
      index++; // closing fence (or end of text, for a block still streaming)
      out.push(codeBlockMarkup(body, language || guessLanguage(body)));
      continue;
    }

    // heading
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      index++; continue;
    }

    // horizontal rule
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { out.push("<hr>"); index++; continue; }

    // blockquote
    if (/^\s*&gt;\s?/.test(line)) {
      const body = [];
      while (index < lines.length && /^\s*&gt;\s?/.test(lines[index])) {
        body.push(lines[index].replace(/^\s*&gt;\s?/, "")); index++;
      }
      out.push(`<blockquote>${body.map((b) => `<p>${renderInline(b)}</p>`).join("")}</blockquote>`);
      continue;
    }

    // unordered list
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*+]\s+/, "")); index++;
      }
      out.push(flushList("ul", items)); continue;
    }

    // ordered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+[.)]\s+/, "")); index++;
      }
      out.push(flushList("ol", items)); continue;
    }

    // blank line
    if (!line.trim()) { index++; continue; }

    // paragraph: consume until a blank line or the start of another construct
    const paragraph = [];
    while (index < lines.length && lines[index].trim() &&
           !/^\s*```/.test(lines[index]) && !/^(#{1,4})\s/.test(lines[index]) &&
           !/^\s*[-*+]\s+/.test(lines[index]) && !/^\s*\d+[.)]\s+/.test(lines[index]) &&
           !/^\s*&gt;\s?/.test(lines[index])) {
      paragraph.push(lines[index]); index++;
    }
    if (looksLikeCode(paragraph)) {
      out.push(codeBlockMarkup(paragraph, guessLanguage(paragraph)));
      continue;
    }
    out.push(`<p>${renderInline(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
  }
  return out.join("");
}

/* A small highlighter rather than a CDN library: this server runs offline, an external
   script would be one more thing to fail, and covering strings, comments, numbers,
   keywords and call sites gets most of the readability for a fraction of the weight.

   Order matters. Strings and comments are matched first and their contents parked, so a
   keyword inside a string is not coloured and a brace inside a comment does nothing. */
const CODE_KEYWORDS = {
  java: "abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while var record sealed true false null",
  python: "and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield True False None self",
  javascript: "async await break case catch class const continue debugger default delete do else export extends finally for function if import in instanceof let new of return static super switch this throw try typeof var void while yield true false null undefined",
  c: "auto break case char const continue default do double else enum extern float for goto if inline int long register return short signed sizeof static struct switch typedef union unsigned void volatile while true false NULL",
  sql: "select from where group by order having insert update delete create table alter drop join inner left right outer on as and or not null distinct limit values into set",
  shell: "if then else elif fi for while do done case esac function return export local echo cd",
};
CODE_KEYWORDS.ts = CODE_KEYWORDS.typescript = CODE_KEYWORDS.js = CODE_KEYWORDS.jsx = CODE_KEYWORDS.javascript;
CODE_KEYWORDS.py = CODE_KEYWORDS.python;
CODE_KEYWORDS.cpp = CODE_KEYWORDS["c++"] = CODE_KEYWORDS.cs = CODE_KEYWORDS.c;
CODE_KEYWORDS.bash = CODE_KEYWORDS.sh = CODE_KEYWORDS.shell;

function highlightCode(escaped, language) {
  const key = (language || "").toLowerCase();
  const keywords = new Set((CODE_KEYWORDS[key] || CODE_KEYWORDS.javascript).split(" "));

  /* One pass, one regex. Every construct is an alternative, so whichever matches first at
     a given position wins and the rest of that text is consumed rather than re-examined.
     Anything not matched is passed through untouched. The input is already HTML-escaped,
     which is why the string cases look for &quot; and &#39;. */
  const scanner = new RegExp([
    "(/\\*[\\s\\S]*?\\*/|//[^\\n]*|#[^\\n]*|--[^\\n]*)",            // 1 comment
    "(&quot;(?:\\\\.|&(?!quot;)|[^&\\n\\\\])*&quot;)",                    // 2 double-quoted
    "(&#39;(?:\\\\.|&(?!#39;)|[^&\\n\\\\])*&#39;)",                       // 3 single-quoted
    "(`[^`\\n]*`)",                                                      // 4 template
    "(\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?[fFdDlL]?\\b)",       // 5 number
    "([A-Za-z_$][\\w$]*)(?=\\s*\\()",                                    // 6 call
    "([A-Za-z_$][\\w$]*)",                                                  // 7 word
  ].join("|"), "g");

  return escaped.replace(scanner, (match, comment, dq, sq, tpl, num, call, word) => {
    if (comment) return `<span class="tok-com">${comment}</span>`;
    if (dq || sq || tpl) return `<span class="tok-str">${dq || sq || tpl}</span>`;
    if (num) return `<span class="tok-num">${num}</span>`;
    if (call) {
      return keywords.has(call) ? `<span class="tok-key">${call}</span>`
                                : `<span class="tok-fun">${call}</span>`;
    }
    if (word) {
      if (keywords.has(word)) return `<span class="tok-key">${word}</span>`;
      if (/^[A-Z][A-Za-z0-9_$]*$/.test(word)) return `<span class="tok-typ">${word}</span>`;
      return word;
    }
    return match;
  });
}

/* The model almost never writes fences, so a block of code arrives as an ordinary
   paragraph. This spots one by shape: braces and semicolons, indentation, or a line that
   opens a declaration in a language people ask about. It is deliberately conservative,
   because turning prose into a code block is worse than leaving code as prose. */
const CODE_OPENERS = /^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*(?:class|interface|enum|record)\s+[A-Z]|^\s*(?:def|function|func|fn)\s+\w|^\s*(?:import|package|#include|using)\s+|^\s*(?:const|let|var)\s+\w+\s*=|^\s*(?:if|for|while|switch)\s*\(.*\)\s*\{/;

function looksLikeCode(lines) {
  if (lines.length < 2) return false;
  const meaningful = lines.filter((line) => line.trim());
  if (meaningful.length < 2) return false;
  const opens = meaningful.some((line) => CODE_OPENERS.test(line));
  if (!opens) return false;
  // and at least a third of the lines carry code punctuation or are indented
  const codey = meaningful.filter((line) =>
    /[{};]\s*$/.test(line) || /^\s{2,}\S/.test(line) || /\w+\s*\([^)]*\)/.test(line)).length;
  return codey / meaningful.length >= 0.34;
}

function guessLanguage(lines) {
  const body = lines.join("\n");
  if (/\b(public|private)\s+(static\s+)?(void|class)\b|System\.out\./.test(body)) return "java";
  if (/^\s*def\s+\w+\s*\(|^\s*from\s+\w+\s+import\b|\bprint\(/m.test(body)) return "python";
  if (/#include\b|\bstd::/.test(body)) return "cpp";
  if (/\b(const|let)\s+\w+\s*=|=>|console\.log/.test(body)) return "javascript";
  if (/^\s*(SELECT|INSERT|UPDATE|CREATE)\b/im.test(body)) return "sql";
  return "code";
}

function codeBlockMarkup(body, language) {
  // a fence the model opens and never closes leaves an empty body, which rendered as
  // a blank slab under the real block
  const joined = body.join("\n").replace(/\s+$/, "");
  if (!joined.trim()) return "";
  const id = `code-${++codeBlockCounter}`;
  return `<div class="code-block"><div class="code-block-head">` +
         `<span class="lang-tag">${escapeHtml(language || "code")}</span>` +
         `<button data-copy-code="${id}"><svg class="icon icon-sm"><use href="#i-copy"/></svg>Copy</button></div>` +
         `<pre><code id="${id}" data-raw="${escapeHtml(joined)}">${highlightCode(joined, language)}</code></pre></div>`;
}

function renderAssistantContent(text, { streaming = false } = {}) {
  // Rendered as ordinary markdown: every reply is shown in full, exactly as the model
  // produced it. Nothing is hidden, folded away or routed into a separate panel.
  let html = renderMarkdown(text);
  if (streaming) html += '<span class="caret"></span>';
  return html;
}
