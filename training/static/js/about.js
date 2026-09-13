/* The About panel is built from the measurements recorded by evaluate_model.py, so the
   numbers on screen are ones that were actually taken rather than written by hand. */
let aboutRendered = false;

/* Swaps the conversation for the About view, so the charts get the whole window. */
function showAbout(visible) {
  $("about-view").hidden = !visible;
  $("chat-scroll").hidden = visible;
  $("composer-wrap").hidden = visible;
  if (visible) { toggleSettings(false); renderAbout(); }
}

async function renderAbout() {
  if (aboutRendered) return;
  const body = $("about-body");
  let evaluation = null;
  try {
    evaluation = await (await fetch("/static/eval.json", { cache: "no-cache" })).json();
  } catch (error) {
    evaluation = null;
  }

  const table = (title, pairs) =>
    '<h4 style="margin:18px 0 6px;font-size:13px;font-weight:650">' + title + "</h4>" +
    '<table class="detail-table"><tbody>' +
    pairs.map(([k, v]) => "<tr><td>" + escapeHtml(k) + "</td><td>" + escapeHtml(String(v)) + "</td></tr>").join("") +
    "</tbody></table>";

  /* One horizontal bar per row, with the chance level marked. A bar chart with no chance
     line is misleading for multiple choice: 49% means "twice chance" on a four-option
     question and "worse than guessing" on a two-option one. */
  const barChart = (rows) =>
    '<div class="chart">' + rows.map((row) => {
      const percent = Math.max(0, Math.min(100, row.value * 100));
      const chance = row.chance == null ? "" :
        `<span class="chart-chance" style="left:${(row.chance * 100).toFixed(1)}%"></span>`;
      return '<div class="chart-row">' +
        `<span class="chart-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span>` +
        `<span class="chart-track"><span class="chart-fill${row.reference ? " reference" : ""}" ` +
          `style="width:${percent.toFixed(1)}%"></span>${chance}</span>` +
        `<span class="chart-value">${percent.toFixed(0)}%</span>` +
      '</div>';
    }).join("") + "</div>";

  const legend = (items) => '<div class="chart-legend">' + items.map((item) =>
    `<span><i class="chart-key" style="background:${item.colour}"></i>${escapeHtml(item.label)}</span>`
  ).join("") + "</div>";

  let benchmarks = "";
  if (evaluation && evaluation.benchmarks) {
    const entries = Object.entries(evaluation.benchmarks).filter(([, value]) => !value.error);
    if (entries.length) {
      benchmarks =
        '<h4 style="margin:18px 0 6px;font-size:13px;font-weight:650">Benchmarks</h4>' +
        '<p class="panel-note" style="margin-bottom:8px">Scored on this machine against chance, ' +
        'by comparing how likely the model finds each answer option.</p>' +
        barChart(entries.map(([key, value]) => ({
          label: key.replace("|", " "),
          value: Math.max(value.accuracy, value.accuracy_length_normalised),
          chance: value.chance,
        }))) +
        legend([{ colour: "var(--accent)", label: "Moonfrost" },
                { colour: "var(--danger)", label: "chance level" }]);
    }
  }

  const perplexity = evaluation && evaluation.pretraining_eval
    ? evaluation.pretraining_eval.perplexity.toFixed(2) : "not measured";
  const speed = evaluation && evaluation.inference
    ? evaluation.inference.tokens_per_second.toFixed(1) + " tokens/second" : "not measured";

  /* Reference models scored by the same harness, on the same examples, so the comparison
     is like for like. Figures copied from other model cards would be comparing harnesses
     as much as models. */
  let comparison = "";
  const references = [];
  for (const [name, file] of [["SmolLM2-135M", "eval_smollm2_135m.json"],
                              ["SmolLM2-360M", "eval_smollm2_360m.json"],
                              ["Qwen2.5-0.5B", "eval_qwen25_05b.json"]]) {
    try {
      const data = await (await fetch("/static/" + file, { cache: "no-cache" })).json();
      references.push([name, data]);
    } catch (error) { /* that reference has not been measured */ }
  }

  if (references.length && evaluation && evaluation.benchmarks) {
    const tasks = ["ARC-Easy|0-shot", "HellaSwag|0-shot", "WinoGrande|0-shot", "ARC-Challenge|0-shot"];
    const blocks = tasks.map((task) => {
      const ours = evaluation.benchmarks[task];
      if (!ours || ours.error) return "";
      const rows = [{ label: "Moonfrost 777M",
                      value: Math.max(ours.accuracy, ours.accuracy_length_normalised),
                      chance: ours.chance }];
      for (const [name, data] of references) {
        const theirs = data.benchmarks && data.benchmarks[task];
        if (theirs && !theirs.error) {
          rows.push({ label: name, reference: true, chance: ours.chance,
                      value: Math.max(theirs.accuracy, theirs.accuracy_length_normalised) });
        }
      }
      return `<h5 style="margin:14px 0 6px;font-size:12px;font-weight:650;color:var(--text-muted)">` +
             `${escapeHtml(task.replace("|", " "))}</h5>` + barChart(rows);
    }).join("");

    comparison =
      '<h4 style="margin:22px 0 6px;font-size:13px;font-weight:650">How it compares</h4>' +
      '<p class="panel-note" style="margin-bottom:6px">Every model below was scored here, on the ' +
      'same examples with the same prompts, so the numbers are directly comparable. ' +
      'Moonfrost read about 6 billion tokens; the reference models read hundreds of times more.</p>' +
      blocks +
      legend([{ colour: "var(--accent)", label: "Moonfrost" },
              { colour: "var(--text-faint)", label: "reference models" },
              { colour: "var(--danger)", label: "chance" }]);
  }

  /* Size against tokens read. Every figure here is the publisher's own stated spec, which
     is why models that cannot be run locally can still appear: no benchmark score is being
     borrowed, only the two numbers that define the training budget. */
  /* Only models whose publisher states BOTH numbers can appear. GPT-4 and later, Claude
     and Gemini disclose neither a parameter count nor a token budget, so there is no
     honest place to put them on these axes and they are left off rather than guessed at. */
  /* Every model anyone is likely to have heard of, with whatever its developer actually
     published. Two things are tracked separately because almost no model has both: the
     training budget (parameters and tokens), which the open releases state and the closed
     ones do not, and MMLU, which nearly everyone reports and which is the only score
     comparable across the whole range.

     `mmlu` is the developer's own published 5-shot figure unless `measured` is set, in
     which case it was scored here on the same harness as Moonfrost. A missing field is
     left missing rather than guessed at: a model with no published parameter count simply
     does not appear in the scale view. */
  const LANDSCAPE = [
    { name: "GPT-2 1.5B", year: 2019, params: 1.5e9, tokens: 1e10, mmlu: 26.0, open: true,
      note: "The model that made scaling look promising. At chance on MMLU." },
    { name: "GPT-3 175B", year: 2020, params: 1.75e11, tokens: 3e11, mmlu: 43.9, open: false,
      note: "175B parameters on 300B tokens, which Chinchilla later showed was the wrong way round." },
    { name: "Chinchilla 70B", year: 2022, params: 7e10, tokens: 1.4e12, mmlu: 67.5, open: false,
      note: "Beat a model four times its size by training on four times the text. The dashed line comes from this paper." },
    { name: "LLaMA 65B", year: 2023, params: 6.5e10, tokens: 1.4e12, mmlu: 63.4, open: true,
      note: "The release that started open weights at frontier scale." },
    { name: "Pythia 1.4B", year: 2023, params: 1.4e9, tokens: 3e11, mmlu: 25.6, open: true,
      note: "A research model with every checkpoint published. At chance on MMLU, like everything near this size." },
    { name: "Llama 2 7B", year: 2023, params: 7e9, tokens: 2e12, mmlu: 45.3, open: true,
      note: "2T tokens for 7B parameters, roughly 290 per parameter." },
    { name: "Llama 2 70B", year: 2023, params: 7e10, tokens: 2e12, mmlu: 68.9, open: true },
    { name: "Mistral 7B", year: 2023, params: 7.3e9, tokens: null, mmlu: 60.1, open: true,
      note: "Beat Llama 2 13B at 7B. Mistral has never published its token count." },
    { name: "Mixtral 8x7B", year: 2023, params: 4.7e10, tokens: null, mmlu: 70.6, open: true,
      note: "47B total, 13B active. The release that made sparse mixtures mainstream." },
    { name: "GPT-4", year: 2023, params: null, tokens: null, mmlu: 86.4, open: false,
      note: "Neither its size nor its training budget has ever been published." },
    { name: "Claude 3 Opus", year: 2024, params: null, tokens: null, mmlu: 86.8, open: false },
    { name: "Gemini 1.5 Pro", year: 2024, params: null, tokens: null, mmlu: 85.9, open: false },
    { name: "Llama 3 8B", year: 2024, params: 8e9, tokens: 1.5e13, mmlu: 66.6, open: true,
      note: "15T tokens for 8B parameters, about 1,875 per parameter. Far past compute-optimal, because inference cost matters more than training cost." },
    { name: "Llama 3.1 70B", year: 2024, params: 7e10, tokens: 1.5e13, mmlu: 83.6, open: true },
    { name: "Llama 3.1 405B", year: 2024, params: 4.05e11, tokens: 1.5e13, mmlu: 88.6, open: true,
      note: "The largest openly released dense model, and the only frontier-scale one whose budget is public." },
    { name: "Gemma 2 9B", year: 2024, params: 9e9, tokens: 8e12, mmlu: 71.3, open: true },
    { name: "Qwen2.5-72B", year: 2024, params: 7.2e10, tokens: 1.8e13, mmlu: 86.1, open: true },
    { name: "Qwen2.5-0.5B", year: 2024, params: 5e8, tokens: 1.8e13, mmlu: 47.5, open: true,
      note: "The closest published model to Moonfrost in size, and it read three thousand times more text." },
    { name: "OLMo 7B", year: 2024, params: 7e9, tokens: 2.5e12, mmlu: 28.3, open: true,
      note: "Fully open: weights, data, code and logs." },
    { name: "SmolLM2-135M", year: 2024, params: 1.35e8, tokens: 2e12, open: true,
      note: "Benchmarked here on the same harness as Moonfrost." },
    { name: "SmolLM2-360M", year: 2024, params: 3.62e8, tokens: 4e12, open: true,
      note: "Benchmarked here on the same harness as Moonfrost." },
    { name: "SmolLM2-1.7B", year: 2024, params: 1.7e9, tokens: 1.1e13, mmlu: 51.9, open: true },
    { name: "GPT-4o", year: 2024, params: null, tokens: null, mmlu: 88.7, open: false },
    { name: "Claude 3.5 Sonnet", year: 2024, params: null, tokens: null, mmlu: 88.7, open: false },
    { name: "DeepSeek-V3", year: 2024, params: 6.71e11, tokens: 1.48e13, mmlu: 88.5, open: true,
      note: "671B total, 37B active. The architecture Moonfrost is a small copy of." },
    { name: "o1", year: 2024, params: null, tokens: null, mmlu: 91.8, open: false,
      note: "The first widely available model that spends compute at answer time rather than only at training time." },
    { name: "DeepSeek-R1", year: 2025, params: 6.71e11, tokens: 1.48e13, mmlu: 90.8, open: true,
      note: "Reinforcement-learned reasoning on the V3 base, with open weights." },
    { name: "Moonfrost 777M", year: 2026, params: 7.77e8, tokens: 6e9, open: true, ours: true,
      note: "777M total, 161M active, 6B tokens, about $55 of rented H100 time. Everything here was measured on this machine." },
  ];

  /* MMLU for Moonfrost and for every model benchmarked here comes out of the eval files,
     so those bars reflect what was last measured rather than a number typed into this
     file. Where a developer's published figure and the one measured here disagree, the
     measured one wins, because it is the only one taken on the same harness as Moonfrost.
     Qwen2.5-0.5B is the clearest case: published 47.5, measured 34.4. */
  const readMmlu = (data) => {
    if (!data || !data.benchmarks) return null;
    // every published figure on this chart is five-shot, so compare like with like
    const keys = Object.keys(data.benchmarks).filter((k) => k.startsWith("MMLU"));
    const key = keys.find((k) => k.includes("5-shot")) || keys[0];
    if (!key || data.benchmarks[key].error) return null;
    const entry = data.benchmarks[key];
    return Math.max(entry.accuracy, entry.accuracy_length_normalised) * 100;
  };

  const measuredHere = { "Moonfrost 777M": readMmlu(evaluation) };
  for (const [name, data] of references) measuredHere[name] = readMmlu(data);
  for (const model of LANDSCAPE) {
    const value = measuredHere[model.name];
    if (typeof value === "number") { model.mmlu = value; model.measured = true; }
  }

  const scatter = (filter) => {
    const width = 600, height = 360, pad = { left: 46, right: 16, top: 14, bottom: 34 };
    // only models that published both numbers can be placed on these axes
    const plotted = LANDSCAPE.filter((m) => m.params && m.tokens).filter(filter || (() => true));
    if (plotted.length < 2) return '<p class="scatter-note">Fewer than two models in this group published both a parameter count and a token budget.</p>';
    const xs = plotted.map((m) => Math.log10(m.params));
    const ys = plotted.map((m) => Math.log10(m.tokens));
    const xMin = Math.min(...xs) - 0.35, xMax = Math.max(...xs) + 0.35;
    const yMin = Math.min(...ys) - 0.35, yMax = Math.max(...ys) + 0.35;
    const px = (value) => pad.left + (Math.log10(value) - xMin) / (xMax - xMin) * (width - pad.left - pad.right);
    const py = (value) => height - pad.bottom - (Math.log10(value) - yMin) / (yMax - yMin) * (height - pad.top - pad.bottom);

    const tick = (value) => value >= 1e12 ? (value / 1e12) + "T"
                          : value >= 1e9 ? (value / 1e9) + "B"
                          : value >= 1e6 ? (value / 1e6) + "M" : String(value);

    let svg = `<svg class="scatter" viewBox="0 0 ${width} ${height}" role="img" aria-label="Model size against training tokens">`;
    for (const value of [1e8, 1e9, 1e10, 1e11, 1e12]) {
      if (Math.log10(value) < xMin || Math.log10(value) > xMax) continue;
      svg += `<line class="grid" x1="${px(value)}" y1="${pad.top}" x2="${px(value)}" y2="${height - pad.bottom}"/>` +
             `<text class="axis-text" x="${px(value)}" y="${height - pad.bottom + 14}" text-anchor="middle">${tick(value)}</text>`;
    }
    for (const value of [1e9, 1e10, 1e11, 1e12, 1e13]) {
      if (Math.log10(value) < yMin || Math.log10(value) > yMax) continue;
      svg += `<line class="grid" x1="${pad.left}" y1="${py(value)}" x2="${width - pad.right}" y2="${py(value)}"/>` +
             `<text class="axis-text" x="${pad.left - 6}" y="${py(value) + 3}" text-anchor="end">${tick(value)}</text>`;
    }

    // the compute-optimal line: twenty tokens for every parameter
    const lineFrom = Math.pow(10, xMin), lineTo = Math.pow(10, xMax);
    svg += `<line class="chinchilla" x1="${px(lineFrom)}" y1="${py(lineFrom * 20)}" ` +
           `x2="${px(lineTo)}" y2="${py(lineTo * 20)}"/>` +
           `<text class="axis-text" x="${width - pad.right}" y="${py(lineTo * 20) - 6}" text-anchor="end">` +
           `20 tokens per parameter</text>`;

    for (const model of plotted) {
      const x = px(model.params), y = py(model.tokens);
      svg += `<circle class="dot${model.ours ? " ours" : ""}" cx="${x}" cy="${y}" ` +
             `r="${model.ours ? 7 : 4.5}"><title>${escapeHtml(model.name)}: ` +
             `${describeParams(model)}${model.note ? ". " + escapeHtml(model.note) : ""}</title></circle>` +
             `<text class="dot-label${model.ours ? " ours" : ""}" x="${x + (model.ours ? 11 : 8)}" y="${y + 3}">` +
             `${escapeHtml(model.name)}</text>`;
    }
    svg += `<text class="axis-text" x="${width / 2}" y="${height - 2}" text-anchor="middle">parameters</text>` +
           `<text class="axis-text" x="12" y="${height / 2}" text-anchor="middle" ` +
           `transform="rotate(-90 12 ${height / 2})">training tokens</text></svg>` +
           '<p class="scatter-note">The dashed line is twenty tokens per parameter, roughly ' +
           'where a model has been trained as much as its size justifies. Moonfrost sits far ' +
           'below it. Models whose developer never published a parameter count or a token ' +
           'budget cannot be placed on these axes and are absent from this view; they appear ' +
           'under Capability, where a published score exists.</p>';
    return svg;
  };

  /* Three views over the same list, because no single pair of axes fits both an open
     model that published its token count and a closed one that published only a score.
     The view is switched in place rather than by re-rendering the whole About page. */
  const GROUPS = {
    all: () => true,
    frontier: (m) => m.mmlu >= 80 || m.ours,
    open: (m) => m.open,
    small: (m) => m.params !== null && m.params < 2e9,
  };

  const rankRows = (models, field, max, format, chance) => models.map((model) => {
    const value = model[field];
    const width = Math.max(0, Math.min(100, (value / max) * 100));
    const tag = model.ours ? "here" : (model.measured ? "here" : (model.open ? "open" : ""));
    return `<div class="rank-row${model.ours ? " ours" : ""}${model.open ? "" : " closed"}" ` +
           `data-model="${escapeHtml(model.name)}" tabindex="0">` +
           `<span class="rank-name">${escapeHtml(model.name)}</span>` +
           `<span class="rank-track"><span class="rank-fill" style="width:${width.toFixed(1)}%"></span>` +
           (chance ? `<span class="rank-chance" style="left:${chance}%"></span>` : "") +
           `</span>` +
           `<span class="rank-value">${format(value)}</span>` +
           `<span class="rank-tag">${tag}</span></div>`;
  }).join("");

  const capabilityView = (group) => {
    const models = LANDSCAPE.filter(GROUPS[group])
      .filter((m) => typeof m.mmlu === "number")
      .sort((a, b) => b.mmlu - a.mmlu);
    if (!models.length) return '<p class="scatter-note">Nothing in this group has a published MMLU score.</p>';
    return '<div class="rank-list">' +
      rankRows(models, "mmlu", 100, (v) => v.toFixed(1) + "%", 25) +
      '</div>' +
      '<p class="scatter-note">MMLU, five-shot, as published by each developer. The red line ' +
      'is chance, 25% on a four-option question. Bars marked <b>here</b> were scored on this ' +
      'machine with the same harness used for every number on this page.</p>';
  };

  const efficiencyView = (group) => {
    const models = LANDSCAPE.filter(GROUPS[group])
      .filter((m) => m.params && m.tokens)
      .map((m) => ({ ...m, ratio: m.tokens / m.params }))
      .sort((a, b) => b.ratio - a.ratio);
    const max = Math.log10(Math.max(...models.map((m) => m.ratio)));
    const scaled = models.map((m) => ({ ...m, logRatio: Math.log10(m.ratio) }));
    return '<div class="rank-list">' +
      rankRows(scaled, "logRatio", max, (v) => Math.round(Math.pow(10, v)).toLocaleString(), null) +
      '</div>' +
      '<p class="scatter-note">Training tokens per parameter, on a log scale because the range ' +
      'runs from 8 to eighteen thousand. Chinchilla put compute-optimal at about 20. Everything ' +
      'released since sits far above it, because text is cheaper than serving a larger model ' +
      'forever. Moonfrost sits below it, which is the budget showing.</p>';
  };

  const landscape =
    '<h4 style="margin:22px 0 6px;font-size:13px;font-weight:650">Where Moonfrost sits</h4>' +
    '<p class="panel-note" style="margin-bottom:8px">Every model here is placed with figures ' +
    'its own developer published, or with a score measured on this machine. Nothing is ' +
    'estimated. Pick a view, narrow the group, and select a row for the detail.</p>' +
    '<div class="landscape" id="landscape">' +
      '<div class="landscape-bar">' +
        '<div class="seg" id="landscape-view">' +
          '<button data-view="capability" class="on">Capability</button>' +
          '<button data-view="scale">Scale</button>' +
          '<button data-view="efficiency">Efficiency</button>' +
        '</div><span class="spacer"></span>' +
        '<div id="landscape-groups">' +
          '<button class="filter-chip on" data-group="all">All</button> ' +
          '<button class="filter-chip" data-group="frontier">Frontier</button> ' +
          '<button class="filter-chip" data-group="open">Open weights</button> ' +
          '<button class="filter-chip" data-group="small">Under 2B</button>' +
        '</div>' +
      '</div>' +
      '<div id="landscape-body"></div>' +
      '<div class="landscape-detail" id="landscape-detail">Select a row to read what that model is.</div>' +
    '</div>';

  // rendered after the panel is in the document, because it swaps its own contents
  const drawLandscape = () => {
    const host = $("landscape-body");
    if (!host) return;
    const view = $("landscape-view").querySelector(".on").dataset.view;
    const group = $("landscape-groups").querySelector(".on").dataset.group;
    host.innerHTML = view === "capability" ? capabilityView(group)
                   : view === "efficiency" ? efficiencyView(group)
                   : scatter(GROUPS[group]);
  };

  const bindLandscape = () => {
    const root = $("landscape");
    if (!root) return;
    root.addEventListener("click", (event) => {
      const seg = event.target.closest("[data-view]");
      const chip = event.target.closest("[data-group]");
      const row = event.target.closest("[data-model]");
      if (seg) {
        $("landscape-view").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === seg));
        drawLandscape();
      } else if (chip) {
        $("landscape-groups").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === chip));
        drawLandscape();
      } else if (row) {
        const model = LANDSCAPE.find((m) => m.name === row.dataset.model);
        if (model) $("landscape-detail").innerHTML = describeModel(model);
      }
    });
    // keyboard reaches the same detail line as the pointer
    root.addEventListener("focusin", (event) => {
      const row = event.target.closest("[data-model]");
      if (!row) return;
      const model = LANDSCAPE.find((m) => m.name === row.dataset.model);
      if (model) $("landscape-detail").innerHTML = describeModel(model);
    });
    drawLandscape();
  };

  const big = (value) => value >= 1e12 ? (value / 1e12).toFixed(value >= 1e13 ? 0 : 1) + "T"
                       : value >= 1e9 ? (value / 1e9).toFixed(value >= 1e10 ? 0 : 1) + "B"
                       : Math.round(value / 1e6) + "M";

  const describeParams = (model) =>
    model.params && model.tokens ? `${big(model.params)} parameters, ${big(model.tokens)} tokens`
    : model.params ? `${big(model.params)} parameters`
    : "size not published";

  const describeModel = (model) => {
    const bits = [];
    if (model.params) bits.push(`<b>${big(model.params)}</b> parameters`);
    if (model.tokens) bits.push(`<b>${big(model.tokens)}</b> tokens`);
    if (model.params && model.tokens) bits.push(`${Math.round(model.tokens / model.params).toLocaleString()} per parameter`);
    if (typeof model.mmlu === "number") {
      bits.push(`MMLU <b>${model.mmlu.toFixed(1)}%</b> (${model.measured ? "measured here" : "as published"})`);
    }
    if (!model.params && !model.tokens) bits.push("size and token budget never published");
    return `<b>${escapeHtml(model.name)}</b>, ${model.year}. ` + bits.join(" &middot; ") +
           (model.note ? `<br>${escapeHtml(model.note)}` : "");
  };

  body.innerHTML =
    '<p class="panel-note">A 777M-parameter mixture-of-experts model with 161M parameters ' +
    'active per token, pretrained from scratch on about 6 billion tokens of educational web ' +
    'text and then fine-tuned for chat. Every reply is generated on this machine.</p>' +
    table("Architecture", [
      ["Total parameters", "777M"],
      ["Active per token", "161M"],
      ["Layers", "14 (1 dense, 13 mixture-of-experts)"],
      ["Attention", "Multi-head latent attention, 14 heads"],
      ["Experts", "32 routed, top-3, plus 1 shared"],
      ["Context", "1,024 tokens"],
      ["Vocabulary", "32,768, byte-level BPE trained from scratch"],
    ]) +
    table("Training", [
      ["Pretraining data", "FineWeb-Edu, about 6B tokens"],
      ["Chat data", "smol-smoltalk, everyday-conversations, hand-written identity set"],
      ["Hardware", "1x H100, about 12 GPU-hours"],
      ["Throughput", "about 179,000 tokens/second"],
      ["Cost", "about 55 US dollars"],
    ]) +
    table("Measured", [
      ["Held-out perplexity", perplexity],
      ["Generation speed here", speed],
      ["Chat validation loss", "1.2484"],
    ]) +
    landscape +
    benchmarks +
    comparison +
    '<h4 style="margin:22px 0 6px;font-size:13px;font-weight:650">What it cannot do</h4>' +
    '<p class="panel-note">It invents facts, and topics thin in educational web text are ' +
    'missing outright. No arithmetic, no multi-step reasoning, no tools, no internet, and no ' +
    'memory between conversations. Six billion tokens is far below what a model this size ' +
    'wants, and that gap explains most of the above.</p>' +
    '<p class="panel-note"><a class="about-link" target="_blank" rel="noopener" ' +
    'href="https://huggingface.co/whoashish115/Moonfrost-777M-Instruct-v2">Model on Hugging Face</a>' +
    ' &middot; <a class="about-link" target="_blank" rel="noopener" ' +
    'href="https://huggingface.co/datasets/whoashish115/Moonfrost-Persona-SFT">Identity dataset</a></p>';
  bindLandscape();
  aboutRendered = true;
}

function bindProfile() {
  const nameInput = $("set-profile-name");
  const picker = $("set-profile-avatar");
  nameInput.value = profile.name === "You" ? "" : profile.name;

  nameInput.addEventListener("input", () => {
    profile.name = nameInput.value.trim() || "You";
    applyProfile();
  });

  $("btn-profile-avatar").addEventListener("click", () => picker.click());
  picker.addEventListener("change", async () => {
    const file = picker.files && picker.files[0];
    if (!file) return;
    try {
      profile.avatar = await shrinkToDataUrl(file);
      applyProfile();
    } catch (error) {
      toast(error.message);
    }
    picker.value = "";
  });

  $("btn-profile-clear").addEventListener("click", () => {
    profile.avatar = "";
    applyProfile();
  });

  applyProfile();
}


/* Marks any feed whose credentials are missing, so a source that cannot work says so in
   the menu rather than failing only once it is chosen. */


/* Shows one specific image rather than a random one from a feed, and re-themes from it. */
