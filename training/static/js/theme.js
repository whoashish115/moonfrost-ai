/* HSL conversion, used to derive hover and soft shades from whatever accent is chosen.
   These went out with the wallpaper code by accident, and applyAppearance() threw on the
   first line of init as a result, which stopped everything after it from running. */
function hexToRgb(hex) {
  const value = String(hex || "").replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  return [0, 2, 4].map((offset) => parseInt(full.slice(offset, offset + 2), 16) || 0);
}

function rgbToHsl([red, green, blue]) {
  const r = red / 255, g = green / 255, b = blue / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const lightness = (max + min) / 2;
  if (max === min) return [0, 0, lightness];
  const delta = max - min;
  const saturation = lightness > 0.5 ? delta / (2 - max - min) : delta / (max + min);
  let hue;
  if (max === r) hue = (g - b) / delta + (g < b ? 6 : 0);
  else if (max === g) hue = (b - r) / delta + 2;
  else hue = (r - g) / delta + 4;
  return [hue * 60, saturation, lightness];
}

function hslToHex(hue, saturation, lightness) {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const second = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
  const match = lightness - chroma / 2;
  const sector = Math.floor((((hue % 360) + 360) % 360) / 60);
  const table = [[chroma, second, 0], [second, chroma, 0], [0, chroma, second],
                 [0, second, chroma], [second, 0, chroma], [chroma, 0, second]][sector];
  return "#" + table.map((channel) =>
    Math.round((channel + match) * 255).toString(16).padStart(2, "0")).join("");
}

/* The colour roles the interface is built from. Every rule uses these variables, so a
   theme is just a set of values for them and nothing needs to know a theme exists. */
const COLOUR_ROLES = [
  ["bg", "Page"],
  ["bg-elevated", "Panels"],
  ["bg-sunken", "Inputs and bubbles"],
  ["sidebar-bg", "Sidebar"],
  ["border", "Borders"],
  ["text", "Text"],
  ["text-muted", "Secondary text"],
  ["accent", "Accent"],
  ["accent-2", "Accent, second stop"],
];

const THEME_PRESETS = {
  moonfrost: {
    label: "Moonfrost",
    // Neutral greys throughout, with the primary the only colour in the interface. The
    // surfaces used to carry a trace of the accent's hue, which tinted the whole app.
    // Neutral greys, eased off both extremes: pure white glared and near-black crushed the
    // separation between surfaces. The primary is the only colour in the interface.
    light: { bg: "#fafafa", "bg-elevated": "#ffffff", "bg-sunken": "#f0f0f1", "sidebar-bg": "#f3f3f4",
             border: "#e4e4e7", text: "#26262a", "text-muted": "#6b6b74",
             accent: "#c2428f", "accent-2": "#c2428f" },
    dark:  { bg: "#1c1c1f", "bg-elevated": "#26262a", "bg-sunken": "#2f2f34", "sidebar-bg": "#212125",
             border: "#3a3a40", text: "#ececee", "text-muted": "#a8a8b0",
             accent: "#d770ae", "accent-2": "#d770ae" },
  },
  slate: {
    label: "Slate",
    light: { bg: "#fbfbfc", "bg-elevated": "#ffffff", "bg-sunken": "#f1f2f4", "sidebar-bg": "#e9eaee",
             border: "#e2e3e8", text: "#1c1d21", "text-muted": "#5d6068", accent: "#3f6ae0" },
    dark:  { bg: "#15171b", "bg-elevated": "#1f2228", "bg-sunken": "#272b32", "sidebar-bg": "#1a1d22",
             border: "#31353d", text: "#f2f3f5", "text-muted": "#a3a8b2", accent: "#6f95ff" },
  },
  ember: {
    label: "Ember",
    light: { bg: "#fffcfa", "bg-elevated": "#ffffff", "bg-sunken": "#f7efe9", "sidebar-bg": "#f2e6dd",
             border: "#eddfd4", text: "#2a1f19", "text-muted": "#6d5b4f", accent: "#e2562b" },
    dark:  { bg: "#17120f", "bg-elevated": "#221b17", "bg-sunken": "#2a221d", "sidebar-bg": "#1c1613",
             border: "#372c25", text: "#f9f2ee", "text-muted": "#bcaba0", accent: "#ff7a4d" },
  },
  forest: {
    label: "Forest",
    light: { bg: "#fbfdfb", "bg-elevated": "#ffffff", "bg-sunken": "#eef4ef", "sidebar-bg": "#e3ece5",
             border: "#dde7df", text: "#1a241c", "text-muted": "#566259", accent: "#2f8f5b" },
    dark:  { bg: "#111714", "bg-elevated": "#1a221d", "bg-sunken": "#212b25", "sidebar-bg": "#151c18",
             border: "#2b382f", text: "#eff5f1", "text-muted": "#a3b3a8", accent: "#46c882" },
  },
  mono: {
    label: "Mono",
    light: { bg: "#ffffff", "bg-elevated": "#ffffff", "bg-sunken": "#f2f2f2", "sidebar-bg": "#eaeaea",
             border: "#e2e2e2", text: "#141414", "text-muted": "#5e5e5e",
             accent: "#2b2b2b", "accent-2": "#525252" },
    dark:  { bg: "#121212", "bg-elevated": "#1c1c1c", "bg-sunken": "#242424", "sidebar-bg": "#171717",
             border: "#2e2e2e", text: "#f0f0f0", "text-muted": "#a8a8a8",
             accent: "#e8e8e8", "accent-2": "#b4b4b4" },
  },
  purple: {
    label: "Purple",
    light: { bg: "#fdfcff", "bg-elevated": "#ffffff", "bg-sunken": "#f3f1f7", "sidebar-bg": "#efecf5",
             border: "#e6e2ee", text: "#221d2b", "text-muted": "#635b70",
             accent: "#7c3aed", "accent-2": "#7c3aed" },
    dark:  { bg: "#17151c", "bg-elevated": "#211d29", "bg-sunken": "#2a2533", "sidebar-bg": "#1b1822",
             border: "#342e3f", text: "#f1eef5", "text-muted": "#aaa1b6",
             accent: "#a78bfa", "accent-2": "#a78bfa" },
  },
  indigo: {
    label: "Indigo",
    light: { bg: "#fcfcff", "bg-elevated": "#ffffff", "bg-sunken": "#f0f1f7", "sidebar-bg": "#eaecf5",
             border: "#e1e3ee", text: "#1b1d2b", "text-muted": "#5b5f70",
             accent: "#4f46e5", "accent-2": "#4f46e5" },
    dark:  { bg: "#15161d", "bg-elevated": "#1f212a", "bg-sunken": "#282a35", "sidebar-bg": "#191b23",
             border: "#313442", text: "#eeeff5", "text-muted": "#a2a6b8",
             accent: "#818cf8", "accent-2": "#818cf8" },
  },
  teal: {
    label: "Teal",
    light: { bg: "#fbfefe", "bg-elevated": "#ffffff", "bg-sunken": "#eef5f5", "sidebar-bg": "#e6efef",
             border: "#dde9e9", text: "#162422", "text-muted": "#54635f",
             accent: "#0d9488", "accent-2": "#0d9488" },
    dark:  { bg: "#111817", "bg-elevated": "#1a2321", "bg-sunken": "#222d2b", "sidebar-bg": "#151d1c",
             border: "#2b3835", text: "#eef5f4", "text-muted": "#a0b2ae",
             accent: "#2dd4bf", "accent-2": "#2dd4bf" },
  },
  amber: {
    label: "Amber",
    light: { bg: "#fffdf9", "bg-elevated": "#ffffff", "bg-sunken": "#f7f3ea", "sidebar-bg": "#f2ece0",
             border: "#ece4d5", text: "#2a2318", "text-muted": "#6b6151",
             accent: "#b45309", "accent-2": "#b45309" },
    dark:  { bg: "#18150f", "bg-elevated": "#231e16", "bg-sunken": "#2c261c", "sidebar-bg": "#1c1812",
             border: "#382f22", text: "#f6f1e8", "text-muted": "#b8ac9a",
             accent: "#fbbf24", "accent-2": "#fbbf24" },
  },
  rose: {
    label: "Rose",
    light: { bg: "#fffcfd", "bg-elevated": "#ffffff", "bg-sunken": "#f8f0f3", "sidebar-bg": "#f3e8ec",
             border: "#eee0e5", text: "#2a1b20", "text-muted": "#6d5a60",
             accent: "#e11d48", "accent-2": "#e11d48" },
    dark:  { bg: "#181214", "bg-elevated": "#231b1e", "bg-sunken": "#2c2226", "sidebar-bg": "#1c1517",
             border: "#38292e", text: "#f6eef1", "text-muted": "#bda7ae",
             accent: "#fb7185", "accent-2": "#fb7185" },
  },
  ocean: {
    label: "Ocean",
    light: { bg: "#fbfdff", "bg-elevated": "#ffffff", "bg-sunken": "#eef4fa", "sidebar-bg": "#e6eef7",
             border: "#dde8f2", text: "#152029", "text-muted": "#526270",
             accent: "#0369a1", "accent-2": "#0369a1" },
    dark:  { bg: "#101619", "bg-elevated": "#192227", "bg-sunken": "#212c32", "sidebar-bg": "#141b1f",
             border: "#2a373f", text: "#edf3f7", "text-muted": "#9fb0bb",
             accent: "#38bdf8", "accent-2": "#38bdf8" },
  },
  sand: {
    label: "Sand",
    light: { bg: "#fdfcfa", "bg-elevated": "#ffffff", "bg-sunken": "#f4f1ec", "sidebar-bg": "#edeae3",
             border: "#e6e2da", text: "#242019", "text-muted": "#655f54",
             accent: "#8a6d3b", "accent-2": "#8a6d3b" },
    dark:  { bg: "#161513", "bg-elevated": "#201e1a", "bg-sunken": "#292620", "sidebar-bg": "#1a1815",
             border: "#343029", text: "#f3f1ec", "text-muted": "#b2aca0",
             accent: "#d6b47c", "accent-2": "#d6b47c" },
  },
  lime: {
    label: "Lime",
    light: { bg: "#fcfefa", "bg-elevated": "#ffffff", "bg-sunken": "#f1f6ea", "sidebar-bg": "#e9f0e0",
             border: "#e0e9d5", text: "#1c2416", "text-muted": "#5a6450",
             accent: "#4d7c0f", "accent-2": "#4d7c0f" },
    dark:  { bg: "#141711", "bg-elevated": "#1d2119", "bg-sunken": "#252b1f", "sidebar-bg": "#171b14",
             border: "#2f3726", text: "#f0f4ea", "text-muted": "#a9b39c",
             accent: "#a3e635", "accent-2": "#a3e635" },
  },
  crimson: {
    label: "Crimson",
    light: { bg: "#fffcfc", "bg-elevated": "#ffffff", "bg-sunken": "#f8f0f0", "sidebar-bg": "#f3e8e8",
             border: "#eee0e0", text: "#291b1b", "text-muted": "#6c5a5a",
             accent: "#be123c", "accent-2": "#be123c" },
    dark:  { bg: "#181313", "bg-elevated": "#231c1c", "bg-sunken": "#2c2323", "sidebar-bg": "#1c1616",
             border: "#382b2b", text: "#f6efef", "text-muted": "#bda9a9",
             accent: "#f43f5e", "accent-2": "#f43f5e" },
  },
  midnight: {
    label: "Midnight",
    light: { bg: "#f7f8fa", "bg-elevated": "#ffffff", "bg-sunken": "#eceef3", "sidebar-bg": "#e4e7ee",
             border: "#dcdfe8", text: "#161a22", "text-muted": "#555c6a",
             accent: "#1e40af", "accent-2": "#1e40af" },
    dark:  { bg: "#0e1017", "bg-elevated": "#171a23", "bg-sunken": "#1f232e", "sidebar-bg": "#12141b",
             border: "#272c38", text: "#eceef4", "text-muted": "#9ba3b4",
             accent: "#60a5fa", "accent-2": "#60a5fa" },
  },
};

// off by default: the panel is useful once, and covers the conversation after that
const APPEARANCE_DEFAULTS = { preset: "moonfrost", contrast: "normal", suggestions: false,
                             light: {}, dark: {} };
let appearance = { ...APPEARANCE_DEFAULTS };

function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/* How far each role moves at low and high contrast, as a shift in HSL lightness. Surfaces
   move toward the page colour on low and away from it on high; text does the opposite, so
   the gap between a panel and the words on it widens or narrows together. */
const CONTRAST_SHIFTS = {
  low:    { bg: 0, "bg-elevated": -0.02, "bg-sunken": -0.02, "sidebar-bg": -0.015,
            border: -0.03, text: -0.10, "text-muted": -0.06 },
  normal: {},
  high:   { bg: 0, "bg-elevated": 0.035, "bg-sunken": 0.035, "sidebar-bg": 0.025,
            border: 0.06, text: 0.10, "text-muted": 0.08 },
};

function shiftLightness(hex, amount, theme) {
  if (!amount) return hex;
  const [hue, saturation, lightness] = rgbToHsl(hexToRgb(hex));
  // "away from the page" means lighter in dark mode and darker in light mode
  const direction = theme === "dark" ? 1 : -1;
  return hslToHex(hue, saturation, Math.max(0.02, Math.min(0.98, lightness + amount * direction)));
}

/* The preset supplies a full palette; anything the reader has changed overrides it; the
   contrast setting is applied last, so it works on a custom colour as well as a preset. */
function paletteFor(theme) {
  const preset = THEME_PRESETS[appearance.preset] || THEME_PRESETS.moonfrost;
  const palette = { ...preset[theme], ...(appearance[theme] || {}) };
  const shifts = CONTRAST_SHIFTS[appearance.contrast] || {};
  const adjusted = { ...palette };
  for (const [role, amount] of Object.entries(shifts)) {
    if (!palette[role]) continue;
    // text moves the other way: lighter text on a dark page is more contrast, not less
    const towardsText = role.startsWith("text");
    adjusted[role] = shiftLightness(palette[role], towardsText ? amount : amount, theme);
  }
  return adjusted;
}

function applyAppearance() {
  const root = document.documentElement;
  const theme = currentTheme();
  const palette = paletteFor(theme);
  for (const [role] of COLOUR_ROLES) {
    if (palette[role]) root.style.setProperty("--" + role, palette[role]);
  }
  // the shades derived from the accent, so hover states follow a custom colour too
  const accent = palette.accent || "#5b5bd6";
  const [hue, saturation, lightness] = rgbToHsl(hexToRgb(accent));
  root.style.setProperty("--accent-hover",
    hslToHex(hue, saturation, theme === "dark" ? Math.min(0.82, lightness + 0.08)
                                               : Math.max(0.12, lightness - 0.08)));
  // Kept nearly colourless: this paints selected rows and chips, and at any real
  // saturation it spread the primary's hue across the whole sidebar.
  root.style.setProperty("--accent-soft", theme === "dark"
    ? `hsl(${hue.toFixed(0)} 12% 18%)`
    : `hsl(${hue.toFixed(0)} 16% 95%)`);
  root.style.setProperty("--bg-rgb", hexToRgb(palette.bg || "#ffffff").join(", "));

  // A preset whose two stops are the same colour, which is the default, produces a flat
  // fill; setting them apart in the colour pickers turns it back into a gradient. The
  // variable stays a gradient either way so nothing downstream needs to care.
  const second = palette["accent-2"] || accent;
  root.style.setProperty("--accent-2", second);
  root.style.setProperty("--accent-gradient", `linear-gradient(135deg, ${accent}, ${second})`);

  // White text on a white accent is invisible, which is exactly what the Mono theme did in
  // dark mode. The label colour follows the accent's luminance instead of being assumed.
  const luminance = ([r, g, b]) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  const brightness = (luminance(hexToRgb(accent)) + luminance(hexToRgb(second))) / 2;
  root.style.setProperty("--on-accent", brightness > 0.6 ? "#101014" : "#ffffff");

  // A neutral hover: one step lighter than the panel in dark, one step darker in light.
  const [bgH, bgS, bgL] = rgbToHsl(hexToRgb(palette["bg-elevated"] || palette.bg || "#ffffff"));
  root.style.setProperty("--hover-surface",
    hslToHex(bgH, bgS, theme === "dark" ? Math.min(0.22, bgL + 0.06) : Math.max(0.88, bgL - 0.05)));

  writeStored(APPEARANCE_STORAGE_KEY, appearance);
}

function renderColourGrid() {
  const grid = $("colour-grid");
  if (!grid) return;
  const palette = paletteFor(currentTheme());
  grid.innerHTML = COLOUR_ROLES.map(([role, label]) =>
    `<label class="colour-row">
       <input type="color" data-role="${role}" value="${palette[role] || "#000000"}">
       <span>${label}</span>
     </label>`).join("");
}

function bindAppearance() {
  const presets = $("set-theme-preset");
  presets.innerHTML = Object.entries(THEME_PRESETS)
    .map(([key, value]) => `<option value="${key}">${value.label}</option>`).join("")
    + '<option value="custom">Custom</option>';
  presets.value = appearance.preset;

  const suggestions = $("set-suggestions");
  suggestions.checked = appearance.suggestions !== false;
  suggestions.addEventListener("change", () => {
    appearance.suggestions = suggestions.checked;
    writeStored(APPEARANCE_STORAGE_KEY, appearance);
    renderSuggestions();
  });

  const contrast = $("set-contrast");
  contrast.value = appearance.contrast || "normal";
  contrast.addEventListener("change", () => {
    appearance.contrast = contrast.value;
    applyAppearance();
    renderColourGrid();
  });

  presets.addEventListener("change", () => {
    if (presets.value === "custom") return;
    appearance.preset = presets.value;
    appearance.light = {};
    appearance.dark = {};      // a preset replaces the palette rather than layering on it
    applyAppearance();
    renderColourGrid();
  });

  $("colour-grid").addEventListener("input", (event) => {
    const picker = event.target.closest("input[type=color]");
    if (!picker) return;
    const theme = currentTheme();
    appearance[theme] = { ...(appearance[theme] || {}), [picker.dataset.role]: picker.value };
    applyAppearance();
  });

  $("btn-reset-colours").addEventListener("click", () => {
    appearance[currentTheme()] = {};
    applyAppearance();
    renderColourGrid();
  });

  renderColourGrid();
}
