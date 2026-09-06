/* getElementById, shortened. It sat in the block that was removed with the wallpaper
   feature, and every function in the file uses it, so init threw on its first line. */
const $ = (id) => document.getElementById(id);

function readStored(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (e) { return fallback; }
}
function writeStored(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* not essential */ }
}
