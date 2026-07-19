import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await readFile(path.join(siteRoot, "index.html"), "utf8");

assert.match(html, /<html lang="en">/);
assert.match(html, /<title>[^<]+<\/title>/);
assert.match(html, /name="description"/);
assert.match(html, /rel="canonical"/);
assert.match(html, /type="module" src="app\.js"/);
assert.doesNotMatch(html, /(?:src|href)="\//, "Assets must remain relative for project Pages");

for (const file of [
  "app.js",
  "engine.mjs",
  "styles.css",
  "assets/eliza-lab-mark.svg",
  "assets/eliza-lab-lockup.svg",
  "assets/demo-poster.svg",
  "assets/demo.mp4",
]) {
  await access(path.join(siteRoot, file));
}

console.log("ELIZA Lab site validation passed.");
