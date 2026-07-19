import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await readFile(path.join(siteRoot, "index.html"), "utf8");
const app = await readFile(path.join(siteRoot, "app.js"), "utf8");
const styles = await readFile(path.join(siteRoot, "styles.css"), "utf8");

assert.match(html, /<html lang="en">/);
assert.match(html, /<title>[^<]+<\/title>/);
assert.match(html, /name="description"/);
assert.match(html, /rel="canonical"/);
assert.match(html, /type="module" src="app\.js"/);
assert.doesNotMatch(html, /(?:src|href)="\//, "Assets must remain relative for project Pages");
assert.match(app, /"message-user"/, "The app must identify user messages");
assert.match(styles, /\.message-user\b/, "User messages must have a matching style");

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
