import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = await readFile(path.join(siteRoot, "index.html"), "utf8");
const app = await readFile(path.join(siteRoot, "app.js"), "utf8");
const styles = await readFile(path.join(siteRoot, "styles.css"), "utf8");
const expectedCsp = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'none'; connect-src 'none'; worker-src 'none'; manifest-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'";
const socialPreviewUrl = "https://ejupi-djenis30.github.io/PsychologistRustBot/assets/social-preview.png";

assert.match(html, /<html lang="en">/);
assert.match(html, /<title>[^<]+<\/title>/);
assert.match(html, /name="description"/);
assert.match(html, /rel="canonical"/);
assert.match(html, /<meta name="referrer" content="no-referrer" \/>/);
assert.match(html, /http-equiv="Content-Security-Policy"/);
assert.ok(html.includes(`content="${expectedCsp}"`), "The static CSP must match the site's required assets");
assert.doesNotMatch(html, /frame-ancestors/, "A meta CSP cannot enforce frame-ancestors");
assert.ok(html.includes(`property="og:image" content="${socialPreviewUrl}"`));
assert.match(html, /property="og:image:width" content="1200"/);
assert.match(html, /property="og:image:height" content="675"/);
assert.match(html, /property="og:image:alt"/);
assert.match(html, /name="twitter:card" content="summary_large_image"/);
assert.ok(html.includes(`name="twitter:image" content="${socialPreviewUrl}"`));
assert.match(html, /name="twitter:image:alt"/);
assert.match(html, /type="module" src="app\.js(?:\?[^"\s]+)?"/);
assert.doesNotMatch(html, /(?:src|href)="\//, "Assets must remain relative for project Pages");
assert.ok(
  html.includes(
    '<a href="https://github.com/ejupi-djenis30/PsychologistRustBot">ELIZA Lab contributors ↗</a>',
  ),
  "The footer must use collective project attribution.",
);
assert.doesNotMatch(html, /Djenis\s+Ejupi/iu, "The public site must not expose a personal byline.");
const skipLink = '<a class="skip-link" href="#main-content">Skip to content</a>';
assert.ok(html.includes(skipLink), "The site must expose a skip link.");
assert.ok(
  html.includes('<main id="main-content" tabindex="-1">'),
  "The skip-link target must be the focusable main landmark.",
);
assert.ok(
  html.indexOf(skipLink) < html.indexOf('<header class="site-header">'),
  "The skip link must appear before the repeated header.",
);
assert.match(app, /"message-user"/, "The app must identify user messages");
assert.match(app, /MAX_TRANSCRIPT_MESSAGES = 80/, "The transcript must remain bounded");
assert.match(app, /boundedCharacters\(input\.value\)/, "The input must use the engine's Unicode-aware limit");
assert.match(html, /maxlength="512"/, "The browser limit must match MAX_INPUT_CHARS");
assert.doesNotMatch(html, /maxlength="2048"/, "The retired input limit must not return");
assert.match(app, /message-safety/, "Safety exits must have a distinct accessible message");
assert.match(styles, /\.message-user\b/, "User messages must have a matching style");
assert.match(styles, /\.message-safety\b/, "Safety messages must have a matching style");
assert.match(styles, /\.skip-link\s*\{/, "The skip link must have a visible style");
assert.match(styles, /\.skip-link:focus-visible\s*\{/, "The skip link needs a keyboard-focus state");

for (const file of [
  "app.js",
  "engine.mjs",
  "styles.css",
  "assets/eliza-lab-mark.svg",
  "assets/eliza-lab-lockup.svg",
  "assets/social-preview.png",
]) {
  await access(path.join(siteRoot, file));
}

const socialPreview = await readFile(path.join(siteRoot, "assets/social-preview.png"));
assert.equal(socialPreview.subarray(0, 8).toString("hex"), "89504e470d0a1a0a", "Social preview must be PNG");
assert.equal(socialPreview.readUInt32BE(16), 1_200, "Social preview width must be 1200 pixels");
assert.equal(socialPreview.readUInt32BE(20), 675, "Social preview height must be 675 pixels");

console.log("ELIZA Lab site validation passed.");
