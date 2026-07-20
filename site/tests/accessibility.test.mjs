import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const siteRoot = new URL("../", import.meta.url);

test("the skip link precedes the header and targets the focusable main landmark", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");
  const skipLink = '<a class="skip-link" href="#main-content">Skip to content</a>';

  assert.ok(html.includes(skipLink));
  assert.ok(html.includes('<main id="main-content" tabindex="-1">'));
  assert.ok(html.indexOf(skipLink) < html.indexOf('<header class="site-header">'));
});

test("the skip link exposes a visible keyboard state and a usable target size", async () => {
  const styles = await readFile(new URL("styles.css", siteRoot), "utf8");

  assert.match(styles, /\.skip-link\s*\{[^}]*min-height:\s*2\.75rem;/s);
  assert.match(styles, /\.skip-link:focus-visible\s*\{[^}]*transform:\s*translateY\(0\);/s);
});
