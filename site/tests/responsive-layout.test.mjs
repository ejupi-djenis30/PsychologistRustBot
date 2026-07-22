import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const siteRoot = new URL("../", import.meta.url);
const read = (relative) => readFile(new URL(relative, siteRoot), "utf8");

test("the hero links directly to the published v1.3.0 release", async () => {
  const html = await read("index.html");

  assert.match(
    html,
    /<a class="button button-release" href="https:\/\/github\.com\/ejupi-djenis30\/PsychologistRustBot\/releases\/tag\/v1\.3\.0">\s*Get the v1\.3\.0 release/s,
  );
});

test("all primary destinations remain visible in the CSS-only mobile navigation", async () => {
  const [html, styles] = await Promise.all([read("index.html"), read("styles.css")]);

  for (const [href, label] of [
    ["#experiment", "Inference"],
    ["#method", "Pipeline"],
    ["#open-set-v3", "Model report"],
    ["#safety", "Boundaries"],
  ]) {
    assert.ok(html.includes(`<a href="${href}">${label}</a>`));
  }

  assert.match(
    styles,
    /@media \(max-width: 960px\)[\s\S]*?\.site-header nav\s*\{[\s\S]*?display:\s*grid;[\s\S]*?grid-template-columns:\s*repeat\(4, minmax\(0, 1fr\)\);/,
  );
  assert.match(
    styles,
    /@media \(max-width: 620px\)[\s\S]*?\.site-header nav\s*\{[\s\S]*?grid-template-columns:\s*repeat\(2, minmax\(0, 1fr\)\);/,
  );
  assert.doesNotMatch(
    styles,
    /@media \(max-width: 960px\)[\s\S]*?\.site-header nav\s*\{[^}]*display:\s*none;/,
  );
});

test("the complete four-step protocol uses a shared mobile card floor at 320px", async () => {
  const [html, styles] = await Promise.all([read("index.html"), read("styles.css")]);
  const protocol = html.match(/<div class="v3-protocol"[\s\S]*?<\/div>/)?.[0] ?? "";

  assert.equal((protocol.match(/<article(?:\s|>)/g) ?? []).length, 4);
  for (const copy of [
    "315 grouped rows",
    "70 rows for model selection and thresholds",
    "70 rows for temperature",
    "70 untouched rows",
  ]) {
    assert.ok(protocol.includes(copy));
  }
  assert.match(
    styles,
    /@media \(max-width: 620px\)[\s\S]*?\.v3-protocol article\s*\{\s*min-block-size:\s*9\.5rem;/,
  );
});
