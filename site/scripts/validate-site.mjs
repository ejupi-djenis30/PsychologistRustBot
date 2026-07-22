import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(siteRoot, "..");
const html = await readFile(path.join(siteRoot, "index.html"), "utf8");
const app = await readFile(path.join(siteRoot, "app.js"), "utf8");
const styles = await readFile(path.join(siteRoot, "styles.css"), "utf8");
const modelBytes = await readFile(path.join(repositoryRoot, "models/eliza-intent-v1.json"));
const reportBytes = await readFile(path.join(repositoryRoot, "reports/eliza-intent-v1.json"));
const v2BundleRoot = path.join(repositoryRoot, "artifacts/eliza-open-set-v2");
const v2ManifestBytes = await readFile(path.join(v2BundleRoot, "manifest.json"));
const v2MetricsBytes = await readFile(path.join(v2BundleRoot, "metrics.json"));
const v2PolicyBytes = await readFile(path.join(v2BundleRoot, "policy.json"));
const model = JSON.parse(modelBytes);
const report = JSON.parse(reportBytes);
const v2Manifest = JSON.parse(v2ManifestBytes);
const v2Metrics = JSON.parse(v2MetricsBytes);
const v2Policy = JSON.parse(v2PolicyBytes);
const expectedCsp = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'none'; connect-src 'self'; worker-src 'none'; manifest-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'";
const socialPreviewUrl = "https://ejupi-djenis30.github.io/PsychologistRustBot/assets/social-preview.png";

assert.match(html, /<html lang="en">/);
assert.match(html, /<title>[^<]+<\/title>/);
assert.match(html, /name="description"/);
assert.match(html, /rel="canonical"/);
assert.match(html, /<meta name="referrer" content="no-referrer" \/>/);
assert.match(html, /http-equiv="Content-Security-Policy"/);
assert.ok(html.includes(`content="${expectedCsp}"`), "The CSP must allow only the static model fetch");
assert.doesNotMatch(html, /frame-ancestors/, "A meta CSP cannot enforce frame-ancestors");
assert.ok(html.includes(`property="og:image" content="${socialPreviewUrl}"`));
assert.match(html, /property="og:image:width" content="1200"/);
assert.match(html, /property="og:image:height" content="675"/);
assert.match(html, /property="og:image:alt"/);
assert.match(html, /name="twitter:card" content="summary_large_image"/);
assert.ok(html.includes(`name="twitter:image" content="${socialPreviewUrl}"`));
assert.match(html, /name="twitter:image:alt"/);
assert.match(html, /type="module" src="app\.js(?:\?[^"\s]+)?"/);
assert.match(app, /\.\/engine\.mjs\?v=[^"\s]+/);
assert.match(app, /\.\/ml-engine\.mjs\?v=[^"\s]+/);
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
assert.match(app, /boundedCharacters\(input\.value\)/, "The input must use the Unicode-aware limit");
assert.doesNotMatch(
  html,
  /\bmaxlength=/,
  "Native maxlength counts UTF-16 units; the code-point-aware JavaScript guard owns the limit",
);
assert.match(app, /for \(const character of value\)/, "The input limit must count code points");
assert.match(app, /message-safety/, "Safety exits must have a distinct accessible message");
assert.match(app, /\.\/data\/eliza-intent-v1\.json/, "The app must load the staged learned model");
assert.match(app, /\.\/data\/open-set-v2\/metrics\.json/, "The app must load the staged v2 report");
assert.match(app, /RULE FALLBACK/, "A model-load failure must be visible");
assert.match(app, /topFeatures/, "The browser trace must expose feature contributions");
assert.match(html, /class="lab-shell" aria-busy="true"/);
assert.match(html, /role="status" aria-live="polite" data-model-status/);
assert.match(html, /data-model-gated disabled/);
assert.match(app, /aria-busy", "false"/);
assert.match(app, /finally\s*\{\s*setModelReady\(\)/s);
assert.match(styles, /\.message-user\b/, "User messages must have a matching style");
assert.match(styles, /\.message-safety\b/, "Safety messages must have a matching style");
assert.match(styles, /\.pipeline-map\s*\{/, "The pipeline diagram must be code-built");
assert.match(styles, /\.v2-protocol\s*\{/, "The v2 protocol diagram must be code-built");
assert.match(styles, /\.skip-link\s*\{/, "The skip link must have a visible style");
assert.match(styles, /\.skip-link:focus-visible\s*\{/, "The skip link needs a keyboard-focus state");

assert.equal(model.schema_version, 1);
assert.equal(model.model_kind, "eliza-intent-softmax");
assert.equal(model.model_version, "1.0.0");
assert.equal(model.dataset_fingerprint, report.dataset_fingerprint);
assert.equal(report.total_examples, 112);
assert.equal(report.holdout_metrics.example_count, 21);
assert.equal(report.holdout_metrics.accuracy, 14 / 21);
assert.ok(Math.abs(report.holdout_metrics.macro_f1 - 0.661224489795918) < 1e-12);
assert.equal(report.holdout_metrics.coverage, 7 / 21);
assert.equal(report.holdout_metrics.selective_accuracy, 6 / 7);
assert.equal(report.ood_metrics.example_count, 20);
assert.equal(report.ood_metrics.accepted_examples, 0);
assert.equal(report.calibration.holdout_used_for_calibration, false);
assert.equal(model.training_config.thresholds.minimum_confidence, 0.45);
assert.equal(model.training_config.thresholds.minimum_margin, 0.2);

assert.equal(v2Manifest.schema_version, 2);
assert.equal(v2Manifest.bundle_kind, "eliza-open-set-bundle");
assert.equal(v2Manifest.model_version, "2.0.0");
assert.equal(v2Metrics.schema_version, 2);
assert.equal(v2Metrics.dataset_sha256, v2Manifest.dataset_sha256);
assert.equal(v2Metrics.split_plan_sha256, v2Manifest.split_plan_sha256);
assert.deepEqual(v2Metrics.partition_counts, {
  calibration: 14,
  development: 14,
  "id-test": 14,
  "ood-development": 20,
  "ood-test": 20,
  train: 70,
});
assert.equal(v2Metrics.threshold_selection.id_test_used, false);
assert.equal(v2Metrics.threshold_selection.ood_test_used, false);
assert.equal(v2Policy.temperature_source, "calibration-partition-temperature-scaling-v2");
assert.equal(v2Policy.threshold_source, "development-plus-ood-development-grid-v2");
assert.equal(v2Metrics.bootstrap_95.resamples, 1_000);
assert.equal(v2Metrics.id_test.example_count, 14);
assert.equal(v2Metrics.ood_test.example_count, 20);
assert.ok(v2Metrics.id_test.accuracy >= v2Metrics.bootstrap_95.id_accuracy.lower_95);
assert.ok(v2Metrics.id_test.accuracy <= v2Metrics.bootstrap_95.id_accuracy.upper_95);
assert.ok(v2Metrics.ood_test.discrimination.auroc >= v2Metrics.bootstrap_95.ood_auroc.lower_95);
assert.ok(v2Metrics.ood_test.discrimination.auroc <= v2Metrics.bootstrap_95.ood_auroc.upper_95);
assert.match(html, /No test row chooses a weight, temperature or threshold\./);
assert.match(html, /OOD FPR is still weak/i);

for (const [metric, visibleValue] of [
  ["holdout-size", "21"],
  ["holdout-accuracy", "14 / 21"],
  ["holdout-macro-f1", "0.661"],
  ["holdout-coverage", "7 / 21"],
  ["ood-accepted", "0 / 20"],
  ["seed", "20260722"],
]) {
  assert.ok(
    html.includes(`data-report-metric="${metric}">${visibleValue}</dd>`),
    `The visible ${metric} value must match the generated report`,
  );
}
assert.match(html, /holdout contains only 21 rows/i);
assert.match(html, /not a general-purpose NLP claim/i);

for (const file of [
  "app.js",
  "engine.mjs",
  "ml-engine.mjs",
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

const stageIndex = process.argv.indexOf("--stage");
if (stageIndex >= 0) {
  const stageArgument = process.argv[stageIndex + 1];
  assert.ok(stageArgument, "--stage requires a directory");
  const stage = path.resolve(repositoryRoot, stageArgument);
  assert.deepEqual(
    await readFile(path.join(stage, "data/eliza-intent-v1.json")),
    modelBytes,
    "The deployed model must be byte-identical to the checked-in artifact",
  );
  assert.deepEqual(
    await readFile(path.join(stage, "data/model-report-v1.json")),
    reportBytes,
    "The deployed report must be byte-identical to the checked-in report",
  );
  for (const file of ["manifest.json", "metrics.json", "model.json", "policy.json", "split-plan.json"]) {
    assert.deepEqual(
      await readFile(path.join(stage, "data/open-set-v2", file)),
      await readFile(path.join(v2BundleRoot, file)),
      `The deployed v2 ${file} must be byte-identical to the checked-in bundle`,
    );
  }
}

console.log("ELIZA Lab site and model-report validation passed.");
