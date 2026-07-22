import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (relative) => readFile(new URL(relative, import.meta.url), "utf8");

test("the v3 protocol bundle keeps fit, calibration, threshold selection and tests separate", async () => {
  const [html, app, runtime, metricsText, policyText, manifestText] = await Promise.all([
    read("../index.html"),
    read("../app.js"),
    read("../open-set-engine.mjs"),
    read("../../artifacts/eliza-open-set-v3/metrics.json"),
    read("../../artifacts/eliza-open-set-v3/policy.json"),
    read("../../artifacts/eliza-open-set-v3/manifest.json"),
  ]);
  const metrics = JSON.parse(metricsText);
  const policy = JSON.parse(policyText);
  const manifest = JSON.parse(manifestText);

  assert.equal(metrics.schema_version, 3);
  assert.equal(policy.schema_version, 3);
  assert.equal(manifest.schema_version, 3);
  assert.equal(metrics.model_version, "3.0.0");
  assert.deepEqual(metrics.threshold_selection.inputs, ["development", "ood-development"]);
  assert.deepEqual(metrics.development_selection.inputs, ["train", "development"]);
  assert.equal(policy.temperature_source, "calibration-partition-temperature-scaling-v3");
  assert.equal(policy.threshold_source, "fixed-development-plus-ood-development-grid-v3");
  assert.equal(metrics.bootstrap_95.resamples, 1_000);
  assert.equal(
    metrics.bootstrap_95.strategy,
    "label-stratified-id-family-and-ood-domain-cluster-percentile-v3",
  );
  assert.equal(metrics.baselines.strategy, "training-only-majority-and-laplace-unigram-naive-bayes-v3");
  assert.equal(metrics.partition_counts["contrast-test"], 28);
  assert.equal(metrics.partition_family_counts["contrast-test"], 14);
  assert.equal(metrics.contrast_test.example_count, 28);
  assert.equal(metrics.contrast_test.pair_count, 14);
  assert.equal(metrics.contrast_test.predictions.length, 28);
  assert.deepEqual(Object.keys(metrics.ood_test.by_stratum).sort(), ["capability", "noise", "semantic"]);
  assert.match(app, /loadOpenSetBundle\("\.\/data\/open-set-v3"\)/);
  assert.doesNotMatch(app, /eliza-intent-v1\.json/);
  for (const name of ["manifest.json", "metrics.json", "model.json", "policy.json", "split-plan.json"]) {
    assert.match(runtime, new RegExp(name.replace(".", "\\.")));
  }
  assert.match(runtime, /subtle\.digest\("SHA-256"/);
  assert.match(html, /majority and unigram baselines/i);
  assert.match(html, /Contrast pair accuracy/i);
  assert.match(html, /not a production-language claim/i);
  assert.doesNotMatch(html, /production[- ]grade (?:model|classifier)/iu);
});

test("the v3 report exposes deterministic uncertainty intervals around point estimates", async () => {
  const metrics = JSON.parse(await read("../../artifacts/eliza-open-set-v3/metrics.json"));
  for (const [point, interval] of [
    [metrics.id_test.accuracy, metrics.bootstrap_95.id_accuracy],
    [metrics.id_test.macro_f1, metrics.bootstrap_95.id_macro_f1],
    [metrics.id_test.calibration.negative_log_likelihood, metrics.bootstrap_95.id_negative_log_likelihood],
    [metrics.id_test.calibration.multiclass_brier, metrics.bootstrap_95.id_multiclass_brier],
    [metrics.id_test.calibration.expected_calibration_error, metrics.bootstrap_95.id_expected_calibration_error],
    [metrics.id_test.aurc, metrics.bootstrap_95.id_aurc],
    [metrics.ood_test.discrimination.auroc, metrics.bootstrap_95.ood_auroc],
    [metrics.ood_test.discrimination.aupr_in_domain, metrics.bootstrap_95.ood_aupr_in_domain],
    [metrics.ood_test.discrimination.fpr_at_95_tpr, metrics.bootstrap_95.ood_fpr_at_95_tpr],
  ]) {
    assert.equal(interval.value, point);
    assert.ok(interval.lower_95 <= point);
    assert.ok(point <= interval.upper_95);
  }
});
