import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (relative) => readFile(new URL(relative, import.meta.url), "utf8");

test("the v2 report keeps fit, calibration, threshold selection and tests separate", async () => {
  const [html, app, metricsText, policyText, manifestText] = await Promise.all([
    read("../index.html"),
    read("../app.js"),
    read("../../artifacts/eliza-open-set-v2/metrics.json"),
    read("../../artifacts/eliza-open-set-v2/policy.json"),
    read("../../artifacts/eliza-open-set-v2/manifest.json"),
  ]);
  const metrics = JSON.parse(metricsText);
  const policy = JSON.parse(policyText);
  const manifest = JSON.parse(manifestText);

  assert.equal(metrics.schema_version, 2);
  assert.equal(policy.schema_version, 2);
  assert.equal(manifest.schema_version, 2);
  assert.equal(metrics.threshold_selection.id_test_used, false);
  assert.equal(metrics.threshold_selection.ood_test_used, false);
  assert.equal(policy.temperature_source, "calibration-partition-temperature-scaling-v2");
  assert.equal(policy.threshold_source, "development-plus-ood-development-grid-v2");
  assert.equal(metrics.bootstrap_95.resamples, 1_000);
  assert.match(app, /data\/open-set-v2\/metrics\.json/);
  assert.match(html, /OOD FPR is still weak/i);
  assert.match(html, /not a production-language claim/i);
  assert.doesNotMatch(html, /production[- ]grade (?:model|classifier)/iu);
});

test("the v2 report exposes deterministic uncertainty intervals around point estimates", async () => {
  const metrics = JSON.parse(await read("../../artifacts/eliza-open-set-v2/metrics.json"));
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
