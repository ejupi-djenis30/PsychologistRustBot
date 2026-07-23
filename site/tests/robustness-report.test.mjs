import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { verifyRobustnessBinding } from "../../scripts/verify-robustness-report.mjs";

const read = (relative) => readFile(new URL(relative, import.meta.url), "utf8");
const clone = (value) => structuredClone(value);

const [report, html, styles, manifest, policy, modelMetrics] = await Promise.all([
  read("../../scripts/tests/fixtures/robustness-report-v1.json").then(JSON.parse),
  read("../index.html"),
  read("../styles.css"),
  read("../../artifacts/eliza-open-set-v3/manifest.json").then(JSON.parse),
  read("../../artifacts/eliza-open-set-v3/policy.json").then(JSON.parse),
  read("../../artifacts/eliza-open-set-v3/metrics.json").then(JSON.parse),
]);

const verify = (overrides = {}) =>
  verifyRobustnessBinding({
    report,
    html,
    styles,
    manifest,
    policy,
    modelMetrics,
    ...overrides,
  });

test("the site metrics are derived from the complete bundle-bound audit report", () => {
  assert.deepEqual(verify(), {
    inputCount: 70,
    variantCount: 490,
    suiteVersion: "1.0.0",
  });
});

test("the verifier rejects forged provenance and inconsistent aggregate metrics", () => {
  const forged = clone(report);
  forged.population = "provided-cases";
  assert.throws(() => verify({ report: forged }), /verified bundle ID-test/u);

  const wrongDigest = clone(report);
  wrongDigest.split_plan_sha256 = "0".repeat(64);
  assert.throws(() => verify({ report: wrongDigest }), /split digest/u);

  const inconsistent = clone(report);
  inconsistent.typographic.label_flips += 1;
  assert.throws(() => verify({ report: inconsistent }), /label agreement/u);
});

test("the verifier reconstructs agreement, mean and maximum family aggregates", () => {
  const inconsistentDecision = clone(report);
  inconsistentDecision.perturbations[4].metrics.decision_agreement = 0.942857143;
  assert.throws(
    () => verify({ report: inconsistentDecision }),
    /typographic aggregate does not reconstruct decision_agreement/u,
  );

  const inconsistentMean = clone(report);
  inconsistentMean.typographic.mean_normalized_js_divergence += 0.001;
  assert.throws(
    () => verify({ report: inconsistentMean }),
    /typographic aggregate does not reconstruct mean_normalized_js_divergence/u,
  );

  const inconsistentMaximum = clone(report);
  inconsistentMaximum.typographic.maximum_absolute_confidence_delta -= 0.001;
  assert.throws(
    () => verify({ report: inconsistentMaximum }),
    /typographic aggregate does not reconstruct maximum_absolute_confidence_delta/u,
  );
});

test("the verifier rejects stale visible values and graph widths", () => {
  assert.throws(
    () => verify({ html: html.replace("95.714%", "95.700%") }),
    /typographic-decision-agreement/u,
  );
  assert.throws(
    () => verify({ styles: styles.replace("width: 7.3315%;", "width: 7.3000%;") }),
    /robustness-bar--drift/u,
  );
});

test("metric bindings reject nested markup instead of trying to sanitize it", () => {
  const injected = html.replace(
    ">100%</strong>",
    '><script type="text/plain">ignored</script>100%</strong>',
  );
  assert.throws(
    () => verify({ html: injected }),
    /must contain plain text only/u,
  );
});

test("CI and Pages verify the site against reports generated in the same job", async () => {
  const [ci, pages] = await Promise.all([
    read("../../.github/workflows/ci.yml"),
    read("../../.github/workflows/pages.yml"),
  ]);
  for (const [name, workflow, reportPath] of [
    ["CI", ci, "target/robustness-id-test-report.json"],
    ["Pages", pages, "target/pages-robustness-id-test-report.json"],
  ]) {
    const audit = workflow.indexOf("robustness audit --bundle-id-test");
    const output = workflow.indexOf(reportPath, audit);
    const verifier = workflow.indexOf("node scripts/verify-robustness-report.mjs", output);
    const verifiedInput = workflow.indexOf(reportPath, verifier);
    assert.ok(audit >= 0, `${name} must generate a robustness report`);
    assert.ok(output > audit, `${name} must write the generated report`);
    assert.ok(verifier > output, `${name} must verify after generation`);
    assert.ok(verifiedInput > verifier, `${name} must verify that same report path`);
  }
});
