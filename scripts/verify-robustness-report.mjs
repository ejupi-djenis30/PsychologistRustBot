import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const reportingScale = 1_000_000_000;

const invariant = (condition, message) => {
  if (!condition) throw new Error(message);
};

const quantize = (value) => Math.round(value * reportingScale) / reportingScale;
const isUnit = (value) => Number.isFinite(value) && value >= 0 && value <= 1;
const close = (left, right) => Math.abs(left - right) <= 2 / reportingScale;
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");

const exactKeys = (value, expected, label) => {
  invariant(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  invariant(
    JSON.stringify(actual) === JSON.stringify(wanted),
    `${label} fields do not match the robustness schema`,
  );
};

const reconstructAgreementCount = (slice, field, label) => {
  if (slice.evaluated_variants === 0) {
    invariant(slice[field] === 0, `${label}.${field} must be zero without evaluated variants`);
    return 0;
  }
  const count = Math.round(slice[field] * slice.evaluated_variants);
  invariant(
    close(slice[field], quantize(count / slice.evaluated_variants)),
    `${label}.${field} does not resolve to an integer agreement count`,
  );
  return count;
};

const validateSlice = (slice, label) => {
  exactKeys(
    slice,
    [
      "evaluated_variants",
      "skipped_applications",
      "label_agreement",
      "acceptance_agreement",
      "decision_agreement",
      "label_flips",
      "acceptance_flips",
      "accepted_to_abstained",
      "abstained_to_accepted",
      "mean_normalized_js_divergence",
      "maximum_normalized_js_divergence",
      "mean_absolute_confidence_delta",
      "maximum_absolute_confidence_delta",
    ],
    label,
  );
  for (const field of [
    "evaluated_variants",
    "skipped_applications",
    "label_flips",
    "acceptance_flips",
    "accepted_to_abstained",
    "abstained_to_accepted",
  ]) {
    invariant(Number.isSafeInteger(slice[field]) && slice[field] >= 0, `${label}.${field} is invalid`);
  }
  for (const field of [
    "label_agreement",
    "acceptance_agreement",
    "decision_agreement",
    "mean_normalized_js_divergence",
    "maximum_normalized_js_divergence",
    "mean_absolute_confidence_delta",
    "maximum_absolute_confidence_delta",
  ]) {
    invariant(isUnit(slice[field]), `${label}.${field} is outside [0, 1]`);
  }
  invariant(
    slice.label_flips <= slice.evaluated_variants &&
      slice.acceptance_flips <= slice.evaluated_variants,
    `${label} flip counts exceed its evaluated variants`,
  );
  invariant(
    slice.accepted_to_abstained + slice.abstained_to_accepted === slice.acceptance_flips,
    `${label} acceptance transitions do not reconstruct its flip count`,
  );
  invariant(
    reconstructAgreementCount(slice, "label_agreement", label) ===
      slice.evaluated_variants - slice.label_flips,
    `${label} label agreement does not match its flip count`,
  );
  invariant(
    reconstructAgreementCount(slice, "acceptance_agreement", label) ===
      slice.evaluated_variants - slice.acceptance_flips,
    `${label} acceptance agreement does not match its flip count`,
  );
  reconstructAgreementCount(slice, "decision_agreement", label);
  invariant(
    slice.mean_normalized_js_divergence <=
      slice.maximum_normalized_js_divergence + 2 / reportingScale,
    `${label} mean JS divergence exceeds its maximum`,
  );
  invariant(
    slice.mean_absolute_confidence_delta <=
      slice.maximum_absolute_confidence_delta + 2 / reportingScale,
    `${label} mean confidence delta exceeds its maximum`,
  );
};

const elementText = (html, key) => {
  const expression = new RegExp(
    `<([a-z][a-z0-9]*)\\b[^>]*\\bdata-robustness-metric="${escapeRegExp(key)}"[^>]*>` +
      "([\\s\\S]*?)<\\/\\1>",
    "iu",
  );
  const match = expression.exec(html);
  invariant(match, `site is missing robustness metric binding ${key}`);
  return match[2].replace(/<[^>]+>/gu, "").replace(/\s+/gu, " ").trim();
};

const cssWidth = (styles, className) => {
  const expression = new RegExp(
    `\\.${escapeRegExp(className)}::after\\s*\\{[^}]*\\bwidth:\\s*([0-9.]+%);`,
    "iu",
  );
  const match = expression.exec(styles);
  invariant(match, `site is missing the ${className} width binding`);
  return match[1];
};

const percent = (value, digits) => `${(value * 100).toFixed(digits)}%`;

export const verifyRobustnessBinding = ({
  report,
  html,
  styles,
  manifest,
  policy,
  modelMetrics,
}) => {
  exactKeys(
    report,
    [
      "report_kind",
      "schema_version",
      "suite_version",
      "population",
      "model_version",
      "model_dataset_sha256",
      "split_plan_sha256",
      "temperature",
      "minimum_confidence",
      "minimum_probability_margin",
      "input_count",
      "baseline_accepted",
      "evaluated_variants",
      "formatting",
      "typographic",
      "perturbations",
    ],
    "robustness report",
  );
  invariant(report.report_kind === "eliza-metamorphic-robustness", "unexpected report kind");
  invariant(report.schema_version === 1, "unexpected robustness report schema");
  invariant(report.suite_version === "1.0.0", "unexpected perturbation suite");
  invariant(report.population === "bundle-id-test", "site metrics require the verified bundle ID-test");
  invariant(report.model_version === manifest.model_version, "report model version is not bundle-bound");
  invariant(
    report.model_dataset_sha256 === manifest.dataset_sha256,
    "report dataset digest is not bundle-bound",
  );
  invariant(
    report.split_plan_sha256 === manifest.split_plan_sha256,
    "report split digest is not bundle-bound",
  );
  invariant(report.temperature === policy.temperature, "report temperature is not policy-bound");
  invariant(
    report.minimum_confidence === policy.minimum_confidence,
    "report confidence threshold is not policy-bound",
  );
  invariant(
    report.minimum_probability_margin === policy.minimum_probability_margin,
    "report margin threshold is not policy-bound",
  );
  invariant(
    report.input_count === modelMetrics.id_test.example_count,
    "report input count is not the frozen ID-test count",
  );
  invariant(
    report.baseline_accepted ===
      modelMetrics.id_test.predictions.filter((prediction) => prediction.accepted).length,
    "report baseline acceptance count is not the verified ID-test ledger",
  );
  invariant(Number.isSafeInteger(report.evaluated_variants), "invalid report variant count");
  validateSlice(report.formatting, "formatting");
  validateSlice(report.typographic, "typographic");

  const expectedPerturbations = [
    ["ascii-letter-case", "formatting"],
    ["horizontal-whitespace", "formatting"],
    ["unicode-compatibility-width", "formatting"],
    ["terminal-punctuation", "formatting"],
    ["single-character-deletion", "typographic"],
    ["adjacent-character-transposition", "typographic"],
    ["single-character-duplication", "typographic"],
  ];
  invariant(
    Array.isArray(report.perturbations) &&
      report.perturbations.length === expectedPerturbations.length,
    "robustness perturbation inventory is incomplete",
  );
  for (const [index, [name, family]] of expectedPerturbations.entries()) {
    const perturbation = report.perturbations[index];
    exactKeys(perturbation, ["name", "family", "metrics"], `perturbation ${index}`);
    invariant(
      perturbation.name === name && perturbation.family === family,
      `unexpected perturbation at index ${index}`,
    );
    validateSlice(perturbation.metrics, `perturbation ${name}`);
  }

  const familyMembers = (family) =>
    report.perturbations
      .filter((perturbation) => perturbation.family === family)
      .map((perturbation) => perturbation.metrics);
  for (const [family, aggregate] of [
    ["formatting", report.formatting],
    ["typographic", report.typographic],
  ]) {
    const members = familyMembers(family);
    const evaluatedVariants = members.reduce(
      (sum, member) => sum + member.evaluated_variants,
      0,
    );
    for (const field of [
      "evaluated_variants",
      "skipped_applications",
      "label_flips",
      "acceptance_flips",
      "accepted_to_abstained",
      "abstained_to_accepted",
    ]) {
      invariant(
        aggregate[field] === members.reduce((sum, member) => sum + member[field], 0),
        `${family} aggregate does not reconstruct ${field}`,
      );
    }

    const agreement = (field) =>
      evaluatedVariants === 0
        ? 0
        : quantize(
            members.reduce(
              (sum, member, index) =>
                sum +
                reconstructAgreementCount(
                  member,
                  field,
                  `${family} perturbation ${index}`,
                ),
              0,
            ) / evaluatedVariants,
          );
    const weightedMean = (field) =>
      evaluatedVariants === 0
        ? 0
        : quantize(
            members.reduce(
              (sum, member) => sum + member[field] * member.evaluated_variants,
              0,
            ) / evaluatedVariants,
          );
    const maximum = (field) => Math.max(...members.map((member) => member[field]));

    for (const [field, expected] of [
      ["label_agreement", agreement("label_agreement")],
      ["acceptance_agreement", agreement("acceptance_agreement")],
      ["decision_agreement", agreement("decision_agreement")],
      [
        "mean_normalized_js_divergence",
        weightedMean("mean_normalized_js_divergence"),
      ],
      [
        "maximum_normalized_js_divergence",
        maximum("maximum_normalized_js_divergence"),
      ],
      [
        "mean_absolute_confidence_delta",
        weightedMean("mean_absolute_confidence_delta"),
      ],
      [
        "maximum_absolute_confidence_delta",
        maximum("maximum_absolute_confidence_delta"),
      ],
    ]) {
      invariant(
        close(aggregate[field], expected),
        `${family} aggregate does not reconstruct ${field}`,
      );
    }
  }
  invariant(
    report.evaluated_variants ===
      report.formatting.evaluated_variants + report.typographic.evaluated_variants,
    "total variant count does not match the family reports",
  );
  invariant(
    report.formatting.label_agreement === 1 &&
      report.formatting.decision_agreement === 1 &&
      report.formatting.maximum_normalized_js_divergence === 0,
    "formatting invariants are not exact",
  );

  const expectedText = new Map([
    ["formatting-decision-agreement", percent(report.formatting.decision_agreement, 0)],
    [
      "formatting-variant-count",
      `${report.formatting.evaluated_variants} variants across four invariants`,
    ],
    ["typographic-label-agreement", percent(report.typographic.label_agreement, 3)],
    [
      "typographic-variant-count",
      `${report.typographic.evaluated_variants} single-edit typo variants`,
    ],
    ["typographic-decision-agreement", percent(report.typographic.decision_agreement, 3)],
    [
      "maximum-js-divergence",
      report.typographic.maximum_normalized_js_divergence.toFixed(6),
    ],
    [
      "audit-population-summary",
      `${report.input_count} FROZEN INPUTS / ${report.evaluated_variants} DETERMINISTIC VARIANTS`,
    ],
  ]);
  for (const [key, value] of expectedText) {
    invariant(elementText(html, key) === value, `site metric ${key} does not match the CI report`);
  }

  const expectedWidths = new Map([
    ["robustness-bar--format", percent(report.formatting.decision_agreement, 0)],
    ["robustness-bar--typo-label", percent(report.typographic.label_agreement, 3)],
    ["robustness-bar--typo-decision", percent(report.typographic.decision_agreement, 3)],
    [
      "robustness-bar--drift",
      percent(report.typographic.maximum_normalized_js_divergence, 4),
    ],
  ]);
  for (const [className, value] of expectedWidths) {
    invariant(cssWidth(styles, className) === value, `site bar ${className} does not match the CI report`);
  }

  return {
    inputCount: report.input_count,
    variantCount: report.evaluated_variants,
    suiteVersion: report.suite_version,
  };
};

export const verifyRobustnessReportFile = async (reportPath) => {
  const [reportText, html, styles, manifestText, policyText, metricsText] = await Promise.all([
    readFile(path.resolve(reportPath), "utf8"),
    readFile(path.join(repositoryRoot, "site/index.html"), "utf8"),
    readFile(path.join(repositoryRoot, "site/styles.css"), "utf8"),
    readFile(path.join(repositoryRoot, "artifacts/eliza-open-set-v3/manifest.json"), "utf8"),
    readFile(path.join(repositoryRoot, "artifacts/eliza-open-set-v3/policy.json"), "utf8"),
    readFile(path.join(repositoryRoot, "artifacts/eliza-open-set-v3/metrics.json"), "utf8"),
  ]);
  return verifyRobustnessBinding({
    report: JSON.parse(reportText),
    html,
    styles,
    manifest: JSON.parse(manifestText),
    policy: JSON.parse(policyText),
    modelMetrics: JSON.parse(metricsText),
  });
};

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const reportPath = process.argv[2];
  if (!reportPath) {
    console.error("Usage: node scripts/verify-robustness-report.mjs REPORT.json");
    process.exitCode = 2;
  } else {
    try {
      const result = await verifyRobustnessReportFile(reportPath);
      console.log(
        `Robustness site metrics verified from suite ${result.suiteVersion}: ` +
          `${result.inputCount} inputs / ${result.variantCount} variants.`,
      );
    } catch (error) {
      console.error(`Robustness report verification failed: ${error.message}`);
      process.exitCode = 1;
    }
  }
}
