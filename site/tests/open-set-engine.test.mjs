import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  EXPECTED_BUNDLE_MANIFEST_SHA256,
  loadOpenSetBundle,
  verifyOpenSetArtifactSemantics,
} from "../open-set-engine.mjs";

const INVENTORY = ["manifest.json", "metrics.json", "model.json", "policy.json", "split-plan.json"];
const bundleRoot = new URL("../../artifacts/eliza-open-set-v3/", import.meta.url);

const readBundle = async () =>
  new Map(
    await Promise.all(
      INVENTORY.map(async (name) => [name, new Uint8Array(await readFile(new URL(name, bundleRoot)))]),
    ),
  );

const bundleFetch = (files, requests = []) => async (url) => {
  const name = new URL(String(url)).pathname.split("/").at(-1);
  requests.push(name);
  const bytes = files.get(name);
  return bytes ? new Response(bytes) : new Response("missing", { status: 404 });
};

const sha256 = async (bytes) =>
  Buffer.from(await webcrypto.subtle.digest("SHA-256", bytes)).toString("hex");

const jsonBytes = (value) => new TextEncoder().encode(`${JSON.stringify(value, null, 2)}\n`);

const parsedBundle = async () => {
  const files = await readBundle();
  const parse = (name) => JSON.parse(new TextDecoder().decode(files.get(name)));
  return {
    files,
    manifest: parse("manifest.json"),
    metrics: parse("metrics.json"),
    model: parse("model.json"),
    policy: parse("policy.json"),
    plan: parse("split-plan.json"),
  };
};

const close = (left, right) => Math.abs(left - right) <= 1e-10;

test("the browser loads and verifies all five v3 bundle artifacts", async () => {
  const files = await readBundle();
  const requests = [];
  const runtime = await loadOpenSetBundle(bundleRoot, {
    crypto: webcrypto,
    fetch: bundleFetch(files, requests),
  });

  assert.equal(runtime.version, "3.0.0");
  assert.deepEqual(new Set(requests), new Set(INVENTORY));
  assert.equal(requests.length, INVENTORY.length);
  assert.equal(runtime.metrics.schema_version, 3);
  assert.equal(runtime.splitPlan.schema_version, 3);
  assert.ok(Object.isFrozen(runtime.metrics));
  assert.equal(await sha256(files.get("manifest.json")), EXPECTED_BUNDLE_MANIFEST_SHA256);
});

test("the browser runtime reproduces the frozen ID, OOD and contrast decisions", async () => {
  const files = await readBundle();
  const runtime = await loadOpenSetBundle(bundleRoot, {
    crypto: webcrypto,
    fetch: bundleFetch(files),
  });
  const idRows = new Map(
    runtime.splitPlan.assignments
      .filter(({ partition }) => partition === "id-test")
      .map((row) => [row.id, row]),
  );
  for (const expected of runtime.metrics.id_test.predictions) {
    const actual = runtime.predict(idRows.get(expected.id).text);
    assert.equal(actual.label, expected.predicted_label, expected.id);
    assert.equal(actual.accepted, expected.accepted, expected.id);
    assert.ok(close(actual.confidence, expected.confidence), expected.id);
    assert.ok(close(actual.probabilityMargin, expected.probability_margin), expected.id);
    for (const [label, probability] of Object.entries(expected.probabilities)) {
      assert.ok(close(actual.probabilities[label], probability), `${expected.id}/${label}`);
    }
  }

  const oodRows = new Map(runtime.splitPlan.ood_test.map((row) => [row.id, row]));
  for (const expected of runtime.metrics.ood_test.predictions) {
    const actual = runtime.predict(oodRows.get(expected.id).text);
    assert.equal(actual.label, expected.predicted_label, expected.id);
    assert.equal(actual.accepted, expected.accepted, expected.id);
    assert.ok(close(actual.confidence, expected.confidence), expected.id);
    assert.ok(close(actual.probabilityMargin, expected.probability_margin), expected.id);
  }

  const contrastRows = new Map(runtime.splitPlan.contrast_test.map((row) => [row.id, row]));
  for (const expected of runtime.metrics.contrast_test.predictions) {
    const actual = runtime.predict(contrastRows.get(expected.id).text);
    assert.equal(actual.label, expected.predicted_label, expected.id);
    assert.equal(actual.accepted, expected.accepted, expected.id);
    assert.ok(close(actual.confidence, expected.confidence), expected.id);
  }

  assert.equal(runtime.predict("").accepted, false);
  assert.equal(runtime.predict("x".repeat(513)).accepted, false);
});

test("a payload changed in transit fails its manifest digest", async () => {
  const files = await readBundle();
  const tampered = new Uint8Array(files.get("model.json"));
  tampered[tampered.length - 2] ^= 1;
  files.set("model.json", tampered);

  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: webcrypto,
      fetch: bundleFetch(files),
    }),
    /model\.json failed SHA-256 verification/u,
  );
});

test("a self-rehashed policy change fails the embedded release trust root", async () => {
  const files = await readBundle();
  const policy = JSON.parse(new TextDecoder().decode(files.get("policy.json")));
  policy.minimum_confidence += 0.01;
  const policyBytes = new TextEncoder().encode(`${JSON.stringify(policy, null, 2)}\n`);
  files.set("policy.json", policyBytes);

  const manifest = JSON.parse(new TextDecoder().decode(files.get("manifest.json")));
  manifest.files["policy.json"] = await sha256(policyBytes);
  files.set("manifest.json", new TextEncoder().encode(`${JSON.stringify(manifest, null, 2)}\n`));

  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: webcrypto,
      fetch: bundleFetch(files),
    }),
    /embedded release trust root/u,
  );
});

test("semantic verification recomputes reported metrics and threshold observations", async () => {
  const base = await parsedBundle();
  const probes = [
    [
      "macro F1",
      (metrics) => {
        metrics.id_test.macro_f1 = 0.99;
        metrics.bootstrap_95.id_macro_f1.value = 0.99;
        metrics.bootstrap_95.id_macro_f1.lower_95 = 0.98;
        metrics.bootstrap_95.id_macro_f1.upper_95 = 1;
      },
      /ID aggregate metrics/u,
    ],
    [
      "ID calibration",
      (metrics) => {
        metrics.id_test.calibration.negative_log_likelihood = 0;
        metrics.bootstrap_95.id_negative_log_likelihood.value = 0;
        metrics.bootstrap_95.id_negative_log_likelihood.lower_95 = 0;
        metrics.bootstrap_95.id_negative_log_likelihood.upper_95 = 0.1;
      },
      /ID aggregate metrics/u,
    ],
    [
      "risk coverage",
      (metrics) => {
        metrics.id_test.aurc = 0;
        metrics.bootstrap_95.id_aurc.value = 0;
        metrics.bootstrap_95.id_aurc.lower_95 = 0;
        metrics.bootstrap_95.id_aurc.upper_95 = 0.1;
      },
      /ID aggregate metrics/u,
    ],
    [
      "OOD discrimination",
      (metrics) => {
        metrics.ood_test.discrimination.auroc = 1;
        metrics.bootstrap_95.ood_auroc.value = 1;
        metrics.bootstrap_95.ood_auroc.lower_95 = 0.99;
        metrics.bootstrap_95.ood_auroc.upper_95 = 1;
      },
      /OOD aggregate metrics/u,
    ],
    [
      "threshold observations",
      (metrics) => {
        metrics.threshold_selection.observed_development_coverage = 0;
      },
      /Threshold observations/u,
    ],
    [
      "calibration partition",
      (metrics) => {
        metrics.calibrated_calibration_partition.multiclass_brier = 0;
      },
      /Calibration-partition metrics/u,
    ],
    [
      "contrast pair accuracy",
      (metrics) => {
        metrics.contrast_test.pair_accuracy =
          metrics.contrast_test.pair_accuracy === 0 ? 1 : 0;
      },
      /Contrast-test report/u,
    ],
  ];
  for (const [name, mutate, expected] of probes) {
    const metrics = structuredClone(base.metrics);
    mutate(metrics);
    await assert.rejects(
      verifyOpenSetArtifactSemantics(
        structuredClone(base.model),
        structuredClone(base.policy),
        metrics,
        structuredClone(base.plan),
        webcrypto,
      ),
      expected,
      name,
    );
  }
});

test("semantic verification binds the contrast source fingerprint", async () => {
  const base = await parsedBundle();
  const metrics = structuredClone(base.metrics);
  metrics.contrast_test_sha256 = "a".repeat(64);
  await assert.rejects(
    verifyOpenSetArtifactSemantics(
      structuredClone(base.model),
      structuredClone(base.policy),
      metrics,
      structuredClone(base.plan),
      webcrypto,
    ),
    /source rows/i,
  );
});

test("a fully rehashed train-row change still fails the release trust root", async () => {
  const { files, manifest, metrics, model, policy, plan } = await parsedBundle();
  const trainRow = plan.assignments.find((row) => row.partition === "train");
  trainRow.text = `${trainRow.text} altered`;
  const replacementDatasetDigest = "a".repeat(64);
  plan.dataset_sha256 = replacementDatasetDigest;
  model.dataset_sha256 = replacementDatasetDigest;
  policy.dataset_sha256 = replacementDatasetDigest;
  metrics.dataset_sha256 = replacementDatasetDigest;
  manifest.dataset_sha256 = replacementDatasetDigest;

  const planBytes = jsonBytes(plan);
  const splitDigest = await sha256(planBytes);
  model.split_plan_sha256 = splitDigest;
  policy.split_plan_sha256 = splitDigest;
  metrics.split_plan_sha256 = splitDigest;
  manifest.split_plan_sha256 = splitDigest;
  files.set("split-plan.json", planBytes);

  for (const [name, value] of [
    ["model.json", model],
    ["policy.json", policy],
    ["metrics.json", metrics],
  ]) {
    files.set(name, jsonBytes(value));
  }
  for (const name of ["metrics.json", "model.json", "policy.json", "split-plan.json"]) {
    manifest.files[name] = await sha256(files.get(name));
  }
  files.set("manifest.json", jsonBytes(manifest));

  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: webcrypto,
      fetch: bundleFetch(files),
    }),
    /embedded release trust root/u,
  );
});

test("declared and streamed oversized artifacts are rejected before parsing", async () => {
  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: webcrypto,
      fetch: async () =>
        new Response("{}", {
          headers: { "content-length": String(8 * 1024 * 1024 + 1) },
        }),
    }),
    /artifact byte budget/u,
  );

  const files = await readBundle();
  const oversizedStream = new ReadableStream({
    pull(controller) {
      controller.enqueue(new Uint8Array(1024 * 1024));
    },
  });
  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: webcrypto,
      fetch: async (url) => {
        const name = new URL(String(url)).pathname.split("/").at(-1);
        if (name === "metrics.json") return new Response(oversizedStream);
        return bundleFetch(files)(url);
      },
    }),
    /artifact byte budget/u,
  );
});

test("verification fails closed when Web Crypto is unavailable", async () => {
  const files = await readBundle();
  let requests = 0;
  await assert.rejects(
    loadOpenSetBundle(bundleRoot, {
      crypto: undefined,
      fetch: async (url) => {
        requests += 1;
        return bundleFetch(files)(url);
      },
    }),
    /Web Crypto SHA-256 is required/u,
  );
  assert.equal(requests, 0);
});
