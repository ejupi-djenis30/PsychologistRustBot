import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ElizaEngine } from "../engine.mjs";
import { IntentModel } from "../ml-engine.mjs";

const TOLERANCE = 1e-12;

const close = (actual, expected, context) => {
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `${context}: expected ${expected}, observed ${actual}`,
  );
};

test("explains a feeling reflection", () => {
  const engine = new ElizaEngine();
  const reply = engine.respond("I feel uncertain about my next step.");

  assert.equal(reply.rule, "feeling-reflection");
  assert.equal(reply.transformed, "uncertain about your next step");
  assert.equal(reply.text, "What makes you feel uncertain about your next step?");
});

test("stops the simulation for urgent-safety language", () => {
  const engine = new ElizaEngine();
  const reply = engine.respond("I might hurt myself");

  assert.equal(reply.rule, "safety-boundary");
  assert.match(reply.text, /emergency number/);
  assert.doesNotMatch(reply.text, /diagnos/i);
});

test("uses phrase boundaries and recognizes explicit safety variants", () => {
  const engine = new ElizaEngine();
  assert.notEqual(engine.respond("I want to skill myself").rule, "safety-boundary");
  assert.equal(engine.respond("I don’t want to live").rule, "safety-boundary");
  assert.equal(engine.respond("I am suicidal").rule, "safety-boundary");
});

test("uses deterministic fallbacks", () => {
  const engine = new ElizaEngine();
  const first = engine.respond("A statement");
  const second = engine.respond("Another statement");

  assert.equal(first.rule, "fallback");
  assert.notEqual(first.text, second.text);
});

test("does not expose transcript state", () => {
  const engine = new ElizaEngine();
  engine.respond("A private fictional sentence");

  assert.deepEqual(Object.keys(engine), []);
  assert.equal(engine.turn, 1);
});

test("matches the shared Rust/browser parity corpus", async () => {
  const corpus = await readFile(new URL("../../fixtures/parity.tsv", import.meta.url), "utf8");
  for (const [index, line] of corpus.split(/\r?\n/u).entries()) {
    if (!line || line.startsWith("#")) continue;
    const [input, rule, response] = line.split("\t", 3);
    assert.ok(input && rule && response, `invalid corpus row ${index + 1}`);
    const reply = new ElizaEngine().respond(input);
    assert.equal(reply.rule, rule, `rule mismatch on row ${index + 1}`);
    assert.equal(reply.text, response, `text mismatch on row ${index + 1}`);
  }
});

test("bounds oversized input without allocating a code-point array", () => {
  const engine = new ElizaEngine();
  assert.equal(engine.respond("🙂".repeat(513)).rule, "input-boundary");
});

test("matches the shared Rust/browser learned-model corpus", async () => {
  const encodedModel = await readFile(
    new URL("../../models/eliza-intent-v1.json", import.meta.url),
    "utf8",
  );
  const model = new IntentModel(JSON.parse(encodedModel));
  const corpus = await readFile(new URL("../../fixtures/ml-parity.tsv", import.meta.url), "utf8");
  for (const [index, line] of corpus.split(/\r?\n/u).entries()) {
    if (!line || line.startsWith("#")) continue;
    const [input, encodedExpectation] = line.split("\t", 2);
    assert.ok(input && encodedExpectation, `invalid learned corpus row ${index + 1}`);
    const expected = JSON.parse(encodedExpectation);
    const prediction = model.predict(input);
    assert.equal(prediction.label, expected.label, `label mismatch on row ${index + 1}`);
    assert.equal(prediction.accepted, expected.accepted, `decision mismatch on row ${index + 1}`);
    close(prediction.confidence, expected.confidence, `confidence on row ${index + 1}`);
    close(prediction.margin, expected.margin, `margin on row ${index + 1}`);
    assert.equal(Object.keys(prediction.probabilities).length, 7);
    for (const [label, probability] of Object.entries(expected.probabilities)) {
      close(prediction.probabilities[label], probability, `${label} probability on row ${index + 1}`);
    }
    expected.top_features.forEach((expectedFeature, featureIndex) => {
      const actual = prediction.topFeatures[featureIndex];
      assert.equal(actual.feature, expectedFeature.feature, `feature ${featureIndex} on row ${index + 1}`);
      close(actual.value, expectedFeature.value, `feature value ${featureIndex} on row ${index + 1}`);
      close(actual.weight, expectedFeature.weight, `feature weight ${featureIndex} on row ${index + 1}`);
      close(
        actual.contribution,
        expectedFeature.contribution,
        `feature contribution ${featureIndex} on row ${index + 1}`,
      );
    });
  }
});

test("rejects malformed or internally inconsistent browser model artifacts", async () => {
  const source = JSON.parse(
    await readFile(new URL("../../models/eliza-intent-v1.json", import.meta.url), "utf8"),
  );
  const corruptions = [
    (model) => {
      model.unexpected = true;
    },
    (model) => {
      model.model_version = "9.0.0";
    },
    (model) => {
      model.dataset_fingerprint = "fnv1a64:1234";
    },
    (model) => {
      model.vectorizer.config.word_ngram_min = 0;
    },
    (model) => {
      model.vectorizer.config.char_ngram_max -= 1;
    },
    (model) => {
      model.vectorizer.vocabulary[1] = model.vectorizer.vocabulary[0];
    },
    (model) => {
      model.vectorizer.inverse_document_frequency[0] = 0.5;
    },
    (model) => {
      model.training_config.thresholds.minimum_confidence = 1.1;
    },
    (model) => {
      model.weights[0].pop();
    },
    (model) => {
      model.training_config.vectorizer.max_features = 32;
      model.vectorizer.config.max_features = 32;
    },
  ];

  corruptions.forEach((corrupt) => {
    const candidate = structuredClone(source);
    corrupt(candidate);
    assert.throws(() => new IntentModel(candidate), TypeError);
  });
});

test("browser inference snapshots the validated artifact", async () => {
  const source = JSON.parse(
    await readFile(new URL("../../models/eliza-intent-v1.json", import.meta.url), "utf8"),
  );
  const model = new IntentModel(source);
  const before = model.predict("Today I feel calm");
  source.weights[0][0] = 100_000;
  source.vectorizer.config.word_ngram_min = 0;
  const after = model.predict("Today I feel calm");

  assert.deepEqual(after, before);
});

test("learned dialogue keeps safety ahead of inference and abstains out of domain", async () => {
  const model = new IntentModel(
    JSON.parse(
      await readFile(new URL("../../models/eliza-intent-v1.json", import.meta.url), "utf8"),
    ),
  );
  const engine = new ElizaEngine(model);
  const safety = engine.respond("hello, I want to die");
  assert.equal(safety.rule, "safety-boundary");
  assert.equal(safety.model, null);

  const unknown = engine.respond("🪐🛰️🧪");
  assert.equal(unknown.rule, "ml-abstain");
  assert.equal(unknown.model.accepted, false);
});
