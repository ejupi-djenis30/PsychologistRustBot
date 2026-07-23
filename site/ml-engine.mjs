const MODEL_SCHEMA_VERSION = 1;
const MODEL_KIND = "eliza-intent-softmax";
const MODEL_VERSION = "1.0.0";
const ALPHANUMERIC = /[\p{Alphabetic}\p{Number}]/u;
const WHITESPACE = /\p{White_Space}+/gu;

const normalize = (value) =>
  String(value)
    .toLowerCase()
    .replace(/[’‘]/gu, "'")
    .replace(WHITESPACE, " ")
    .replace(/^ | $/gu, "");

const tokenize = (value) => {
  const tokens = [];
  let current = "";
  const flush = () => {
    const token = current.replace(/^'+|'+$/gu, "");
    if (token) tokens.push(token);
    current = "";
  };
  for (const character of normalize(value)) {
    if (character === "'" || ALPHANUMERIC.test(character)) current += character;
    else flush();
  }
  flush();
  return tokens;
};

const extractTerms = (text, config) => {
  const tokens = tokenize(text);
  const terms = [];
  for (let size = config.word_ngram_min; size <= config.word_ngram_max; size += 1) {
    for (let start = 0; start + size <= tokens.length; start += 1) {
      terms.push(`w${size}:${tokens.slice(start, start + size).join("_")}`);
    }
  }
  const characters = Array.from(`^${tokens.join(" ")}$`);
  for (let size = config.char_ngram_min; size <= config.char_ngram_max; size += 1) {
    for (let start = 0; start + size <= characters.length; start += 1) {
      terms.push(`c${size}:${characters.slice(start, start + size).join("")}`);
    }
  }
  return terms;
};

const finiteNumber = (value, description) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${description} must be a finite number`);
  }
  return value;
};

const record = (value, description) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${description} must be an object`);
  }
  return value;
};

const exactKeys = (value, expected, description) => {
  const actual = Object.keys(record(value, description)).sort();
  const allowed = [...expected].sort();
  if (actual.length !== allowed.length || actual.some((key, index) => key !== allowed[index])) {
    throw new TypeError(`${description} has missing or unknown fields`);
  }
};

const boundedInteger = (value, minimum, maximum, description) => {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${description} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
};

const boundedNumber = (value, minimum, maximum, description) => {
  finiteNumber(value, description);
  if (value < minimum || value > maximum) {
    throw new TypeError(`${description} must be between ${minimum} and ${maximum}`);
  }
  return value;
};

const validateVectorizerConfig = (config, description) => {
  exactKeys(
    config,
    [
      "word_ngram_min",
      "word_ngram_max",
      "char_ngram_min",
      "char_ngram_max",
      "min_document_frequency",
      "max_features",
    ],
    description,
  );
  const wordMin = boundedInteger(config.word_ngram_min, 1, 3, `${description}.word_ngram_min`);
  const wordMax = boundedInteger(config.word_ngram_max, 1, 3, `${description}.word_ngram_max`);
  const charMin = boundedInteger(config.char_ngram_min, 2, 6, `${description}.char_ngram_min`);
  const charMax = boundedInteger(config.char_ngram_max, 2, 6, `${description}.char_ngram_max`);
  if (wordMin > wordMax || charMin > charMax) {
    throw new TypeError(`${description} contains a reversed n-gram range`);
  }
  boundedInteger(
    config.min_document_frequency,
    1,
    1_000_000,
    `${description}.min_document_frequency`,
  );
  boundedInteger(config.max_features, 32, 100_000, `${description}.max_features`);
};

const sameVectorizerConfig = (left, right) =>
  [
    "word_ngram_min",
    "word_ngram_max",
    "char_ngram_min",
    "char_ngram_max",
    "min_document_frequency",
    "max_features",
  ].every((key) => left[key] === right[key]);

const validFeature = (feature, config) => {
  if (typeof feature !== "string") return false;
  const match = /^([wc])(\d+):(.+)$/su.exec(feature);
  if (!match || match[3].includes(":")) return false;
  const [, kind, encodedSize, payload] = match;
  const size = Number(encodedSize);
  if (!Number.isSafeInteger(size)) return false;
  if (kind === "w") {
    if (size < config.word_ngram_min || size > config.word_ngram_max) return false;
    const tokens = payload.split("_");
    return (
      tokens.length === size &&
      tokens.every((token) => {
        const characters = Array.from(token);
        return (
          characters.length > 0 &&
          ALPHANUMERIC.test(characters[0]) &&
          ALPHANUMERIC.test(characters.at(-1)) &&
          characters.every((character) => character === "'" || ALPHANUMERIC.test(character))
        );
      })
    );
  }
  return (
    size >= config.char_ngram_min &&
    size <= config.char_ngram_max &&
    Array.from(payload).length === size &&
    Array.from(payload).every(
      (character) =>
        character === "'" ||
        character === "^" ||
        character === "$" ||
        character === " " ||
        ALPHANUMERIC.test(character),
    )
  );
};

const codePointCompare = (left, right) => {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0));
  const rightPoints = Array.from(right, (character) => character.codePointAt(0));
  const length = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < length; index += 1) {
    if (leftPoints[index] !== rightPoints[index]) return leftPoints[index] - rightPoints[index];
  }
  return leftPoints.length - rightPoints.length;
};

export class IntentModel {
  #model;
  #featureIndex;

  constructor(model) {
    const snapshot = structuredClone(record(model, "Model"));
    exactKeys(
      snapshot,
      [
        "schema_version",
        "model_kind",
        "model_version",
        "dataset_fingerprint",
        "training_config",
        "labels",
        "vectorizer",
        "weights",
        "biases",
      ],
      "Model",
    );
    if (
      snapshot.schema_version !== MODEL_SCHEMA_VERSION ||
      snapshot.model_kind !== MODEL_KIND ||
      snapshot.model_version !== MODEL_VERSION
    ) {
      throw new TypeError("Unsupported model schema, kind, or version");
    }
    if (!/^fnv1a64:[0-9a-f]{16}$/u.test(snapshot.dataset_fingerprint)) {
      throw new TypeError("Model fingerprint is malformed");
    }

    const training = snapshot.training_config;
    exactKeys(
      training,
      ["seed", "epochs", "learning_rate", "l2_penalty", "holdout_fraction", "vectorizer", "thresholds"],
      "Training config",
    );
    boundedInteger(training.seed, 0, Number.MAX_SAFE_INTEGER, "Training seed");
    boundedInteger(training.epochs, 1, 10_000, "Training epochs");
    boundedNumber(training.learning_rate, 0.000_001, 10, "Learning rate");
    boundedNumber(training.l2_penalty, 0, 1, "L2 penalty");
    boundedNumber(training.holdout_fraction, 0.05, 0.5, "Holdout fraction");
    validateVectorizerConfig(training.vectorizer, "Training vectorizer config");
    exactKeys(training.thresholds, ["minimum_confidence", "minimum_margin"], "Thresholds");
    boundedNumber(training.thresholds.minimum_confidence, 0, 1, "Minimum confidence");
    boundedNumber(training.thresholds.minimum_margin, 0, 1, "Minimum margin");

    const labels = snapshot.labels;
    const vectorizer = snapshot.vectorizer;
    if (
      !Array.isArray(labels) ||
      labels.length < 2 ||
      labels.some((label) => typeof label !== "string" || label.length === 0) ||
      new Set(labels).size !== labels.length
    ) {
      throw new TypeError("Model labels must be non-empty and unique");
    }
    exactKeys(vectorizer, ["config", "vocabulary", "inverse_document_frequency"], "Vectorizer");
    validateVectorizerConfig(vectorizer.config, "Serialized vectorizer config");
    if (!sameVectorizerConfig(training.vectorizer, vectorizer.config)) {
      throw new TypeError("Training and serialized vectorizer configs differ");
    }
    if (
      !vectorizer ||
      !Array.isArray(vectorizer.vocabulary) ||
      !Array.isArray(vectorizer.inverse_document_frequency) ||
      vectorizer.vocabulary.length === 0 ||
      vectorizer.vocabulary.length !== vectorizer.inverse_document_frequency.length ||
      vectorizer.vocabulary.length > vectorizer.config.max_features ||
      vectorizer.vocabulary.some((feature) => !validFeature(feature, vectorizer.config)) ||
      new Set(vectorizer.vocabulary).size !== vectorizer.vocabulary.length
    ) {
      throw new TypeError("Model vectorizer violates its feature contract");
    }
    if (
      !Array.isArray(snapshot.weights) ||
      snapshot.weights.length !== labels.length ||
      snapshot.weights.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== vectorizer.vocabulary.length ||
          row.some((weight) => typeof weight !== "number" || !Number.isFinite(weight)),
      ) ||
      !Array.isArray(snapshot.biases) ||
      snapshot.biases.length !== labels.length ||
      snapshot.biases.some((bias) => typeof bias !== "number" || !Number.isFinite(bias))
    ) {
      throw new TypeError("Model parameters are not finite and rectangular");
    }
    vectorizer.inverse_document_frequency.forEach((value) => {
      finiteNumber(value, "Inverse document frequency");
      if (value < 1) throw new TypeError("Inverse document frequency must be at least one");
    });

    this.#model = snapshot;
    this.#featureIndex = new Map(
      vectorizer.vocabulary.map((feature, index) => [feature, index]),
    );
  }

  get version() {
    return this.#model.model_version;
  }

  get fingerprint() {
    return this.#model.dataset_fingerprint;
  }

  #transform(text) {
    const counts = new Map();
    for (const term of extractTerms(text, this.#model.vectorizer.config)) {
      const index = this.#featureIndex.get(term);
      if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
    }
    const values = Array.from(counts, ([index, count]) => [
      index,
      (1 + Math.log(count)) * this.#model.vectorizer.inverse_document_frequency[index],
    ]).sort(([left], [right]) => left - right);
    const norm = Math.sqrt(values.reduce((total, [, value]) => total + value * value, 0));
    return norm > 0 ? values.map(([index, value]) => [index, value / norm]) : [];
  }

  predict(text) {
    const features = this.#transform(text);
    const logits = this.#model.weights.map(
      (row, classIndex) =>
        this.#model.biases[classIndex] +
        features.reduce((total, [feature, value]) => total + row[feature] * value, 0),
    );
    const maximum = Math.max(...logits);
    const exponentials = logits.map((logit) => Math.exp(logit - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map((value) => value / total);
    const ranking = probabilities
      .map((probability, index) => ({ index, probability }))
      .sort((left, right) => right.probability - left.probability || left.index - right.index);
    const top = ranking[0];
    const confidence = top.probability;
    const margin = confidence - ranking[1].probability;
    const thresholds = this.#model.training_config.thresholds;
    const accepted =
      features.length > 0 &&
      confidence >= thresholds.minimum_confidence &&
      margin >= thresholds.minimum_margin;
    const topFeatures = features
      .map(([feature, value]) => ({
        feature: this.#model.vectorizer.vocabulary[feature],
        value,
        weight: this.#model.weights[top.index][feature],
        contribution: value * this.#model.weights[top.index][feature],
      }))
      .filter(({ contribution }) => contribution > 0)
      .sort(
        (left, right) =>
          right.contribution - left.contribution || codePointCompare(left.feature, right.feature),
      )
      .slice(0, 8);

    return {
      label: this.#model.labels[top.index],
      accepted,
      confidence,
      margin,
      probabilities: Object.fromEntries(
        this.#model.labels.map((label, index) => [label, probabilities[index]]),
      ),
      topFeatures,
    };
  }
}
