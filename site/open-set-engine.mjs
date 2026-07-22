const SCHEMA_VERSION = 3;
const MODEL_KIND = "eliza-open-set-linear";
const MODEL_VERSION = "3.0.0";
const BUNDLE_KIND = "eliza-open-set-bundle";
const BUNDLE_VERSION = "3.0.0";
const MAX_INPUT_CHARS = 512;
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;
const MAX_BUNDLE_BYTES = 16 * 1024 * 1024;
const MAX_EXAMPLES = 100_000;
const MAX_CLASSES = 256;
const MAX_PARAMETER_MAGNITUDE = 1_000_000;
const MAX_IDF = 64;
const JSON_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const PAYLOAD_NAMES = ["metrics.json", "model.json", "policy.json", "split-plan.json"];
const ALPHANUMERIC = /[\p{Alphabetic}\p{Number}]/u;
const WHITESPACE = /\p{White_Space}+/gu;
const SHA256 = /^[0-9a-f]{64}$/u;
const IDENTIFIER = /^[A-Za-z0-9_-]{1,128}$/u;
const LABEL = /^[a-z-]{1,64}$/u;
const PARTITIONS = ["train", "development", "calibration", "id-test"];
const OOD_STRATA = ["semantic", "capability", "noise"];
const CONTRAST_VARIANTS = ["a", "b"];
const MIN_PARAPHRASES_PER_FAMILY = 3;
export const EXPECTED_BUNDLE_MANIFEST_SHA256 =
  "cad018ec176542dc9cd04e826e76dbda27bbf302847faa76e8f5800123b0c114";

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

const finiteNumber = (value, description) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${description} must be finite`);
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

const boundedInteger = (value, minimum, maximum, description) => {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${description} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
};

const requireSha256 = (value, description) => {
  if (typeof value !== "string" || !SHA256.test(value)) {
    throw new TypeError(`${description} must be a lowercase SHA-256 digest`);
  }
  return value;
};

const requireIdentifier = (value, description) => {
  if (typeof value !== "string" || !IDENTIFIER.test(value)) {
    throw new TypeError(`${description} is invalid`);
  }
  return value;
};

const requireLabel = (value, description) => {
  if (typeof value !== "string" || !LABEL.test(value)) {
    throw new TypeError(`${description} is invalid`);
  }
  return value;
};

const requireText = (value, description) => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`${description} must not be empty`);
  }
  let count = 0;
  for (const _character of value) {
    count += 1;
    if (count > MAX_INPUT_CHARS) throw new TypeError(`${description} is too long`);
  }
  return value;
};

const normalize = (value) =>
  String(value)
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[’‘]/gu, "'")
    .replace(WHITESPACE, " ")
    .trim();

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

const featureIdentity = (value) => tokenize(value).join(" ");

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

const validateDevelopmentSelectionConfig = (selection) => {
  exactKeys(
    selection,
    ["max_features_candidates", "l2_penalty_candidates", "macro_f1_tolerance"],
    "Development selection config",
  );
  for (const [name, candidates, minimum, maximum, integer] of [
    ["max features", selection.max_features_candidates, 32, 100_000, true],
    ["L2 penalty", selection.l2_penalty_candidates, 0, 1, false],
  ]) {
    if (!Array.isArray(candidates) || candidates.length === 0 || candidates.length > 8) {
      throw new TypeError(`Development ${name} candidates have an invalid size`);
    }
    candidates.forEach((candidate, index) => {
      if (integer) boundedInteger(candidate, minimum, maximum, `Development ${name} candidate`);
      else boundedNumber(candidate, minimum, maximum, `Development ${name} candidate`);
      if (index > 0 && candidates[index - 1] >= candidate) {
        throw new TypeError(`Development ${name} candidates must be strictly ascending`);
      }
    });
  }
  boundedNumber(selection.macro_f1_tolerance, 0, 0.05, "Development macro-F1 tolerance");
};

const validFeature = (feature, config) => {
  if (typeof feature !== "string") return false;
  const match = /^([wc])(\d+):(.+)$/su.exec(feature);
  if (!match) return false;
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
  const characters = Array.from(payload);
  return (
    size >= config.char_ngram_min &&
    size <= config.char_ngram_max &&
    characters.length === size &&
    characters.every(
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

const parseJson = (bytes, description) => {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    throw new TypeError(`${description} is not canonical UTF-8 JSON`, { cause: error });
  }
};

const artifactUrl = (baseUrl, name) => `${String(baseUrl).replace(/\/+$/u, "")}/${name}`;

const fetchBytes = async (fetchImpl, url, description, budget) => {
  const response = await fetchImpl(url, {
    cache: "no-cache",
    credentials: "same-origin",
  });
  if (!response || response.ok !== true) {
    throw new Error(`${description} request failed`);
  }
  const declaredLength = response.headers?.get?.("content-length");
  if (declaredLength !== null && declaredLength !== undefined) {
    if (!/^\d+$/u.test(declaredLength)) {
      throw new TypeError(`${description} has an invalid Content-Length`);
    }
    const length = Number(declaredLength);
    if (
      !Number.isSafeInteger(length) ||
      length <= 0 ||
      length > MAX_ARTIFACT_BYTES ||
      budget.consumed + length > MAX_BUNDLE_BYTES
    ) {
      throw new TypeError(`${description} exceeds the artifact byte budget`);
    }
  }
  let bytes;
  if (typeof response.body?.getReader === "function") {
    const reader = response.body.getReader();
    const chunks = [];
    let length = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
      length += chunk.byteLength;
      if (length > MAX_ARTIFACT_BYTES || budget.consumed + length > MAX_BUNDLE_BYTES) {
        await reader.cancel().catch(() => {});
        throw new TypeError(`${description} exceeds the artifact byte budget`);
      }
      chunks.push(chunk);
    }
    bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  } else if (typeof response.arrayBuffer === "function") {
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    throw new TypeError(`${description} response has no readable body`);
  }
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_ARTIFACT_BYTES) {
    throw new TypeError(`${description} has an invalid byte length`);
  }
  if (budget.consumed + bytes.byteLength > MAX_BUNDLE_BYTES) {
    throw new TypeError("Model bundle exceeds the total byte budget");
  }
  budget.consumed += bytes.byteLength;
  return bytes;
};

const sha256Hex = async (cryptoProvider, bytes) => {
  const digest = await cryptoProvider.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
};

const fingerprintRows = async (cryptoProvider, rows) =>
  sha256Hex(
    cryptoProvider,
    new TextEncoder().encode([...rows].sort(codePointCompare).join("\n")),
  );

const validateSourceFingerprints = async (cryptoProvider, plan, model, metrics) => {
  const datasetDigest = await fingerprintRows(
    cryptoProvider,
    plan.assignments.map(
      (row) => `${row.id}\t${row.group_id}\t${row.label}\t${featureIdentity(row.text)}`,
    ),
  );
  const oodDigest = async (rows) =>
    fingerprintRows(
      cryptoProvider,
      rows.map(
        (row) =>
          `${row.id}\t${row.family_id}\t${row.domain_group}\t${row.stratum}\t${featureIdentity(row.text)}`,
      ),
    );
  const [oodDevelopmentDigest, oodTestDigest] = await Promise.all([
    oodDigest(plan.ood_development),
    oodDigest(plan.ood_test),
  ]);
  const contrastDigest = await fingerprintRows(
    cryptoProvider,
    plan.contrast_test.map(
      (row) =>
        `${row.id}\t${row.pair_id}\t${row.variant}\t${row.label}\t${featureIdentity(row.text)}`,
    ),
  );
  if (
    datasetDigest !== plan.dataset_sha256 ||
    datasetDigest !== model.dataset_sha256 ||
    datasetDigest !== metrics.dataset_sha256 ||
    oodDevelopmentDigest !== metrics.ood_development_sha256 ||
    oodTestDigest !== metrics.ood_test_sha256 ||
    contrastDigest !== metrics.contrast_test_sha256
  ) {
    throw new TypeError("Source rows do not reproduce their recorded SHA-256 fingerprints");
  }
};

const validateManifest = (manifest) => {
  exactKeys(
    manifest,
    [
      "schema_version",
      "bundle_kind",
      "bundle_version",
      "model_version",
      "dataset_sha256",
      "split_plan_sha256",
      "files",
    ],
    "Bundle manifest",
  );
  if (
    manifest.schema_version !== SCHEMA_VERSION ||
    manifest.bundle_kind !== BUNDLE_KIND ||
    manifest.bundle_version !== BUNDLE_VERSION ||
    manifest.model_version !== MODEL_VERSION
  ) {
    throw new TypeError("Unsupported bundle identity");
  }
  requireSha256(manifest.dataset_sha256, "Manifest dataset digest");
  requireSha256(manifest.split_plan_sha256, "Manifest split-plan digest");
  exactKeys(manifest.files, PAYLOAD_NAMES, "Bundle file inventory");
  for (const name of PAYLOAD_NAMES) requireSha256(manifest.files[name], `${name} digest`);
  if (manifest.files["split-plan.json"] !== manifest.split_plan_sha256) {
    throw new TypeError("Manifest split-plan digests disagree");
  }
};

const validateModel = (model) => {
  exactKeys(
    model,
    [
      "schema_version",
      "model_kind",
      "model_version",
      "dataset_sha256",
      "split_plan_sha256",
      "training_config",
      "labels",
      "vectorizer",
      "weights",
      "biases",
    ],
    "Open-set model",
  );
  if (
    model.schema_version !== SCHEMA_VERSION ||
    model.model_kind !== MODEL_KIND ||
    model.model_version !== MODEL_VERSION
  ) {
    throw new TypeError("Unsupported open-set model identity");
  }
  requireSha256(model.dataset_sha256, "Model dataset digest");
  requireSha256(model.split_plan_sha256, "Model split-plan digest");
  exactKeys(
    model.training_config,
    ["seed", "epochs", "learning_rate", "l2_penalty", "vectorizer", "development_selection"],
    "Training config",
  );
  boundedInteger(model.training_config.seed, 0, JSON_SAFE_INTEGER, "Training seed");
  boundedInteger(model.training_config.epochs, 1, 10_000, "Training epochs");
  boundedNumber(model.training_config.learning_rate, 0.000_001, 10, "Learning rate");
  boundedNumber(model.training_config.l2_penalty, 0, 1, "L2 penalty");
  validateVectorizerConfig(model.training_config.vectorizer, "Training vectorizer config");
  validateDevelopmentSelectionConfig(model.training_config.development_selection);
  if (
    !model.training_config.development_selection.max_features_candidates.includes(
      model.training_config.vectorizer.max_features,
    ) ||
    !model.training_config.development_selection.l2_penalty_candidates.includes(
      model.training_config.l2_penalty,
    )
  ) {
    throw new TypeError("Fitted training config is absent from its development candidate grid");
  }

  if (
    !Array.isArray(model.labels) ||
    model.labels.length < 2 ||
    model.labels.length > MAX_CLASSES ||
    model.labels.some((label, index) => {
      try {
        requireLabel(label, `Model label ${index}`);
        return false;
      } catch {
        return true;
      }
    }) ||
    new Set(model.labels).size !== model.labels.length
  ) {
    throw new TypeError("Model labels must be valid and unique");
  }

  exactKeys(
    model.vectorizer,
    ["config", "vocabulary", "inverse_document_frequency"],
    "Serialized vectorizer",
  );
  validateVectorizerConfig(model.vectorizer.config, "Serialized vectorizer config");
  if (!sameVectorizerConfig(model.training_config.vectorizer, model.vectorizer.config)) {
    throw new TypeError("Training and serialized vectorizer configs differ");
  }
  const vocabulary = model.vectorizer.vocabulary;
  const idf = model.vectorizer.inverse_document_frequency;
  if (
    !Array.isArray(vocabulary) ||
    !Array.isArray(idf) ||
    vocabulary.length === 0 ||
    vocabulary.length !== idf.length ||
    vocabulary.length > model.vectorizer.config.max_features ||
    new Set(vocabulary).size !== vocabulary.length ||
    vocabulary.some((feature) => !validFeature(feature, model.vectorizer.config)) ||
    idf.some((value) => typeof value !== "number" || !Number.isFinite(value) || value < 1 || value > MAX_IDF)
  ) {
    throw new TypeError("Serialized vectorizer violates its shape contract");
  }
  if (
    !Array.isArray(model.weights) ||
    model.weights.length !== model.labels.length ||
    model.weights.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== vocabulary.length ||
        row.some(
          (weight) =>
            typeof weight !== "number" ||
            !Number.isFinite(weight) ||
            Math.abs(weight) > MAX_PARAMETER_MAGNITUDE,
        ),
    ) ||
    !Array.isArray(model.biases) ||
    model.biases.length !== model.labels.length ||
    model.biases.some(
      (bias) =>
        typeof bias !== "number" ||
        !Number.isFinite(bias) ||
        Math.abs(bias) > MAX_PARAMETER_MAGNITUDE,
    )
  ) {
    throw new TypeError("Model parameters must be finite, bounded, and rectangular");
  }
};

const validatePolicy = (policy, model) => {
  exactKeys(
    policy,
    [
      "schema_version",
      "model_version",
      "dataset_sha256",
      "split_plan_sha256",
      "temperature",
      "minimum_confidence",
      "minimum_probability_margin",
      "temperature_source",
      "threshold_source",
      "calibration_example_count",
      "development_example_count",
      "ood_development_example_count",
    ],
    "Open-set policy",
  );
  if (
    policy.schema_version !== SCHEMA_VERSION ||
    policy.model_version !== model.model_version ||
    policy.dataset_sha256 !== model.dataset_sha256 ||
    policy.split_plan_sha256 !== model.split_plan_sha256 ||
    policy.temperature_source !== "calibration-partition-temperature-scaling-v3" ||
    policy.threshold_source !== "fixed-development-plus-ood-development-grid-v3"
  ) {
    throw new TypeError("Open-set policy does not match its model and provenance");
  }
  boundedNumber(policy.temperature, 0.05, 20, "Policy temperature");
  boundedNumber(policy.minimum_confidence, 0, 1, "Minimum confidence");
  boundedNumber(policy.minimum_probability_margin, 0, 1, "Minimum probability margin");
  boundedInteger(policy.calibration_example_count, 1, MAX_EXAMPLES, "Calibration count");
  boundedInteger(policy.development_example_count, 1, MAX_EXAMPLES, "Development count");
  boundedInteger(
    policy.ood_development_example_count,
    1,
    MAX_EXAMPLES,
    "OOD development count",
  );
};

const validatePlan = (plan, model) => {
  exactKeys(
    plan,
    [
      "schema_version",
      "strategy",
      "seed",
      "dataset_sha256",
      "assignments",
      "ood_development",
      "ood_test",
      "contrast_test",
    ],
    "Split plan",
  );
  if (
    plan.schema_version !== SCHEMA_VERSION ||
    plan.strategy !== "group-stratified-scaled-four-way-v3" ||
    plan.seed !== model.training_config.seed ||
    plan.dataset_sha256 !== model.dataset_sha256
  ) {
    throw new TypeError("Split plan does not match the model provenance");
  }
  if (!Array.isArray(plan.assignments) || plan.assignments.length === 0 || plan.assignments.length > MAX_EXAMPLES) {
    throw new TypeError("Split plan assignments have an invalid size");
  }
  if (
    !Array.isArray(plan.ood_development) ||
    !Array.isArray(plan.ood_test) ||
    !Array.isArray(plan.contrast_test)
  ) {
    throw new TypeError("Split plan evaluation populations must be arrays");
  }
  for (const [name, rows] of [
    ["OOD development", plan.ood_development],
    ["OOD test", plan.ood_test],
    ["Contrast test", plan.contrast_test],
  ]) {
    if (rows.length === 0 || rows.length > MAX_EXAMPLES) {
      throw new TypeError(`${name} has an invalid size`);
    }
  }

  const ids = new Set();
  const groups = new Map();
  const supervisedFamilySizes = new Map();
  const identities = new Set();
  const labelsByPartition = new Map(PARTITIONS.map((partition) => [partition, new Set()]));
  let previousId = null;
  for (const [index, assignment] of plan.assignments.entries()) {
    exactKeys(assignment, ["id", "group_id", "label", "text", "partition"], `Assignment ${index}`);
    requireIdentifier(assignment.id, `Assignment ${index} id`);
    requireIdentifier(assignment.group_id, `Assignment ${index} group`);
    requireLabel(assignment.label, `Assignment ${index} label`);
    requireText(assignment.text, `Assignment ${index} text`);
    if (!PARTITIONS.includes(assignment.partition)) {
      throw new TypeError(`Assignment ${index} has an invalid partition`);
    }
    if (previousId !== null && codePointCompare(previousId, assignment.id) >= 0) {
      throw new TypeError("Split assignments must have unique ascending ids");
    }
    previousId = assignment.id;
    if (ids.has(assignment.id)) throw new TypeError("Split assignments contain duplicate ids");
    ids.add(assignment.id);
    const identity = featureIdentity(assignment.text);
    if (identities.has(identity)) throw new TypeError("Split assignments contain feature-equivalent text");
    identities.add(identity);
    const owner = groups.get(assignment.group_id);
    if (owner && (owner.label !== assignment.label || owner.partition !== assignment.partition)) {
      throw new TypeError("A supervised group crosses labels or partitions");
    }
    groups.set(assignment.group_id, { label: assignment.label, partition: assignment.partition });
    supervisedFamilySizes.set(
      assignment.group_id,
      (supervisedFamilySizes.get(assignment.group_id) ?? 0) + 1,
    );
    labelsByPartition.get(assignment.partition).add(assignment.label);
  }
  const supervisedFamilySupport = new Set(supervisedFamilySizes.values());
  if (
    supervisedFamilySupport.size !== 1 ||
    Math.min(...supervisedFamilySupport) < MIN_PARAPHRASES_PER_FAMILY
  ) {
    throw new TypeError("Supervised paraphrase families must have equal multi-prompt support");
  }
  const expectedLabels = [...model.labels].sort(codePointCompare);
  for (const partition of PARTITIONS) {
    const actualLabels = [...labelsByPartition.get(partition)].sort(codePointCompare);
    if (
      actualLabels.length !== expectedLabels.length ||
      actualLabels.some((label, index) => label !== expectedLabels[index])
    ) {
      throw new TypeError("Every supervised partition must contain every model label");
    }
  }

  const families = { development: new Set(), test: new Set() };
  const domains = { development: new Set(), test: new Set() };
  for (const [population, rows] of [
    ["development", plan.ood_development],
    ["test", plan.ood_test],
  ]) {
    previousId = null;
    const familyOwnership = new Map();
    const domainOwnership = new Map();
    const rowsByStratum = new Map(OOD_STRATA.map((stratum) => [stratum, 0]));
    const domainsByStratum = new Map(OOD_STRATA.map((stratum) => [stratum, new Set()]));
    for (const [index, row] of rows.entries()) {
      exactKeys(
        row,
        ["id", "family_id", "domain_group", "stratum", "text"],
        `OOD ${population} row ${index}`,
      );
      requireIdentifier(row.id, `OOD ${population} row ${index} id`);
      requireIdentifier(row.family_id, `OOD ${population} row ${index} family`);
      requireIdentifier(row.domain_group, `OOD ${population} row ${index} domain`);
      if (!OOD_STRATA.includes(row.stratum)) {
        throw new TypeError(`OOD ${population} row ${index} has an invalid stratum`);
      }
      requireText(row.text, `OOD ${population} row ${index} text`);
      if (previousId !== null && codePointCompare(previousId, row.id) >= 0) {
        throw new TypeError(`OOD ${population} ids must be unique and ascending`);
      }
      previousId = row.id;
      const identity = featureIdentity(row.text);
      if (
        ids.has(row.id) ||
        groups.has(row.family_id) ||
        groups.has(row.domain_group) ||
        identities.has(identity)
      ) {
        throw new TypeError("Experimental populations overlap by id, family, or feature identity");
      }
      ids.add(row.id);
      identities.add(identity);
      families[population].add(row.family_id);
      domains[population].add(row.domain_group);
      const owner = familyOwnership.get(row.family_id);
      if (owner && (owner.domain !== row.domain_group || owner.stratum !== row.stratum)) {
        throw new TypeError("An OOD family crosses domain groups or strata");
      }
      familyOwnership.set(row.family_id, {
        domain: row.domain_group,
        stratum: row.stratum,
        count: (owner?.count ?? 0) + 1,
      });
      const domainOwner = domainOwnership.get(row.domain_group);
      if (domainOwner && domainOwner.stratum !== row.stratum) {
        throw new TypeError("An OOD domain crosses strata");
      }
      const domainFamilies = domainOwner?.families ?? new Set();
      domainFamilies.add(row.family_id);
      domainOwnership.set(row.domain_group, { stratum: row.stratum, families: domainFamilies });
      rowsByStratum.set(row.stratum, rowsByStratum.get(row.stratum) + 1);
      domainsByStratum.get(row.stratum).add(row.domain_group);
    }
    const familySupport = new Set(
      Array.from(familyOwnership.values(), ({ count }) => count),
    );
    const stratumRowSupport = new Set(rowsByStratum.values());
    const stratumDomainSupport = new Set(
      Array.from(domainsByStratum.values(), (domainSet) => domainSet.size),
    );
    if (
      familySupport.size !== 1 ||
      Math.min(...familySupport) < MIN_PARAPHRASES_PER_FAMILY ||
      Array.from(domainOwnership.values()).some(({ families: domainFamilies }) => domainFamilies.size < 2) ||
      stratumRowSupport.size !== 1 ||
      Math.min(...stratumRowSupport) === 0 ||
      stratumDomainSupport.size !== 1 ||
      Math.min(...stratumDomainSupport) === 0
    ) {
      throw new TypeError("OOD families, domains, and strata do not satisfy the balanced support contract");
    }
  }
  if ([...families.development].some((family) => families.test.has(family))) {
    throw new TypeError("OOD development and OOD test overlap by domain family");
  }
  if ([...domains.development].some((domain) => domains.test.has(domain))) {
    throw new TypeError("OOD development and OOD test overlap by broader domain group");
  }

  previousId = null;
  const pairOwnership = new Map();
  const contrastLabelCounts = new Map();
  for (const [index, row] of plan.contrast_test.entries()) {
    exactKeys(
      row,
      ["id", "pair_id", "variant", "label", "text"],
      `Contrast-test row ${index}`,
    );
    requireIdentifier(row.id, `Contrast-test row ${index} id`);
    requireIdentifier(row.pair_id, `Contrast-test row ${index} pair`);
    requireLabel(row.label, `Contrast-test row ${index} label`);
    requireText(row.text, `Contrast-test row ${index} text`);
    if (!CONTRAST_VARIANTS.includes(row.variant)) {
      throw new TypeError(`Contrast-test row ${index} has an invalid variant`);
    }
    if (previousId !== null && codePointCompare(previousId, row.id) >= 0) {
      throw new TypeError("Contrast-test ids must be unique and ascending");
    }
    previousId = row.id;
    const identity = featureIdentity(row.text);
    if (
      ids.has(row.id) ||
      groups.has(row.pair_id) ||
      families.development.has(row.pair_id) ||
      families.test.has(row.pair_id) ||
      domains.development.has(row.pair_id) ||
      domains.test.has(row.pair_id) ||
      identities.has(identity)
    ) {
      throw new TypeError(
        "Contrast test overlaps another population by id, pair, or feature identity",
      );
    }
    ids.add(row.id);
    identities.add(identity);
    const owner = pairOwnership.get(row.pair_id) ?? { variants: new Set(), labels: new Set() };
    if (owner.variants.has(row.variant)) {
      throw new TypeError("A contrast pair repeats a variant");
    }
    owner.variants.add(row.variant);
    owner.labels.add(row.label);
    pairOwnership.set(row.pair_id, owner);
    contrastLabelCounts.set(row.label, (contrastLabelCounts.get(row.label) ?? 0) + 1);
  }
  if (
    pairOwnership.size === 0 ||
    [...pairOwnership.values()].some(
      ({ variants, labels }) => variants.size !== 2 || labels.size !== 2,
    )
  ) {
    throw new TypeError("Every contrast pair must contain a and b with different labels");
  }
  const actualContrastLabels = [...contrastLabelCounts.keys()].sort(codePointCompare);
  const contrastSupports = new Set(contrastLabelCounts.values());
  if (
    actualContrastLabels.length !== expectedLabels.length ||
    actualContrastLabels.some((label, index) => label !== expectedLabels[index]) ||
    contrastSupports.size !== 1 ||
    Math.min(...contrastSupports) < 2
  ) {
    throw new TypeError("The contrast test must contain every label with equal support");
  }
};

const validateMetricEstimate = (estimate, point, description) => {
  exactKeys(estimate, ["value", "lower_95", "upper_95"], description);
  const value = finiteNumber(estimate.value, `${description}.value`);
  const lower = finiteNumber(estimate.lower_95, `${description}.lower_95`);
  const upper = finiteNumber(estimate.upper_95, `${description}.upper_95`);
  if (Math.abs(value - point) > 1e-12 || lower > value || value > upper) {
    throw new TypeError(`${description} does not contain its point estimate`);
  }
};

const validateCalibrationMetrics = (calibration, description) => {
  exactKeys(
    calibration,
    ["negative_log_likelihood", "multiclass_brier", "expected_calibration_error", "ece_bins"],
    description,
  );
  boundedNumber(calibration.negative_log_likelihood, 0, Number.MAX_VALUE, `${description} NLL`);
  boundedNumber(calibration.multiclass_brier, 0, 2, `${description} Brier score`);
  boundedNumber(calibration.expected_calibration_error, 0, 1, `${description} ECE`);
  boundedInteger(calibration.ece_bins, 1, MAX_EXAMPLES, `${description} ECE bins`);
};

const developmentCandidateIsBetter = (candidate, current, macroF1Tolerance) => {
  const macroF1Difference = candidate.macro_f1 - current.macro_f1;
  if (macroF1Difference > macroF1Tolerance) return true;
  if (macroF1Difference < -macroF1Tolerance) return false;
  for (const [left, right, higherIsBetter] of [
    [candidate.max_features, current.max_features, false],
    [candidate.l2_penalty, current.l2_penalty, true],
    [candidate.accuracy, current.accuracy, true],
    [candidate.negative_log_likelihood, current.negative_log_likelihood, false],
    [candidate.multiclass_brier, current.multiclass_brier, false],
  ]) {
    if (left === right) continue;
    return higherIsBetter ? left > right : left < right;
  }
  return false;
};

const validateBaselineEvaluation = (evaluation, model, description) => {
  exactKeys(
    evaluation,
    ["accuracy", "macro_f1", "confusion_matrix", "per_class"],
    description,
  );
  boundedNumber(evaluation.accuracy, 0, 1, `${description} accuracy`);
  boundedNumber(evaluation.macro_f1, 0, 1, `${description} macro F1`);
  if (
    !Array.isArray(evaluation.confusion_matrix) ||
    evaluation.confusion_matrix.length !== model.labels.length ||
    evaluation.confusion_matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== model.labels.length ||
        row.some((value) => !Number.isSafeInteger(value) || value < 0),
    ) ||
    !Array.isArray(evaluation.per_class) ||
    evaluation.per_class.length !== model.labels.length
  ) {
    throw new TypeError(`${description} has an invalid class-report shape`);
  }
};

const validateMetrics = (metrics, model, policy, plan) => {
  exactKeys(
    metrics,
    [
      "schema_version",
      "model_version",
      "dataset_sha256",
      "split_plan_sha256",
      "ood_development_sha256",
      "ood_test_sha256",
      "contrast_test_sha256",
      "partition_counts",
      "partition_family_counts",
      "paraphrases_per_family",
      "ood_domain_counts",
      "ood_stratum_counts",
      "development_selection",
      "threshold_selection",
      "uncalibrated_calibration_partition",
      "calibrated_calibration_partition",
      "id_test",
      "baselines",
      "contrast_test",
      "ood_test",
      "bootstrap_95",
      "limitations",
    ],
    "Open-set metrics",
  );
  if (
    metrics.schema_version !== SCHEMA_VERSION ||
    metrics.model_version !== model.model_version ||
    metrics.dataset_sha256 !== model.dataset_sha256 ||
    metrics.split_plan_sha256 !== model.split_plan_sha256
  ) {
    throw new TypeError("Metrics do not match the model provenance");
  }
  requireSha256(metrics.ood_development_sha256, "OOD development digest");
  requireSha256(metrics.ood_test_sha256, "OOD test digest");
  requireSha256(metrics.contrast_test_sha256, "Contrast test digest");
  if (
    new Set([
      metrics.dataset_sha256,
      metrics.ood_development_sha256,
      metrics.ood_test_sha256,
      metrics.contrast_test_sha256,
    ]).size !== 4
  ) {
    throw new TypeError("Evaluation populations must have distinct source fingerprints");
  }
  exactKeys(
    metrics.partition_counts,
    [...PARTITIONS, "ood-development", "ood-test", "contrast-test"],
    "Partition counts",
  );
  const actualCounts = Object.fromEntries(PARTITIONS.map((partition) => [partition, 0]));
  const actualFamilies = Object.fromEntries(PARTITIONS.map((partition) => [partition, new Set()]));
  for (const assignment of plan.assignments) {
    actualCounts[assignment.partition] += 1;
    actualFamilies[assignment.partition].add(assignment.group_id);
  }
  actualCounts["ood-development"] = plan.ood_development.length;
  actualCounts["ood-test"] = plan.ood_test.length;
  actualCounts["contrast-test"] = plan.contrast_test.length;
  actualFamilies["ood-development"] = new Set(plan.ood_development.map((row) => row.family_id));
  actualFamilies["ood-test"] = new Set(plan.ood_test.map((row) => row.family_id));
  actualFamilies["contrast-test"] = new Set(plan.contrast_test.map((row) => row.pair_id));
  for (const [partition, count] of Object.entries(actualCounts)) {
    if (metrics.partition_counts[partition] !== count) {
      throw new TypeError(`Metrics contain the wrong ${partition} count`);
    }
  }
  exactKeys(
    metrics.partition_family_counts,
    [...PARTITIONS, "ood-development", "ood-test", "contrast-test"],
    "Partition family counts",
  );
  for (const [partition, families] of Object.entries(actualFamilies)) {
    if (metrics.partition_family_counts[partition] !== families.size) {
      throw new TypeError(`Metrics contain the wrong ${partition} family count`);
    }
  }
  const supervisedFamilySizes = new Map();
  for (const assignment of plan.assignments) {
    supervisedFamilySizes.set(
      assignment.group_id,
      (supervisedFamilySizes.get(assignment.group_id) ?? 0) + 1,
    );
  }
  const familySupport = new Set(supervisedFamilySizes.values());
  if (
    familySupport.size !== 1 ||
    metrics.paraphrases_per_family !== familySupport.values().next().value
  ) {
    throw new TypeError("Metrics contain the wrong supervised family support");
  }
  exactKeys(metrics.ood_domain_counts, ["ood-development", "ood-test"], "OOD domain counts");
  exactKeys(metrics.ood_stratum_counts, ["ood-development", "ood-test"], "OOD stratum counts");
  for (const [partition, rows] of [
    ["ood-development", plan.ood_development],
    ["ood-test", plan.ood_test],
  ]) {
    if (metrics.ood_domain_counts[partition] !== new Set(rows.map((row) => row.domain_group)).size) {
      throw new TypeError(`Metrics contain the wrong ${partition} domain count`);
    }
    exactKeys(metrics.ood_stratum_counts[partition], OOD_STRATA, `${partition} stratum counts`);
    for (const stratum of OOD_STRATA) {
      if (
        metrics.ood_stratum_counts[partition][stratum] !==
        rows.filter((row) => row.stratum === stratum).length
      ) {
        throw new TypeError(`Metrics contain the wrong ${partition} ${stratum} count`);
      }
    }
  }
  if (
    policy.calibration_example_count !== actualCounts.calibration ||
    policy.development_example_count !== actualCounts.development ||
    policy.ood_development_example_count !== actualCounts["ood-development"]
  ) {
    throw new TypeError("Policy provenance counts do not match the split plan");
  }

  const selection = metrics.development_selection;
  exactKeys(
    selection,
    [
      "strategy",
      "seed",
      "macro_f1_tolerance",
      "training_example_count",
      "training_family_count",
      "development_example_count",
      "development_family_count",
      "candidates",
      "selected_index",
      "inputs",
    ],
    "Development selection report",
  );
  const expectedCandidateGrid = model.training_config.development_selection.max_features_candidates
    .flatMap((maxFeatures) =>
      model.training_config.development_selection.l2_penalty_candidates.map((l2Penalty) => [
        maxFeatures,
        l2Penalty,
      ]),
    );
  if (
    selection.strategy !== "train-fit-development-f1-epsilon-parsimony-accuracy-nll-brier-v3" ||
    selection.seed !== model.training_config.seed ||
    selection.macro_f1_tolerance !==
      model.training_config.development_selection.macro_f1_tolerance ||
    selection.training_example_count !== actualCounts.train ||
    selection.training_family_count !== actualFamilies.train.size ||
    selection.development_example_count !== actualCounts.development ||
    selection.development_family_count !== actualFamilies.development.size ||
    !Array.isArray(selection.inputs) ||
    selection.inputs.length !== 2 ||
    selection.inputs[0] !== "train" ||
    selection.inputs[1] !== "development" ||
    !Array.isArray(selection.candidates) ||
    selection.candidates.length !== expectedCandidateGrid.length
  ) {
    throw new TypeError("Development-only model selection has invalid provenance");
  }
  selection.candidates.forEach((candidate, index) => {
    exactKeys(
      candidate,
      [
        "max_features",
        "l2_penalty",
        "accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "multiclass_brier",
      ],
      `Development candidate ${index}`,
    );
    const [expectedFeatures, expectedL2] = expectedCandidateGrid[index];
    if (candidate.max_features !== expectedFeatures || candidate.l2_penalty !== expectedL2) {
      throw new TypeError("Development candidates do not match their declared grid");
    }
    boundedNumber(candidate.accuracy, 0, 1, `Development candidate ${index} accuracy`);
    boundedNumber(candidate.macro_f1, 0, 1, `Development candidate ${index} macro F1`);
    boundedNumber(
      candidate.negative_log_likelihood,
      0,
      Number.MAX_VALUE,
      `Development candidate ${index} NLL`,
    );
    boundedNumber(candidate.multiclass_brier, 0, 2, `Development candidate ${index} Brier`);
  });
  boundedInteger(
    selection.selected_index,
    0,
    selection.candidates.length - 1,
    "Selected development candidate",
  );
  const reproducedSelection = selection.candidates.reduce(
    (best, candidate, index) =>
      developmentCandidateIsBetter(
        candidate,
        selection.candidates[best],
        selection.macro_f1_tolerance,
      )
        ? index
        : best,
    0,
  );
  const selectedCandidate = selection.candidates[selection.selected_index];
  if (
    selection.selected_index !== reproducedSelection ||
    selectedCandidate.max_features !== model.training_config.vectorizer.max_features ||
    selectedCandidate.l2_penalty !== model.training_config.l2_penalty
  ) {
    throw new TypeError("Fitted model disagrees with development-only selection");
  }

  exactKeys(
    metrics.threshold_selection,
    [
      "strategy",
      "evaluated_candidate_count",
      "feasible_candidate_count",
      "development_example_count",
      "ood_development_example_count",
      "minimum_development_selective_accuracy",
      "maximum_ood_development_coverage",
      "selected_confidence",
      "selected_probability_margin",
      "observed_development_coverage",
      "observed_development_selective_accuracy",
      "observed_ood_development_coverage",
      "inputs",
    ],
    "Threshold selection report",
  );
  if (
    metrics.threshold_selection.strategy !== policy.threshold_source ||
    metrics.threshold_selection.development_example_count !== policy.development_example_count ||
    metrics.threshold_selection.ood_development_example_count !== policy.ood_development_example_count ||
    metrics.threshold_selection.selected_confidence !== policy.minimum_confidence ||
    metrics.threshold_selection.selected_probability_margin !== policy.minimum_probability_margin ||
    metrics.threshold_selection.minimum_development_selective_accuracy !== 0.75 ||
    metrics.threshold_selection.maximum_ood_development_coverage !== 0.1 ||
    metrics.threshold_selection.evaluated_candidate_count !== 49 ||
    !Number.isSafeInteger(metrics.threshold_selection.feasible_candidate_count) ||
    metrics.threshold_selection.feasible_candidate_count < 1 ||
    metrics.threshold_selection.feasible_candidate_count >
      metrics.threshold_selection.evaluated_candidate_count ||
    !Array.isArray(metrics.threshold_selection.inputs) ||
    metrics.threshold_selection.inputs.length !== 2 ||
    metrics.threshold_selection.inputs[0] !== "development" ||
    metrics.threshold_selection.inputs[1] !== "ood-development"
  ) {
    throw new TypeError("Threshold report disagrees with the frozen policy");
  }
  for (const value of [
    metrics.threshold_selection.observed_development_coverage,
    metrics.threshold_selection.observed_development_selective_accuracy,
    metrics.threshold_selection.observed_ood_development_coverage,
  ]) {
    boundedNumber(value, 0, 1, "Observed threshold metric");
  }

  if (!Array.isArray(metrics.limitations) || metrics.limitations.some((item) => typeof item !== "string")) {
    throw new TypeError("Metrics limitations must be text");
  }
  exactKeys(
    metrics.id_test,
    [
      "example_count",
      "accuracy",
      "macro_f1",
      "labels",
      "confusion_matrix",
      "per_class",
      "coverage",
      "selective_accuracy",
      "calibration",
      "aurc",
      "risk_coverage_curve",
      "predictions",
    ],
    "ID-test report",
  );
  exactKeys(
    metrics.ood_test,
    [
      "example_count",
      "accepted_examples",
      "coverage",
      "discrimination",
      "by_stratum",
      "predictions",
    ],
    "OOD-test report",
  );
  exactKeys(
    metrics.contrast_test,
    [
      "example_count",
      "pair_count",
      "accuracy",
      "macro_f1",
      "pair_accuracy",
      "prediction_flip_rate",
      "coverage",
      "confusion_matrix",
      "per_class",
      "predictions",
    ],
    "Contrast-test report",
  );
  if (
    metrics.id_test.example_count !== actualCounts["id-test"] ||
    metrics.ood_test.example_count !== actualCounts["ood-test"] ||
    metrics.contrast_test.example_count !== actualCounts["contrast-test"] ||
    metrics.contrast_test.pair_count !== actualFamilies["contrast-test"].size ||
    !Array.isArray(metrics.id_test.predictions) ||
    !Array.isArray(metrics.ood_test.predictions) ||
    !Array.isArray(metrics.contrast_test.predictions) ||
    metrics.id_test.predictions.length !== actualCounts["id-test"] ||
    metrics.ood_test.predictions.length !== actualCounts["ood-test"] ||
    metrics.contrast_test.predictions.length !== actualCounts["contrast-test"]
  ) {
    throw new TypeError("Evaluation prediction counts disagree with the split plan");
  }
  validateCalibrationMetrics(
    metrics.uncalibrated_calibration_partition,
    "Uncalibrated calibration partition",
  );
  validateCalibrationMetrics(
    metrics.calibrated_calibration_partition,
    "Calibrated calibration partition",
  );
  validateCalibrationMetrics(metrics.id_test.calibration, "ID-test calibration");
  for (const value of [
    metrics.id_test.accuracy,
    metrics.id_test.macro_f1,
    metrics.id_test.coverage,
    metrics.id_test.aurc,
    metrics.ood_test.coverage,
    metrics.contrast_test.accuracy,
    metrics.contrast_test.macro_f1,
    metrics.contrast_test.pair_accuracy,
    metrics.contrast_test.prediction_flip_rate,
    metrics.contrast_test.coverage,
  ]) boundedNumber(value, 0, 1, "Evaluation metric");
  if (
    !Array.isArray(metrics.contrast_test.confusion_matrix) ||
    metrics.contrast_test.confusion_matrix.length !== model.labels.length ||
    metrics.contrast_test.confusion_matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== model.labels.length ||
        row.some((value) => !Number.isSafeInteger(value) || value < 0),
    ) ||
    !Array.isArray(metrics.contrast_test.per_class) ||
    metrics.contrast_test.per_class.length !== model.labels.length
  ) {
    throw new TypeError("Contrast-test class report has an invalid shape");
  }
  if (metrics.id_test.selective_accuracy !== null) {
    boundedNumber(metrics.id_test.selective_accuracy, 0, 1, "Selective accuracy");
  }
  boundedInteger(
    metrics.ood_test.accepted_examples,
    0,
    metrics.ood_test.example_count,
    "OOD accepted examples",
  );
  exactKeys(metrics.ood_test.discrimination, ["auroc", "aupr_in_domain", "fpr_at_95_tpr"], "OOD discrimination");
  for (const value of Object.values(metrics.ood_test.discrimination)) boundedNumber(value, 0, 1, "OOD metric");
  exactKeys(metrics.ood_test.by_stratum, OOD_STRATA, "OOD metrics by stratum");
  for (const stratum of OOD_STRATA) {
    const summary = metrics.ood_test.by_stratum[stratum];
    exactKeys(
      summary,
      ["example_count", "accepted_examples", "coverage", "discrimination"],
      `${stratum} OOD summary`,
    );
    boundedInteger(summary.example_count, 1, MAX_EXAMPLES, `${stratum} OOD count`);
    boundedInteger(
      summary.accepted_examples,
      0,
      summary.example_count,
      `${stratum} accepted OOD count`,
    );
    boundedNumber(summary.coverage, 0, 1, `${stratum} OOD coverage`);
    exactKeys(
      summary.discrimination,
      ["auroc", "aupr_in_domain", "fpr_at_95_tpr"],
      `${stratum} OOD discrimination`,
    );
    for (const value of Object.values(summary.discrimination)) {
      boundedNumber(value, 0, 1, `${stratum} OOD discrimination metric`);
    }
  }
  if (
    !Array.isArray(metrics.id_test.risk_coverage_curve) ||
    metrics.id_test.risk_coverage_curve.length === 0 ||
    metrics.id_test.risk_coverage_curve.length > metrics.id_test.example_count
  ) {
    throw new TypeError("ID risk-coverage curve has an invalid length");
  }
  metrics.id_test.risk_coverage_curve.forEach((point, index) => {
    exactKeys(point, ["accepted", "coverage", "risk"], `Risk-coverage point ${index}`);
    if (
      point.accepted < 1 ||
      point.accepted > metrics.id_test.example_count ||
      (index > 0 && point.accepted <= metrics.id_test.risk_coverage_curve[index - 1].accepted)
    ) {
      throw new TypeError("Risk-coverage accept counts are not ordered");
    }
    if (!approximatelyEqual(point.coverage, point.accepted / metrics.id_test.example_count)) {
      throw new TypeError("Risk-coverage point has the wrong coverage");
    }
    boundedNumber(point.risk, 0, 1, `Risk-coverage point ${index} risk`);
  });

  exactKeys(
    metrics.baselines,
    [
      "strategy",
      "inputs",
      "evaluation_partition",
      "training_example_count",
      "training_family_count",
      "majority_label",
      "majority",
      "unigram_naive_bayes",
      "learned_minus_unigram_accuracy",
      "learned_minus_unigram_macro_f1",
    ],
    "Baseline report",
  );
  if (
    metrics.baselines.strategy !== "training-only-majority-and-laplace-unigram-naive-bayes-v3" ||
    !Array.isArray(metrics.baselines.inputs) ||
    metrics.baselines.inputs.length !== 1 ||
    metrics.baselines.inputs[0] !== "train" ||
    metrics.baselines.evaluation_partition !== "id-test" ||
    metrics.baselines.training_example_count !== actualCounts.train ||
    metrics.baselines.training_family_count !== actualFamilies.train.size ||
    !model.labels.includes(metrics.baselines.majority_label)
  ) {
    throw new TypeError("Baseline report has invalid train-only provenance");
  }
  finiteNumber(
    metrics.baselines.learned_minus_unigram_accuracy,
    "Learned-minus-unigram accuracy",
  );
  finiteNumber(
    metrics.baselines.learned_minus_unigram_macro_f1,
    "Learned-minus-unigram macro F1",
  );
  validateBaselineEvaluation(metrics.baselines.majority, model, "Majority baseline");
  validateBaselineEvaluation(
    metrics.baselines.unigram_naive_bayes,
    model,
    "Unigram Naive Bayes baseline",
  );

  exactKeys(
    metrics.bootstrap_95,
    [
      "strategy",
      "seed",
      "resamples",
      "confidence_level",
      "id_accuracy",
      "id_macro_f1",
      "id_negative_log_likelihood",
      "id_multiclass_brier",
      "id_expected_calibration_error",
      "id_aurc",
      "ood_auroc",
      "ood_aupr_in_domain",
      "ood_fpr_at_95_tpr",
    ],
    "Bootstrap report",
  );
  if (
    metrics.bootstrap_95.strategy !==
      "label-stratified-id-family-and-ood-domain-cluster-percentile-v3"
  ) {
    throw new TypeError("Bootstrap strategy is unsupported");
  }
  boundedInteger(metrics.bootstrap_95.seed, 0, JSON_SAFE_INTEGER, "Bootstrap seed");
  boundedInteger(metrics.bootstrap_95.resamples, 100, 20_000, "Bootstrap resamples");
  boundedNumber(metrics.bootstrap_95.confidence_level, 0, 1, "Bootstrap confidence level");
  for (const [estimate, point, description] of [
    [metrics.bootstrap_95.id_accuracy, metrics.id_test.accuracy, "ID accuracy interval"],
    [metrics.bootstrap_95.id_macro_f1, metrics.id_test.macro_f1, "ID macro-F1 interval"],
    [
      metrics.bootstrap_95.id_negative_log_likelihood,
      metrics.id_test.calibration.negative_log_likelihood,
      "ID NLL interval",
    ],
    [
      metrics.bootstrap_95.id_multiclass_brier,
      metrics.id_test.calibration.multiclass_brier,
      "ID Brier interval",
    ],
    [
      metrics.bootstrap_95.id_expected_calibration_error,
      metrics.id_test.calibration.expected_calibration_error,
      "ID ECE interval",
    ],
    [metrics.bootstrap_95.id_aurc, metrics.id_test.aurc, "ID AURC interval"],
    [metrics.bootstrap_95.ood_auroc, metrics.ood_test.discrimination.auroc, "OOD AUROC interval"],
    [metrics.bootstrap_95.ood_aupr_in_domain, metrics.ood_test.discrimination.aupr_in_domain, "OOD AUPR interval"],
    [metrics.bootstrap_95.ood_fpr_at_95_tpr, metrics.ood_test.discrimination.fpr_at_95_tpr, "OOD FPR95 interval"],
  ]) validateMetricEstimate(estimate, point, description);
};

const deepFreeze = (value) => {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
};

class VerifiedOpenSetBundle {
  #model;
  #policy;
  #metrics;
  #plan;
  #featureIndex;

  constructor(model, policy, metrics, plan) {
    this.#model = model;
    this.#policy = policy;
    this.#metrics = metrics;
    this.#plan = plan;
    this.#featureIndex = new Map(model.vectorizer.vocabulary.map((feature, index) => [feature, index]));
  }

  get version() {
    return this.#model.model_version;
  }

  get metrics() {
    return this.#metrics;
  }

  get splitPlan() {
    return this.#plan;
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

  predict(input) {
    const text = String(input ?? "");
    let characterCount = 0;
    let oversized = false;
    for (const _character of text) {
      characterCount += 1;
      if (characterCount > MAX_INPUT_CHARS) {
        oversized = true;
        break;
      }
    }
    const features = this.#transform(oversized ? "" : text);
    const logits = this.#model.weights.map(
      (row, classIndex) =>
        this.#model.biases[classIndex] +
        features.reduce((total, [feature, value]) => total + row[feature] * value, 0),
    );
    const scaled = logits.map((logit) => logit / this.#policy.temperature);
    const maximum = Math.max(...scaled);
    const exponentials = scaled.map((logit) => Math.exp(logit - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map((value) => value / total);
    const ranking = probabilities
      .map((probability, index) => ({ index, probability }))
      .sort((left, right) => right.probability - left.probability || left.index - right.index);
    const top = ranking[0];
    const runnerUp = ranking[1];
    const confidence = top.probability;
    const probabilityMargin = confidence - runnerUp.probability;
    const logitMargin = logits[top.index] - logits[runnerUp.index];
    const accepted =
      features.length > 0 &&
      confidence >= this.#policy.minimum_confidence &&
      probabilityMargin >= this.#policy.minimum_probability_margin;
    const biasDifference = this.#model.biases[top.index] - this.#model.biases[runnerUp.index];
    const contributions = features.map(([feature, value]) => {
      const topWeight = this.#model.weights[top.index][feature];
      const runnerUpWeight = this.#model.weights[runnerUp.index][feature];
      return {
        feature: this.#model.vectorizer.vocabulary[feature],
        value,
        weight: topWeight - runnerUpWeight,
        topWeight,
        runnerUpWeight,
        contribution: value * (topWeight - runnerUpWeight),
      };
    });
    const featureContributionSum = contributions.reduce(
      (sum, contribution) => sum + contribution.contribution,
      0,
    );
    contributions.sort(
      (left, right) =>
        Math.abs(right.contribution) - Math.abs(left.contribution) ||
        codePointCompare(left.feature, right.feature),
    );
    const topFeatures = contributions.slice(0, 8);
    const probabilityRecord = Object.fromEntries(
      this.#model.labels.map((label, index) => [label, probabilities[index]]),
    );
    return {
      label: this.#model.labels[top.index],
      runnerUpLabel: this.#model.labels[runnerUp.index],
      accepted,
      confidence,
      margin: probabilityMargin,
      probabilityMargin,
      logitMargin,
      probabilities: probabilityRecord,
      topFeatures,
      explanation: {
        topLabel: this.#model.labels[top.index],
        runnerUpLabel: this.#model.labels[runnerUp.index],
        biasDifference,
        featureContributionSum,
        reconstructedLogitMargin: biasDifference + featureContributionSum,
        topContributions: topFeatures,
      },
    };
  }
}

const approximatelyEqual = (left, right) =>
  Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= 1e-10;

const calibrationFromPredictions = (predictions, bins) => {
  const count = predictions.length;
  const negativeLogLikelihood =
    predictions.reduce(
      (total, prediction) =>
        total - Math.log(Math.max(prediction.probabilities[prediction.actual_label], 1e-15)),
      0,
    ) / count;
  const multiclassBrier =
    predictions.reduce(
      (total, prediction) =>
        total +
        Object.entries(prediction.probabilities).reduce(
          (rowTotal, [label, probability]) =>
            rowTotal + (probability - Number(label === prediction.actual_label)) ** 2,
          0,
        ),
      0,
    ) / count;
  let expectedCalibrationError = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    const lower = bin / bins;
    const upper = (bin + 1) / bins;
    const members = predictions.filter(
      (prediction) =>
        prediction.confidence >= lower &&
        (prediction.confidence < upper || (bin + 1 === bins && prediction.confidence <= upper)),
    );
    if (members.length === 0) continue;
    const accuracy = members.filter((prediction) => prediction.correct).length / members.length;
    const confidence =
      members.reduce((total, prediction) => total + prediction.confidence, 0) / members.length;
    expectedCalibrationError += (members.length / count) * Math.abs(accuracy - confidence);
  }
  return {
    negative_log_likelihood: negativeLogLikelihood,
    multiclass_brier: multiclassBrier,
    expected_calibration_error: expectedCalibrationError,
    ece_bins: bins,
  };
};

const calibrationMatches = (recorded, reproduced) =>
  recorded.ece_bins === reproduced.ece_bins &&
  approximatelyEqual(recorded.negative_log_likelihood, reproduced.negative_log_likelihood) &&
  approximatelyEqual(recorded.multiclass_brier, reproduced.multiclass_brier) &&
  approximatelyEqual(recorded.expected_calibration_error, reproduced.expected_calibration_error);

const summarizePerClass = (labels, predictions) => {
  const labelIndex = new Map(labels.map((label, index) => [label, index]));
  const confusionMatrix = Array.from({ length: labels.length }, () =>
    Array.from({ length: labels.length }, () => 0),
  );
  for (const prediction of predictions) {
    confusionMatrix[labelIndex.get(prediction.actual_label)][
      labelIndex.get(prediction.predicted_label)
    ] += 1;
  }
  const perClass = labels.map((label, index) => {
    const support = confusionMatrix[index].reduce((total, value) => total + value, 0);
    const predicted = confusionMatrix.reduce((total, row) => total + row[index], 0);
    const truePositive = confusionMatrix[index][index];
    const precision = predicted === 0 ? 0 : truePositive / predicted;
    const recall = support === 0 ? 0 : truePositive / support;
    return {
      label,
      support,
      predicted,
      true_positive: truePositive,
      precision,
      recall,
      f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
    };
  });
  return {
    confusionMatrix,
    perClass,
    macroF1: perClass.reduce((total, metrics) => total + metrics.f1, 0) / perClass.length,
  };
};

const riskCoverage = (predictions) => {
  const ranked = [...predictions].sort(
    (left, right) =>
      right.confidence - left.confidence || codePointCompare(left.id, right.id),
  );
  const curve = [];
  let correct = 0;
  let index = 0;
  let aurc = 0;
  while (index < ranked.length) {
    const confidence = ranked[index].confidence;
    const start = index;
    while (index < ranked.length && ranked[index].confidence === confidence) {
      correct += Number(ranked[index].correct);
      index += 1;
    }
    const risk = 1 - correct / index;
    aurc += risk * ((index - start) / ranked.length);
    curve.push({ accepted: index, coverage: index / ranked.length, risk });
  }
  return { curve, aurc };
};

const discriminationFromScores = (idScores, oodScores) => {
  const ranked = [
    ...idScores.map((score) => ({ score, inDomain: true })),
    ...oodScores.map((score) => ({ score, inDomain: false })),
  ].sort((left, right) => right.score - left.score);
  let truePositives = 0;
  let falsePositives = 0;
  let aurocWins = 0;
  let auprInDomain = 0;
  let index = 0;
  while (index < ranked.length) {
    const score = ranked[index].score;
    let groupPositives = 0;
    let groupNegatives = 0;
    while (index < ranked.length && ranked[index].score === score) {
      if (ranked[index].inDomain) groupPositives += 1;
      else groupNegatives += 1;
      index += 1;
    }
    const lowerScoringNegatives = oodScores.length - falsePositives - groupNegatives;
    aurocWins += groupPositives * lowerScoringNegatives + 0.5 * groupPositives * groupNegatives;
    truePositives += groupPositives;
    falsePositives += groupNegatives;
    if (groupPositives > 0) {
      auprInDomain +=
        (truePositives / (truePositives + falsePositives)) *
        (groupPositives / idScores.length);
    }
  }
  const sortedId = [...idScores].sort((left, right) => right - left);
  const targetRank = Math.min(Math.max(Math.ceil(0.95 * sortedId.length) - 1, 0), sortedId.length - 1);
  const threshold = sortedId[targetRank];
  return {
    auroc: aurocWins / (idScores.length * oodScores.length),
    aupr_in_domain: auprInDomain,
    fpr_at_95_tpr: oodScores.filter((score) => score >= threshold).length / oodScores.length,
  };
};

const uncalibratedProbabilities = (calibrated, temperature) => {
  const powered = Object.fromEntries(
    Object.entries(calibrated).map(([label, probability]) => [label, probability ** temperature]),
  );
  const total = Object.values(powered).reduce((sum, probability) => sum + probability, 0);
  return Object.fromEntries(
    Object.entries(powered).map(([label, probability]) => [label, probability / total]),
  );
};

const summarizeBaselinePredictions = (labels, predictions) => {
  const summary = summarizePerClass(labels, predictions);
  const correct = predictions.filter(
    (prediction) => prediction.actual_label === prediction.predicted_label,
  ).length;
  return {
    accuracy: correct / predictions.length,
    macro_f1: summary.macroF1,
    confusion_matrix: summary.confusionMatrix,
    per_class: summary.perClass,
  };
};

const reproduceBaselines = (plan, labels, learned) => {
  const training = plan.assignments.filter((row) => row.partition === "train");
  const idTest = plan.assignments.filter((row) => row.partition === "id-test");
  const documentCounts = new Map(labels.map((label) => [label, 0]));
  const tokenCounts = new Map(labels.map((label) => [label, new Map()]));
  const tokenTotals = new Map(labels.map((label) => [label, 0]));
  const vocabulary = new Set();
  for (const row of training) {
    documentCounts.set(row.label, documentCounts.get(row.label) + 1);
    for (const token of tokenize(row.text)) {
      vocabulary.add(token);
      const counts = tokenCounts.get(row.label);
      counts.set(token, (counts.get(token) ?? 0) + 1);
      tokenTotals.set(row.label, tokenTotals.get(row.label) + 1);
    }
  }
  const majorityLabel = [...labels].sort(
    (left, right) =>
      documentCounts.get(right) - documentCounts.get(left) || codePointCompare(left, right),
  )[0];
  const majorityPredictions = idTest.map((row) => ({
    actual_label: row.label,
    predicted_label: majorityLabel,
  }));
  const unigramPredictions = idTest.map((row) => {
    const tokens = tokenize(row.text);
    let bestLabel = labels[0];
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const label of labels) {
      let score = Math.log(
        (documentCounts.get(label) + 1) / (training.length + labels.length),
      );
      const denominator = tokenTotals.get(label) + vocabulary.size;
      for (const token of tokens) {
        if (vocabulary.has(token)) {
          score += Math.log(((tokenCounts.get(label).get(token) ?? 0) + 1) / denominator);
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }
    return { actual_label: row.label, predicted_label: bestLabel };
  });
  const majority = summarizeBaselinePredictions(labels, majorityPredictions);
  const unigramNaiveBayes = summarizeBaselinePredictions(labels, unigramPredictions);
  return {
    strategy: "training-only-majority-and-laplace-unigram-naive-bayes-v3",
    inputs: ["train"],
    evaluation_partition: "id-test",
    training_example_count: training.length,
    training_family_count: new Set(training.map((row) => row.group_id)).size,
    majority_label: majorityLabel,
    majority,
    unigram_naive_bayes: unigramNaiveBayes,
    learned_minus_unigram_accuracy: learned.accuracy - unigramNaiveBayes.accuracy,
    learned_minus_unigram_macro_f1: learned.macro_f1 - unigramNaiveBayes.macro_f1,
  };
};

const baselineEvaluationMatches = (recorded, reproduced) =>
  approximatelyEqual(recorded.accuracy, reproduced.accuracy) &&
  approximatelyEqual(recorded.macro_f1, reproduced.macro_f1) &&
  JSON.stringify(recorded.confusion_matrix) === JSON.stringify(reproduced.confusion_matrix) &&
  recorded.per_class.length === reproduced.per_class.length &&
  recorded.per_class.every((entry, index) => {
    const expected = reproduced.per_class[index];
    return (
      entry.label === expected.label &&
      entry.support === expected.support &&
      entry.predicted === expected.predicted &&
      entry.true_positive === expected.true_positive &&
      approximatelyEqual(entry.precision, expected.precision) &&
      approximatelyEqual(entry.recall, expected.recall) &&
      approximatelyEqual(entry.f1, expected.f1)
    );
  });

const validateRecordedPredictions = (runtime, model, policy) => {
  const idRows = new Map(
    runtime.splitPlan.assignments
      .filter((assignment) => assignment.partition === "id-test")
      .map((assignment) => [assignment.id, assignment]),
  );
  let acceptedId = 0;
  let correctId = 0;
  let acceptedCorrectId = 0;
  const recordedId = [];
  for (const [index, recorded] of runtime.metrics.id_test.predictions.entries()) {
    exactKeys(
      recorded,
      ["id", "actual_label", "predicted_label", "correct", "accepted", "confidence", "probability_margin", "probabilities"],
      `Recorded ID prediction ${index}`,
    );
    const row = idRows.get(recorded.id);
    if (!row || row.label !== recorded.actual_label) throw new TypeError("Recorded ID prediction has no split-plan row");
    const prediction = runtime.predict(row.text);
    if (
      prediction.label !== recorded.predicted_label ||
      prediction.accepted !== recorded.accepted ||
      recorded.correct !== (recorded.predicted_label === recorded.actual_label) ||
      !approximatelyEqual(prediction.confidence, recorded.confidence) ||
      !approximatelyEqual(prediction.probabilityMargin, recorded.probability_margin)
    ) {
      throw new TypeError("Recorded ID prediction does not reproduce in the browser runtime");
    }
    exactKeys(recorded.probabilities, Object.keys(prediction.probabilities), `Recorded ID probabilities ${index}`);
    for (const [label, probability] of Object.entries(prediction.probabilities)) {
      if (!approximatelyEqual(probability, recorded.probabilities[label])) {
        throw new TypeError("Recorded ID probabilities do not reproduce in the browser runtime");
      }
    }
    idRows.delete(recorded.id);
    correctId += Number(recorded.correct);
    acceptedId += Number(recorded.accepted);
    acceptedCorrectId += Number(recorded.accepted && recorded.correct);
    recordedId.push(recorded);
  }
  if (idRows.size !== 0) throw new TypeError("Metrics omit ID-test rows");
  const idCount = runtime.metrics.id_test.example_count;
  const selectiveAccuracy = acceptedId === 0 ? null : acceptedCorrectId / acceptedId;
  const perClass = summarizePerClass(model.labels, recordedId);
  if (
    !Array.isArray(runtime.metrics.id_test.labels) ||
    runtime.metrics.id_test.labels.length !== model.labels.length ||
    runtime.metrics.id_test.labels.some((label, index) => label !== model.labels[index]) ||
    !Array.isArray(runtime.metrics.id_test.confusion_matrix) ||
    runtime.metrics.id_test.confusion_matrix.length !== perClass.confusionMatrix.length ||
    runtime.metrics.id_test.confusion_matrix.some(
      (row, index) =>
        !Array.isArray(row) ||
        row.length !== perClass.confusionMatrix[index].length ||
        row.some((value, column) => value !== perClass.confusionMatrix[index][column]),
    ) ||
    !Array.isArray(runtime.metrics.id_test.per_class) ||
    runtime.metrics.id_test.per_class.length !== perClass.perClass.length
  ) {
    throw new TypeError("ID per-class report or confusion matrix disagrees with its predictions");
  }
  runtime.metrics.id_test.per_class.forEach((recorded, index) => {
    exactKeys(
      recorded,
      ["label", "support", "predicted", "true_positive", "precision", "recall", "f1"],
      `ID per-class metric ${index}`,
    );
    const reproduced = perClass.perClass[index];
    if (
      recorded.label !== reproduced.label ||
      recorded.support !== reproduced.support ||
      recorded.predicted !== reproduced.predicted ||
      recorded.true_positive !== reproduced.true_positive ||
      !approximatelyEqual(recorded.precision, reproduced.precision) ||
      !approximatelyEqual(recorded.recall, reproduced.recall) ||
      !approximatelyEqual(recorded.f1, reproduced.f1)
    ) {
      throw new TypeError("ID per-class metrics disagree with their predictions");
    }
  });
  const reproducedCalibration = calibrationFromPredictions(
    recordedId,
    runtime.metrics.id_test.calibration.ece_bins,
  );
  const reproducedRisk = riskCoverage(recordedId);
  if (
    !approximatelyEqual(runtime.metrics.id_test.accuracy, correctId / idCount) ||
    !approximatelyEqual(runtime.metrics.id_test.macro_f1, perClass.macroF1) ||
    !approximatelyEqual(runtime.metrics.id_test.coverage, acceptedId / idCount) ||
    (selectiveAccuracy === null
      ? runtime.metrics.id_test.selective_accuracy !== null
      : !approximatelyEqual(runtime.metrics.id_test.selective_accuracy, selectiveAccuracy)) ||
    !calibrationMatches(runtime.metrics.id_test.calibration, reproducedCalibration) ||
    !approximatelyEqual(runtime.metrics.id_test.aurc, reproducedRisk.aurc) ||
    runtime.metrics.id_test.risk_coverage_curve.length !== reproducedRisk.curve.length ||
    runtime.metrics.id_test.risk_coverage_curve.some((point, index) => {
      const reproduced = reproducedRisk.curve[index];
      return (
        point.accepted !== reproduced.accepted ||
        !approximatelyEqual(point.coverage, reproduced.coverage) ||
        !approximatelyEqual(point.risk, reproduced.risk)
      );
    })
  ) {
    throw new TypeError("ID aggregate metrics disagree with their predictions");
  }

  const reproducedBaselines = reproduceBaselines(
    runtime.splitPlan,
    model.labels,
    runtime.metrics.id_test,
  );
  const recordedBaselines = runtime.metrics.baselines;
  if (
    recordedBaselines.strategy !== reproducedBaselines.strategy ||
    JSON.stringify(recordedBaselines.inputs) !== JSON.stringify(reproducedBaselines.inputs) ||
    recordedBaselines.evaluation_partition !== reproducedBaselines.evaluation_partition ||
    recordedBaselines.training_example_count !== reproducedBaselines.training_example_count ||
    recordedBaselines.training_family_count !== reproducedBaselines.training_family_count ||
    recordedBaselines.majority_label !== reproducedBaselines.majority_label ||
    !baselineEvaluationMatches(recordedBaselines.majority, reproducedBaselines.majority) ||
    !baselineEvaluationMatches(
      recordedBaselines.unigram_naive_bayes,
      reproducedBaselines.unigram_naive_bayes,
    ) ||
    !approximatelyEqual(
      recordedBaselines.learned_minus_unigram_accuracy,
      reproducedBaselines.learned_minus_unigram_accuracy,
    ) ||
    !approximatelyEqual(
      recordedBaselines.learned_minus_unigram_macro_f1,
      reproducedBaselines.learned_minus_unigram_macro_f1,
    )
  ) {
    throw new TypeError("Training-only baselines do not reproduce from the experiment plan");
  }

  const contrastRows = new Map(
    runtime.splitPlan.contrast_test.map((row) => [row.id, row]),
  );
  const contrastPairs = new Map();
  const recordedContrast = [];
  let acceptedContrast = 0;
  let correctContrast = 0;
  for (const [index, recorded] of runtime.metrics.contrast_test.predictions.entries()) {
    exactKeys(
      recorded,
      [
        "id",
        "pair_id",
        "variant",
        "actual_label",
        "predicted_label",
        "correct",
        "accepted",
        "confidence",
      ],
      `Recorded contrast prediction ${index}`,
    );
    const row = contrastRows.get(recorded.id);
    if (
      !row ||
      recorded.pair_id !== row.pair_id ||
      recorded.variant !== row.variant ||
      recorded.actual_label !== row.label ||
      !model.labels.includes(recorded.predicted_label) ||
      typeof recorded.correct !== "boolean" ||
      typeof recorded.accepted !== "boolean"
    ) {
      throw new TypeError("Recorded contrast prediction has no matching split-plan row");
    }
    const prediction = runtime.predict(row.text);
    if (
      recorded.predicted_label !== prediction.label ||
      recorded.correct !== (recorded.predicted_label === recorded.actual_label) ||
      recorded.accepted !== prediction.accepted ||
      !approximatelyEqual(recorded.confidence, prediction.confidence)
    ) {
      throw new TypeError("Recorded contrast prediction does not reproduce in the browser runtime");
    }
    contrastRows.delete(recorded.id);
    const pair = contrastPairs.get(recorded.pair_id) ?? [];
    pair.push(recorded);
    contrastPairs.set(recorded.pair_id, pair);
    acceptedContrast += Number(recorded.accepted);
    correctContrast += Number(recorded.correct);
    recordedContrast.push(recorded);
  }
  const contrastSummary = summarizePerClass(model.labels, recordedContrast);
  const reproducedContrast = {
    accuracy: correctContrast / recordedContrast.length,
    macro_f1: contrastSummary.macroF1,
    confusion_matrix: contrastSummary.confusionMatrix,
    per_class: contrastSummary.perClass,
  };
  const correctContrastPairs = [...contrastPairs.values()].filter(
    (members) => members.length === 2 && members.every((prediction) => prediction.correct),
  ).length;
  const flippedContrastPairs = [...contrastPairs.values()].filter(
    (members) =>
      members.length === 2 && members[0].predicted_label !== members[1].predicted_label,
  ).length;
  if (
    contrastRows.size !== 0 ||
    [...contrastPairs.values()].some((members) => members.length !== 2) ||
    runtime.metrics.contrast_test.example_count !== recordedContrast.length ||
    runtime.metrics.contrast_test.pair_count !== contrastPairs.size ||
    !baselineEvaluationMatches(runtime.metrics.contrast_test, reproducedContrast) ||
    !approximatelyEqual(
      runtime.metrics.contrast_test.pair_accuracy,
      correctContrastPairs / contrastPairs.size,
    ) ||
    !approximatelyEqual(
      runtime.metrics.contrast_test.prediction_flip_rate,
      flippedContrastPairs / contrastPairs.size,
    ) ||
    !approximatelyEqual(
      runtime.metrics.contrast_test.coverage,
      acceptedContrast / recordedContrast.length,
    )
  ) {
    throw new TypeError("Contrast-test report does not reproduce from its frozen paired rows");
  }

  const oodRows = new Map(runtime.splitPlan.ood_test.map((row) => [row.id, row]));
  let acceptedOod = 0;
  const recordedOod = [];
  for (const [index, recorded] of runtime.metrics.ood_test.predictions.entries()) {
    exactKeys(
      recorded,
      [
        "id",
        "family_id",
        "domain_group",
        "stratum",
        "predicted_label",
        "accepted",
        "confidence",
        "probability_margin",
      ],
      `Recorded OOD prediction ${index}`,
    );
    const row = oodRows.get(recorded.id);
    if (!row) throw new TypeError("Recorded OOD prediction has no split-plan row");
    const prediction = runtime.predict(row.text);
    if (
      prediction.label !== recorded.predicted_label ||
      prediction.accepted !== recorded.accepted ||
      recorded.family_id !== row.family_id ||
      recorded.domain_group !== row.domain_group ||
      recorded.stratum !== row.stratum ||
      !approximatelyEqual(prediction.confidence, recorded.confidence) ||
      !approximatelyEqual(prediction.probabilityMargin, recorded.probability_margin)
    ) {
      throw new TypeError("Recorded OOD prediction does not reproduce in the browser runtime");
    }
    acceptedOod += Number(recorded.accepted);
    oodRows.delete(recorded.id);
    recordedOod.push(recorded);
  }
  const oodCount = runtime.metrics.ood_test.example_count;
  const reproducedDiscrimination = discriminationFromScores(
    recordedId.map((prediction) => prediction.confidence),
    recordedOod.map((prediction) => prediction.confidence),
  );
  const idScores = recordedId.map((prediction) => prediction.confidence);
  const reproducedByStratum = Object.fromEntries(
    OOD_STRATA.map((stratum) => {
      const members = recordedOod.filter((prediction) => prediction.stratum === stratum);
      const accepted = members.filter((prediction) => prediction.accepted).length;
      return [
        stratum,
        {
          example_count: members.length,
          accepted_examples: accepted,
          coverage: accepted / members.length,
          discrimination: discriminationFromScores(
            idScores,
            members.map((prediction) => prediction.confidence),
          ),
        },
      ];
    }),
  );
  const stratumSummariesMatch = OOD_STRATA.every((stratum) => {
    const recorded = runtime.metrics.ood_test.by_stratum[stratum];
    const reproduced = reproducedByStratum[stratum];
    return (
      recorded.example_count === reproduced.example_count &&
      recorded.accepted_examples === reproduced.accepted_examples &&
      approximatelyEqual(recorded.coverage, reproduced.coverage) &&
      approximatelyEqual(recorded.discrimination.auroc, reproduced.discrimination.auroc) &&
      approximatelyEqual(
        recorded.discrimination.aupr_in_domain,
        reproduced.discrimination.aupr_in_domain,
      ) &&
      approximatelyEqual(
        recorded.discrimination.fpr_at_95_tpr,
        reproduced.discrimination.fpr_at_95_tpr,
      )
    );
  });
  if (
    oodRows.size !== 0 ||
    runtime.metrics.ood_test.accepted_examples !== acceptedOod ||
    !approximatelyEqual(runtime.metrics.ood_test.coverage, acceptedOod / oodCount) ||
    !approximatelyEqual(
      runtime.metrics.ood_test.discrimination.auroc,
      reproducedDiscrimination.auroc,
    ) ||
    !approximatelyEqual(
      runtime.metrics.ood_test.discrimination.aupr_in_domain,
      reproducedDiscrimination.aupr_in_domain,
    ) ||
    !approximatelyEqual(
      runtime.metrics.ood_test.discrimination.fpr_at_95_tpr,
      reproducedDiscrimination.fpr_at_95_tpr,
    ) ||
    !stratumSummariesMatch
  ) {
    throw new TypeError("OOD aggregate metrics disagree with their predictions");
  }

  const developmentRows = runtime.splitPlan.assignments.filter(
    (assignment) => assignment.partition === "development",
  );
  const developmentPredictions = developmentRows.map((row) => ({
    actual: row.label,
    prediction: runtime.predict(row.text),
  }));
  const acceptedDevelopment = developmentPredictions.filter(
    ({ prediction }) => prediction.accepted,
  );
  const acceptedDevelopmentCorrect = acceptedDevelopment.filter(
    ({ actual, prediction }) => actual === prediction.label,
  ).length;
  const oodDevelopmentPredictions = runtime.splitPlan.ood_development.map((row) =>
    runtime.predict(row.text),
  );
  const threshold = runtime.metrics.threshold_selection;
  if (
    acceptedDevelopment.length === 0 ||
    !approximatelyEqual(
      threshold.observed_development_coverage,
      acceptedDevelopment.length / developmentPredictions.length,
    ) ||
    !approximatelyEqual(
      threshold.observed_development_selective_accuracy,
      acceptedDevelopmentCorrect / acceptedDevelopment.length,
    ) ||
    !approximatelyEqual(
      threshold.observed_ood_development_coverage,
      oodDevelopmentPredictions.filter((prediction) => prediction.accepted).length /
        oodDevelopmentPredictions.length,
    )
  ) {
    throw new TypeError("Threshold observations do not reproduce from development data");
  }

  const calibrationRows = runtime.splitPlan.assignments.filter(
    (assignment) => assignment.partition === "calibration",
  );
  const calibratedRows = calibrationRows.map((row) => {
    const prediction = runtime.predict(row.text);
    return {
      actual_label: row.label,
      predicted_label: prediction.label,
      correct: prediction.label === row.label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
    };
  });
  const uncalibratedRows = calibratedRows.map((row) => {
    const probabilities = uncalibratedProbabilities(row.probabilities, policy.temperature);
    const ranking = model.labels
      .map((label, index) => ({ label, index, probability: probabilities[label] }))
      .sort(
        (left, right) => right.probability - left.probability || left.index - right.index,
      );
    return {
      actual_label: row.actual_label,
      predicted_label: ranking[0].label,
      correct: ranking[0].label === row.actual_label,
      confidence: ranking[0].probability,
      probabilities,
    };
  });
  const calibratedMetrics = calibrationFromPredictions(
    calibratedRows,
    runtime.metrics.calibrated_calibration_partition.ece_bins,
  );
  const uncalibratedMetrics = calibrationFromPredictions(
    uncalibratedRows,
    runtime.metrics.uncalibrated_calibration_partition.ece_bins,
  );
  if (
    !calibrationMatches(
      runtime.metrics.calibrated_calibration_partition,
      calibratedMetrics,
    ) ||
    !calibrationMatches(
      runtime.metrics.uncalibrated_calibration_partition,
      uncalibratedMetrics,
    )
  ) {
    throw new TypeError("Calibration-partition metrics do not reproduce from the model");
  }
};

export const verifyOpenSetArtifactSemantics = async (
  model,
  policy,
  metrics,
  plan,
  cryptoProvider = globalThis.crypto,
) => {
  if (!cryptoProvider?.subtle || typeof cryptoProvider.subtle.digest !== "function") {
    throw new TypeError("Web Crypto SHA-256 is required to verify artifact semantics");
  }
  validateModel(model);
  validatePolicy(policy, model);
  validatePlan(plan, model);
  validateMetrics(metrics, model, policy, plan);
  await validateSourceFingerprints(cryptoProvider, plan, model, metrics);
  const runtime = new VerifiedOpenSetBundle(
    deepFreeze(model),
    deepFreeze(policy),
    deepFreeze(metrics),
    deepFreeze(plan),
  );
  validateRecordedPredictions(runtime, model, policy);
  return Object.freeze(runtime);
};

export const loadOpenSetBundle = async (baseUrl, options = {}) => {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  const cryptoProvider = Object.hasOwn(options, "crypto") ? options.crypto : globalThis.crypto;
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required");
  if (!cryptoProvider?.subtle || typeof cryptoProvider.subtle.digest !== "function") {
    throw new TypeError("Web Crypto SHA-256 is required to verify the model bundle");
  }
  const budget = { consumed: 0 };

  const manifestBytes = await fetchBytes(
    fetchImpl,
    artifactUrl(baseUrl, "manifest.json"),
    "Bundle manifest",
    budget,
  );
  const manifestDigest = await sha256Hex(cryptoProvider, manifestBytes);
  if (manifestDigest !== EXPECTED_BUNDLE_MANIFEST_SHA256) {
    throw new TypeError("Bundle manifest does not match the embedded release trust root");
  }
  const manifest = parseJson(manifestBytes, "Bundle manifest");
  validateManifest(manifest);

  const payloadEntries = [];
  for (const name of PAYLOAD_NAMES) {
    const bytes = await fetchBytes(fetchImpl, artifactUrl(baseUrl, name), name, budget);
    const digest = await sha256Hex(cryptoProvider, bytes);
    if (digest !== manifest.files[name]) throw new TypeError(`${name} failed SHA-256 verification`);
    payloadEntries.push([name, parseJson(bytes, name)]);
  }
  const payloads = Object.fromEntries(payloadEntries);
  const model = payloads["model.json"];
  const policy = payloads["policy.json"];
  const metrics = payloads["metrics.json"];
  const plan = payloads["split-plan.json"];

  if (
    manifest.model_version !== model.model_version ||
    manifest.dataset_sha256 !== model.dataset_sha256 ||
    manifest.split_plan_sha256 !== model.split_plan_sha256
  ) {
    throw new TypeError("Manifest and payload provenance disagree");
  }
  return verifyOpenSetArtifactSemantics(model, policy, metrics, plan, cryptoProvider);
};
