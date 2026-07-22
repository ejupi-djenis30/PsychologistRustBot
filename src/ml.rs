//! Deterministic, local intent classification for ELIZA Lab.
//!
//! This module deliberately keeps the complete learning pipeline inspectable: TSV parsing,
//! stratified splitting, vocabulary fitting, TF-IDF feature extraction, multinomial logistic
//! regression, uncertainty-aware prediction, evaluation, and versioned JSON serialization.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MODEL_SCHEMA_VERSION: u32 = 1;
pub const MODEL_KIND: &str = "eliza-intent-softmax";
pub const DEFAULT_SEED: u64 = 20_260_722;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
static TEMP_FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
pub enum MlError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidDataset(String),
    InvalidConfiguration(String),
    InvalidModel(String),
}

impl fmt::Display for MlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::InvalidDataset(message) => write!(formatter, "invalid dataset: {message}"),
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid training configuration: {message}")
            }
            Self::InvalidModel(message) => write!(formatter, "invalid model: {message}"),
        }
    }
}

impl std::error::Error for MlError {}

impl From<std::io::Error> for MlError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for MlError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledExample {
    pub id: String,
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataset {
    examples: Vec<LabeledExample>,
}

impl Dataset {
    pub fn bundled() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/intents-v1.tsv"))
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the file is empty".into()))?;
        if header != "id\tlabel\ttext" {
            return Err(MlError::InvalidDataset(
                "the header must be exactly `id\\tlabel\\ttext`".into(),
            ));
        }

        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut normalized_texts = HashSet::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} must contain three tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let label = fields[1].trim();
            let text = fields[2].trim();
            if id.is_empty() || label.is_empty() || text.is_empty() {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} contains an empty field"
                )));
            }
            if !id.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            }) {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} has an invalid example id"
                )));
            }
            if !label.chars().all(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
            }) {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} has an invalid label"
                )));
            }
            if text.chars().count() > crate::MAX_INPUT_CHARS {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} exceeds the input boundary"
                )));
            }
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!(
                    "duplicate example id `{id}`"
                )));
            }
            let normalized = normalize_text(text);
            if !normalized_texts.insert(normalized) {
                return Err(MlError::InvalidDataset(format!(
                    "line {line_number} duplicates normalized text"
                )));
            }
            examples.push(LabeledExample {
                id: id.to_owned(),
                label: label.to_owned(),
                text: text.to_owned(),
            });
        }

        if examples.is_empty() {
            return Err(MlError::InvalidDataset("no examples were found".into()));
        }
        let dataset = Self { examples };
        dataset.validate_class_support()?;
        Ok(dataset)
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn examples(&self) -> &[LabeledExample] {
        &self.examples
    }

    pub fn labels(&self) -> Vec<String> {
        self.examples
            .iter()
            .map(|example| example.label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn class_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for example in &self.examples {
            *counts.entry(example.label.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Stable content fingerprint for reproducibility, not a cryptographic integrity check.
    pub fn fingerprint(&self) -> String {
        let mut rows = self
            .examples
            .iter()
            .map(|example| {
                format!(
                    "{}\t{}\t{}",
                    example.id,
                    example.label,
                    normalize_text(&example.text)
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        format!(
            "fnv1a64:{:016x}",
            stable_hash(rows.join("\n").as_bytes(), 0)
        )
    }

    pub fn stratified_split(
        &self,
        holdout_fraction: f64,
        seed: u64,
    ) -> Result<DatasetSplit, MlError> {
        if !(0.05..=0.5).contains(&holdout_fraction) {
            return Err(MlError::InvalidConfiguration(
                "holdout_fraction must be between 0.05 and 0.5".into(),
            ));
        }
        self.validate_class_support()?;

        let mut by_label: BTreeMap<&str, Vec<&LabeledExample>> = BTreeMap::new();
        for example in &self.examples {
            by_label.entry(&example.label).or_default().push(example);
        }

        let mut train = Vec::new();
        let mut holdout = Vec::new();
        for examples in by_label.values_mut() {
            examples.sort_by(|left, right| {
                let left_hash = split_hash(left, seed);
                let right_hash = split_hash(right, seed);
                left_hash
                    .cmp(&right_hash)
                    .then_with(|| left.id.cmp(&right.id))
            });
            let holdout_count = ((examples.len() as f64 * holdout_fraction).round() as usize)
                .clamp(1, examples.len() - 1);
            for (index, example) in examples.iter().enumerate() {
                if index < holdout_count {
                    holdout.push((*example).clone());
                } else {
                    train.push((*example).clone());
                }
            }
        }
        train.sort_by(|left, right| left.id.cmp(&right.id));
        holdout.sort_by(|left, right| left.id.cmp(&right.id));

        Ok(DatasetSplit {
            train,
            holdout,
            seed,
            holdout_fraction,
            dataset_fingerprint: self.fingerprint(),
        })
    }

    fn validate_class_support(&self) -> Result<(), MlError> {
        let counts = self.class_counts();
        if counts.len() < 2 {
            return Err(MlError::InvalidDataset(
                "at least two labels are required".into(),
            ));
        }
        if let Some((label, count)) = counts.iter().find(|(_, count)| **count < 5) {
            return Err(MlError::InvalidDataset(format!(
                "label `{label}` has {count} examples; at least five are required"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSplit {
    train: Vec<LabeledExample>,
    holdout: Vec<LabeledExample>,
    seed: u64,
    holdout_fraction: f64,
    dataset_fingerprint: String,
}

impl DatasetSplit {
    pub fn training_examples(&self) -> &[LabeledExample] {
        &self.train
    }

    pub fn holdout_examples(&self) -> &[LabeledExample] {
        &self.holdout
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn holdout_fraction(&self) -> f64 {
        self.holdout_fraction
    }

    pub fn dataset_fingerprint(&self) -> &str {
        &self.dataset_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OodExample {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OodDataset {
    examples: Vec<OodExample>,
}

impl OodDataset {
    pub fn bundled() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/ood-v1.tsv"))
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the OOD file is empty".into()))?;
        if header != "id\ttext" {
            return Err(MlError::InvalidDataset(
                "the OOD header must be exactly `id\\ttext`".into(),
            ));
        }
        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut texts = HashSet::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 2 {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} must contain two tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let text = fields[1].trim();
            if id.is_empty() || text.is_empty() {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} contains an empty field"
                )));
            }
            if !id.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            }) {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} has an invalid example id"
                )));
            }
            if text.chars().count() > crate::MAX_INPUT_CHARS {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} exceeds the input boundary"
                )));
            }
            if !ids.insert(id.to_owned()) || !texts.insert(normalize_text(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} duplicates an id or normalized text"
                )));
            }
            examples.push(OodExample {
                id: id.to_owned(),
                text: text.to_owned(),
            });
        }
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "the OOD dataset contains no examples".into(),
            ));
        }
        Ok(Self { examples })
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn examples(&self) -> &[OodExample] {
        &self.examples
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VectorizerConfig {
    pub word_ngram_min: usize,
    pub word_ngram_max: usize,
    pub char_ngram_min: usize,
    pub char_ngram_max: usize,
    pub min_document_frequency: usize,
    pub max_features: usize,
}

impl Default for VectorizerConfig {
    fn default() -> Self {
        Self {
            word_ngram_min: 1,
            word_ngram_max: 2,
            char_ngram_min: 3,
            char_ngram_max: 5,
            min_document_frequency: 1,
            max_features: 512,
        }
    }
}

impl VectorizerConfig {
    fn validate(&self) -> Result<(), MlError> {
        if self.word_ngram_min == 0
            || self.word_ngram_min > self.word_ngram_max
            || self.word_ngram_max > 3
        {
            return Err(MlError::InvalidConfiguration(
                "word n-grams must define a range between 1 and 3".into(),
            ));
        }
        if self.char_ngram_min < 2
            || self.char_ngram_min > self.char_ngram_max
            || self.char_ngram_max > 6
        {
            return Err(MlError::InvalidConfiguration(
                "character n-grams must define a range between 2 and 6".into(),
            ));
        }
        if self.min_document_frequency == 0
            || self.min_document_frequency > 1_000_000
            || !(32..=100_000).contains(&self.max_features)
        {
            return Err(MlError::InvalidConfiguration(
                "min_document_frequency must be between 1 and 1000000 and max_features between 32 and 100000"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecisionThresholds {
    pub minimum_confidence: f64,
    pub minimum_margin: f64,
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.48,
            minimum_margin: 0.16,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2_penalty: f64,
    pub holdout_fraction: f64,
    pub vectorizer: VectorizerConfig,
    pub thresholds: DecisionThresholds,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            epochs: 600,
            learning_rate: 0.8,
            l2_penalty: 0.0005,
            holdout_fraction: 0.2,
            vectorizer: VectorizerConfig::default(),
            thresholds: DecisionThresholds::default(),
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), MlError> {
        if self.seed > MAX_JSON_SAFE_INTEGER {
            return Err(MlError::InvalidConfiguration(format!(
                "seed must not exceed the JSON-safe integer ceiling {MAX_JSON_SAFE_INTEGER}"
            )));
        }
        if self.epochs == 0 || self.epochs > 10_000 {
            return Err(MlError::InvalidConfiguration(
                "epochs must be between 1 and 10000".into(),
            ));
        }
        if !self.learning_rate.is_finite() || !(0.000_001..=10.0).contains(&self.learning_rate) {
            return Err(MlError::InvalidConfiguration(
                "learning_rate must be finite and between 0.000001 and 10".into(),
            ));
        }
        if !self.l2_penalty.is_finite() || !(0.0..=1.0).contains(&self.l2_penalty) {
            return Err(MlError::InvalidConfiguration(
                "l2_penalty must be finite and between 0 and 1".into(),
            ));
        }
        if !(0.05..=0.5).contains(&self.holdout_fraction) {
            return Err(MlError::InvalidConfiguration(
                "holdout_fraction must be between 0.05 and 0.5".into(),
            ));
        }
        self.vectorizer.validate()?;
        validate_thresholds(&self.thresholds)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TfidfVectorizer {
    pub config: VectorizerConfig,
    pub vocabulary: Vec<String>,
    pub inverse_document_frequency: Vec<f64>,
}

impl TfidfVectorizer {
    fn fit(examples: &[LabeledExample], config: VectorizerConfig) -> Result<Self, MlError> {
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for example in examples {
            let terms = extract_terms(&example.text, &config)
                .into_iter()
                .collect::<HashSet<_>>();
            for term in terms {
                *document_frequency.entry(term).or_insert(0) += 1;
            }
        }
        let mut candidates = document_frequency
            .into_iter()
            .filter(|(_, frequency)| *frequency >= config.min_document_frequency)
            .collect::<Vec<_>>();
        candidates.sort_by(
            |(left_term, left_frequency), (right_term, right_frequency)| {
                right_frequency
                    .cmp(left_frequency)
                    .then_with(|| left_term.cmp(right_term))
            },
        );
        candidates.truncate(config.max_features);
        if candidates.is_empty() {
            return Err(MlError::InvalidDataset(
                "the training split produced an empty vocabulary".into(),
            ));
        }

        let document_count = examples.len() as f64;
        let (vocabulary, inverse_document_frequency) = candidates
            .into_iter()
            .map(|(term, frequency)| {
                let idf = quantize(((1.0 + document_count) / (1.0 + frequency as f64)).ln() + 1.0);
                (term, idf)
            })
            .unzip();
        Ok(Self {
            config,
            vocabulary,
            inverse_document_frequency,
        })
    }

    fn validate(&self) -> Result<(), MlError> {
        self.config.validate().map_err(|error| {
            MlError::InvalidModel(format!(
                "the serialized vectorizer config is invalid: {error}"
            ))
        })?;
        if self.vocabulary.is_empty()
            || self.vocabulary.len() != self.inverse_document_frequency.len()
            || self.vocabulary.len() > self.config.max_features
        {
            return Err(MlError::InvalidModel(
                "the vocabulary and IDF vectors must be non-empty, aligned, and within max_features"
                    .into(),
            ));
        }
        let mut unique = HashSet::new();
        for (term, idf) in self.vocabulary.iter().zip(&self.inverse_document_frequency) {
            if !valid_feature_name(term, &self.config)
                || !unique.insert(term)
                || !idf.is_finite()
                || *idf < 1.0
            {
                return Err(MlError::InvalidModel(
                    "the vectorizer contains an invalid feature".into(),
                ));
            }
        }
        Ok(())
    }

    fn transform(&self, text: &str) -> Vec<(usize, f64)> {
        let index = self
            .vocabulary
            .iter()
            .enumerate()
            .map(|(position, term)| (term.as_str(), position))
            .collect::<HashMap<_, _>>();
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for term in extract_terms(text, &self.config) {
            if let Some(position) = index.get(term.as_str()) {
                *counts.entry(*position).or_insert(0) += 1;
            }
        }
        let mut values = counts
            .into_iter()
            .map(|(position, count)| {
                let term_frequency = 1.0 + (count as f64).ln();
                (
                    position,
                    term_frequency * self.inverse_document_frequency[position],
                )
            })
            .collect::<Vec<_>>();
        values.sort_by_key(|(position, _)| *position);
        let norm = values
            .iter()
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for (_, value) in &mut values {
                *value /= norm;
            }
        }
        values
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IntentModel {
    pub schema_version: u32,
    pub model_kind: String,
    pub model_version: String,
    pub dataset_fingerprint: String,
    pub training_config: TrainingConfig,
    pub labels: Vec<String>,
    pub vectorizer: TfidfVectorizer,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl IntentModel {
    /// Loads the versioned model shipped inside the CLI release binary.
    pub fn bundled() -> Result<Self, MlError> {
        Self::from_json(include_str!("../models/eliza-intent-v1.json"))
    }

    pub fn train(
        dataset: &Dataset,
        config: TrainingConfig,
    ) -> Result<(Self, TrainingReport), MlError> {
        config.validate()?;
        let split = dataset.stratified_split(config.holdout_fraction, config.seed)?;
        let labels = dataset.labels();
        let label_index = labels
            .iter()
            .enumerate()
            .map(|(index, label)| (label.as_str(), index))
            .collect::<HashMap<_, _>>();
        let vectorizer =
            TfidfVectorizer::fit(split.training_examples(), config.vectorizer.clone())?;
        let features = split
            .training_examples()
            .iter()
            .map(|example| vectorizer.transform(&example.text))
            .collect::<Vec<_>>();
        let targets = split
            .training_examples()
            .iter()
            .map(|example| label_index[example.label.as_str()])
            .collect::<Vec<_>>();

        let mut weights = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
        let mut biases = vec![0.0; labels.len()];
        let sample_count = split.training_examples().len() as f64;
        for epoch in 0..config.epochs {
            let mut weight_gradient = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
            let mut bias_gradient = vec![0.0; labels.len()];
            for (row, target) in features.iter().zip(&targets) {
                let probabilities = probabilities_for(row, &weights, &biases);
                for class in 0..labels.len() {
                    let error = probabilities[class] - usize::from(class == *target) as f64;
                    bias_gradient[class] += error;
                    for (feature, value) in row {
                        weight_gradient[class][*feature] += error * value;
                    }
                }
            }
            let learning_rate = config.learning_rate / (1.0 + epoch as f64 * 0.025).sqrt();
            for class in 0..labels.len() {
                biases[class] =
                    quantize(biases[class] - learning_rate * bias_gradient[class] / sample_count);
                for feature in 0..vectorizer.vocabulary.len() {
                    let gradient = weight_gradient[class][feature] / sample_count
                        + config.l2_penalty * weights[class][feature];
                    weights[class][feature] =
                        quantize(weights[class][feature] - learning_rate * gradient);
                }
            }
        }

        let model = Self {
            schema_version: MODEL_SCHEMA_VERSION,
            model_kind: MODEL_KIND.into(),
            model_version: "1.0.0".into(),
            dataset_fingerprint: dataset.fingerprint(),
            training_config: config,
            labels,
            vectorizer,
            weights,
            biases,
        };
        model.validate()?;
        let training_metrics = model.evaluate(split.training_examples())?;
        let holdout_metrics = model.evaluate(split.holdout_examples())?;
        let report = TrainingReport {
            schema_version: 1,
            model_kind: model.model_kind.clone(),
            model_version: model.model_version.clone(),
            dataset_fingerprint: model.dataset_fingerprint.clone(),
            seed: split.seed(),
            total_examples: dataset.examples.len(),
            class_counts: dataset.class_counts(),
            training_example_ids: split
                .training_examples()
                .iter()
                .map(|example| example.id.clone())
                .collect(),
            holdout_example_ids: split
                .holdout_examples()
                .iter()
                .map(|example| example.id.clone())
                .collect(),
            vocabulary_size: model.vectorizer.vocabulary.len(),
            training_metrics,
            holdout_metrics,
            calibration: None,
            ood_metrics: None,
        };
        Ok((model, report))
    }

    pub fn from_json(input: &str) -> Result<Self, MlError> {
        let model: Self = serde_json::from_str(input)?;
        model.validate()?;
        Ok(model)
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    pub fn to_json_pretty(&self) -> Result<String, MlError> {
        self.validate()?;
        Ok(format!("{}\n", serde_json::to_string_pretty(self)?))
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), MlError> {
        write_json_file(path.as_ref(), self.to_json_pretty()?.as_bytes())
    }

    pub fn validate(&self) -> Result<(), MlError> {
        if self.schema_version != MODEL_SCHEMA_VERSION {
            return Err(MlError::InvalidModel(format!(
                "unsupported schema version {}; expected {MODEL_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.model_kind != MODEL_KIND || self.model_version != "1.0.0" {
            return Err(MlError::InvalidModel(
                "unsupported model kind or model version".into(),
            ));
        }
        self.training_config.validate()?;
        self.vectorizer.validate()?;
        if self.training_config.vectorizer != self.vectorizer.config {
            return Err(MlError::InvalidModel(
                "training_config.vectorizer must exactly match vectorizer.config".into(),
            ));
        }
        if self.labels.len() < 2 || self.labels.len() != self.weights.len() {
            return Err(MlError::InvalidModel(
                "labels and weight rows must be aligned".into(),
            ));
        }
        if self.biases.len() != self.labels.len() {
            return Err(MlError::InvalidModel(
                "labels and biases must be aligned".into(),
            ));
        }
        let mut unique_labels = HashSet::new();
        for label in &self.labels {
            if label.is_empty() || !unique_labels.insert(label) {
                return Err(MlError::InvalidModel(
                    "model labels must be non-empty and unique".into(),
                ));
            }
        }
        for (row, bias) in self.weights.iter().zip(&self.biases) {
            if row.len() != self.vectorizer.vocabulary.len()
                || row.iter().any(|weight| !weight.is_finite())
                || !bias.is_finite()
            {
                return Err(MlError::InvalidModel(
                    "model parameters are not finite and rectangular".into(),
                ));
            }
        }
        if !valid_dataset_fingerprint(&self.dataset_fingerprint) {
            return Err(MlError::InvalidModel(
                "dataset_fingerprint must be `fnv1a64:` followed by 16 lowercase hexadecimal digits"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn predict(&self, text: &str) -> Prediction {
        if self.validate().is_err() {
            return rejected_invalid_model_prediction(&self.labels);
        }
        self.predict_validated(text)
    }

    fn predict_validated(&self, text: &str) -> Prediction {
        let features = self.vectorizer.transform(text);
        let probabilities = probabilities_for(&features, &self.weights, &self.biases);
        let mut ranking = probabilities
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        ranking.sort_by(|(left_index, left), (right_index, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });
        let (top_index, confidence) = ranking[0];
        let margin = confidence - ranking[1].1;
        let thresholds = &self.training_config.thresholds;
        let accepted = confidence >= thresholds.minimum_confidence
            && margin >= thresholds.minimum_margin
            && !features.is_empty();
        let probability_by_label = self
            .labels
            .iter()
            .cloned()
            .zip(probabilities)
            .collect::<BTreeMap<_, _>>();

        let mut contributions = features
            .into_iter()
            .map(|(feature, value)| FeatureContribution {
                feature: self.vectorizer.vocabulary[feature].clone(),
                value,
                weight: self.weights[top_index][feature],
                contribution: value * self.weights[top_index][feature],
            })
            .filter(|contribution| contribution.contribution > 0.0)
            .collect::<Vec<_>>();
        contributions.sort_by(|left, right| {
            right
                .contribution
                .total_cmp(&left.contribution)
                .then_with(|| left.feature.cmp(&right.feature))
        });
        contributions.truncate(8);

        Prediction {
            label: self.labels[top_index].clone(),
            accepted,
            confidence,
            margin,
            probabilities: probability_by_label,
            top_features: contributions,
        }
    }

    pub fn evaluate(&self, examples: &[LabeledExample]) -> Result<EvaluationMetrics, MlError> {
        self.validate()?;
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "evaluation requires at least one example".into(),
            ));
        }
        let label_index = self
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| (label.as_str(), index))
            .collect::<HashMap<_, _>>();
        let mut confusion = vec![vec![0usize; self.labels.len()]; self.labels.len()];
        let mut correct = 0usize;
        let mut accepted = 0usize;
        let mut accepted_correct = 0usize;
        let mut log_loss = 0.0;
        let mut evaluated_predictions = Vec::with_capacity(examples.len());
        for example in examples {
            let actual = *label_index.get(example.label.as_str()).ok_or_else(|| {
                MlError::InvalidDataset(format!(
                    "evaluation label `{}` is absent from the model",
                    example.label
                ))
            })?;
            let prediction = self.predict_validated(&example.text);
            let predicted = label_index[prediction.label.as_str()];
            confusion[actual][predicted] += 1;
            let is_correct = actual == predicted;
            correct += usize::from(is_correct);
            if prediction.accepted {
                accepted += 1;
                accepted_correct += usize::from(is_correct);
            }
            log_loss -= prediction.probabilities[&example.label].max(1e-15).ln();
            evaluated_predictions.push(EvaluatedPrediction {
                id: example.id.clone(),
                actual_label: example.label.clone(),
                predicted_label: prediction.label,
                correct: is_correct,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
                margin: prediction.margin,
            });
        }

        let mut per_class = Vec::new();
        for (class, label) in self.labels.iter().enumerate() {
            let true_positive = confusion[class][class] as f64;
            let support = confusion[class].iter().sum::<usize>();
            let predicted = confusion.iter().map(|row| row[class]).sum::<usize>();
            let precision = safe_ratio(true_positive, predicted as f64);
            let recall = safe_ratio(true_positive, support as f64);
            let f1 = if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };
            per_class.push(ClassMetrics {
                label: label.clone(),
                precision,
                recall,
                f1,
                support,
            });
        }
        let count = examples.len();
        Ok(EvaluationMetrics {
            example_count: count,
            accuracy: correct as f64 / count as f64,
            macro_f1: per_class.iter().map(|metrics| metrics.f1).sum::<f64>()
                / per_class.len() as f64,
            log_loss: log_loss / count as f64,
            coverage: accepted as f64 / count as f64,
            selective_accuracy: (accepted > 0).then_some(accepted_correct as f64 / accepted as f64),
            rejected_examples: count - accepted,
            labels: self.labels.clone(),
            confusion_matrix: confusion,
            per_class,
            predictions: evaluated_predictions,
        })
    }

    /// Chooses an abstention operating point from a provenanced training split and a disjoint OOD
    /// fixture. The typed split keeps holdout rows inaccessible to the scoring loop, and this
    /// method revalidates the split before making the provenance claim in the report.
    pub fn calibrate_thresholds(
        &mut self,
        split: &DatasetSplit,
        ood_dataset: &OodDataset,
        maximum_ood_coverage: f64,
        minimum_training_selective_accuracy: f64,
    ) -> Result<ThresholdCalibration, MlError> {
        self.validate()?;
        self.validate_calibration_sources(split, ood_dataset)?;
        let training_examples = split.training_examples();
        let ood_examples = ood_dataset.examples();
        if training_examples.is_empty() || ood_examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "threshold calibration requires training and OOD examples".into(),
            ));
        }
        if !(0.0..=1.0).contains(&maximum_ood_coverage)
            || !(0.0..=1.0).contains(&minimum_training_selective_accuracy)
        {
            return Err(MlError::InvalidConfiguration(
                "calibration targets must be between 0 and 1".into(),
            ));
        }

        let training_scores = training_examples
            .iter()
            .map(|example| {
                let prediction = self.predict_validated(&example.text);
                let has_features = !self.vectorizer.transform(&example.text).is_empty();
                (
                    prediction.label == example.label,
                    prediction.confidence,
                    prediction.margin,
                    has_features,
                )
            })
            .collect::<Vec<_>>();
        let ood_scores = ood_examples
            .iter()
            .map(|example| {
                let prediction = self.predict_validated(&example.text);
                (
                    prediction.confidence,
                    prediction.margin,
                    !self.vectorizer.transform(&example.text).is_empty(),
                )
            })
            .collect::<Vec<_>>();

        #[derive(Clone, Copy)]
        struct Candidate {
            confidence: f64,
            margin: f64,
            training_coverage: f64,
            training_selective_accuracy: f64,
            ood_coverage: f64,
        }

        let mut best: Option<Candidate> = None;
        for confidence_step in 15..=80 {
            let confidence = confidence_step as f64 / 100.0;
            for margin_step in 0..=80 {
                let margin = margin_step as f64 / 100.0;
                let mut training_accepted = 0usize;
                let mut training_correct = 0usize;
                for (correct, observed_confidence, observed_margin, has_features) in
                    &training_scores
                {
                    if *has_features
                        && *observed_confidence >= confidence
                        && *observed_margin >= margin
                    {
                        training_accepted += 1;
                        training_correct += usize::from(*correct);
                    }
                }
                if training_accepted == 0 {
                    continue;
                }
                let training_selective_accuracy =
                    training_correct as f64 / training_accepted as f64;
                if training_selective_accuracy < minimum_training_selective_accuracy {
                    continue;
                }
                let ood_accepted = ood_scores
                    .iter()
                    .filter(|(observed_confidence, observed_margin, has_features)| {
                        *has_features
                            && *observed_confidence >= confidence
                            && *observed_margin >= margin
                    })
                    .count();
                let ood_coverage = ood_accepted as f64 / ood_scores.len() as f64;
                if ood_coverage > maximum_ood_coverage {
                    continue;
                }
                let candidate = Candidate {
                    confidence,
                    margin,
                    training_coverage: training_accepted as f64 / training_scores.len() as f64,
                    training_selective_accuracy,
                    ood_coverage,
                };
                let is_better = match best {
                    None => true,
                    Some(current) => {
                        candidate.training_coverage > current.training_coverage
                            || (candidate.training_coverage == current.training_coverage
                                && candidate.ood_coverage < current.ood_coverage)
                            || (candidate.training_coverage == current.training_coverage
                                && candidate.ood_coverage == current.ood_coverage
                                && candidate.margin > current.margin)
                            || (candidate.training_coverage == current.training_coverage
                                && candidate.ood_coverage == current.ood_coverage
                                && candidate.margin == current.margin
                                && candidate.confidence > current.confidence)
                    }
                };
                if is_better {
                    best = Some(candidate);
                }
            }
        }
        let best = best.ok_or_else(|| {
            MlError::InvalidConfiguration(
                "no threshold pair satisfies the requested calibration targets".into(),
            )
        })?;
        self.training_config.thresholds = DecisionThresholds {
            minimum_confidence: best.confidence,
            minimum_margin: best.margin,
        };
        Ok(ThresholdCalibration {
            strategy: "grid-search-training-plus-ood-v1".into(),
            training_example_count: training_examples.len(),
            ood_example_count: ood_examples.len(),
            maximum_ood_coverage,
            minimum_training_selective_accuracy,
            selected_thresholds: self.training_config.thresholds.clone(),
            observed_training_coverage: best.training_coverage,
            observed_training_selective_accuracy: best.training_selective_accuracy,
            observed_ood_coverage: best.ood_coverage,
            holdout_used_for_calibration: false,
        })
    }

    fn validate_calibration_sources(
        &self,
        split: &DatasetSplit,
        ood_dataset: &OodDataset,
    ) -> Result<(), MlError> {
        if split.training_examples().is_empty() || split.holdout_examples().is_empty() {
            return Err(MlError::InvalidDataset(
                "calibration requires a non-empty training and holdout split".into(),
            ));
        }
        if split.seed() != self.training_config.seed
            || split.holdout_fraction() != self.training_config.holdout_fraction
            || split.dataset_fingerprint() != self.dataset_fingerprint
        {
            return Err(MlError::InvalidDataset(
                "calibration split provenance does not match the trained model".into(),
            ));
        }

        let mut supervised = split.training_examples().to_vec();
        supervised.extend_from_slice(split.holdout_examples());
        let reconstructed = Dataset {
            examples: supervised,
        };
        reconstructed.validate_class_support()?;
        if reconstructed.fingerprint() != split.dataset_fingerprint()
            || reconstructed.labels() != self.labels
        {
            return Err(MlError::InvalidDataset(
                "calibration split content does not match its model provenance".into(),
            ));
        }
        let expected = reconstructed.stratified_split(split.holdout_fraction(), split.seed())?;
        if expected.training_examples() != split.training_examples()
            || expected.holdout_examples() != split.holdout_examples()
        {
            return Err(MlError::InvalidDataset(
                "calibration split is not the deterministic split recorded by the model".into(),
            ));
        }

        let supervised_ids = reconstructed
            .examples()
            .iter()
            .map(|example| example.id.as_str())
            .collect::<HashSet<_>>();
        let supervised_texts = reconstructed
            .examples()
            .iter()
            .map(|example| normalize_text(&example.text))
            .collect::<HashSet<_>>();
        for example in ood_dataset.examples() {
            if supervised_ids.contains(example.id.as_str())
                || supervised_texts.contains(&normalize_text(&example.text))
            {
                return Err(MlError::InvalidDataset(format!(
                    "OOD example `{}` overlaps the supervised calibration corpus",
                    example.id
                )));
            }
        }
        Ok(())
    }

    /// Measures abstention on an explicitly out-of-domain corpus. OOD rows have no target
    /// class, so this report intentionally makes no accuracy claim.
    pub fn evaluate_ood(&self, examples: &[OodExample]) -> Result<OodMetrics, MlError> {
        self.validate()?;
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "OOD evaluation requires at least one example".into(),
            ));
        }
        let mut accepted = 0usize;
        let mut confidence_total = 0.0;
        let mut margin_total = 0.0;
        let predictions = examples
            .iter()
            .map(|example| {
                let prediction = self.predict_validated(&example.text);
                accepted += usize::from(prediction.accepted);
                confidence_total += prediction.confidence;
                margin_total += prediction.margin;
                OodPrediction {
                    id: example.id.clone(),
                    predicted_label: prediction.label,
                    accepted: prediction.accepted,
                    confidence: prediction.confidence,
                    margin: prediction.margin,
                }
            })
            .collect::<Vec<_>>();
        let count = examples.len();
        Ok(OodMetrics {
            example_count: count,
            accepted_examples: accepted,
            rejected_examples: count - accepted,
            coverage: accepted as f64 / count as f64,
            abstention_rate: (count - accepted) as f64 / count as f64,
            mean_confidence: confidence_total / count as f64,
            mean_margin: margin_total / count as f64,
            predictions,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub margin: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub top_features: Vec<FeatureContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeatureContribution {
    pub feature: String,
    pub value: f64,
    pub weight: f64,
    pub contribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClassMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationMetrics {
    pub example_count: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub log_loss: f64,
    pub coverage: f64,
    pub selective_accuracy: Option<f64>,
    pub rejected_examples: usize,
    pub labels: Vec<String>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<ClassMetrics>,
    pub predictions: Vec<EvaluatedPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatedPrediction {
    pub id: String,
    pub actual_label: String,
    pub predicted_label: String,
    pub correct: bool,
    pub accepted: bool,
    pub confidence: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThresholdCalibration {
    pub strategy: String,
    pub training_example_count: usize,
    pub ood_example_count: usize,
    pub maximum_ood_coverage: f64,
    pub minimum_training_selective_accuracy: f64,
    pub selected_thresholds: DecisionThresholds,
    pub observed_training_coverage: f64,
    pub observed_training_selective_accuracy: f64,
    pub observed_ood_coverage: f64,
    pub holdout_used_for_calibration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodPrediction {
    pub id: String,
    pub predicted_label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodMetrics {
    pub example_count: usize,
    pub accepted_examples: usize,
    pub rejected_examples: usize,
    pub coverage: f64,
    pub abstention_rate: f64,
    pub mean_confidence: f64,
    pub mean_margin: f64,
    pub predictions: Vec<OodPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingReport {
    pub schema_version: u32,
    pub model_kind: String,
    pub model_version: String,
    pub dataset_fingerprint: String,
    pub seed: u64,
    pub total_examples: usize,
    pub class_counts: BTreeMap<String, usize>,
    pub training_example_ids: Vec<String>,
    pub holdout_example_ids: Vec<String>,
    pub vocabulary_size: usize,
    pub training_metrics: EvaluationMetrics,
    pub holdout_metrics: EvaluationMetrics,
    pub calibration: Option<ThresholdCalibration>,
    pub ood_metrics: Option<OodMetrics>,
}

impl TrainingReport {
    pub fn to_json_pretty(&self) -> Result<String, MlError> {
        Ok(format!("{}\n", serde_json::to_string_pretty(self)?))
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), MlError> {
        write_json_file(path.as_ref(), self.to_json_pretty()?.as_bytes())
    }
}

/// Writes the model and its report as one recoverable transaction. Both payloads are serialized
/// and synced before either destination changes. Existing destinations are moved aside together;
/// if either install fails, the previous pair is restored.
pub fn write_training_artifacts(
    model: &IntentModel,
    model_path: impl AsRef<Path>,
    report: &TrainingReport,
    report_path: impl AsRef<Path>,
) -> Result<(), MlError> {
    let model_json = model.to_json_pretty()?;
    let report_json = report.to_json_pretty()?;
    write_json_pair(
        model_path.as_ref(),
        model_json.as_bytes(),
        report_path.as_ref(),
        report_json.as_bytes(),
        false,
    )
}

struct StagedJsonFile {
    target: std::path::PathBuf,
    temporary: std::path::PathBuf,
    backup: std::path::PathBuf,
    had_original: bool,
    installed: bool,
}

fn stage_json_file(path: &Path, content: &[u8]) -> Result<StagedJsonFile, MlError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    if path.exists() && !path.is_file() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("artifact destination {} is not a file", path.display()),
        )));
    }
    let directory = parent.unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output path has no file name",
        ))
    })?;
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = directory.join(format!(
        ".{}.tmp-{}-{sequence}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    let backup = directory.join(format!(
        ".{}.bak-{}-{sequence}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(content)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(MlError::Io(error));
    }
    Ok(StagedJsonFile {
        target: path.to_path_buf(),
        temporary,
        backup,
        had_original: false,
        installed: false,
    })
}

fn backup_original(staged: &mut StagedJsonFile) -> Result<(), std::io::Error> {
    if staged.target.exists() {
        fs::rename(&staged.target, &staged.backup)?;
        staged.had_original = true;
    }
    Ok(())
}

fn restore_staged_files(files: &mut [&mut StagedJsonFile]) -> Result<(), std::io::Error> {
    let mut rollback_error = None;
    for staged in files.iter_mut().rev() {
        if staged.installed && staged.target.exists() {
            if let Err(error) = fs::remove_file(&staged.target) {
                rollback_error.get_or_insert(error);
                continue;
            }
            staged.installed = false;
        }
        if staged.had_original && staged.backup.exists() {
            if let Err(error) = fs::rename(&staged.backup, &staged.target) {
                rollback_error.get_or_insert(error);
            } else {
                staged.had_original = false;
            }
        }
        if staged.temporary.exists() {
            let _ = fs::remove_file(&staged.temporary);
        }
    }
    rollback_error.map_or(Ok(()), Err)
}

fn transaction_error(operation: std::io::Error, rollback: Option<std::io::Error>) -> MlError {
    let message = match rollback {
        Some(rollback) => {
            format!("artifact transaction failed: {operation}; rollback also failed: {rollback}")
        }
        None => format!("artifact transaction failed and was rolled back: {operation}"),
    };
    MlError::Io(std::io::Error::other(message))
}

fn write_json_pair(
    first_path: &Path,
    first_content: &[u8],
    second_path: &Path,
    second_content: &[u8],
    inject_failure_after_first_install: bool,
) -> Result<(), MlError> {
    if artifact_collision_key(first_path)? == artifact_collision_key(second_path)? {
        return Err(MlError::InvalidConfiguration(
            "model and report destinations must be distinct, including through symlink aliases"
                .into(),
        ));
    }
    let mut first = stage_json_file(first_path, first_content)?;
    let mut second = match stage_json_file(second_path, second_content) {
        Ok(staged) => staged,
        Err(error) => {
            let _ = fs::remove_file(&first.temporary);
            return Err(error);
        }
    };

    if let Err(error) = backup_original(&mut first).and_then(|_| backup_original(&mut second)) {
        let rollback = restore_staged_files(&mut [&mut first, &mut second]).err();
        return Err(transaction_error(error, rollback));
    }
    if let Err(error) = fs::rename(&first.temporary, &first.target) {
        let rollback = restore_staged_files(&mut [&mut first, &mut second]).err();
        return Err(transaction_error(error, rollback));
    }
    first.installed = true;

    let second_install = if inject_failure_after_first_install {
        Err(std::io::Error::other("injected second-artifact failure"))
    } else {
        fs::rename(&second.temporary, &second.target)
    };
    if let Err(error) = second_install {
        let rollback = restore_staged_files(&mut [&mut first, &mut second]).err();
        return Err(transaction_error(error, rollback));
    }
    second.installed = true;

    for staged in [&mut first, &mut second] {
        if staged.had_original && staged.backup.exists() {
            fs::remove_file(&staged.backup)?;
            staged.had_original = false;
        }
    }
    Ok(())
}

fn artifact_collision_key(path: &Path) -> Result<String, MlError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let existing_ancestor = absolute
        .ancestors()
        .find(|ancestor| ancestor.exists())
        .ok_or_else(|| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot find an existing ancestor for {}", path.display()),
            ))
        })?;
    let unresolved_suffix = absolute.strip_prefix(existing_ancestor).map_err(|error| {
        MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("cannot resolve {}: {error}", path.display()),
        ))
    })?;
    let resolved = existing_ancestor.canonicalize()?.join(unresolved_suffix);
    let mut normalized = PathBuf::new();
    for component in resolved.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    let key = normalized.to_string_lossy().into_owned();
    Ok(if cfg!(windows) {
        key.to_lowercase()
    } else {
        key
    })
}

fn write_json_file(path: &Path, content: &[u8]) -> Result<(), MlError> {
    let mut staged = stage_json_file(path, content)?;
    if let Err(error) = backup_original(&mut staged) {
        let rollback = restore_staged_files(&mut [&mut staged]).err();
        return Err(transaction_error(error, rollback));
    }
    if let Err(error) = fs::rename(&staged.temporary, &staged.target) {
        let rollback = restore_staged_files(&mut [&mut staged]).err();
        return Err(transaction_error(error, rollback));
    }
    staged.installed = true;
    if staged.had_original && staged.backup.exists() {
        fs::remove_file(&staged.backup)?;
    }
    Ok(())
}

fn validate_thresholds(thresholds: &DecisionThresholds) -> Result<(), MlError> {
    if !thresholds.minimum_confidence.is_finite()
        || !(0.0..=1.0).contains(&thresholds.minimum_confidence)
        || !thresholds.minimum_margin.is_finite()
        || !(0.0..=1.0).contains(&thresholds.minimum_margin)
    {
        return Err(MlError::InvalidConfiguration(
            "decision thresholds must be finite values between 0 and 1".into(),
        ));
    }
    Ok(())
}

fn valid_dataset_fingerprint(value: &str) -> bool {
    value.strip_prefix("fnv1a64:").is_some_and(|digest| {
        digest.len() == 16
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    })
}

fn valid_feature_name(term: &str, config: &VectorizerConfig) -> bool {
    let Some((prefix, payload)) = term.split_once(':') else {
        return false;
    };
    if payload.is_empty() || payload.contains(':') {
        return false;
    }
    let Some((kind, encoded_size)) = prefix.split_at_checked(1) else {
        return false;
    };
    let Ok(size) = encoded_size.parse::<usize>() else {
        return false;
    };
    match kind {
        "w" if (config.word_ngram_min..=config.word_ngram_max).contains(&size) => {
            let tokens = payload.split('_').collect::<Vec<_>>();
            tokens.len() == size
                && tokens.iter().all(|token| {
                    token.chars().next().is_some_and(char::is_alphanumeric)
                        && token.chars().next_back().is_some_and(char::is_alphanumeric)
                        && token
                            .chars()
                            .all(|character| character.is_alphanumeric() || character == '\'')
                })
        }
        "c" if (config.char_ngram_min..=config.char_ngram_max).contains(&size) => {
            payload.chars().count() == size
                && payload.chars().all(|character| {
                    character.is_alphanumeric() || matches!(character, '\'' | '^' | '$' | ' ')
                })
        }
        _ => false,
    }
}

fn rejected_invalid_model_prediction(labels: &[String]) -> Prediction {
    let label = labels
        .first()
        .filter(|label| !label.is_empty())
        .cloned()
        .unwrap_or_else(|| "invalid-model".into());
    let probabilities = BTreeMap::from([(label.clone(), 0.0)]);
    Prediction {
        label,
        accepted: false,
        confidence: 0.0,
        margin: 0.0,
        probabilities,
        top_features: Vec::new(),
    }
}

fn probabilities_for(features: &[(usize, f64)], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    let mut logits = weights
        .iter()
        .zip(biases)
        .map(|(row, bias)| {
            *bias
                + features
                    .iter()
                    .map(|(feature, value)| row.get(*feature).copied().unwrap_or(0.0) * value)
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for logit in &mut logits {
        *logit = (*logit - maximum).exp();
    }
    let total = logits.iter().sum::<f64>();
    for probability in &mut logits {
        *probability /= total;
    }
    logits
}

fn extract_terms(text: &str, config: &VectorizerConfig) -> Vec<String> {
    if config.validate().is_err() {
        return Vec::new();
    }
    let tokens = tokenize(text);
    let mut terms = Vec::new();
    for size in config.word_ngram_min..=config.word_ngram_max {
        for window in tokens.windows(size) {
            terms.push(format!("w{size}:{}", window.join("_")));
        }
    }
    let normalized = format!("^{}$", tokens.join(" "));
    let characters = normalized.chars().collect::<Vec<_>>();
    for size in config.char_ngram_min..=config.char_ngram_max {
        for window in characters.windows(size) {
            terms.push(format!("c{size}:{}", window.iter().collect::<String>()));
        }
    }
    terms
}

fn tokenize(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);
    normalized
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .map(|word| word.trim_matches('\''))
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(crate) fn normalize_text(value: &str) -> String {
    let lowercase = value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| match character {
            '’' | '‘' => '\'',
            _ => character,
        })
        .collect::<String>();
    lowercase.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn split_hash(example: &LabeledExample, seed: u64) -> u64 {
    stable_hash(
        format!(
            "{}\u{1f}{}\u{1f}{}",
            example.id,
            example.label,
            normalize_text(&example.text)
        )
        .as_bytes(),
        seed,
    )
}

fn stable_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Quantization makes serialized training artifacts byte-reproducible and avoids platform-sized
/// noise in the final decimal place. Twelve decimal places are far below this model's resolution.
fn quantize(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(source, destination)
    }

    #[cfg(windows)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(source, destination)
    }

    const MINI_DATASET: &str = "id\tlabel\ttext\n\
g1\tgreeting\thello there\n\
g2\tgreeting\thi eliza\n\
g3\tgreeting\tgood morning\n\
g4\tgreeting\tgreetings\n\
g5\tgreeting\thello again\n\
q1\tquestion\twhat should I do?\n\
q2\tquestion\twhere do I begin?\n\
q3\tquestion\tcan this work?\n\
q4\tquestion\thow do I start?\n\
q5\tquestion\twhich option is clearer?\n";

    const MINI_OOD: &str = "id\ttext\n\
o1\tcalculate a satellite orbit\n\
o2\ttranslate this paragraph\n";

    #[test]
    fn parser_rejects_duplicate_normalized_text() {
        let invalid = format!("{MINI_DATASET}q6\tquestion\t  WHAT should I do?  \n");
        assert!(matches!(
            Dataset::from_tsv(&invalid),
            Err(MlError::InvalidDataset(message)) if message.contains("duplicates normalized text")
        ));
    }

    #[test]
    fn strict_tsv_parsers_reject_extra_columns() {
        let supervised = format!("{MINI_DATASET}q6\tquestion\ta valid question\textra\n");
        assert!(matches!(
            Dataset::from_tsv(&supervised),
            Err(MlError::InvalidDataset(message)) if message.contains("three tab-separated fields")
        ));
        assert!(matches!(
            OodDataset::from_tsv("id\ttext\no1\tunrelated text\textra\n"),
            Err(MlError::InvalidDataset(message)) if message.contains("two tab-separated fields")
        ));
    }

    #[test]
    fn split_is_stratified_disjoint_and_reproducible() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let left = dataset.stratified_split(0.2, 42).unwrap();
        let right = dataset.stratified_split(0.2, 42).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.holdout_examples().len(), 2);
        let train_ids = left
            .training_examples()
            .iter()
            .map(|example| &example.id)
            .collect::<HashSet<_>>();
        assert!(left
            .holdout_examples()
            .iter()
            .all(|example| !train_ids.contains(&example.id)));
        assert_eq!(
            left.holdout_examples()
                .iter()
                .map(|example| &example.label)
                .collect::<BTreeSet<_>>()
                .len(),
            2
        );
    }

    #[test]
    fn training_and_serialization_are_deterministic() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let config = TrainingConfig {
            epochs: 40,
            vectorizer: VectorizerConfig {
                max_features: 128,
                ..VectorizerConfig::default()
            },
            ..TrainingConfig::default()
        };
        let (left, left_report) = IntentModel::train(&dataset, config.clone()).unwrap();
        let (right, right_report) = IntentModel::train(&dataset, config).unwrap();
        assert_eq!(left, right);
        assert_eq!(left_report, right_report);

        assert_eq!(
            left.to_json_pretty().unwrap(),
            right.to_json_pretty().unwrap()
        );
        assert_eq!(
            left_report.to_json_pretty().unwrap(),
            right_report.to_json_pretty().unwrap()
        );

        let encoded = left.to_json_pretty().unwrap();
        let decoded = IntentModel::from_json(&encoded).unwrap();
        assert_eq!(decoded, left);
        assert_eq!(decoded.predict("hello lab"), left.predict("hello lab"));
    }

    #[test]
    fn an_empty_feature_vector_is_never_accepted() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        let prediction = model.predict("🪐🪐🪐");
        assert!(!prediction.accepted);
        assert!(prediction.top_features.is_empty());
    }

    #[test]
    fn vocabulary_is_fit_without_holdout_terms() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let split = dataset.stratified_split(0.2, 42).unwrap();
        let config = VectorizerConfig {
            max_features: 1_000,
            ..VectorizerConfig::default()
        };
        let train_terms = split
            .training_examples()
            .iter()
            .flat_map(|example| extract_terms(&example.text, &config))
            .collect::<HashSet<_>>();
        let holdout_only_terms = split
            .holdout_examples()
            .iter()
            .flat_map(|example| extract_terms(&example.text, &config))
            .filter(|term| !train_terms.contains(term))
            .collect::<HashSet<_>>();
        assert!(!holdout_only_terms.is_empty());

        let vectorizer = TfidfVectorizer::fit(split.training_examples(), config).unwrap();
        assert!(vectorizer
            .vocabulary
            .iter()
            .all(|term| !holdout_only_terms.contains(term)));
    }

    #[test]
    fn strict_model_validation_rejects_schema_and_parameter_corruption() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();

        let mut unknown_field = serde_json::to_value(&model).unwrap();
        unknown_field
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), serde_json::Value::Bool(true));
        assert!(IntentModel::from_json(&unknown_field.to_string()).is_err());

        let mut wrong_version = model.clone();
        wrong_version.schema_version += 1;
        assert!(wrong_version.validate().is_err());

        let mut non_finite = model.clone();
        non_finite.weights[0][0] = f64::NAN;
        assert!(non_finite.validate().is_err());
        assert!(non_finite.to_json_pretty().is_err());

        let mut non_rectangular = model;
        non_rectangular.weights[0].pop();
        assert!(non_rectangular.validate().is_err());
    }

    #[test]
    fn strict_model_validation_covers_vectorizer_and_fingerprint_invariants() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();

        let mut invalid_serialized_config = model.clone();
        invalid_serialized_config.vectorizer.config.word_ngram_min = 0;
        assert!(invalid_serialized_config.validate().is_err());

        let mut mismatched_config = model.clone();
        mismatched_config.vectorizer.config.char_ngram_max -= 1;
        assert!(mismatched_config.validate().is_err());

        let mut duplicate_feature = model.clone();
        duplicate_feature.vectorizer.vocabulary[1] =
            duplicate_feature.vectorizer.vocabulary[0].clone();
        assert!(duplicate_feature.validate().is_err());

        let mut malformed_feature = model.clone();
        malformed_feature.vectorizer.vocabulary[0] = "w0:broken".into();
        assert!(malformed_feature.validate().is_err());

        let mut invalid_idf = model.clone();
        invalid_idf.vectorizer.inverse_document_frequency[0] = 0.5;
        assert!(invalid_idf.validate().is_err());

        for fingerprint in [
            "fnv1a64:1234",
            "fnv1a64:0123456789ABCDEf",
            "sha256:0123456789abcdef",
            "fnv1a64:0123456789abcdeg",
        ] {
            let mut invalid_fingerprint = model.clone();
            invalid_fingerprint.dataset_fingerprint = fingerprint.into();
            assert!(invalid_fingerprint.validate().is_err(), "{fingerprint}");
        }
    }

    #[test]
    fn training_seed_respects_the_cross_runtime_json_integer_boundary() {
        let maximum = TrainingConfig {
            seed: MAX_JSON_SAFE_INTEGER,
            ..TrainingConfig::default()
        };
        assert!(maximum.validate().is_ok());
        let too_large = TrainingConfig {
            seed: MAX_JSON_SAFE_INTEGER + 1,
            ..TrainingConfig::default()
        };
        assert!(matches!(
            too_large.validate(),
            Err(MlError::InvalidConfiguration(message)) if message.contains("JSON-safe")
        ));
    }

    #[test]
    fn prediction_is_panic_free_even_when_serde_validation_is_bypassed() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        let mut encoded = serde_json::to_value(model).unwrap();
        encoded["vectorizer"]["config"]["word_ngram_min"] = 0.into();
        let malformed: IntentModel = serde_json::from_value(encoded).unwrap();

        let prediction = std::panic::catch_unwind(|| malformed.predict("the a''b ended early"))
            .expect("malformed deserialized models must abstain instead of panicking");
        assert!(!prediction.accepted);
        assert_eq!(prediction.confidence, 0.0);
        assert!(prediction.top_features.is_empty());
    }

    #[test]
    fn calibration_rejects_ood_overlap_and_split_provenance_mismatch() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        let split = dataset
            .stratified_split(
                model.training_config.holdout_fraction,
                model.training_config.seed,
            )
            .unwrap();

        let overlapping_text = split.training_examples()[0].text.clone();
        let overlapping =
            OodDataset::from_tsv(&format!("id\ttext\no1\t{overlapping_text}\n")).unwrap();
        let mut candidate = model.clone();
        assert!(matches!(
            candidate.calibrate_thresholds(&split, &overlapping, 1.0, 0.0),
            Err(MlError::InvalidDataset(message)) if message.contains("overlaps")
        ));

        let altered = MINI_DATASET.replace("hello there", "hello from elsewhere");
        let altered_dataset = Dataset::from_tsv(&altered).unwrap();
        let altered_split = altered_dataset
            .stratified_split(
                model.training_config.holdout_fraction,
                model.training_config.seed,
            )
            .unwrap();
        let ood = OodDataset::from_tsv(MINI_OOD).unwrap();
        assert!(matches!(
            candidate.calibrate_thresholds(&altered_split, &ood, 1.0, 0.0),
            Err(MlError::InvalidDataset(message)) if message.contains("provenance")
        ));
    }

    #[test]
    fn calibration_rejects_a_tampered_training_holdout_boundary() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (mut model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        let mut split = dataset
            .stratified_split(
                model.training_config.holdout_fraction,
                model.training_config.seed,
            )
            .unwrap();
        split.holdout[0] = split.train[0].clone();
        let ood = OodDataset::from_tsv(MINI_OOD).unwrap();

        assert!(model.calibrate_thresholds(&split, &ood, 1.0, 0.0).is_err());
    }

    #[test]
    fn artifact_pair_transaction_restores_both_previous_files() {
        let directory = std::env::temp_dir().join(format!(
            "eliza-artifact-transaction-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("model.json");
        let report_path = directory.join("report.json");
        fs::write(&model_path, b"old-model").unwrap();
        fs::write(&report_path, b"old-report").unwrap();

        let result = write_json_pair(&model_path, b"new-model", &report_path, b"new-report", true);
        assert!(result.is_err());
        assert_eq!(fs::read(&model_path).unwrap(), b"old-model");
        assert_eq!(fs::read(&report_path).unwrap(), b"old-report");
        assert!(fs::read_dir(&directory).unwrap().all(|entry| {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            !name.contains(".tmp-") && !name.contains(".bak-")
        }));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn artifact_pair_serializes_everything_before_touching_destinations() {
        let dataset = Dataset::from_tsv(MINI_DATASET).unwrap();
        let (mut model, report) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        model.vectorizer.config.word_ngram_min = 0;
        let directory = std::env::temp_dir().join(format!(
            "eliza-artifact-preserialize-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("model.json");
        let report_path = directory.join("report.json");
        fs::write(&model_path, b"old-model").unwrap();
        fs::write(&report_path, b"old-report").unwrap();

        assert!(write_training_artifacts(&model, &model_path, &report, &report_path).is_err());
        assert_eq!(fs::read(&model_path).unwrap(), b"old-model");
        assert_eq!(fs::read(&report_path).unwrap(), b"old-report");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn artifact_pair_rejects_identical_destinations_before_staging() {
        let directory = std::env::temp_dir().join(format!(
            "eliza-artifact-collision-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let destination = directory.join("artifact.json");
        fs::write(&destination, b"original").unwrap();

        assert!(write_json_pair(&destination, b"model", &destination, b"report", false).is_err());
        assert_eq!(fs::read(&destination).unwrap(), b"original");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn artifact_pair_rejects_symlink_aliased_future_destinations() {
        let directory = std::env::temp_dir().join(format!(
            "eliza-artifact-alias-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let real_parent = directory.join("real");
        let alias_parent = directory.join("alias");
        fs::create_dir_all(&real_parent).unwrap();
        if let Err(error) = symlink_directory(&real_parent, &alias_parent) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                fs::remove_dir_all(directory).unwrap();
                return;
            }
            panic!("failed to create test directory symlink: {error}");
        }
        if fs::read_dir(&alias_parent).is_err() {
            #[cfg(unix)]
            fs::remove_file(&alias_parent).unwrap();
            #[cfg(windows)]
            fs::remove_dir(&alias_parent).unwrap();
            fs::remove_dir_all(directory).unwrap();
            return;
        }

        let model_path = real_parent.join("future/artifact.json");
        let report_path = alias_parent.join("future/artifact.json");
        assert_eq!(
            artifact_collision_key(&model_path).unwrap(),
            artifact_collision_key(&report_path).unwrap()
        );
        assert!(write_json_pair(&model_path, b"model", &report_path, b"report", false).is_err());
        assert!(!real_parent.join("future").exists());

        #[cfg(unix)]
        fs::remove_file(&alias_parent).unwrap();
        #[cfg(windows)]
        fs::remove_dir(&alias_parent).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
}
