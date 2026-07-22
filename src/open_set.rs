//! Leak-resistant, local open-set intent classification.
//!
//! This module is the version-two experimental path. It deliberately keeps the legacy model and
//! CLI stable while adding group-aware data partitions, probability calibration, independent OOD
//! evaluation, cryptographically linked artifacts, and a compiled inference representation.

use crate::ml::{MlError, VectorizerConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const OPEN_SET_SCHEMA_VERSION: u32 = 2;
pub const OPEN_SET_MODEL_VERSION: &str = "2.0.0";
pub const OPEN_SET_MODEL_KIND: &str = "eliza-open-set-linear";
pub const DEFAULT_BOOTSTRAP_RESAMPLES: usize = 1_000;
const BUNDLE_INVENTORY: [&str; 5] = [
    "manifest.json",
    "metrics.json",
    "model.json",
    "policy.json",
    "split-plan.json",
];
const MAX_EXAMPLES: usize = 100_000;
const MAX_GROUPS_PER_LABEL: usize = 20_000;
const MAX_CLASSES: usize = 256;
const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JSONL_BYTES: usize = crate::MAX_INPUT_CHARS * 4 + 16_384;
const MAX_BOOTSTRAP_SAMPLED_ROWS: usize = 5_000_000;
const MAX_PARAMETER_MAGNITUDE: f64 = 1_000_000.0;
const MAX_IDF: f64 = 64.0;
static BUNDLE_TEMP_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedExample {
    pub id: String,
    pub group_id: String,
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedDataset {
    examples: Vec<GroupedExample>,
}

impl GroupedDataset {
    pub fn bundled() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/intents-v2.tsv"))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        let path = path.as_ref();
        reject_oversized_file(path, MAX_JSON_BYTES, "grouped dataset")?;
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the grouped dataset is empty".into()))?;
        if header != "id\tgroup_id\tlabel\ttext" {
            return Err(MlError::InvalidDataset(
                "the grouped header must be exactly `id\\tgroup_id\\tlabel\\ttext`".into(),
            ));
        }

        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut normalized_texts = HashSet::new();
        let mut group_labels: HashMap<String, String> = HashMap::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(MlError::InvalidDataset(format!(
                    "grouped line {line_number} must contain four tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let group_id = fields[1].trim();
            let label = fields[2].trim();
            let text = fields[3].trim();
            validate_identifier(id, "example id", line_number)?;
            validate_identifier(group_id, "group id", line_number)?;
            validate_label(label, line_number)?;
            validate_text(text, line_number, "grouped")?;
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!(
                    "duplicate grouped example id `{id}`"
                )));
            }
            if !normalized_texts.insert(normalize_text(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "grouped line {line_number} duplicates normalized text"
                )));
            }
            match group_labels.get(group_id) {
                Some(existing) if existing != label => {
                    return Err(MlError::InvalidDataset(format!(
                        "group `{group_id}` crosses labels `{existing}` and `{label}`"
                    )))
                }
                Some(_) => {}
                None => {
                    group_labels.insert(group_id.to_owned(), label.to_owned());
                }
            }
            examples.push(GroupedExample {
                id: id.to_owned(),
                group_id: group_id.to_owned(),
                label: label.to_owned(),
                text: text.to_owned(),
            });
            if examples.len() > MAX_EXAMPLES {
                return Err(MlError::InvalidDataset(format!(
                    "the grouped dataset exceeds {MAX_EXAMPLES} examples"
                )));
            }
        }
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "the grouped dataset contains no examples".into(),
            ));
        }
        let dataset = Self { examples };
        dataset.validate_partition_support()?;
        Ok(dataset)
    }

    pub fn examples(&self) -> &[GroupedExample] {
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

    pub fn fingerprint_sha256(&self) -> String {
        let mut rows = self
            .examples
            .iter()
            .map(|example| {
                format!(
                    "{}\t{}\t{}\t{}",
                    example.id,
                    example.group_id,
                    example.label,
                    normalize_text(&example.text)
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        sha256_hex(rows.join("\n").as_bytes())
    }

    fn validate_partition_support(&self) -> Result<(), MlError> {
        let mut groups_by_label: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
        for example in &self.examples {
            groups_by_label
                .entry(&example.label)
                .or_default()
                .insert(&example.group_id);
        }
        if groups_by_label.len() < 2 || groups_by_label.len() > MAX_CLASSES {
            return Err(MlError::InvalidDataset(format!(
                "the grouped dataset must contain between 2 and {MAX_CLASSES} labels"
            )));
        }
        for (label, groups) in groups_by_label {
            if groups.len() < 4 {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` has {} groups; at least four are required",
                    groups.len()
                )));
            }
            if groups.len() > MAX_GROUPS_PER_LABEL {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` exceeds the group limit"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetOodExample {
    pub id: String,
    pub group_id: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetOodDataset {
    examples: Vec<OpenSetOodExample>,
}

impl OpenSetOodDataset {
    pub fn bundled_development() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/ood-dev-v2.tsv"))
    }

    pub fn bundled_test() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/ood-test-v2.tsv"))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        let path = path.as_ref();
        reject_oversized_file(path, MAX_JSON_BYTES, "OOD dataset")?;
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the OOD dataset is empty".into()))?;
        if header != "id\tgroup_id\ttext" {
            return Err(MlError::InvalidDataset(
                "the OOD header must be exactly `id\\tgroup_id\\ttext`".into(),
            ));
        }
        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut groups = HashSet::new();
        let mut texts = HashSet::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} must contain three tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let group_id = fields[1].trim();
            let text = fields[2].trim();
            validate_identifier(id, "OOD id", line_number)?;
            validate_identifier(group_id, "OOD group id", line_number)?;
            validate_text(text, line_number, "OOD")?;
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!("duplicate OOD id `{id}`")));
            }
            if !groups.insert(group_id.to_owned()) {
                return Err(MlError::InvalidDataset(format!(
                    "duplicate OOD group id `{group_id}`; OOD rows must be independently grouped"
                )));
            }
            if !texts.insert(normalize_text(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} duplicates normalized text"
                )));
            }
            examples.push(OpenSetOodExample {
                id: id.to_owned(),
                group_id: group_id.to_owned(),
                text: text.to_owned(),
            });
            if examples.len() > MAX_EXAMPLES {
                return Err(MlError::InvalidDataset(format!(
                    "the OOD dataset exceeds {MAX_EXAMPLES} examples"
                )));
            }
        }
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "the OOD dataset contains no examples".into(),
            ));
        }
        Ok(Self { examples })
    }

    pub fn examples(&self) -> &[OpenSetOodExample] {
        &self.examples
    }

    pub fn fingerprint_sha256(&self) -> String {
        let mut rows = self
            .examples
            .iter()
            .map(|example| {
                format!(
                    "{}\t{}\t{}",
                    example.id,
                    example.group_id,
                    normalize_text(&example.text)
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        sha256_hex(rows.join("\n").as_bytes())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum PartitionKind {
    Train,
    Development,
    Calibration,
    IdTest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitAssignment {
    pub id: String,
    pub group_id: String,
    pub label: String,
    pub partition: PartitionKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitPlanManifest {
    pub schema_version: u32,
    pub strategy: String,
    pub seed: u64,
    pub dataset_sha256: String,
    pub assignments: Vec<SplitAssignment>,
}

impl SplitPlanManifest {
    fn validate_contract(&self) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.strategy != "group-stratified-scaled-four-way-v2"
            || self.seed > 9_007_199_254_740_991
            || !valid_sha256(&self.dataset_sha256)
            || self.assignments.is_empty()
            || self.assignments.len() > MAX_EXAMPLES
        {
            return Err(MlError::InvalidDataset(
                "the split-plan manifest has an invalid identity or size".into(),
            ));
        }

        let mut ids = HashSet::new();
        let mut group_ownership: HashMap<&str, (&str, PartitionKind)> = HashMap::new();
        let mut groups_by_label: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut labels_by_partition: BTreeMap<PartitionKind, BTreeSet<&str>> = BTreeMap::new();
        let mut previous_id: Option<&str> = None;
        for (index, assignment) in self.assignments.iter().enumerate() {
            validate_identifier(&assignment.id, "split assignment id", index + 1)?;
            validate_identifier(&assignment.group_id, "split assignment group id", index + 1)?;
            validate_label(&assignment.label, index + 1)?;
            if previous_id.is_some_and(|previous| previous >= assignment.id.as_str())
                || !ids.insert(assignment.id.as_str())
            {
                return Err(MlError::InvalidDataset(
                    "split assignments must have unique, ascending ids".into(),
                ));
            }
            previous_id = Some(&assignment.id);
            match group_ownership.insert(
                assignment.group_id.as_str(),
                (assignment.label.as_str(), assignment.partition),
            ) {
                Some((label, partition))
                    if label != assignment.label || partition != assignment.partition =>
                {
                    return Err(MlError::InvalidDataset(format!(
                        "split group `{}` crosses labels or partitions",
                        assignment.group_id
                    )))
                }
                _ => {}
            }
            labels_by_partition
                .entry(assignment.partition)
                .or_default()
                .insert(&assignment.label);
            groups_by_label
                .entry(&assignment.label)
                .or_default()
                .insert(&assignment.group_id);
        }
        let expected_labels = labels_by_partition
            .get(&PartitionKind::Train)
            .cloned()
            .unwrap_or_default();
        if expected_labels.len() < 2
            || [
                PartitionKind::Train,
                PartitionKind::Development,
                PartitionKind::Calibration,
                PartitionKind::IdTest,
            ]
            .iter()
            .any(|partition| labels_by_partition.get(partition) != Some(&expected_labels))
        {
            return Err(MlError::InvalidDataset(
                "every split partition must contain the same two or more labels".into(),
            ));
        }
        for (label, groups) in groups_by_label {
            if groups.len() < 4 || groups.len() > MAX_GROUPS_PER_LABEL {
                return Err(MlError::InvalidDataset(format!(
                    "split label `{label}` has an invalid group count"
                )));
            }
            let mut ordered_groups = groups.into_iter().collect::<Vec<_>>();
            ordered_groups.sort_by(|left, right| {
                group_split_hash(label, left, self.seed)
                    .cmp(&group_split_hash(label, right, self.seed))
                    .then_with(|| left.cmp(right))
            });
            let evaluation_groups = evaluation_group_quota(ordered_groups.len());
            for (group_index, group_id) in ordered_groups.into_iter().enumerate() {
                let expected_partition = if group_index < evaluation_groups {
                    PartitionKind::IdTest
                } else if group_index < evaluation_groups * 2 {
                    PartitionKind::Calibration
                } else if group_index < evaluation_groups * 3 {
                    PartitionKind::Development
                } else {
                    PartitionKind::Train
                };
                if group_ownership[group_id].1 != expected_partition {
                    return Err(MlError::InvalidDataset(format!(
                        "split group `{group_id}` does not match the declared strategy"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SplitPlan {
    manifest: SplitPlanManifest,
    train: Vec<GroupedExample>,
    development: Vec<GroupedExample>,
    calibration: Vec<GroupedExample>,
    id_test: Vec<GroupedExample>,
}

impl SplitPlan {
    pub fn build(dataset: &GroupedDataset, seed: u64) -> Result<Self, MlError> {
        dataset.validate_partition_support()?;
        let mut grouped: BTreeMap<String, BTreeMap<String, Vec<GroupedExample>>> = BTreeMap::new();
        for example in dataset.examples() {
            grouped
                .entry(example.label.clone())
                .or_default()
                .entry(example.group_id.clone())
                .or_default()
                .push(example.clone());
        }

        let mut assignments = Vec::with_capacity(dataset.examples().len());
        let mut train = Vec::new();
        let mut development = Vec::new();
        let mut calibration = Vec::new();
        let mut id_test = Vec::new();
        for (label, groups) in grouped {
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_by(|(left, _), (right, _)| {
                group_split_hash(&label, left, seed)
                    .cmp(&group_split_hash(&label, right, seed))
                    .then_with(|| left.cmp(right))
            });
            if groups.len() < 4 {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` cannot populate four group-disjoint partitions"
                )));
            }
            let evaluation_groups = evaluation_group_quota(groups.len());
            for (group_index, (_, mut examples)) in groups.into_iter().enumerate() {
                examples.sort_by(|left, right| left.id.cmp(&right.id));
                let partition = if group_index < evaluation_groups {
                    PartitionKind::IdTest
                } else if group_index < evaluation_groups * 2 {
                    PartitionKind::Calibration
                } else if group_index < evaluation_groups * 3 {
                    PartitionKind::Development
                } else {
                    PartitionKind::Train
                };
                for example in examples {
                    assignments.push(SplitAssignment {
                        id: example.id.clone(),
                        group_id: example.group_id.clone(),
                        label: example.label.clone(),
                        partition,
                    });
                    match partition {
                        PartitionKind::Train => train.push(example),
                        PartitionKind::Development => development.push(example),
                        PartitionKind::Calibration => calibration.push(example),
                        PartitionKind::IdTest => id_test.push(example),
                    }
                }
            }
        }
        assignments.sort_by(|left, right| left.id.cmp(&right.id));
        for partition in [&mut train, &mut development, &mut calibration, &mut id_test] {
            partition.sort_by(|left, right| left.id.cmp(&right.id));
        }
        let plan = Self {
            manifest: SplitPlanManifest {
                schema_version: OPEN_SET_SCHEMA_VERSION,
                strategy: "group-stratified-scaled-four-way-v2".into(),
                seed,
                dataset_sha256: dataset.fingerprint_sha256(),
                assignments,
            },
            train,
            development,
            calibration,
            id_test,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn train(&self) -> &[GroupedExample] {
        &self.train
    }

    pub fn development(&self) -> &[GroupedExample] {
        &self.development
    }

    pub fn calibration(&self) -> &[GroupedExample] {
        &self.calibration
    }

    pub fn id_test(&self) -> &[GroupedExample] {
        &self.id_test
    }

    pub fn manifest(&self) -> &SplitPlanManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> Result<String, MlError> {
        Ok(sha256_hex(&canonical_json(&self.manifest)?))
    }

    fn validate(&self) -> Result<(), MlError> {
        self.manifest.validate_contract()?;
        let partitions = [
            (PartitionKind::Train, self.train()),
            (PartitionKind::Development, self.development()),
            (PartitionKind::Calibration, self.calibration()),
            (PartitionKind::IdTest, self.id_test()),
        ];
        let mut seen_ids = HashSet::new();
        let mut expected_assignments = Vec::with_capacity(self.manifest.assignments.len());
        let mut group_partition: HashMap<&str, PartitionKind> = HashMap::new();
        let mut labels_by_partition: BTreeMap<PartitionKind, BTreeSet<&str>> = BTreeMap::new();
        for (kind, examples) in partitions {
            if examples.is_empty() {
                return Err(MlError::InvalidDataset(format!(
                    "partition {kind:?} is empty"
                )));
            }
            for example in examples {
                if !seen_ids.insert(example.id.as_str()) {
                    return Err(MlError::InvalidDataset(format!(
                        "example `{}` crosses partitions",
                        example.id
                    )));
                }
                if let Some(previous) = group_partition.insert(&example.group_id, kind) {
                    if previous != kind {
                        return Err(MlError::InvalidDataset(format!(
                            "group `{}` crosses {previous:?} and {kind:?}",
                            example.group_id
                        )));
                    }
                }
                labels_by_partition
                    .entry(kind)
                    .or_default()
                    .insert(&example.label);
                expected_assignments.push(SplitAssignment {
                    id: example.id.clone(),
                    group_id: example.group_id.clone(),
                    label: example.label.clone(),
                    partition: kind,
                });
            }
        }
        let expected_labels = labels_by_partition
            .get(&PartitionKind::Train)
            .cloned()
            .unwrap_or_default();
        if labels_by_partition
            .values()
            .any(|labels| *labels != expected_labels)
        {
            return Err(MlError::InvalidDataset(
                "all four partitions must contain the same labels".into(),
            ));
        }
        expected_assignments.sort_by(|left, right| left.id.cmp(&right.id));
        if self.manifest.assignments != expected_assignments {
            return Err(MlError::InvalidDataset(
                "split assignments do not exactly match the partition examples".into(),
            ));
        }
        Ok(())
    }
}

fn evaluation_group_quota(group_count: usize) -> usize {
    ((group_count + 5) / 10).max(1).min((group_count - 1) / 3)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetTrainingConfig {
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2_penalty: f64,
    pub vectorizer: VectorizerConfig,
}

impl Default for OpenSetTrainingConfig {
    fn default() -> Self {
        Self {
            seed: 20_260_722,
            epochs: 600,
            learning_rate: 0.8,
            l2_penalty: 0.0005,
            vectorizer: VectorizerConfig::default(),
        }
    }
}

impl OpenSetTrainingConfig {
    fn validate(&self) -> Result<(), MlError> {
        if self.seed > 9_007_199_254_740_991 {
            return Err(MlError::InvalidConfiguration(
                "the open-set seed exceeds the JSON-safe integer ceiling".into(),
            ));
        }
        if !(1..=10_000).contains(&self.epochs) {
            return Err(MlError::InvalidConfiguration(
                "open-set epochs must be between 1 and 10000".into(),
            ));
        }
        if !self.learning_rate.is_finite() || !(0.000_001..=10.0).contains(&self.learning_rate) {
            return Err(MlError::InvalidConfiguration(
                "open-set learning_rate must be finite and between 0.000001 and 10".into(),
            ));
        }
        if !self.l2_penalty.is_finite() || !(0.0..=1.0).contains(&self.l2_penalty) {
            return Err(MlError::InvalidConfiguration(
                "open-set l2_penalty must be finite and between 0 and 1".into(),
            ));
        }
        validate_vectorizer_config(&self.vectorizer)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetVectorizer {
    pub config: VectorizerConfig,
    pub vocabulary: Vec<String>,
    pub inverse_document_frequency: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetModelV2 {
    pub schema_version: u32,
    pub model_kind: String,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub training_config: OpenSetTrainingConfig,
    pub labels: Vec<String>,
    pub vectorizer: OpenSetVectorizer,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl OpenSetModelV2 {
    pub fn validate(&self) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.model_kind != OPEN_SET_MODEL_KIND
            || self.model_version != OPEN_SET_MODEL_VERSION
            || !valid_sha256(&self.dataset_sha256)
            || !valid_sha256(&self.split_plan_sha256)
        {
            return Err(MlError::InvalidModel(
                "the open-set model identity is invalid".into(),
            ));
        }
        self.training_config.validate()?;
        self.vectorizer.validate()?;
        if self.training_config.vectorizer != self.vectorizer.config {
            return Err(MlError::InvalidModel(
                "open-set training and serialized vectorizer configs differ".into(),
            ));
        }
        if self.labels.len() < 2
            || self.labels.len() > MAX_CLASSES
            || self.labels.len() != self.weights.len()
            || self.labels.len() != self.biases.len()
            || self.labels.iter().collect::<HashSet<_>>().len() != self.labels.len()
        {
            return Err(MlError::InvalidModel(
                "open-set labels, weights, and biases are not aligned".into(),
            ));
        }
        for label in &self.labels {
            if label.is_empty()
                || !label
                    .chars()
                    .all(|character| character.is_ascii_lowercase() || character == '-')
            {
                return Err(MlError::InvalidModel(
                    "the open-set model contains an invalid label".into(),
                ));
            }
        }
        for (row, bias) in self.weights.iter().zip(&self.biases) {
            if row.len() != self.vectorizer.vocabulary.len()
                || row
                    .iter()
                    .any(|weight| !weight.is_finite() || weight.abs() > MAX_PARAMETER_MAGNITUDE)
                || !bias.is_finite()
                || bias.abs() > MAX_PARAMETER_MAGNITUDE
            {
                return Err(MlError::InvalidModel(
                    "open-set parameters must be finite, bounded, and rectangular".into(),
                ));
            }
        }
        Ok(())
    }
}

impl OpenSetVectorizer {
    fn fit(examples: &[GroupedExample], config: VectorizerConfig) -> Result<Self, MlError> {
        validate_vectorizer_config(&config)?;
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for example in examples {
            for term in extract_terms(&example.text, &config)
                .into_iter()
                .collect::<HashSet<_>>()
            {
                *document_frequency.entry(term).or_insert(0) += 1;
            }
        }
        let mut candidates = document_frequency
            .into_iter()
            .filter(|(_, count)| *count >= config.min_document_frequency)
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_term, left_count), (right_term, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_term.cmp(right_term))
        });
        candidates.truncate(config.max_features);
        if candidates.is_empty() {
            return Err(MlError::InvalidDataset(
                "the open-set training partition produced an empty vocabulary".into(),
            ));
        }
        let document_count = examples.len() as f64;
        let (vocabulary, inverse_document_frequency) = candidates
            .into_iter()
            .map(|(term, count)| {
                let idf = quantize(((1.0 + document_count) / (1.0 + count as f64)).ln() + 1.0);
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
        validate_vectorizer_config(&self.config).map_err(|error| {
            MlError::InvalidModel(format!(
                "the open-set vectorizer config is invalid: {error}"
            ))
        })?;
        if self.vocabulary.is_empty()
            || self.vocabulary.len() != self.inverse_document_frequency.len()
            || self.vocabulary.len() > self.config.max_features
            || self.vocabulary.iter().collect::<HashSet<_>>().len() != self.vocabulary.len()
            || self
                .inverse_document_frequency
                .iter()
                .any(|value| !value.is_finite() || !(1.0..=MAX_IDF).contains(value))
        {
            return Err(MlError::InvalidModel(
                "the open-set vectorizer violates its shape contract".into(),
            ));
        }
        Ok(())
    }
}

fn fit_model(plan: &SplitPlan, config: OpenSetTrainingConfig) -> Result<OpenSetModelV2, MlError> {
    config.validate()?;
    let labels = plan
        .train()
        .iter()
        .map(|example| example.label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if labels.len() < 2 {
        return Err(MlError::InvalidDataset(
            "open-set training requires at least two labels".into(),
        ));
    }
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let vectorizer = OpenSetVectorizer::fit(plan.train(), config.vectorizer.clone())?;
    let feature_index = build_feature_index(&vectorizer.vocabulary);
    let features = plan
        .train()
        .iter()
        .map(|example| transform(&vectorizer, &feature_index, &example.text))
        .collect::<Vec<_>>();
    let targets = plan
        .train()
        .iter()
        .map(|example| label_index[example.label.as_str()])
        .collect::<Vec<_>>();
    let mut weights = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
    let mut biases = vec![0.0; labels.len()];
    let sample_count = plan.train().len() as f64;
    for epoch in 0..config.epochs {
        let mut weight_gradient = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
        let mut bias_gradient = vec![0.0; labels.len()];
        for (row, target) in features.iter().zip(&targets) {
            let probabilities = softmax(&logits_for(row, &weights, &biases), 1.0);
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
    let model = OpenSetModelV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_kind: OPEN_SET_MODEL_KIND.into(),
        model_version: OPEN_SET_MODEL_VERSION.into(),
        dataset_sha256: plan.manifest.dataset_sha256.clone(),
        split_plan_sha256: plan.manifest_sha256()?,
        training_config: config,
        labels,
        vectorizer,
        weights,
        biases,
    };
    model.validate()?;
    Ok(model)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetPolicyV2 {
    pub schema_version: u32,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub temperature: f64,
    pub minimum_confidence: f64,
    pub minimum_probability_margin: f64,
    pub temperature_source: String,
    pub threshold_source: String,
    pub calibration_example_count: usize,
    pub development_example_count: usize,
    pub ood_development_example_count: usize,
}

impl OpenSetPolicyV2 {
    fn validate_against(&self, model: &OpenSetModelV2) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.model_version != model.model_version
            || self.dataset_sha256 != model.dataset_sha256
            || self.split_plan_sha256 != model.split_plan_sha256
            || self.temperature_source != "calibration-partition-temperature-scaling-v2"
            || self.threshold_source != "development-plus-ood-development-grid-v2"
        {
            return Err(MlError::InvalidModel(
                "the open-set policy does not match its model and provenance".into(),
            ));
        }
        if !self.temperature.is_finite() || !(0.05..=20.0).contains(&self.temperature) {
            return Err(MlError::InvalidModel(
                "the open-set temperature must be finite and between 0.05 and 20".into(),
            ));
        }
        if !self.minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_confidence)
            || !self.minimum_probability_margin.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_probability_margin)
            || self.calibration_example_count == 0
            || self.development_example_count == 0
            || self.ood_development_example_count == 0
        {
            return Err(MlError::InvalidModel(
                "the open-set operating policy contains invalid thresholds or provenance counts"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastiveContribution {
    pub feature: String,
    pub value: f64,
    pub top_weight: f64,
    pub runner_up_weight: f64,
    pub contribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastiveExplanation {
    pub top_label: String,
    pub runner_up_label: String,
    pub bias_difference: f64,
    pub feature_contribution_sum: f64,
    pub reconstructed_logit_margin: f64,
    pub top_contributions: Vec<ContrastiveContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetPrediction {
    pub label: String,
    pub runner_up_label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
    pub logit_margin: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub explanation: ContrastiveExplanation,
}

#[derive(Debug, Clone)]
pub struct CompiledModel {
    model: OpenSetModelV2,
    policy: OpenSetPolicyV2,
    feature_index: HashMap<String, usize>,
}

impl CompiledModel {
    pub fn new(model: OpenSetModelV2, policy: OpenSetPolicyV2) -> Result<Self, MlError> {
        model.validate()?;
        policy.validate_against(&model)?;
        let feature_index = build_feature_index(&model.vectorizer.vocabulary);
        Ok(Self {
            model,
            policy,
            feature_index,
        })
    }

    pub fn model(&self) -> &OpenSetModelV2 {
        &self.model
    }

    pub fn policy(&self) -> &OpenSetPolicyV2 {
        &self.policy
    }

    pub fn predict(&self, text: &str) -> OpenSetPrediction {
        let features = transform(&self.model.vectorizer, &self.feature_index, text);
        let logits = logits_for(&features, &self.model.weights, &self.model.biases);
        prediction_from_scores(&self.model, &self.policy, &features, &logits)
    }

    pub fn predict_batch<'a, I>(&self, inputs: I) -> Vec<OpenSetPrediction>
    where
        I: IntoIterator<Item = &'a str>,
    {
        inputs.into_iter().map(|text| self.predict(text)).collect()
    }
}

fn prediction_from_scores(
    model: &OpenSetModelV2,
    policy: &OpenSetPolicyV2,
    features: &[(usize, f64)],
    logits: &[f64],
) -> OpenSetPrediction {
    let probabilities = softmax(logits, policy.temperature);
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
    let (runner_up_index, runner_up_probability) = ranking[1];
    let probability_margin = confidence - runner_up_probability;
    let logit_margin = logits[top_index] - logits[runner_up_index];
    let accepted = !features.is_empty()
        && confidence >= policy.minimum_confidence
        && probability_margin >= policy.minimum_probability_margin;
    let bias_difference = model.biases[top_index] - model.biases[runner_up_index];
    let mut all_contributions = features
        .iter()
        .map(|(feature, value)| {
            let top_weight = model.weights[top_index][*feature];
            let runner_up_weight = model.weights[runner_up_index][*feature];
            ContrastiveContribution {
                feature: model.vectorizer.vocabulary[*feature].clone(),
                value: *value,
                top_weight,
                runner_up_weight,
                contribution: *value * (top_weight - runner_up_weight),
            }
        })
        .collect::<Vec<_>>();
    let feature_contribution_sum = all_contributions
        .iter()
        .map(|contribution| contribution.contribution)
        .sum::<f64>();
    all_contributions.sort_by(|left, right| {
        right
            .contribution
            .abs()
            .total_cmp(&left.contribution.abs())
            .then_with(|| left.feature.cmp(&right.feature))
    });
    all_contributions.truncate(8);
    OpenSetPrediction {
        label: model.labels[top_index].clone(),
        runner_up_label: model.labels[runner_up_index].clone(),
        accepted,
        confidence,
        probability_margin,
        logit_margin,
        probabilities: model.labels.iter().cloned().zip(probabilities).collect(),
        explanation: ContrastiveExplanation {
            top_label: model.labels[top_index].clone(),
            runner_up_label: model.labels[runner_up_index].clone(),
            bias_difference,
            feature_contribution_sum,
            reconstructed_logit_margin: bias_difference + feature_contribution_sum,
            top_contributions: all_contributions,
        },
    }
}

fn calibrate_temperature(
    model: &OpenSetModelV2,
    examples: &[GroupedExample],
) -> Result<f64, MlError> {
    if examples.is_empty() {
        return Err(MlError::InvalidDataset(
            "temperature scaling requires a non-empty calibration partition".into(),
        ));
    }
    let feature_index = build_feature_index(&model.vectorizer.vocabulary);
    let label_index = model
        .labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let scored = examples
        .iter()
        .map(|example| {
            let target = label_index
                .get(example.label.as_str())
                .copied()
                .ok_or_else(|| {
                    MlError::InvalidDataset(format!(
                        "calibration label `{}` is absent from the model",
                        example.label
                    ))
                })?;
            let features = transform(&model.vectorizer, &feature_index, &example.text);
            Ok((logits_for(&features, &model.weights, &model.biases), target))
        })
        .collect::<Result<Vec<_>, MlError>>()?;

    let loss = |log_temperature: f64| {
        let temperature = log_temperature.exp();
        scored
            .iter()
            .map(|(logits, target)| -softmax(logits, temperature)[*target].max(1e-15).ln())
            .sum::<f64>()
            / scored.len() as f64
    };
    let mut left = 0.05_f64.ln();
    let mut right = 20.0_f64.ln();
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut middle_left = right - golden * (right - left);
    let mut middle_right = left + golden * (right - left);
    let mut loss_left = loss(middle_left);
    let mut loss_right = loss(middle_right);
    for _ in 0..96 {
        if loss_left <= loss_right {
            right = middle_right;
            middle_right = middle_left;
            loss_right = loss_left;
            middle_left = right - golden * (right - left);
            loss_left = loss(middle_left);
        } else {
            left = middle_left;
            middle_left = middle_right;
            loss_left = loss_right;
            middle_right = left + golden * (right - left);
            loss_right = loss(middle_right);
        }
    }
    Ok(quantize(((left + right) / 2.0).exp().clamp(0.05, 20.0)))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThresholdSelection {
    pub strategy: String,
    pub development_example_count: usize,
    pub ood_development_example_count: usize,
    pub minimum_development_selective_accuracy: f64,
    pub maximum_ood_development_coverage: f64,
    pub selected_confidence: f64,
    pub selected_probability_margin: f64,
    pub observed_development_coverage: f64,
    pub observed_development_selective_accuracy: f64,
    pub observed_ood_development_coverage: f64,
    pub id_test_used: bool,
    pub ood_test_used: bool,
}

fn select_thresholds(
    model: &OpenSetModelV2,
    temperature: f64,
    development: &[GroupedExample],
    ood_development: &OpenSetOodDataset,
) -> Result<ThresholdSelection, MlError> {
    if development.is_empty() || ood_development.examples().is_empty() {
        return Err(MlError::InvalidDataset(
            "threshold selection requires development and OOD-development examples".into(),
        ));
    }
    let temporary_policy = OpenSetPolicyV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: 0.0,
        minimum_probability_margin: 0.0,
        temperature_source: "calibration-partition-temperature-scaling-v2".into(),
        threshold_source: "development-plus-ood-development-grid-v2".into(),
        calibration_example_count: 1,
        development_example_count: development.len(),
        ood_development_example_count: ood_development.examples().len(),
    };
    let runtime = CompiledModel::new(model.clone(), temporary_policy)?;
    let development_scores = development
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            (
                prediction.accepted,
                prediction.label == example.label,
                prediction.confidence,
                prediction.probability_margin,
            )
        })
        .collect::<Vec<_>>();
    let ood_scores = ood_development
        .examples()
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            (
                prediction.accepted,
                prediction.confidence,
                prediction.probability_margin,
            )
        })
        .collect::<Vec<_>>();

    const MIN_SELECTIVE_ACCURACY: f64 = 0.75;
    const MAX_OOD_COVERAGE: f64 = 0.10;
    #[derive(Clone, Copy)]
    struct Candidate {
        confidence: f64,
        margin: f64,
        development_coverage: f64,
        development_selective_accuracy: f64,
        ood_coverage: f64,
    }
    let mut best = None;
    for confidence_step in 15..=95 {
        let confidence = confidence_step as f64 / 100.0;
        for margin_step in 0..=90 {
            let margin = margin_step as f64 / 100.0;
            let accepted = development_scores
                .iter()
                .filter(|(has_features, _, observed_confidence, observed_margin)| {
                    *has_features
                        && *observed_confidence >= confidence
                        && *observed_margin >= margin
                })
                .collect::<Vec<_>>();
            if accepted.is_empty() {
                continue;
            }
            let development_selective_accuracy = accepted
                .iter()
                .filter(|(_, correct, _, _)| *correct)
                .count() as f64
                / accepted.len() as f64;
            if development_selective_accuracy < MIN_SELECTIVE_ACCURACY {
                continue;
            }
            let ood_accepted = ood_scores
                .iter()
                .filter(|(has_features, observed_confidence, observed_margin)| {
                    *has_features
                        && *observed_confidence >= confidence
                        && *observed_margin >= margin
                })
                .count();
            let ood_coverage = ood_accepted as f64 / ood_scores.len() as f64;
            if ood_coverage > MAX_OOD_COVERAGE {
                continue;
            }
            let candidate = Candidate {
                confidence,
                margin,
                development_coverage: accepted.len() as f64 / development_scores.len() as f64,
                development_selective_accuracy,
                ood_coverage,
            };
            let is_better = best.map_or(true, |current: Candidate| {
                candidate.development_coverage > current.development_coverage
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            > current.development_selective_accuracy)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage < current.ood_coverage)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage == current.ood_coverage
                        && candidate.margin > current.margin)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage == current.ood_coverage
                        && candidate.margin == current.margin
                        && candidate.confidence > current.confidence)
            });
            if is_better {
                best = Some(candidate);
            }
        }
    }
    let best = best.ok_or_else(|| {
        MlError::InvalidConfiguration(
            "no development/OOD-development threshold satisfies the locked v2 policy".into(),
        )
    })?;
    Ok(ThresholdSelection {
        strategy: "development-plus-ood-development-grid-v2".into(),
        development_example_count: development.len(),
        ood_development_example_count: ood_development.examples().len(),
        minimum_development_selective_accuracy: MIN_SELECTIVE_ACCURACY,
        maximum_ood_development_coverage: MAX_OOD_COVERAGE,
        selected_confidence: best.confidence,
        selected_probability_margin: best.margin,
        observed_development_coverage: best.development_coverage,
        observed_development_selective_accuracy: best.development_selective_accuracy,
        observed_ood_development_coverage: best.ood_coverage,
        id_test_used: false,
        ood_test_used: false,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricEstimate {
    pub value: f64,
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationMetrics {
    pub negative_log_likelihood: f64,
    pub multiclass_brier: f64,
    pub expected_calibration_error: f64,
    pub ece_bins: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RiskCoveragePoint {
    pub accepted: usize,
    pub coverage: f64,
    pub risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatedOpenSetPrediction {
    pub id: String,
    pub actual_label: String,
    pub predicted_label: String,
    pub correct: bool,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IdEvaluationV2 {
    pub example_count: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub coverage: f64,
    pub selective_accuracy: Option<f64>,
    pub calibration: CalibrationMetrics,
    pub aurc: f64,
    pub risk_coverage_curve: Vec<RiskCoveragePoint>,
    pub predictions: Vec<EvaluatedOpenSetPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodEvaluatedPrediction {
    pub id: String,
    pub predicted_label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodDiscriminationMetrics {
    pub auroc: f64,
    pub aupr_in_domain: f64,
    pub fpr_at_95_tpr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodEvaluationV2 {
    pub example_count: usize,
    pub accepted_examples: usize,
    pub coverage: f64,
    pub discrimination: OodDiscriminationMetrics,
    pub predictions: Vec<OodEvaluatedPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BootstrapReport {
    pub strategy: String,
    pub seed: u64,
    pub resamples: usize,
    pub confidence_level: f64,
    pub id_accuracy: MetricEstimate,
    pub id_macro_f1: MetricEstimate,
    pub id_negative_log_likelihood: MetricEstimate,
    pub id_multiclass_brier: MetricEstimate,
    pub id_expected_calibration_error: MetricEstimate,
    pub id_aurc: MetricEstimate,
    pub ood_auroc: MetricEstimate,
    pub ood_aupr_in_domain: MetricEstimate,
    pub ood_fpr_at_95_tpr: MetricEstimate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetMetricsV2 {
    pub schema_version: u32,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub ood_development_sha256: String,
    pub ood_test_sha256: String,
    pub partition_counts: BTreeMap<String, usize>,
    pub threshold_selection: ThresholdSelection,
    pub uncalibrated_calibration_partition: CalibrationMetrics,
    pub calibrated_calibration_partition: CalibrationMetrics,
    pub id_test: IdEvaluationV2,
    pub ood_test: OodEvaluationV2,
    pub bootstrap_95: BootstrapReport,
    pub limitations: Vec<String>,
}

fn evaluate_id(
    runtime: &CompiledModel,
    examples: &[GroupedExample],
) -> Result<IdEvaluationV2, MlError> {
    if examples.is_empty() {
        return Err(MlError::InvalidDataset(
            "ID evaluation requires at least one example".into(),
        ));
    }
    let predictions = examples
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            EvaluatedOpenSetPrediction {
                id: example.id.clone(),
                actual_label: example.label.clone(),
                predicted_label: prediction.label.clone(),
                correct: prediction.label == example.label,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
                probability_margin: prediction.probability_margin,
                probabilities: prediction.probabilities,
            }
        })
        .collect::<Vec<_>>();
    Ok(summarize_id_predictions(&runtime.model.labels, predictions))
}

fn summarize_id_predictions(
    labels: &[String],
    predictions: Vec<EvaluatedOpenSetPrediction>,
) -> IdEvaluationV2 {
    let example_count = predictions.len();
    let correct = predictions
        .iter()
        .filter(|prediction| prediction.correct)
        .count();
    let accepted = predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .collect::<Vec<_>>();
    let accepted_correct = accepted
        .iter()
        .filter(|prediction| prediction.correct)
        .count();
    let calibration = calibration_metrics(&predictions, 10);
    let (risk_coverage_curve, aurc) = risk_coverage(&predictions);
    IdEvaluationV2 {
        example_count,
        accuracy: correct as f64 / example_count as f64,
        macro_f1: macro_f1(labels, &predictions),
        coverage: accepted.len() as f64 / example_count as f64,
        selective_accuracy: (!accepted.is_empty())
            .then_some(accepted_correct as f64 / accepted.len() as f64),
        calibration,
        aurc,
        risk_coverage_curve,
        predictions,
    }
}

fn calibration_metrics(
    predictions: &[EvaluatedOpenSetPrediction],
    bins: usize,
) -> CalibrationMetrics {
    let count = predictions.len() as f64;
    let negative_log_likelihood = predictions
        .iter()
        .map(|prediction| {
            -prediction.probabilities[&prediction.actual_label]
                .max(1e-15)
                .ln()
        })
        .sum::<f64>()
        / count;
    let multiclass_brier = predictions
        .iter()
        .map(|prediction| {
            prediction
                .probabilities
                .iter()
                .map(|(label, probability)| {
                    let target = usize::from(label == &prediction.actual_label) as f64;
                    (probability - target).powi(2)
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / count;
    let mut expected_calibration_error = 0.0;
    for bin in 0..bins {
        let lower = bin as f64 / bins as f64;
        let upper = (bin + 1) as f64 / bins as f64;
        let members = predictions
            .iter()
            .filter(|prediction| {
                prediction.confidence >= lower
                    && (prediction.confidence < upper
                        || (bin + 1 == bins && prediction.confidence <= upper))
            })
            .collect::<Vec<_>>();
        if members.is_empty() {
            continue;
        }
        let accuracy = members
            .iter()
            .filter(|prediction| prediction.correct)
            .count() as f64
            / members.len() as f64;
        let confidence = members
            .iter()
            .map(|prediction| prediction.confidence)
            .sum::<f64>()
            / members.len() as f64;
        expected_calibration_error += members.len() as f64 / count * (accuracy - confidence).abs();
    }
    CalibrationMetrics {
        negative_log_likelihood,
        multiclass_brier,
        expected_calibration_error,
        ece_bins: bins,
    }
}

fn macro_f1(labels: &[String], predictions: &[EvaluatedOpenSetPrediction]) -> f64 {
    labels
        .iter()
        .map(|label| {
            let true_positive = predictions
                .iter()
                .filter(|prediction| {
                    prediction.actual_label == *label && prediction.predicted_label == *label
                })
                .count() as f64;
            let false_positive = predictions
                .iter()
                .filter(|prediction| {
                    prediction.actual_label != *label && prediction.predicted_label == *label
                })
                .count() as f64;
            let false_negative = predictions
                .iter()
                .filter(|prediction| {
                    prediction.actual_label == *label && prediction.predicted_label != *label
                })
                .count() as f64;
            let precision = safe_ratio(true_positive, true_positive + false_positive);
            let recall = safe_ratio(true_positive, true_positive + false_negative);
            safe_ratio(2.0 * precision * recall, precision + recall)
        })
        .sum::<f64>()
        / labels.len() as f64
}

fn risk_coverage(predictions: &[EvaluatedOpenSetPrediction]) -> (Vec<RiskCoveragePoint>, f64) {
    let mut ranked = predictions.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut correct = 0usize;
    let mut index = 0usize;
    let mut aurc = 0.0;
    let mut curve = Vec::with_capacity(ranked.len());
    while index < ranked.len() {
        let confidence = ranked[index].confidence;
        let start = index;
        while index < ranked.len() && ranked[index].confidence.total_cmp(&confidence).is_eq() {
            correct += usize::from(ranked[index].correct);
            index += 1;
        }
        let accepted = index;
        let risk = 1.0 - correct as f64 / accepted as f64;
        aurc += risk * (index - start) as f64 / ranked.len() as f64;
        curve.push(RiskCoveragePoint {
            accepted,
            coverage: accepted as f64 / ranked.len() as f64,
            risk,
        });
    }
    (curve, aurc)
}

fn evaluate_ood(
    runtime: &CompiledModel,
    id_evaluation: &IdEvaluationV2,
    ood: &OpenSetOodDataset,
) -> Result<OodEvaluationV2, MlError> {
    if ood.examples().is_empty() {
        return Err(MlError::InvalidDataset(
            "OOD evaluation requires at least one example".into(),
        ));
    }
    let predictions = ood
        .examples()
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            OodEvaluatedPrediction {
                id: example.id.clone(),
                predicted_label: prediction.label,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
                probability_margin: prediction.probability_margin,
            }
        })
        .collect::<Vec<_>>();
    let accepted_examples = predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .count();
    let id_scores = id_evaluation
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let ood_scores = predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    Ok(OodEvaluationV2 {
        example_count: predictions.len(),
        accepted_examples,
        coverage: accepted_examples as f64 / predictions.len() as f64,
        discrimination: discrimination_metrics(&id_scores, &ood_scores),
        predictions,
    })
}

fn discrimination_metrics(id_scores: &[f64], ood_scores: &[f64]) -> OodDiscriminationMetrics {
    let mut ranked = id_scores
        .iter()
        .copied()
        .map(|score| (score, true))
        .chain(ood_scores.iter().copied().map(|score| (score, false)))
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_score, _), (right_score, _)| right_score.total_cmp(left_score));
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut auroc_wins = 0.0;
    let mut aupr_in_domain = 0.0;
    let mut index = 0usize;
    while index < ranked.len() {
        let score = ranked[index].0;
        let mut group_positives = 0usize;
        let mut group_negatives = 0usize;
        while index < ranked.len() && ranked[index].0.total_cmp(&score).is_eq() {
            if ranked[index].1 {
                group_positives += 1;
            } else {
                group_negatives += 1;
            }
            index += 1;
        }
        let lower_scoring_negatives = ood_scores.len() - false_positives - group_negatives;
        auroc_wins += (group_positives * lower_scoring_negatives) as f64
            + 0.5 * (group_positives * group_negatives) as f64;
        true_positives += group_positives;
        false_positives += group_negatives;
        if group_positives > 0 {
            let precision = true_positives as f64 / (true_positives + false_positives) as f64;
            let recall_increment = group_positives as f64 / id_scores.len() as f64;
            aupr_in_domain += precision * recall_increment;
        }
    }
    let auroc = auroc_wins / (id_scores.len() * ood_scores.len()) as f64;

    let mut sorted_id = id_scores.to_vec();
    sorted_id.sort_by(|left, right| right.total_cmp(left));
    let target_rank = ((0.95 * sorted_id.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted_id.len() - 1);
    let threshold = sorted_id[target_rank];
    let fpr_at_95_tpr = ood_scores
        .iter()
        .filter(|score| **score >= threshold)
        .count() as f64
        / ood_scores.len() as f64;
    OodDiscriminationMetrics {
        auroc,
        aupr_in_domain,
        fpr_at_95_tpr,
    }
}

fn bootstrap_report(
    labels: &[String],
    id: &IdEvaluationV2,
    ood: &OodEvaluationV2,
    seed: u64,
    resamples: usize,
) -> Result<BootstrapReport, MlError> {
    if !(100..=20_000).contains(&resamples) {
        return Err(MlError::InvalidConfiguration(
            "bootstrap resamples must be between 100 and 20000".into(),
        ));
    }
    let sampled_rows = id
        .predictions
        .len()
        .checked_add(ood.predictions.len())
        .and_then(|population| population.checked_mul(resamples))
        .ok_or_else(|| {
            MlError::InvalidConfiguration("bootstrap workload overflows its size boundary".into())
        })?;
    if sampled_rows > MAX_BOOTSTRAP_SAMPLED_ROWS {
        return Err(MlError::InvalidConfiguration(format!(
            "bootstrap workload exceeds {MAX_BOOTSTRAP_SAMPLED_ROWS} sampled rows"
        )));
    }
    let mut by_label: BTreeMap<&str, Vec<&EvaluatedOpenSetPrediction>> = BTreeMap::new();
    for prediction in &id.predictions {
        by_label
            .entry(&prediction.actual_label)
            .or_default()
            .push(prediction);
    }
    let mut rng = DeterministicRng::new(seed ^ 0x6a09_e667_f3bc_c909);
    let mut id_accuracy = Vec::with_capacity(resamples);
    let mut id_macro_f1 = Vec::with_capacity(resamples);
    let mut id_nll = Vec::with_capacity(resamples);
    let mut id_brier = Vec::with_capacity(resamples);
    let mut id_ece = Vec::with_capacity(resamples);
    let mut id_aurc = Vec::with_capacity(resamples);
    let mut ood_auroc = Vec::with_capacity(resamples);
    let mut ood_aupr = Vec::with_capacity(resamples);
    let mut ood_fpr95 = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sampled_id = Vec::with_capacity(id.predictions.len());
        for members in by_label.values() {
            for _ in 0..members.len() {
                sampled_id.push((*members[rng.index(members.len())]).clone());
            }
        }
        let summary = summarize_id_predictions(labels, sampled_id);
        let sampled_ood = (0..ood.predictions.len())
            .map(|_| ood.predictions[rng.index(ood.predictions.len())].clone())
            .collect::<Vec<_>>();
        let id_scores = summary
            .predictions
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        let ood_scores = sampled_ood
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        let discrimination = discrimination_metrics(&id_scores, &ood_scores);
        id_accuracy.push(summary.accuracy);
        id_macro_f1.push(summary.macro_f1);
        id_nll.push(summary.calibration.negative_log_likelihood);
        id_brier.push(summary.calibration.multiclass_brier);
        id_ece.push(summary.calibration.expected_calibration_error);
        id_aurc.push(summary.aurc);
        ood_auroc.push(discrimination.auroc);
        ood_aupr.push(discrimination.aupr_in_domain);
        ood_fpr95.push(discrimination.fpr_at_95_tpr);
    }
    Ok(BootstrapReport {
        strategy: "label-stratified-id-row-and-population-stratified-ood-percentile-v2".into(),
        seed,
        resamples,
        confidence_level: 0.95,
        id_accuracy: estimate(id.accuracy, id_accuracy),
        id_macro_f1: estimate(id.macro_f1, id_macro_f1),
        id_negative_log_likelihood: estimate(id.calibration.negative_log_likelihood, id_nll),
        id_multiclass_brier: estimate(id.calibration.multiclass_brier, id_brier),
        id_expected_calibration_error: estimate(id.calibration.expected_calibration_error, id_ece),
        id_aurc: estimate(id.aurc, id_aurc),
        ood_auroc: estimate(ood.discrimination.auroc, ood_auroc),
        ood_aupr_in_domain: estimate(ood.discrimination.aupr_in_domain, ood_aupr),
        ood_fpr_at_95_tpr: estimate(ood.discrimination.fpr_at_95_tpr, ood_fpr95),
    })
}

fn estimate(value: f64, mut samples: Vec<f64>) -> MetricEstimate {
    samples.sort_by(f64::total_cmp);
    let lower = ((samples.len() - 1) as f64 * 0.025).floor() as usize;
    let upper = ((samples.len() - 1) as f64 * 0.975).ceil() as usize;
    MetricEstimate {
        value,
        lower_95: samples[lower],
        upper_95: samples[upper.min(samples.len() - 1)],
    }
}

fn calibration_partition_metrics(
    model: &OpenSetModelV2,
    examples: &[GroupedExample],
    temperature: f64,
) -> Result<CalibrationMetrics, MlError> {
    let policy = OpenSetPolicyV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: 0.0,
        minimum_probability_margin: 0.0,
        temperature_source: "calibration-partition-temperature-scaling-v2".into(),
        threshold_source: "development-plus-ood-development-grid-v2".into(),
        calibration_example_count: examples.len(),
        development_example_count: 1,
        ood_development_example_count: 1,
    };
    let runtime = CompiledModel::new(model.clone(), policy)?;
    Ok(evaluate_id(&runtime, examples)?.calibration)
}

#[derive(Debug, Clone)]
pub struct OpenSetExperimentResult {
    pub model: OpenSetModelV2,
    pub policy: OpenSetPolicyV2,
    pub metrics: OpenSetMetricsV2,
    pub split_plan: SplitPlanManifest,
}

pub fn run_open_set_experiment(
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
    config: OpenSetTrainingConfig,
    bootstrap_resamples: usize,
) -> Result<OpenSetExperimentResult, MlError> {
    reject_cross_dataset_overlap(dataset, ood_development, ood_test)?;
    let plan = SplitPlan::build(dataset, config.seed)?;
    let model = fit_model(&plan, config.clone())?;

    // The temperature has access only to the calibration partition.
    let temperature = calibrate_temperature(&model, plan.calibration())?;
    let uncalibrated_calibration_partition =
        calibration_partition_metrics(&model, plan.calibration(), 1.0)?;
    let calibrated_calibration_partition =
        calibration_partition_metrics(&model, plan.calibration(), temperature)?;

    // Thresholds have access only to ID development and OOD development. The test partitions are
    // intentionally not parameters of `select_thresholds`.
    let threshold_selection =
        select_thresholds(&model, temperature, plan.development(), ood_development)?;
    let policy = OpenSetPolicyV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: threshold_selection.selected_confidence,
        minimum_probability_margin: threshold_selection.selected_probability_margin,
        temperature_source: "calibration-partition-temperature-scaling-v2".into(),
        threshold_source: "development-plus-ood-development-grid-v2".into(),
        calibration_example_count: plan.calibration().len(),
        development_example_count: plan.development().len(),
        ood_development_example_count: ood_development.examples().len(),
    };
    let runtime = CompiledModel::new(model.clone(), policy.clone())?;

    // Only after the fitted model and operating policy are frozen do we evaluate the two tests.
    let id_test = evaluate_id(&runtime, plan.id_test())?;
    let ood_test_evaluation = evaluate_ood(&runtime, &id_test, ood_test)?;
    let bootstrap_95 = bootstrap_report(
        &model.labels,
        &id_test,
        &ood_test_evaluation,
        config.seed,
        bootstrap_resamples,
    )?;
    let partition_counts = BTreeMap::from([
        ("train".into(), plan.train().len()),
        ("development".into(), plan.development().len()),
        ("calibration".into(), plan.calibration().len()),
        ("id-test".into(), plan.id_test().len()),
        ("ood-development".into(), ood_development.examples().len()),
        ("ood-test".into(), ood_test.examples().len()),
    ]);
    let metrics = OpenSetMetricsV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: OPEN_SET_MODEL_VERSION.into(),
        dataset_sha256: dataset.fingerprint_sha256(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        ood_development_sha256: ood_development.fingerprint_sha256(),
        ood_test_sha256: ood_test.fingerprint_sha256(),
        partition_counts,
        threshold_selection,
        uncalibrated_calibration_partition,
        calibrated_calibration_partition,
        id_test,
        ood_test: ood_test_evaluation,
        bootstrap_95,
        limitations: vec![
            "All current examples are synthetic and English-only.".into(),
            "The four-way ID split is group-disjoint but still small; confidence intervals are therefore wide.".into(),
            "OOD discrimination is measured only on the checked-in synthetic OOD-test fixture.".into(),
            "ID confidence intervals resample rows within labels, not groups; with one ID-test group per label, they do not estimate between-group variation.".into(),
            "This classifier is not suitable for clinical, safety, employment, or other decisions about people.".into(),
        ],
    };
    Ok(OpenSetExperimentResult {
        model,
        policy,
        metrics,
        split_plan: plan.manifest,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundleManifestV2 {
    pub schema_version: u32,
    pub bundle_kind: String,
    pub bundle_version: String,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VerifiedBundle {
    pub manifest: BundleManifestV2,
    pub model: OpenSetModelV2,
    pub policy: OpenSetPolicyV2,
    pub metrics: OpenSetMetricsV2,
    pub split_plan: SplitPlanManifest,
}

impl VerifiedBundle {
    pub fn compile(self) -> Result<CompiledModel, MlError> {
        CompiledModel::new(self.model, self.policy)
    }
}

pub fn write_bundle(
    directory: impl AsRef<Path>,
    result: &OpenSetExperimentResult,
) -> Result<BundleManifestV2, MlError> {
    let directory = directory.as_ref();
    validate_bundle_destination(directory)?;
    let payloads = BTreeMap::from([
        ("metrics.json".to_string(), canonical_json(&result.metrics)?),
        ("model.json".to_string(), canonical_json(&result.model)?),
        ("policy.json".to_string(), canonical_json(&result.policy)?),
        (
            "split-plan.json".to_string(),
            canonical_json(&result.split_plan)?,
        ),
    ]);
    let parent = directory.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = directory
        .file_name()
        .ok_or_else(|| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "bundle destination has no directory name",
            ))
        })?
        .to_string_lossy();
    let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.staging-{}-{sequence}", std::process::id()));
    let backup = parent.join(format!(".{name}.backup-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let files = payloads
        .iter()
        .map(|(name, bytes)| (name.clone(), sha256_hex(bytes)))
        .collect::<BTreeMap<_, _>>();
    let manifest = BundleManifestV2 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        bundle_kind: "eliza-open-set-bundle".into(),
        bundle_version: "2.0.0".into(),
        model_version: result.model.model_version.clone(),
        dataset_sha256: result.model.dataset_sha256.clone(),
        split_plan_sha256: result.model.split_plan_sha256.clone(),
        files,
    };

    for (name, bytes) in payloads {
        if let Err(error) = write_atomic(&staging.join(name), &bytes) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    }
    // The manifest is installed last. A partially written directory therefore never verifies.
    if let Err(error) = write_atomic(&staging.join("manifest.json"), &canonical_json(&manifest)?) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if let Err(error) = verify_bundle(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }

    let had_original = directory.exists();
    if had_original {
        if let Err(error) = fs::rename(directory, &backup) {
            let _ = fs::remove_dir_all(&staging);
            return Err(MlError::Io(error));
        }
    }
    if let Err(error) = fs::rename(&staging, directory) {
        let restore_error = if had_original {
            fs::rename(&backup, directory).err()
        } else {
            None
        };
        let _ = fs::remove_dir_all(&staging);
        let message = restore_error.map_or_else(
            || format!("bundle installation failed and the prior bundle was restored: {error}"),
            |restore| {
                format!(
                    "bundle installation failed: {error}; restoring the prior bundle also failed: {restore}"
                )
            },
        );
        return Err(MlError::Io(std::io::Error::other(message)));
    }
    if let Err(error) = verify_bundle(directory) {
        let displacement_error = fs::rename(directory, &staging).err();
        let restore_error = if had_original && displacement_error.is_none() {
            fs::rename(&backup, directory).err()
        } else {
            None
        };
        if displacement_error.is_none() && restore_error.is_none() {
            let _ = fs::remove_dir_all(&staging);
        }
        let message = match (displacement_error, restore_error) {
            (None, None) if had_original => {
                format!("installed bundle failed verification and the prior bundle was restored: {error}")
            }
            (None, None) => format!("installed bundle failed verification and was removed: {error}"),
            (Some(displacement), _) => format!(
                "installed bundle failed verification: {error}; moving it aside also failed: {displacement}"
            ),
            (None, Some(restore)) => format!(
                "installed bundle failed verification: {error}; restoring the prior bundle also failed: {restore}"
            ),
        };
        return Err(MlError::Io(std::io::Error::other(message)));
    }
    if had_original {
        fs::remove_dir_all(&backup)?;
    }
    Ok(manifest)
}

pub fn verify_bundle(directory: impl AsRef<Path>) -> Result<VerifiedBundle, MlError> {
    let directory = directory.as_ref();
    validate_bundle_inventory(directory)?;
    let manifest_path = directory.join("manifest.json");
    let manifest: BundleManifestV2 = read_bounded_json(&manifest_path, "bundle manifest")?;
    validate_manifest(&manifest)?;
    for (name, expected_sha256) in &manifest.files {
        if !valid_sha256(expected_sha256) {
            return Err(MlError::InvalidModel(format!(
                "bundle file `{name}` has a malformed SHA-256"
            )));
        }
        let path = directory.join(name);
        reject_symlink_or_non_file(&path, name)?;
        reject_oversized_file(&path, MAX_JSON_BYTES, name)?;
        let observed = sha256_hex(&fs::read(&path)?);
        if &observed != expected_sha256 {
            return Err(MlError::InvalidModel(format!(
                "bundle file `{name}` failed SHA-256 verification"
            )));
        }
    }
    let model: OpenSetModelV2 = read_bounded_json(&directory.join("model.json"), "model")?;
    let policy: OpenSetPolicyV2 = read_bounded_json(&directory.join("policy.json"), "policy")?;
    let metrics: OpenSetMetricsV2 = read_bounded_json(&directory.join("metrics.json"), "metrics")?;
    let split_plan: SplitPlanManifest =
        read_bounded_json(&directory.join("split-plan.json"), "split plan")?;
    validate_bundle_artifacts(&manifest, &model, &policy, &metrics, &split_plan)?;
    Ok(VerifiedBundle {
        manifest,
        model,
        policy,
        metrics,
        split_plan,
    })
}

fn validate_bundle_destination(directory: &Path) -> Result<(), MlError> {
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(MlError::Io(error)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "bundle destination {} must be a real directory",
                directory.display()
            ),
        )));
    }
    if fs::read_dir(directory)?.next().transpose()?.is_some() {
        verify_bundle(directory).map_err(|error| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to replace non-empty directory {} because it is not a verified v2 bundle: {error}",
                    directory.display()
                ),
            ))
        })?;
    }
    Ok(())
}

fn validate_bundle_inventory(directory: &Path) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(MlError::InvalidModel(
            "the bundle root must be a real directory".into(),
        ));
    }
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            MlError::InvalidModel("the bundle contains a non-UTF-8 file name".into())
        })?;
        observed.insert(name);
    }
    let expected = BUNDLE_INVENTORY
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(MlError::InvalidModel(
            "the bundle inventory must contain exactly the five v2 files".into(),
        ));
    }
    Ok(())
}

pub fn embedded_bundle() -> Result<VerifiedBundle, MlError> {
    let bytes = BTreeMap::from([
        (
            "metrics.json",
            include_bytes!("../artifacts/eliza-open-set-v2/metrics.json").as_slice(),
        ),
        (
            "model.json",
            include_bytes!("../artifacts/eliza-open-set-v2/model.json").as_slice(),
        ),
        (
            "policy.json",
            include_bytes!("../artifacts/eliza-open-set-v2/policy.json").as_slice(),
        ),
        (
            "split-plan.json",
            include_bytes!("../artifacts/eliza-open-set-v2/split-plan.json").as_slice(),
        ),
    ]);
    let manifest: BundleManifestV2 = serde_json::from_slice(include_bytes!(
        "../artifacts/eliza-open-set-v2/manifest.json"
    ))?;
    validate_manifest(&manifest)?;
    for (name, content) in &bytes {
        if sha256_hex(content) != manifest.files[*name] {
            return Err(MlError::InvalidModel(format!(
                "embedded bundle file `{name}` failed SHA-256 verification"
            )));
        }
    }
    let model: OpenSetModelV2 = serde_json::from_slice(bytes["model.json"])?;
    let policy: OpenSetPolicyV2 = serde_json::from_slice(bytes["policy.json"])?;
    let metrics: OpenSetMetricsV2 = serde_json::from_slice(bytes["metrics.json"])?;
    let split_plan: SplitPlanManifest = serde_json::from_slice(bytes["split-plan.json"])?;
    validate_bundle_artifacts(&manifest, &model, &policy, &metrics, &split_plan)?;
    Ok(VerifiedBundle {
        manifest,
        model,
        policy,
        metrics,
        split_plan,
    })
}

fn validate_manifest(manifest: &BundleManifestV2) -> Result<(), MlError> {
    if manifest.schema_version != OPEN_SET_SCHEMA_VERSION
        || manifest.bundle_kind != "eliza-open-set-bundle"
        || manifest.bundle_version != "2.0.0"
        || manifest.model_version != OPEN_SET_MODEL_VERSION
        || !valid_sha256(&manifest.dataset_sha256)
        || !valid_sha256(&manifest.split_plan_sha256)
        || manifest.files.keys().cloned().collect::<BTreeSet<_>>()
            != BTreeSet::from([
                "metrics.json".to_string(),
                "model.json".to_string(),
                "policy.json".to_string(),
                "split-plan.json".to_string(),
            ])
    {
        return Err(MlError::InvalidModel(
            "the bundle manifest violates the v2 contract".into(),
        ));
    }
    Ok(())
}

fn validate_bundle_artifacts(
    manifest: &BundleManifestV2,
    model: &OpenSetModelV2,
    policy: &OpenSetPolicyV2,
    metrics: &OpenSetMetricsV2,
    split_plan: &SplitPlanManifest,
) -> Result<(), MlError> {
    model.validate()?;
    policy.validate_against(model)?;
    split_plan.validate_contract()?;
    if manifest.dataset_sha256 != model.dataset_sha256
        || manifest.split_plan_sha256 != model.split_plan_sha256
        || metrics.schema_version != OPEN_SET_SCHEMA_VERSION
        || metrics.model_version != model.model_version
        || metrics.dataset_sha256 != model.dataset_sha256
        || metrics.split_plan_sha256 != model.split_plan_sha256
        || split_plan.schema_version != OPEN_SET_SCHEMA_VERSION
        || split_plan.dataset_sha256 != model.dataset_sha256
        || split_plan.seed != model.training_config.seed
        || sha256_hex(&canonical_json(split_plan)?) != model.split_plan_sha256
    {
        return Err(MlError::InvalidModel(
            "bundle artifacts disagree about their model or data provenance".into(),
        ));
    }
    validate_metrics_contract(metrics, model, policy, split_plan)?;
    Ok(())
}

fn validate_metrics_contract(
    metrics: &OpenSetMetricsV2,
    model: &OpenSetModelV2,
    policy: &OpenSetPolicyV2,
    split_plan: &SplitPlanManifest,
) -> Result<(), MlError> {
    let mut split_counts = BTreeMap::from([
        (PartitionKind::Train, 0usize),
        (PartitionKind::Development, 0usize),
        (PartitionKind::Calibration, 0usize),
        (PartitionKind::IdTest, 0usize),
    ]);
    for assignment in &split_plan.assignments {
        *split_counts
            .get_mut(&assignment.partition)
            .expect("all typed partitions are initialized") += 1;
    }
    let expected_counts = BTreeMap::from([
        (
            "calibration".into(),
            split_counts[&PartitionKind::Calibration],
        ),
        (
            "development".into(),
            split_counts[&PartitionKind::Development],
        ),
        ("id-test".into(), split_counts[&PartitionKind::IdTest]),
        (
            "ood-development".into(),
            policy.ood_development_example_count,
        ),
        ("ood-test".into(), metrics.ood_test.example_count),
        ("train".into(), split_counts[&PartitionKind::Train]),
    ]);
    let thresholds = &metrics.threshold_selection;
    if metrics.partition_counts != expected_counts
        || policy.calibration_example_count != split_counts[&PartitionKind::Calibration]
        || policy.development_example_count != split_counts[&PartitionKind::Development]
        || thresholds.strategy != "development-plus-ood-development-grid-v2"
        || thresholds.development_example_count != policy.development_example_count
        || thresholds.ood_development_example_count != policy.ood_development_example_count
        || thresholds.selected_confidence != policy.minimum_confidence
        || thresholds.selected_probability_margin != policy.minimum_probability_margin
        || thresholds.id_test_used
        || thresholds.ood_test_used
        || thresholds.minimum_development_selective_accuracy != 0.75
        || thresholds.maximum_ood_development_coverage != 0.10
        || !valid_unit_interval(thresholds.observed_development_coverage)
        || !valid_unit_interval(thresholds.observed_development_selective_accuracy)
        || !valid_unit_interval(thresholds.observed_ood_development_coverage)
        || !valid_sha256(&metrics.ood_development_sha256)
        || !valid_sha256(&metrics.ood_test_sha256)
        || metrics.ood_development_sha256 == metrics.ood_test_sha256
        || metrics.ood_development_sha256 == metrics.dataset_sha256
        || metrics.ood_test_sha256 == metrics.dataset_sha256
    {
        return Err(MlError::InvalidModel(
            "the v2 metrics disagree with the split or frozen policy".into(),
        ));
    }
    validate_calibration_metrics(&metrics.uncalibrated_calibration_partition)?;
    validate_calibration_metrics(&metrics.calibrated_calibration_partition)?;
    if metrics
        .calibrated_calibration_partition
        .negative_log_likelihood
        > metrics
            .uncalibrated_calibration_partition
            .negative_log_likelihood
            + 1e-12
    {
        return Err(MlError::InvalidModel(
            "the recorded temperature increases calibration NLL".into(),
        ));
    }

    let label_set = model
        .labels
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let split_label_set = split_plan
        .assignments
        .iter()
        .map(|assignment| assignment.label.as_str())
        .collect::<BTreeSet<_>>();
    let id_assignments = split_plan
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == PartitionKind::IdTest)
        .map(|assignment| (assignment.id.as_str(), assignment.label.as_str()))
        .collect::<BTreeMap<_, _>>();
    if split_label_set != label_set || id_assignments.len() != metrics.id_test.example_count {
        return Err(MlError::InvalidModel(
            "the model labels or ID-test ledger disagree with the split plan".into(),
        ));
    }
    let mut id_ids = HashSet::new();
    for prediction in &metrics.id_test.predictions {
        if id_assignments.get(prediction.id.as_str()) != Some(&prediction.actual_label.as_str()) {
            return Err(MlError::InvalidModel(
                "the ID-test prediction ledger disagrees with the split plan".into(),
            ));
        }
        let probability_labels = prediction
            .probabilities
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let probability_sum = prediction.probabilities.values().sum::<f64>();
        let mut ranking = model
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| (index, label, prediction.probabilities[label]))
            .collect::<Vec<_>>();
        ranking.sort_by(|(left_index, _, left), (right_index, _, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });
        let expected_margin = ranking[0].2 - ranking[1].2;
        if !id_ids.insert(prediction.id.as_str())
            || !label_set.contains(prediction.actual_label.as_str())
            || !label_set.contains(prediction.predicted_label.as_str())
            || prediction.predicted_label.as_str() != ranking[0].1.as_str()
            || prediction.correct != (prediction.actual_label == prediction.predicted_label)
            || probability_labels != label_set
            || prediction
                .probabilities
                .values()
                .any(|value| !valid_unit_interval(*value))
            || (probability_sum - 1.0).abs() > 1e-9
            || !approximately_equal(
                prediction.confidence,
                prediction.probabilities[&prediction.predicted_label],
            )
            || !approximately_equal(prediction.probability_margin, expected_margin)
            || (prediction.accepted
                && (prediction.confidence < policy.minimum_confidence
                    || prediction.probability_margin < policy.minimum_probability_margin))
        {
            return Err(MlError::InvalidModel(
                "the ID-test prediction ledger is internally inconsistent".into(),
            ));
        }
    }
    let reproduced_id =
        summarize_id_predictions(&model.labels, metrics.id_test.predictions.clone());
    let risk_curve_matches = metrics.id_test.risk_coverage_curve.len()
        == reproduced_id.risk_coverage_curve.len()
        && metrics
            .id_test
            .risk_coverage_curve
            .iter()
            .zip(&reproduced_id.risk_coverage_curve)
            .all(|(recorded, reproduced)| {
                recorded.accepted == reproduced.accepted
                    && approximately_equal(recorded.coverage, reproduced.coverage)
                    && approximately_equal(recorded.risk, reproduced.risk)
            });
    if metrics.id_test.example_count != reproduced_id.example_count
        || !approximately_equal(metrics.id_test.accuracy, reproduced_id.accuracy)
        || !approximately_equal(metrics.id_test.macro_f1, reproduced_id.macro_f1)
        || !approximately_equal(metrics.id_test.coverage, reproduced_id.coverage)
        || !optional_metric_matches(
            metrics.id_test.selective_accuracy,
            reproduced_id.selective_accuracy,
        )
        || !calibration_metrics_match(&metrics.id_test.calibration, &reproduced_id.calibration)
        || !approximately_equal(metrics.id_test.aurc, reproduced_id.aurc)
        || !risk_curve_matches
        || metrics.id_test.predictions != reproduced_id.predictions
        || metrics.id_test.example_count != split_counts[&PartitionKind::IdTest]
    {
        return Err(MlError::InvalidModel(
            "the ID-test summary does not match its prediction ledger".into(),
        ));
    }

    let mut ood_ids = HashSet::new();
    for prediction in &metrics.ood_test.predictions {
        if id_ids.contains(prediction.id.as_str())
            || !ood_ids.insert(prediction.id.as_str())
            || !label_set.contains(prediction.predicted_label.as_str())
            || !valid_unit_interval(prediction.confidence)
            || !valid_unit_interval(prediction.probability_margin)
            || (prediction.accepted
                && (prediction.confidence < policy.minimum_confidence
                    || prediction.probability_margin < policy.minimum_probability_margin))
        {
            return Err(MlError::InvalidModel(
                "the OOD-test prediction ledger is internally inconsistent".into(),
            ));
        }
    }
    let accepted_ood = metrics
        .ood_test
        .predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .count();
    let id_scores = metrics
        .id_test
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let ood_scores = metrics
        .ood_test
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    if metrics.ood_test.example_count == 0
        || metrics.ood_test.example_count != metrics.ood_test.predictions.len()
        || metrics.ood_test.accepted_examples != accepted_ood
        || metrics.ood_test.coverage != accepted_ood as f64 / metrics.ood_test.example_count as f64
        || !discrimination_metrics_match(
            &metrics.ood_test.discrimination,
            &discrimination_metrics(&id_scores, &ood_scores),
        )
    {
        return Err(MlError::InvalidModel(
            "the OOD-test summary does not match its prediction ledger".into(),
        ));
    }

    let bootstrap = &metrics.bootstrap_95;
    if bootstrap.strategy != "label-stratified-id-row-and-population-stratified-ood-percentile-v2"
        || bootstrap.seed != model.training_config.seed
        || !(100..=20_000).contains(&bootstrap.resamples)
        || bootstrap.confidence_level != 0.95
        || metrics.limitations.is_empty()
    {
        return Err(MlError::InvalidModel(
            "the bootstrap report has invalid provenance".into(),
        ));
    }
    for (estimate, value, minimum, maximum) in [
        (&bootstrap.id_accuracy, metrics.id_test.accuracy, 0.0, 1.0),
        (&bootstrap.id_macro_f1, metrics.id_test.macro_f1, 0.0, 1.0),
        (
            &bootstrap.id_negative_log_likelihood,
            metrics.id_test.calibration.negative_log_likelihood,
            0.0,
            f64::INFINITY,
        ),
        (
            &bootstrap.id_multiclass_brier,
            metrics.id_test.calibration.multiclass_brier,
            0.0,
            2.0,
        ),
        (
            &bootstrap.id_expected_calibration_error,
            metrics.id_test.calibration.expected_calibration_error,
            0.0,
            1.0,
        ),
        (&bootstrap.id_aurc, metrics.id_test.aurc, 0.0, 1.0),
        (
            &bootstrap.ood_auroc,
            metrics.ood_test.discrimination.auroc,
            0.0,
            1.0,
        ),
        (
            &bootstrap.ood_aupr_in_domain,
            metrics.ood_test.discrimination.aupr_in_domain,
            0.0,
            1.0,
        ),
        (
            &bootstrap.ood_fpr_at_95_tpr,
            metrics.ood_test.discrimination.fpr_at_95_tpr,
            0.0,
            1.0,
        ),
    ] {
        if !approximately_equal(estimate.value, value)
            || !estimate.value.is_finite()
            || !estimate.lower_95.is_finite()
            || !estimate.upper_95.is_finite()
            || estimate.lower_95 < minimum
            || estimate.upper_95 > maximum
            || estimate.lower_95 > estimate.upper_95
        {
            return Err(MlError::InvalidModel(
                "a bootstrap estimate violates its metric contract".into(),
            ));
        }
    }
    Ok(())
}

fn validate_calibration_metrics(metrics: &CalibrationMetrics) -> Result<(), MlError> {
    if !metrics.negative_log_likelihood.is_finite()
        || metrics.negative_log_likelihood < 0.0
        || !metrics.multiclass_brier.is_finite()
        || !(0.0..=2.0).contains(&metrics.multiclass_brier)
        || !valid_unit_interval(metrics.expected_calibration_error)
        || metrics.ece_bins == 0
    {
        return Err(MlError::InvalidModel(
            "calibration metrics are outside their supported range".into(),
        ));
    }
    Ok(())
}

fn valid_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

fn optional_metric_matches(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => approximately_equal(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn calibration_metrics_match(left: &CalibrationMetrics, right: &CalibrationMetrics) -> bool {
    approximately_equal(left.negative_log_likelihood, right.negative_log_likelihood)
        && approximately_equal(left.multiclass_brier, right.multiclass_brier)
        && approximately_equal(
            left.expected_calibration_error,
            right.expected_calibration_error,
        )
        && left.ece_bins == right.ece_bins
}

fn discrimination_metrics_match(
    left: &OodDiscriminationMetrics,
    right: &OodDiscriminationMetrics,
) -> bool {
    approximately_equal(left.auroc, right.auroc)
        && approximately_equal(left.aupr_in_domain, right.aupr_in_domain)
        && approximately_equal(left.fpr_at_95_tpr, right.fpr_at_95_tpr)
}

pub fn reproduce_bundle(
    directory: impl AsRef<Path>,
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
) -> Result<(), MlError> {
    let verified = verify_bundle(directory)?;
    let resamples = verified.metrics.bootstrap_95.resamples;
    let reproduced = run_open_set_experiment(
        dataset,
        ood_development,
        ood_test,
        verified.model.training_config.clone(),
        resamples,
    )?;
    let expected_payloads = BTreeMap::from([
        ("metrics.json", canonical_json(&reproduced.metrics)?),
        ("model.json", canonical_json(&reproduced.model)?),
        ("policy.json", canonical_json(&reproduced.policy)?),
        ("split-plan.json", canonical_json(&reproduced.split_plan)?),
    ]);
    for (name, bytes) in expected_payloads {
        let expected =
            verified.manifest.files.get(name).ok_or_else(|| {
                MlError::InvalidModel(format!("bundle manifest is missing `{name}`"))
            })?;
        if sha256_hex(&bytes) != *expected {
            return Err(MlError::InvalidModel(format!(
                "reproduced `{name}` does not match the verified bundle"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchInput {
    id: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct BatchOutput<'a> {
    id: &'a str,
    prediction: OpenSetPrediction,
}

pub fn predict_jsonl(
    runtime: &CompiledModel,
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<usize, MlError> {
    let mut count = 0usize;
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader
            .take((MAX_JSONL_BYTES + 1) as u64)
            .read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        if bytes > MAX_JSONL_BYTES {
            if !line.ends_with('\n') {
                drain_to_newline(reader)?;
            }
            return Err(MlError::InvalidDataset(format!(
                "JSONL line {} exceeds the input boundary",
                count + 1
            )));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        enforce_batch_capacity(count)?;
        let input: BatchInput = serde_json::from_str(trimmed)?;
        validate_identifier(&input.id, "batch id", count + 1)?;
        validate_text(&input.text, count + 1, "batch")?;
        serde_json::to_writer(
            &mut *writer,
            &BatchOutput {
                id: &input.id,
                prediction: runtime.predict(&input.text),
            },
        )?;
        writer.write_all(b"\n")?;
        count += 1;
    }
    Ok(count)
}

fn enforce_batch_capacity(processed_rows: usize) -> Result<(), MlError> {
    if processed_rows >= MAX_EXAMPLES {
        return Err(MlError::InvalidDataset(format!(
            "JSONL input exceeds {MAX_EXAMPLES} rows"
        )));
    }
    Ok(())
}

fn drain_to_newline(reader: &mut impl BufRead) -> Result<(), MlError> {
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(());
        }
        if let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            reader.consume(position + 1);
            return Ok(());
        }
        let consumed = buffer.len();
        reader.consume(consumed);
    }
}

fn reject_cross_dataset_overlap(
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
) -> Result<(), MlError> {
    let supervised_ids = dataset
        .examples()
        .iter()
        .map(|example| example.id.as_str())
        .collect::<HashSet<_>>();
    let supervised_groups = dataset
        .examples()
        .iter()
        .map(|example| example.group_id.as_str())
        .collect::<HashSet<_>>();
    let supervised_texts = dataset
        .examples()
        .iter()
        .map(|example| normalize_text(&example.text))
        .collect::<HashSet<_>>();
    let mut ood_ids = HashSet::new();
    let mut ood_groups = HashSet::new();
    let mut ood_texts = HashSet::new();
    for (name, ood) in [("OOD development", ood_development), ("OOD test", ood_test)] {
        for example in ood.examples() {
            let normalized = normalize_text(&example.text);
            if supervised_ids.contains(example.id.as_str())
                || supervised_groups.contains(example.group_id.as_str())
                || supervised_texts.contains(&normalized)
                || !ood_ids.insert(example.id.as_str())
                || !ood_groups.insert(example.group_id.as_str())
                || !ood_texts.insert(normalized)
            {
                return Err(MlError::InvalidDataset(format!(
                    "{name} example `{}` overlaps another experimental population",
                    example.id
                )));
            }
        }
    }
    Ok(())
}

fn validate_identifier(value: &str, name: &str, line: usize) -> Result<(), MlError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(MlError::InvalidDataset(format!(
            "line {line} has an invalid {name}"
        )));
    }
    Ok(())
}

fn validate_label(value: &str, line: usize) -> Result<(), MlError> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .chars()
            .all(|character| character.is_ascii_lowercase() || character == '-')
    {
        return Err(MlError::InvalidDataset(format!(
            "line {line} has an invalid label"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, line: usize, context: &str) -> Result<(), MlError> {
    if value.trim().is_empty() || value.chars().count() > crate::MAX_INPUT_CHARS {
        return Err(MlError::InvalidDataset(format!(
            "{context} line {line} has empty or oversized text"
        )));
    }
    Ok(())
}

fn validate_vectorizer_config(config: &VectorizerConfig) -> Result<(), MlError> {
    if config.word_ngram_min == 0
        || config.word_ngram_min > config.word_ngram_max
        || config.word_ngram_max > 3
        || config.char_ngram_min < 2
        || config.char_ngram_min > config.char_ngram_max
        || config.char_ngram_max > 6
        || config.min_document_frequency == 0
        || config.min_document_frequency > 1_000_000
        || !(32..=100_000).contains(&config.max_features)
    {
        return Err(MlError::InvalidConfiguration(
            "the open-set vectorizer configuration is outside its supported bounds".into(),
        ));
    }
    Ok(())
}

fn normalize_text(value: &str) -> String {
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

fn tokenize(value: &str) -> Vec<String> {
    normalize_text(value)
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .map(|word| word.trim_matches('\''))
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

fn extract_terms(text: &str, config: &VectorizerConfig) -> Vec<String> {
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

fn build_feature_index(vocabulary: &[String]) -> HashMap<String, usize> {
    vocabulary
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, feature)| (feature, index))
        .collect()
}

fn transform(
    vectorizer: &OpenSetVectorizer,
    index: &HashMap<String, usize>,
    text: &str,
) -> Vec<(usize, f64)> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for term in extract_terms(text, &vectorizer.config) {
        if let Some(position) = index.get(&term) {
            *counts.entry(*position).or_insert(0) += 1;
        }
    }
    let mut values = counts
        .into_iter()
        .map(|(position, count)| {
            (
                position,
                (1.0 + (count as f64).ln()) * vectorizer.inverse_document_frequency[position],
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

fn logits_for(features: &[(usize, f64)], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(biases)
        .map(|(row, bias)| {
            *bias
                + features
                    .iter()
                    .map(|(feature, value)| row[*feature] * value)
                    .sum::<f64>()
        })
        .collect()
}

fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled = logits
        .iter()
        .map(|logit| *logit / temperature)
        .collect::<Vec<_>>();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exponentials = scaled
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    for probability in &mut exponentials {
        *probability /= total;
    }
    exponentials
}

fn quantize(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, MlError> {
    Ok(format!("{}\n", serde_json::to_string_pretty(value)?).into_bytes())
}

fn group_split_hash(label: &str, group_id: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in label
        .as_bytes()
        .iter()
        .chain([0x1f].iter())
        .chain(group_id.as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next_u64() % length as u64) as usize
    }
}

fn reject_oversized_file(path: &Path, maximum: u64, context: &str) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{context} {} must be a regular file", path.display()),
        )));
    }
    if metadata.len() > maximum {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{context} exceeds the {maximum}-byte boundary"),
        )));
    }
    Ok(())
}

fn reject_symlink_or_non_file(path: &Path, name: &str) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(MlError::InvalidModel(format!(
            "bundle file `{name}` must be a regular file"
        )));
    }
    Ok(())
}

fn read_bounded_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    context: &str,
) -> Result<T, MlError> {
    reject_oversized_file(path, MAX_JSON_BYTES, context)?;
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), MlError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if path.exists() && !path.is_file() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("artifact destination {} is not a file", path.display()),
        )));
    }
    let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .ok_or_else(|| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "artifact destination has no file name",
            ))
        })?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(MlError::Io(error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "eliza-open-set-{name}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn bundled_experiment(resamples: usize) -> OpenSetExperimentResult {
        run_open_set_experiment(
            &GroupedDataset::bundled().unwrap(),
            &OpenSetOodDataset::bundled_development().unwrap(),
            &OpenSetOodDataset::bundled_test().unwrap(),
            OpenSetTrainingConfig::default(),
            resamples,
        )
        .unwrap()
    }

    #[test]
    fn split_plan_is_group_disjoint_complete_and_deterministic() {
        let dataset = GroupedDataset::bundled().unwrap();
        let left = SplitPlan::build(&dataset, 20_260_722).unwrap();
        let right = SplitPlan::build(&dataset, 20_260_722).unwrap();
        assert_eq!(left.manifest, right.manifest);
        assert_eq!(left.manifest.assignments.len(), dataset.examples().len());

        let mut group_partitions = HashMap::new();
        for assignment in &left.manifest.assignments {
            if let Some(previous) =
                group_partitions.insert(&assignment.group_id, assignment.partition)
            {
                assert_eq!(previous, assignment.partition);
            }
        }
        assert_eq!(left.train().len(), 70);
        assert_eq!(left.development().len(), 14);
        assert_eq!(left.calibration().len(), 14);
        assert_eq!(left.id_test().len(), 14);
    }

    #[test]
    fn split_group_quotas_scale_without_emptying_training() {
        assert_eq!(evaluation_group_quota(4), 1);
        assert_eq!(evaluation_group_quota(8), 1);
        assert_eq!(evaluation_group_quota(15), 2);
        assert_eq!(evaluation_group_quota(100), 10);
        assert_eq!(evaluation_group_quota(20_000), 2_000);
    }

    #[test]
    fn split_manifest_recomputes_the_declared_hash_and_quota_strategy() {
        let dataset = GroupedDataset::bundled().unwrap();
        let plan = SplitPlan::build(&dataset, 20_260_722).unwrap();
        let mut manifest = plan.manifest.clone();
        let moved_group = manifest
            .assignments
            .iter()
            .find(|assignment| assignment.partition == PartitionKind::Train)
            .unwrap()
            .group_id
            .clone();
        for assignment in &mut manifest.assignments {
            if assignment.group_id == moved_group {
                assignment.partition = PartitionKind::IdTest;
            }
        }
        let error = manifest.validate_contract().unwrap_err();
        assert!(error.to_string().contains("declared strategy"));
    }

    #[test]
    fn independent_ood_populations_cannot_overlap() {
        let dataset = GroupedDataset::bundled().unwrap();
        let development = OpenSetOodDataset::bundled_development().unwrap();
        let mut test = OpenSetOodDataset::bundled_test().unwrap();
        test.examples[0].text = development.examples()[0].text.clone();
        assert!(reject_cross_dataset_overlap(&dataset, &development, &test).is_err());
    }

    #[test]
    fn temperature_scaling_does_not_increase_calibration_nll() {
        let result = bundled_experiment(100);
        assert!(
            result
                .metrics
                .calibrated_calibration_partition
                .negative_log_likelihood
                <= result
                    .metrics
                    .uncalibrated_calibration_partition
                    .negative_log_likelihood
                    + 1e-12
        );
        assert!(!result.metrics.threshold_selection.id_test_used);
        assert!(!result.metrics.threshold_selection.ood_test_used);
    }

    #[test]
    fn threshold_search_cannot_turn_an_oov_row_into_coverage() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model.clone(), result.policy.clone()).unwrap();
        let oov_prediction = runtime.predict("🪐");
        assert!(!oov_prediction.accepted);
        let development = vec![GroupedExample {
            id: "oov-development".into(),
            group_id: "oov-development-group".into(),
            label: oov_prediction.label,
            text: "🪐".into(),
        }];
        assert!(select_thresholds(
            &result.model,
            result.policy.temperature,
            &development,
            &OpenSetOodDataset::bundled_development().unwrap(),
        )
        .is_err());
    }

    #[test]
    fn contrastive_explanation_reconstructs_the_top_two_logit_margin() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        for text in [
            "Today I feel calm",
            "I want to finish the prototype",
            "Calculate an orbital period",
        ] {
            let prediction = runtime.predict(text);
            assert!(
                (prediction.logit_margin - prediction.explanation.reconstructed_logit_margin).abs()
                    <= 1e-10,
                "{text}: {} != {}",
                prediction.logit_margin,
                prediction.explanation.reconstructed_logit_margin
            );
            assert_eq!(prediction.label, prediction.explanation.top_label);
            assert_eq!(
                prediction.runner_up_label,
                prediction.explanation.runner_up_label
            );
        }
    }

    #[test]
    fn compiled_batch_matches_individual_prediction() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        let inputs = [
            "Hello there",
            "My plan needs review",
            "Balance this reaction",
        ];
        let batch = runtime.predict_batch(inputs.iter().copied());
        let individual = inputs
            .iter()
            .map(|input| runtime.predict(input))
            .collect::<Vec<_>>();
        assert_eq!(batch, individual);
    }

    #[test]
    fn embedded_bundle_is_digest_verified_and_compilable() {
        let verified = embedded_bundle().unwrap();
        assert_eq!(verified.manifest.model_version, OPEN_SET_MODEL_VERSION);
        let prediction = verified.compile().unwrap().predict("Today I feel calm");
        assert!(prediction.confidence.is_finite());
    }

    #[test]
    fn bootstrap_intervals_are_deterministic_and_contain_point_estimates() {
        let first = bundled_experiment(100);
        let second = bundled_experiment(100);
        assert_eq!(first.metrics.bootstrap_95, second.metrics.bootstrap_95);
        for estimate in [
            &first.metrics.bootstrap_95.id_accuracy,
            &first.metrics.bootstrap_95.id_macro_f1,
            &first.metrics.bootstrap_95.id_negative_log_likelihood,
            &first.metrics.bootstrap_95.id_multiclass_brier,
            &first.metrics.bootstrap_95.id_expected_calibration_error,
            &first.metrics.bootstrap_95.id_aurc,
            &first.metrics.bootstrap_95.ood_auroc,
            &first.metrics.bootstrap_95.ood_aupr_in_domain,
            &first.metrics.bootstrap_95.ood_fpr_at_95_tpr,
        ] {
            assert!(estimate.lower_95 <= estimate.value);
            assert!(estimate.value <= estimate.upper_95);
        }
    }

    #[test]
    fn discrimination_metrics_handle_ties_by_score_group() {
        let metrics = discrimination_metrics(&[0.9, 0.5], &[0.5, 0.1]);
        assert!((metrics.auroc - 0.875).abs() <= 1e-12);
        assert!((metrics.aupr_in_domain - (5.0 / 6.0)).abs() <= 1e-12);
        assert!((metrics.fpr_at_95_tpr - 0.5).abs() <= 1e-12);
    }

    #[test]
    fn risk_coverage_groups_confidence_ties_and_ignores_id_order() {
        let make = |id: &str, confidence: f64, correct: bool| EvaluatedOpenSetPrediction {
            id: id.into(),
            actual_label: "a".into(),
            predicted_label: if correct { "a".into() } else { "b".into() },
            correct,
            accepted: true,
            confidence,
            probability_margin: 0.0,
            probabilities: BTreeMap::new(),
        };
        let left = vec![
            make("one", 0.9, true),
            make("alpha", 0.5, true),
            make("omega", 0.5, false),
            make("four", 0.1, false),
        ];
        let right = vec![
            make("renamed-one", 0.9, true),
            make("zulu", 0.5, true),
            make("able", 0.5, false),
            make("renamed-four", 0.1, false),
        ];
        let (left_curve, left_aurc) = risk_coverage(&left);
        let (right_curve, right_aurc) = risk_coverage(&right);
        assert_eq!(left_curve, right_curve);
        assert_eq!(
            left_curve
                .iter()
                .map(|point| point.accepted)
                .collect::<Vec<_>>(),
            vec![1, 3, 4]
        );
        assert!((left_aurc - 7.0 / 24.0).abs() <= 1e-12);
        assert_eq!(left_aurc, right_aurc);
    }

    #[test]
    fn model_validation_rejects_overflow_scale_parameters() {
        let result = bundled_experiment(100);
        let mut huge_weight = result.model.clone();
        huge_weight.weights[0][0] = f64::MAX;
        assert!(huge_weight.validate().is_err());

        let mut huge_bias = result.model.clone();
        huge_bias.biases[0] = f64::MAX;
        assert!(huge_bias.validate().is_err());

        let mut huge_idf = result.model;
        huge_idf.vectorizer.inverse_document_frequency[0] = f64::MAX;
        assert!(huge_idf.validate().is_err());
    }

    #[test]
    fn metrics_reject_ids_shared_by_id_and_ood_tests() {
        let result = bundled_experiment(100);
        let mut metrics = result.metrics;
        metrics.ood_test.predictions[0].id = metrics.id_test.predictions[0].id.clone();
        let error =
            validate_metrics_contract(&metrics, &result.model, &result.policy, &result.split_plan)
                .unwrap_err();
        assert!(error.to_string().contains("OOD-test prediction ledger"));
    }

    #[test]
    fn bundle_verification_detects_tampering_and_reproduction_matches() {
        let directory = TestDirectory::new("bundle");
        let result = bundled_experiment(100);
        write_bundle(&directory.0, &result).unwrap();
        write_bundle(&directory.0, &result).unwrap();
        let verified = verify_bundle(&directory.0).unwrap();
        assert_eq!(verified.model, result.model);
        reproduce_bundle(
            &directory.0,
            &GroupedDataset::bundled().unwrap(),
            &OpenSetOodDataset::bundled_development().unwrap(),
            &OpenSetOodDataset::bundled_test().unwrap(),
        )
        .unwrap();

        let model_path = directory.0.join("model.json");
        let mut bytes = fs::read(&model_path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        fs::write(model_path, bytes).unwrap();
        assert!(verify_bundle(&directory.0).is_err());
    }

    #[test]
    fn bundle_writer_never_replaces_an_unrelated_non_empty_directory() {
        let root = TestDirectory::new("bundle-destination-guard");
        let unrelated = root.0.join("unrelated");
        fs::create_dir(&unrelated).unwrap();
        let sentinel = unrelated.join("keep-me.txt");
        fs::write(&sentinel, b"important").unwrap();
        let result = bundled_experiment(100);

        let error = write_bundle(&unrelated, &result).unwrap_err();
        assert!(error.to_string().contains("refusing to replace"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"important");

        let bundle = root.0.join("bundle");
        write_bundle(&bundle, &result).unwrap();
        let unexpected = bundle.join("unexpected.txt");
        fs::write(&unexpected, b"not part of the contract").unwrap();
        assert!(verify_bundle(&bundle).is_err());
        assert!(write_bundle(&bundle, &result).is_err());
        assert!(unexpected.exists());
    }

    #[test]
    fn bundle_verification_rejects_self_rehashed_semantic_mismatches() {
        let root = TestDirectory::new("bundle-semantic-tamper");
        let result = bundled_experiment(100);
        let policy_bundle = root.0.join("policy-bundle");
        write_bundle(&policy_bundle, &result).unwrap();

        let policy_path = policy_bundle.join("policy.json");
        let mut policy: OpenSetPolicyV2 =
            serde_json::from_slice(&fs::read(&policy_path).unwrap()).unwrap();
        policy.minimum_confidence -= 0.01;
        let policy_bytes = canonical_json(&policy).unwrap();
        fs::write(&policy_path, &policy_bytes).unwrap();

        let manifest_path = policy_bundle.join("manifest.json");
        let mut manifest: BundleManifestV2 =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest
            .files
            .insert("policy.json".into(), sha256_hex(&policy_bytes));
        fs::write(&manifest_path, canonical_json(&manifest).unwrap()).unwrap();

        let error = verify_bundle(&policy_bundle).unwrap_err();
        assert!(error.to_string().contains("frozen policy"));

        let ledger_bundle = root.0.join("ledger-bundle");
        write_bundle(&ledger_bundle, &result).unwrap();
        let metrics_path = ledger_bundle.join("metrics.json");
        let mut metrics: OpenSetMetricsV2 =
            serde_json::from_slice(&fs::read(&metrics_path).unwrap()).unwrap();
        let replacement_label = result
            .model
            .labels
            .iter()
            .find(|label| **label != metrics.id_test.predictions[0].actual_label)
            .unwrap()
            .clone();
        metrics.id_test.predictions[0].actual_label = replacement_label;
        metrics.id_test.predictions[0].correct = metrics.id_test.predictions[0].actual_label
            == metrics.id_test.predictions[0].predicted_label;
        let metrics_bytes = canonical_json(&metrics).unwrap();
        fs::write(&metrics_path, &metrics_bytes).unwrap();
        let ledger_manifest_path = ledger_bundle.join("manifest.json");
        let mut ledger_manifest: BundleManifestV2 =
            serde_json::from_slice(&fs::read(&ledger_manifest_path).unwrap()).unwrap();
        ledger_manifest
            .files
            .insert("metrics.json".into(), sha256_hex(&metrics_bytes));
        fs::write(
            &ledger_manifest_path,
            canonical_json(&ledger_manifest).unwrap(),
        )
        .unwrap();
        let error = verify_bundle(&ledger_bundle).unwrap_err();
        assert!(error.to_string().contains("split plan"));
    }

    #[test]
    fn jsonl_batch_is_bounded_and_preserves_ids() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        let input = b"{\"id\":\"row-1\",\"text\":\"Today I feel calm\"}\n{\"id\":\"row-2\",\"text\":\"Hello there\"}\n";
        let mut output = Vec::new();
        let count = predict_jsonl(&runtime, &mut &input[..], &mut output).unwrap();
        assert_eq!(count, 2);
        let lines = String::from_utf8(output).unwrap();
        assert!(lines.contains("\"id\":\"row-1\""));
        assert!(lines.contains("\"id\":\"row-2\""));

        let oversized = format!(
            "{{\"id\":\"row-3\",\"text\":\"{}\"}}\n",
            "x".repeat(MAX_JSONL_BYTES)
        );
        let error =
            predict_jsonl(&runtime, &mut oversized.as_bytes(), &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("exceeds the input boundary"));
        assert!(enforce_batch_capacity(MAX_EXAMPLES - 1).is_ok());
        assert!(enforce_batch_capacity(MAX_EXAMPLES).is_err());
    }
}
