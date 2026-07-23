//! Local, aggregate-only robustness evaluation for the open-set classifier.
//!
//! The audit applies deterministic metamorphic transformations to bounded JSONL input. Formatting
//! transformations are expected preprocessing invariants. Typographic transformations are
//! explicitly stress tests rather than meaning-preserving guarantees. Reports contain aggregate
//! measurements only: input identifiers and text are validated in memory and never serialized.

use crate::open_set::{CompiledModel, OpenSetPrediction, PartitionKind, VerifiedBundle};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::io::{BufRead, Read};

pub const ROBUSTNESS_REPORT_SCHEMA_VERSION: u32 = 1;
pub const ROBUSTNESS_REPORT_KIND: &str = "eliza-metamorphic-robustness";
pub const ROBUSTNESS_SUITE_VERSION: &str = "1.0.0";
const MAX_AUDIT_ROWS: usize = 100_000;
const MAX_AUDIT_PHYSICAL_LINES: usize = 100_000;
const MAX_JSONL_BYTES: usize = crate::MAX_INPUT_CHARS * 4 + 16_384;
const MAX_AUDIT_TOTAL_BYTES: usize = 64 * 1024 * 1024;
const REPORTING_SCALE: f64 = 1_000_000_000.0;

#[derive(Debug)]
pub enum RobustnessError {
    Io(std::io::Error),
    InvalidInput(String),
    InvalidBundle(String),
    InvalidPolicy(String),
    GateFailed(String),
}

impl fmt::Display for RobustnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "robustness I/O error: {error}"),
            Self::InvalidInput(message) => write!(formatter, "invalid robustness input: {message}"),
            Self::InvalidBundle(message) => {
                write!(formatter, "invalid robustness bundle: {message}")
            }
            Self::InvalidPolicy(message) => {
                write!(formatter, "invalid robustness policy: {message}")
            }
            Self::GateFailed(message) => write!(formatter, "robustness gate failed: {message}"),
        }
    }
}

impl std::error::Error for RobustnessError {}

impl From<std::io::Error> for RobustnessError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PerturbationFamily {
    Formatting,
    Typographic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RobustnessPopulation {
    Jsonl,
    ProvidedCases,
    BundleIdTest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RobustnessSlice {
    pub evaluated_variants: usize,
    pub skipped_applications: usize,
    pub label_agreement: f64,
    pub acceptance_agreement: f64,
    pub decision_agreement: f64,
    pub label_flips: usize,
    pub acceptance_flips: usize,
    pub accepted_to_abstained: usize,
    pub abstained_to_accepted: usize,
    pub mean_normalized_js_divergence: f64,
    pub maximum_normalized_js_divergence: f64,
    pub mean_absolute_confidence_delta: f64,
    pub maximum_absolute_confidence_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PerturbationReport {
    pub name: String,
    pub family: PerturbationFamily,
    pub metrics: RobustnessSlice,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RobustnessReport {
    pub report_kind: String,
    pub schema_version: u32,
    pub suite_version: String,
    pub population: RobustnessPopulation,
    pub model_version: String,
    pub model_dataset_sha256: String,
    pub split_plan_sha256: String,
    pub temperature: f64,
    pub minimum_confidence: f64,
    pub minimum_probability_margin: f64,
    pub input_count: usize,
    pub baseline_accepted: usize,
    pub evaluated_variants: usize,
    pub formatting: RobustnessSlice,
    pub typographic: RobustnessSlice,
    pub perturbations: Vec<PerturbationReport>,
    #[serde(skip)]
    raw_gate_evidence: Option<RawGateEvidence>,
}

/// Optional regression thresholds for the robustness report.
///
/// Formatting is fail-closed by default because every formatting transformation is designed to
/// normalize to the same model features. Typographic stability has no universal target, so callers
/// must opt into a threshold that matches their own release policy.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustnessGate {
    pub minimum_formatting_label_agreement: f64,
    pub minimum_formatting_decision_agreement: f64,
    pub maximum_formatting_js_divergence: f64,
    pub minimum_typographic_label_agreement: Option<f64>,
    pub minimum_typographic_decision_agreement: Option<f64>,
    pub maximum_typographic_js_divergence: Option<f64>,
}

impl Default for RobustnessGate {
    fn default() -> Self {
        Self {
            minimum_formatting_label_agreement: 1.0,
            minimum_formatting_decision_agreement: 1.0,
            maximum_formatting_js_divergence: 0.0,
            minimum_typographic_label_agreement: None,
            minimum_typographic_decision_agreement: None,
            maximum_typographic_js_divergence: None,
        }
    }
}

impl RobustnessGate {
    pub fn validate(&self) -> Result<(), RobustnessError> {
        for (name, value) in [
            (
                "minimum formatting label agreement",
                self.minimum_formatting_label_agreement,
            ),
            (
                "minimum formatting decision agreement",
                self.minimum_formatting_decision_agreement,
            ),
            (
                "maximum formatting JS divergence",
                self.maximum_formatting_js_divergence,
            ),
        ] {
            validate_unit_interval(value, name)?;
        }
        if let Some(value) = self.minimum_typographic_label_agreement {
            validate_unit_interval(value, "minimum typographic label agreement")?;
        }
        if let Some(value) = self.minimum_typographic_decision_agreement {
            validate_unit_interval(value, "minimum typographic decision agreement")?;
        }
        if let Some(value) = self.maximum_typographic_js_divergence {
            validate_unit_interval(value, "maximum typographic JS divergence")?;
        }
        Ok(())
    }

    pub fn enforce(&self, report: &RobustnessReport) -> Result<(), RobustnessError> {
        self.validate()?;
        let raw = report.raw_gate_evidence.as_ref().ok_or_else(|| {
            RobustnessError::GateFailed(
                "raw audit evidence is unavailable; rerun the audit before applying a gate".into(),
            )
        })?;
        if raw.formatting.report() != report.formatting
            || raw.typographic.report() != report.typographic
        {
            return Err(RobustnessError::GateFailed(
                "serialized metrics do not match the raw audit evidence".into(),
            ));
        }

        let formatting_label_agreement = raw.formatting.label_agreement();
        let formatting_decision_agreement = raw.formatting.decision_agreement();
        let formatting_js_divergence = raw.formatting.maximum_js_divergence;
        let typographic_label_agreement = raw.typographic.label_agreement();
        let typographic_decision_agreement = raw.typographic.decision_agreement();
        let typographic_js_divergence = raw.typographic.maximum_js_divergence;
        let mut failures = Vec::new();
        if raw.formatting.evaluated_variants == 0 {
            failures.push("the formatting audit produced no variants".to_owned());
        }
        if formatting_label_agreement < self.minimum_formatting_label_agreement {
            failures.push(format!(
                "formatting label agreement {:.12} is below {:.12}",
                formatting_label_agreement, self.minimum_formatting_label_agreement
            ));
        }
        if formatting_decision_agreement < self.minimum_formatting_decision_agreement {
            failures.push(format!(
                "formatting decision agreement {:.12} is below {:.12}",
                formatting_decision_agreement, self.minimum_formatting_decision_agreement
            ));
        }
        if formatting_js_divergence > self.maximum_formatting_js_divergence {
            failures.push(format!(
                "formatting JS divergence {:.12} exceeds {:.12}",
                formatting_js_divergence, self.maximum_formatting_js_divergence
            ));
        }
        if let Some(minimum) = self.minimum_typographic_label_agreement {
            if raw.typographic.evaluated_variants == 0 {
                failures.push("the typographic audit produced no variants".to_owned());
            } else if typographic_label_agreement < minimum {
                failures.push(format!(
                    "typographic label agreement {:.12} is below {:.12}",
                    typographic_label_agreement, minimum
                ));
            }
        }
        if let Some(minimum) = self.minimum_typographic_decision_agreement {
            if raw.typographic.evaluated_variants == 0 {
                failures.push("the typographic audit produced no variants".to_owned());
            } else if typographic_decision_agreement < minimum {
                failures.push(format!(
                    "typographic decision agreement {:.12} is below {:.12}",
                    typographic_decision_agreement, minimum
                ));
            }
        }
        if let Some(maximum) = self.maximum_typographic_js_divergence {
            if raw.typographic.evaluated_variants == 0 {
                failures.push("the typographic audit produced no variants".to_owned());
            } else if typographic_js_divergence > maximum {
                failures.push(format!(
                    "typographic JS divergence {:.12} exceeds {:.12}",
                    typographic_js_divergence, maximum
                ));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(RobustnessError::GateFailed(failures.join("; ")))
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RobustnessCase {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Copy)]
struct Perturbation {
    name: &'static str,
    family: PerturbationFamily,
    apply: fn(&str) -> Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct RunningMetrics {
    evaluated_variants: usize,
    skipped_applications: usize,
    label_agreements: usize,
    acceptance_agreements: usize,
    decision_agreements: usize,
    label_flips: usize,
    acceptance_flips: usize,
    accepted_to_abstained: usize,
    abstained_to_accepted: usize,
    js_divergence_sum: f64,
    maximum_js_divergence: f64,
    confidence_delta_sum: f64,
    maximum_confidence_delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct RawGateEvidence {
    formatting: RunningMetrics,
    typographic: RunningMetrics,
}

#[derive(Debug, Clone, Copy)]
struct AuditLimits {
    maximum_rows: usize,
    maximum_physical_lines: usize,
    maximum_line_bytes: usize,
    maximum_total_bytes: usize,
}

const DEFAULT_AUDIT_LIMITS: AuditLimits = AuditLimits {
    maximum_rows: MAX_AUDIT_ROWS,
    maximum_physical_lines: MAX_AUDIT_PHYSICAL_LINES,
    maximum_line_bytes: MAX_JSONL_BYTES,
    maximum_total_bytes: MAX_AUDIT_TOTAL_BYTES,
};

struct AuditState<'a> {
    runtime: &'a CompiledModel,
    population: RobustnessPopulation,
    perturbations: [Perturbation; 7],
    accumulators: Vec<RunningMetrics>,
    ids: HashSet<String>,
    input_count: usize,
    baseline_accepted: usize,
    maximum_rows: usize,
}

impl<'a> AuditState<'a> {
    fn new(
        runtime: &'a CompiledModel,
        population: RobustnessPopulation,
        maximum_rows: usize,
    ) -> Self {
        let perturbations = perturbations();
        Self {
            runtime,
            population,
            accumulators: perturbations
                .iter()
                .map(|_| RunningMetrics::default())
                .collect(),
            perturbations,
            ids: HashSet::new(),
            input_count: 0,
            baseline_accepted: 0,
            maximum_rows,
        }
    }

    fn add(&mut self, input: RobustnessCase, line_number: usize) -> Result<(), RobustnessError> {
        if self.input_count >= self.maximum_rows {
            return Err(RobustnessError::InvalidInput(format!(
                "robustness input exceeds {} rows",
                self.maximum_rows
            )));
        }
        validate_input(&input, line_number, &mut self.ids)?;
        let baseline = self.runtime.predict(&input.text);
        self.baseline_accepted += usize::from(baseline.accepted);

        for (index, perturbation) in self.perturbations.iter().enumerate() {
            match (perturbation.apply)(&input.text) {
                Some(variant)
                    if variant != input.text
                        && variant.chars().count() <= crate::MAX_INPUT_CHARS =>
                {
                    let prediction = self.runtime.predict(&variant);
                    self.accumulators[index].observe(&baseline, &prediction);
                }
                _ => self.accumulators[index].skipped_applications += 1,
            }
        }
        self.input_count += 1;
        Ok(())
    }

    fn finish(self) -> Result<RobustnessReport, RobustnessError> {
        if self.input_count == 0 {
            return Err(RobustnessError::InvalidInput(
                "robustness audit requires at least one input".into(),
            ));
        }

        let mut formatting = RunningMetrics::default();
        let mut typographic = RunningMetrics::default();
        let perturbation_reports = self
            .perturbations
            .iter()
            .zip(&self.accumulators)
            .map(|(perturbation, metrics)| {
                match perturbation.family {
                    PerturbationFamily::Formatting => formatting.merge(metrics),
                    PerturbationFamily::Typographic => typographic.merge(metrics),
                }
                PerturbationReport {
                    name: perturbation.name.into(),
                    family: perturbation.family,
                    metrics: metrics.report(),
                }
            })
            .collect::<Vec<_>>();

        let formatting_report = formatting.report();
        let typographic_report = typographic.report();
        Ok(RobustnessReport {
            report_kind: ROBUSTNESS_REPORT_KIND.into(),
            schema_version: ROBUSTNESS_REPORT_SCHEMA_VERSION,
            suite_version: ROBUSTNESS_SUITE_VERSION.into(),
            population: self.population,
            model_version: self.runtime.model().model_version.clone(),
            model_dataset_sha256: self.runtime.model().dataset_sha256.clone(),
            split_plan_sha256: self.runtime.model().split_plan_sha256.clone(),
            temperature: self.runtime.policy().temperature,
            minimum_confidence: self.runtime.policy().minimum_confidence,
            minimum_probability_margin: self.runtime.policy().minimum_probability_margin,
            input_count: self.input_count,
            baseline_accepted: self.baseline_accepted,
            evaluated_variants: formatting.evaluated_variants + typographic.evaluated_variants,
            formatting: formatting_report,
            typographic: typographic_report,
            perturbations: perturbation_reports,
            raw_gate_evidence: Some(RawGateEvidence {
                formatting,
                typographic,
            }),
        })
    }
}

impl RunningMetrics {
    fn observe(&mut self, baseline: &OpenSetPrediction, variant: &OpenSetPrediction) {
        self.evaluated_variants += 1;
        let labels_agree = baseline.label == variant.label;
        let acceptance_agrees = baseline.accepted == variant.accepted;
        let decisions_agree = match (baseline.accepted, variant.accepted) {
            (true, true) => labels_agree,
            (false, false) => true,
            _ => false,
        };
        self.label_agreements += usize::from(labels_agree);
        self.acceptance_agreements += usize::from(acceptance_agrees);
        self.decision_agreements += usize::from(decisions_agree);
        self.label_flips += usize::from(!labels_agree);
        self.acceptance_flips += usize::from(!acceptance_agrees);
        self.accepted_to_abstained += usize::from(baseline.accepted && !variant.accepted);
        self.abstained_to_accepted += usize::from(!baseline.accepted && variant.accepted);

        let divergence = normalized_jensen_shannon(&baseline.probabilities, &variant.probabilities);
        self.js_divergence_sum += divergence;
        self.maximum_js_divergence = self.maximum_js_divergence.max(divergence);
        let confidence_delta = (baseline.confidence - variant.confidence).abs();
        self.confidence_delta_sum += confidence_delta;
        self.maximum_confidence_delta = self.maximum_confidence_delta.max(confidence_delta);
    }

    fn merge(&mut self, other: &Self) {
        self.evaluated_variants += other.evaluated_variants;
        self.skipped_applications += other.skipped_applications;
        self.label_agreements += other.label_agreements;
        self.acceptance_agreements += other.acceptance_agreements;
        self.decision_agreements += other.decision_agreements;
        self.label_flips += other.label_flips;
        self.acceptance_flips += other.acceptance_flips;
        self.accepted_to_abstained += other.accepted_to_abstained;
        self.abstained_to_accepted += other.abstained_to_accepted;
        self.js_divergence_sum += other.js_divergence_sum;
        self.maximum_js_divergence = self.maximum_js_divergence.max(other.maximum_js_divergence);
        self.confidence_delta_sum += other.confidence_delta_sum;
        self.maximum_confidence_delta = self
            .maximum_confidence_delta
            .max(other.maximum_confidence_delta);
    }

    fn report(&self) -> RobustnessSlice {
        let denominator = self.evaluated_variants as f64;
        RobustnessSlice {
            evaluated_variants: self.evaluated_variants,
            skipped_applications: self.skipped_applications,
            label_agreement: reported_ratio(self.label_agreements, denominator),
            acceptance_agreement: reported_ratio(self.acceptance_agreements, denominator),
            decision_agreement: reported_ratio(self.decision_agreements, denominator),
            label_flips: self.label_flips,
            acceptance_flips: self.acceptance_flips,
            accepted_to_abstained: self.accepted_to_abstained,
            abstained_to_accepted: self.abstained_to_accepted,
            mean_normalized_js_divergence: reported_mean(self.js_divergence_sum, denominator),
            maximum_normalized_js_divergence: reported(self.maximum_js_divergence),
            mean_absolute_confidence_delta: reported_mean(self.confidence_delta_sum, denominator),
            maximum_absolute_confidence_delta: reported(self.maximum_confidence_delta),
        }
    }

    fn label_agreement(&self) -> f64 {
        raw_ratio(self.label_agreements, self.evaluated_variants)
    }

    fn decision_agreement(&self) -> f64 {
        raw_ratio(self.decision_agreements, self.evaluated_variants)
    }
}

/// Reads bounded JSONL objects with `id` and `text`, evaluates deterministic perturbations and
/// returns an aggregate-only report. Neither identifier nor text is included in the result.
pub fn audit_jsonl(
    runtime: &CompiledModel,
    reader: &mut impl BufRead,
) -> Result<RobustnessReport, RobustnessError> {
    audit_jsonl_with_limits(runtime, reader, DEFAULT_AUDIT_LIMITS)
}

fn audit_jsonl_with_limits(
    runtime: &CompiledModel,
    reader: &mut impl BufRead,
    limits: AuditLimits,
) -> Result<RobustnessReport, RobustnessError> {
    let mut state = AuditState::new(runtime, RobustnessPopulation::Jsonl, limits.maximum_rows);
    let mut line = String::new();
    let mut physical_lines = 0usize;
    let mut total_bytes = 0usize;

    loop {
        line.clear();
        let bytes = reader
            .take((limits.maximum_line_bytes + 1) as u64)
            .read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        physical_lines = physical_lines.checked_add(1).ok_or_else(|| {
            RobustnessError::InvalidInput("physical line counter overflowed".into())
        })?;
        if physical_lines > limits.maximum_physical_lines {
            return Err(RobustnessError::InvalidInput(format!(
                "robustness JSONL input exceeds {} physical lines",
                limits.maximum_physical_lines
            )));
        }
        if bytes > limits.maximum_line_bytes {
            return Err(RobustnessError::InvalidInput(format!(
                "robustness JSONL line {physical_lines} exceeds the per-line byte boundary"
            )));
        }
        total_bytes = total_bytes.checked_add(bytes).ok_or_else(|| {
            RobustnessError::InvalidInput("total input byte counter overflowed".into())
        })?;
        if total_bytes > limits.maximum_total_bytes {
            return Err(RobustnessError::InvalidInput(format!(
                "robustness JSONL input exceeds the {} byte total boundary",
                limits.maximum_total_bytes
            )));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let input = serde_json::from_str(trimmed).map_err(|_| {
            RobustnessError::InvalidInput(format!(
                "robustness JSONL line {physical_lines} does not match the required object schema"
            ))
        })?;
        state.add(input, physical_lines)?;
    }
    state.finish()
}

/// Audits already parsed cases without serializing them. This is used for the frozen bundle
/// regression population and is also available to callers that do not need JSONL.
pub fn audit_cases(
    runtime: &CompiledModel,
    cases: impl IntoIterator<Item = RobustnessCase>,
) -> Result<RobustnessReport, RobustnessError> {
    audit_cases_for_population(runtime, cases, RobustnessPopulation::ProvidedCases)
}

/// Audits the final ID-test owned by a semantically verified bundle.
///
/// `VerifiedBundle` cannot be constructed outside `open_set`, and this function consumes it before
/// compiling inference. Callers therefore cannot label arbitrary cases as the bundle ID-test.
pub fn audit_bundle_id_test(bundle: VerifiedBundle) -> Result<RobustnessReport, RobustnessError> {
    let cases = bundle
        .split_plan()
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == PartitionKind::IdTest)
        .map(|assignment| RobustnessCase {
            id: assignment.id.clone(),
            text: assignment.text.clone(),
        })
        .collect::<Vec<_>>();
    let expected_examples = bundle.metrics().id_test.example_count;
    if cases.len() != expected_examples {
        return Err(RobustnessError::InvalidBundle(
            "the verified ID-test count is inconsistent".into(),
        ));
    }
    let runtime = bundle.compile().map_err(|_| {
        RobustnessError::InvalidBundle(
            "the semantically verified model could not be compiled".into(),
        )
    })?;
    audit_cases_for_population(&runtime, cases, RobustnessPopulation::BundleIdTest)
}

fn audit_cases_for_population(
    runtime: &CompiledModel,
    cases: impl IntoIterator<Item = RobustnessCase>,
    population: RobustnessPopulation,
) -> Result<RobustnessReport, RobustnessError> {
    let mut state = AuditState::new(runtime, population, MAX_AUDIT_ROWS);
    for (index, case) in cases.into_iter().enumerate() {
        state.add(case, index + 1)?;
    }
    state.finish()
}

fn perturbations() -> [Perturbation; 7] {
    [
        Perturbation {
            name: "ascii-letter-case",
            family: PerturbationFamily::Formatting,
            apply: swap_ascii_case,
        },
        Perturbation {
            name: "horizontal-whitespace",
            family: PerturbationFamily::Formatting,
            apply: stress_whitespace,
        },
        Perturbation {
            name: "unicode-compatibility-width",
            family: PerturbationFamily::Formatting,
            apply: fullwidth_ascii,
        },
        Perturbation {
            name: "terminal-punctuation",
            family: PerturbationFamily::Formatting,
            apply: add_terminal_punctuation,
        },
        Perturbation {
            name: "single-character-deletion",
            family: PerturbationFamily::Typographic,
            apply: delete_internal_character,
        },
        Perturbation {
            name: "adjacent-character-transposition",
            family: PerturbationFamily::Typographic,
            apply: transpose_internal_characters,
        },
        Perturbation {
            name: "single-character-duplication",
            family: PerturbationFamily::Typographic,
            apply: duplicate_internal_character,
        },
    ]
}

fn validate_input(
    input: &RobustnessCase,
    line: usize,
    ids: &mut HashSet<String>,
) -> Result<(), RobustnessError> {
    if input.id.is_empty()
        || input.id.len() > 128
        || !input
            .id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(RobustnessError::InvalidInput(format!(
            "robustness line {line} has an invalid id"
        )));
    }
    if !ids.insert(input.id.clone()) {
        return Err(RobustnessError::InvalidInput(format!(
            "robustness line {line} duplicates an earlier id"
        )));
    }
    if input.text.trim().is_empty() || input.text.chars().count() > crate::MAX_INPUT_CHARS {
        return Err(RobustnessError::InvalidInput(format!(
            "robustness line {line} has empty or oversized text"
        )));
    }
    Ok(())
}

fn normalized_jensen_shannon(
    left: &std::collections::BTreeMap<String, f64>,
    right: &std::collections::BTreeMap<String, f64>,
) -> f64 {
    let mut divergence = 0.0;
    for (label, left_probability) in left {
        let right_probability = right.get(label).copied().unwrap_or(0.0);
        let midpoint = (left_probability + right_probability) / 2.0;
        if *left_probability > 0.0 && midpoint > 0.0 {
            divergence += 0.5 * left_probability * (left_probability / midpoint).ln();
        }
        if right_probability > 0.0 && midpoint > 0.0 {
            divergence += 0.5 * right_probability * (right_probability / midpoint).ln();
        }
    }
    (divergence / std::f64::consts::LN_2).clamp(0.0, 1.0)
}

fn swap_ascii_case(text: &str) -> Option<String> {
    let transformed = text
        .chars()
        .map(|character| {
            if character.is_ascii_lowercase() {
                character.to_ascii_uppercase()
            } else if character.is_ascii_uppercase() {
                character.to_ascii_lowercase()
            } else {
                character
            }
        })
        .collect::<String>();
    (transformed != text).then_some(transformed)
}

fn stress_whitespace(text: &str) -> Option<String> {
    let mut changed = false;
    let transformed = text
        .chars()
        .map(|character| {
            if character == ' ' {
                changed = true;
                '\t'
            } else if character == '\t' {
                changed = true;
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    if changed {
        Some(transformed)
    } else if text.chars().count() < crate::MAX_INPUT_CHARS {
        Some(format!(" {text}"))
    } else {
        None
    }
}

fn fullwidth_ascii(text: &str) -> Option<String> {
    let mut changed = false;
    let transformed = text
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                changed = true;
                char::from_u32(character as u32 + 0xFEE0)
                    .expect("ASCII alphanumerics have fullwidth compatibility forms")
            } else {
                character
            }
        })
        .collect::<String>();
    changed.then_some(transformed)
}

fn add_terminal_punctuation(text: &str) -> Option<String> {
    if text.chars().count() < crate::MAX_INPUT_CHARS {
        Some(format!("{text}!"))
    } else {
        let mut characters = text.chars().collect::<Vec<_>>();
        let last = characters.last_mut()?;
        if last.is_ascii_punctuation() {
            *last = if *last == '!' { '.' } else { '!' };
            Some(characters.into_iter().collect())
        } else {
            None
        }
    }
}

fn delete_internal_character(text: &str) -> Option<String> {
    let mut characters = text.chars().collect::<Vec<_>>();
    let (start, length) = longest_alphanumeric_run(&characters, 5)?;
    characters.remove(start + length / 2);
    Some(characters.into_iter().collect())
}

fn transpose_internal_characters(text: &str) -> Option<String> {
    let mut characters = text.chars().collect::<Vec<_>>();
    let (start, length) = longest_alphanumeric_run(&characters, 4)?;
    let center = start + length / 2;
    let index = (start..start + length - 1)
        .filter(|candidate| characters[*candidate] != characters[*candidate + 1])
        .min_by_key(|candidate| candidate.abs_diff(center))?;
    characters.swap(index, index + 1);
    Some(characters.into_iter().collect())
}

fn duplicate_internal_character(text: &str) -> Option<String> {
    if text.chars().count() >= crate::MAX_INPUT_CHARS {
        return None;
    }
    let mut characters = text.chars().collect::<Vec<_>>();
    let (start, length) = longest_alphanumeric_run(&characters, 3)?;
    let index = start + length / 2;
    let character = characters[index];
    characters.insert(index, character);
    Some(characters.into_iter().collect())
}

fn longest_alphanumeric_run(characters: &[char], minimum_length: usize) -> Option<(usize, usize)> {
    let mut best = None;
    let mut start = 0usize;
    while start < characters.len() {
        if !characters[start].is_alphanumeric() {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < characters.len() && characters[end].is_alphanumeric() {
            end += 1;
        }
        let length = end - start;
        if length >= minimum_length && best.map_or(true, |(_, best_length)| length > best_length) {
            best = Some((start, length));
        }
        start = end;
    }
    best
}

fn validate_unit_interval(value: f64, name: &str) -> Result<(), RobustnessError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(RobustnessError::InvalidPolicy(format!(
            "{name} must be finite and between zero and one"
        )));
    }
    Ok(())
}

fn reported(value: f64) -> f64 {
    (value * REPORTING_SCALE).round() / REPORTING_SCALE
}

fn reported_ratio(numerator: usize, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        reported(numerator as f64 / denominator)
    }
}

fn reported_mean(sum: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        reported(sum / denominator)
    }
}

fn raw_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_set::embedded_bundle;
    use std::io::Cursor;

    fn runtime() -> CompiledModel {
        embedded_bundle().unwrap().compile().unwrap()
    }

    #[test]
    fn deterministic_report_keeps_prompts_and_ids_out_of_output() {
        let input = concat!(
            "{\"id\":\"case-a\",\"text\":\"Hello, I intend to make a concrete plan\"}\n",
            "{\"id\":\"case-b\",\"text\":\"Today I feel uncertain about the outcome\"}\n"
        );
        let first = audit_jsonl(&runtime(), &mut Cursor::new(input)).unwrap();
        let second = audit_jsonl(&runtime(), &mut Cursor::new(input)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.schema_version, ROBUSTNESS_REPORT_SCHEMA_VERSION);
        assert_eq!(first.report_kind, ROBUSTNESS_REPORT_KIND);
        assert_eq!(first.suite_version, ROBUSTNESS_SUITE_VERSION);
        assert_eq!(first.population, RobustnessPopulation::Jsonl);
        assert_eq!(first.model_version, "3.0.0");
        assert_eq!(first.input_count, 2);
        assert_eq!(first.perturbations.len(), 7);
        assert!(first.evaluated_variants >= 12);
        assert_eq!(first.formatting.decision_agreement, 1.0);
        assert_eq!(first.formatting.maximum_normalized_js_divergence, 0.0);
        RobustnessGate::default().enforce(&first).unwrap();

        let serialized = serde_json::to_string(&first).unwrap();
        assert!(!serialized.contains("case-a"));
        assert!(!serialized.contains("concrete plan"));
        assert!(!serialized.contains("uncertain about"));
        let restored: RobustnessReport = serde_json::from_str(&serialized).unwrap();
        assert_eq!(serde_json::to_string(&restored).unwrap(), serialized);
        assert!(RobustnessGate::default()
            .enforce(&restored)
            .unwrap_err()
            .to_string()
            .contains("raw audit evidence is unavailable"));
    }

    #[test]
    fn malformed_duplicate_and_empty_inputs_fail_closed() {
        for (input, expected) in [
            ("", "requires at least one"),
            (
                "{\"id\":\"case\",\"text\":\"hello\",\"private_token\":\"secret-value\"}\n",
                "does not match the required object schema",
            ),
            (
                "{\"id\":\"case\",\"text\":\"hello\"}\n{\"id\":\"case\",\"text\":\"again\"}\n",
                "duplicates an earlier id",
            ),
            ("{\"id\":\"not valid\",\"text\":\"hello\"}\n", "invalid id"),
            (
                "{\"id\":\"case\",\"text\":\"   \"}\n",
                "empty or oversized text",
            ),
        ] {
            let error = audit_jsonl(&runtime(), &mut Cursor::new(input)).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "{error} did not contain {expected}"
            );
        }

        let oversized = serde_json::json!({
            "id": "case",
            "text": "x".repeat(crate::MAX_INPUT_CHARS + 1)
        })
        .to_string();
        let error = audit_jsonl(&runtime(), &mut Cursor::new(oversized)).unwrap_err();
        assert!(error.to_string().contains("empty or oversized text"));

        for private_fragment in ["private_token", "secret-value", "\"id\"", "\"text\""] {
            let error = audit_jsonl(
                &runtime(),
                &mut Cursor::new(
                    "{\"id\":\"case\",\"text\":\"hello\",\"private_token\":\"secret-value\"}\n",
                ),
            )
            .unwrap_err()
            .to_string();
            assert!(
                !error.contains(private_fragment),
                "parser error exposed `{private_fragment}`: {error}"
            );
        }
    }

    #[test]
    fn jsonl_limits_count_physical_lines_and_total_bytes_without_draining() {
        let runtime = runtime();
        let oversized_input = format!("{}\nunused-tail", "x".repeat(64));
        let mut oversized_reader = Cursor::new(oversized_input.as_bytes());
        let error = audit_jsonl_with_limits(
            &runtime,
            &mut oversized_reader,
            AuditLimits {
                maximum_rows: 10,
                maximum_physical_lines: 10,
                maximum_line_bytes: 16,
                maximum_total_bytes: 100,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("per-line byte boundary"));
        assert_eq!(oversized_reader.position(), 17);
        assert!(oversized_reader.position() < oversized_input.len() as u64);

        let error = audit_jsonl_with_limits(
            &runtime,
            &mut Cursor::new("\n\n\n"),
            AuditLimits {
                maximum_rows: 10,
                maximum_physical_lines: 2,
                maximum_line_bytes: 16,
                maximum_total_bytes: 100,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds 2 physical lines"));

        let error = audit_jsonl_with_limits(
            &runtime,
            &mut Cursor::new("\n\n"),
            AuditLimits {
                maximum_rows: 10,
                maximum_physical_lines: 10,
                maximum_line_bytes: 16,
                maximum_total_bytes: 1,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("1 byte total boundary"));
    }

    #[test]
    fn perturbations_are_bounded_and_family_semantics_are_explicit() {
        let source = "Useful systems remain inspectable";
        for perturbation in perturbations() {
            let variant = (perturbation.apply)(source).unwrap();
            assert_ne!(variant, source);
            assert!(variant.chars().count() <= crate::MAX_INPUT_CHARS);
        }
        assert_eq!(
            perturbations()
                .iter()
                .filter(|item| item.family == PerturbationFamily::Formatting)
                .count(),
            4
        );
        assert_eq!(
            perturbations()
                .iter()
                .filter(|item| item.family == PerturbationFamily::Typographic)
                .count(),
            3
        );
    }

    #[test]
    fn gate_validates_ranges_and_reports_all_selected_failures() {
        let mut report = audit_jsonl(
            &runtime(),
            &mut Cursor::new("{\"id\":\"case\",\"text\":\"A deliberate longer sentence\"}\n"),
        )
        .unwrap();
        let invalid = RobustnessGate {
            minimum_formatting_decision_agreement: f64::NAN,
            ..RobustnessGate::default()
        };
        assert!(invalid.validate().is_err());

        let raw = report.raw_gate_evidence.as_mut().unwrap();
        raw.typographic.label_agreements = 0;
        raw.typographic.decision_agreements = 0;
        raw.typographic.maximum_js_divergence = 0.5;
        report.typographic = raw.typographic.report();
        let strict_typographic = RobustnessGate {
            minimum_typographic_label_agreement: Some(1.0),
            minimum_typographic_decision_agreement: Some(1.0),
            maximum_typographic_js_divergence: Some(0.1),
            ..RobustnessGate::default()
        };
        let error = strict_typographic.enforce(&report).unwrap_err().to_string();
        assert!(error.contains("typographic label agreement"));
        assert!(error.contains("typographic decision agreement"));
        assert!(error.contains("typographic JS divergence"));

        let mut hidden_label_flip = report;
        let raw = hidden_label_flip.raw_gate_evidence.as_mut().unwrap();
        raw.formatting.label_agreements = 0;
        raw.formatting.decision_agreements = raw.formatting.evaluated_variants;
        raw.formatting.maximum_js_divergence = 0.0;
        hidden_label_flip.formatting = raw.formatting.report();
        assert!(RobustnessGate::default()
            .enforce(&hidden_label_flip)
            .unwrap_err()
            .to_string()
            .contains("formatting label agreement"));
    }

    #[test]
    fn gate_uses_unrounded_evidence_and_rejects_deserialized_reports() {
        let mut report = audit_jsonl(
            &runtime(),
            &mut Cursor::new("{\"id\":\"case\",\"text\":\"A deliberate longer sentence\"}\n"),
        )
        .unwrap();
        let raw = report.raw_gate_evidence.as_mut().unwrap();
        raw.formatting.maximum_js_divergence = 0.4 / REPORTING_SCALE;
        report.formatting = raw.formatting.report();
        assert_eq!(report.formatting.maximum_normalized_js_divergence, 0.0);
        assert!(RobustnessGate::default()
            .enforce(&report)
            .unwrap_err()
            .to_string()
            .contains("formatting JS divergence"));

        let serialized = serde_json::to_string(&report).unwrap();
        let restored: RobustnessReport = serde_json::from_str(&serialized).unwrap();
        assert!(RobustnessGate::default()
            .enforce(&restored)
            .unwrap_err()
            .to_string()
            .contains("raw audit evidence is unavailable"));
    }

    #[test]
    fn only_a_verified_bundle_can_claim_the_bundle_id_test_population() {
        let bundle_report = audit_bundle_id_test(embedded_bundle().unwrap()).unwrap();
        assert_eq!(bundle_report.population, RobustnessPopulation::BundleIdTest);
        assert_eq!(bundle_report.input_count, 70);

        let provided_report = audit_cases(
            &runtime(),
            [RobustnessCase {
                id: "case".into(),
                text: "A deliberate fictional sentence".into(),
            }],
        )
        .unwrap();
        assert_eq!(
            provided_report.population,
            RobustnessPopulation::ProvidedCases
        );
    }
}
