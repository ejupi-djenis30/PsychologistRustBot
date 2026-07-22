use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static TEMP_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eliza-lab-cli-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_eliza-lab")
}

fn project_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn run_train(directory: &TestDirectory, model: &str, report: &str) -> Output {
    Command::new(binary())
        .current_dir(directory.path())
        .args([
            "train",
            "--dataset",
            project_path("fixtures/intents-v1.tsv").to_str().unwrap(),
            "--ood",
            project_path("fixtures/ood-v1.tsv").to_str().unwrap(),
            "--output",
            model,
            "--report",
            report,
        ])
        .output()
        .unwrap()
}

#[test]
fn legacy_once_keeps_its_stable_trace_contract() {
    let output = Command::new(binary())
        .args(["--once", "I feel calm today"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("rule=feeling-reflection turn=1"));
    assert!(!stdout.contains("boundary="));
}

#[test]
fn current_directory_outputs_are_byte_reproducible() {
    let directory = TestDirectory::new();
    let first = run_train(&directory, "model-a.json", "report-a.json");
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = run_train(&directory, "model-b.json", "report-b.json");
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );

    assert_eq!(
        fs::read(directory.path().join("model-a.json")).unwrap(),
        fs::read(directory.path().join("model-b.json")).unwrap()
    );
    assert_eq!(
        fs::read(directory.path().join("report-a.json")).unwrap(),
        fs::read(directory.path().join("report-b.json")).unwrap()
    );
}

#[test]
fn training_refuses_to_overwrite_an_input_or_alias_its_outputs() {
    let dataset = project_path("fixtures/intents-v1.tsv");
    let original = fs::read(&dataset).unwrap();
    let output = Command::new(binary())
        .args([
            "train",
            "--dataset",
            dataset.to_str().unwrap(),
            "--ood",
            project_path("fixtures/ood-v1.tsv").to_str().unwrap(),
            "--output",
            dataset.to_str().unwrap(),
            "--report",
            dataset.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("collides"));
    assert_eq!(fs::read(dataset).unwrap(), original);
}

#[test]
fn training_rejects_threshold_options_instead_of_silently_overriding_them() {
    let directory = TestDirectory::new();
    let dataset = project_path("fixtures/intents-v1.tsv");
    let ood = project_path("fixtures/ood-v1.tsv");

    for (index, option) in ["--minimum-confidence", "--minimum-margin"]
        .into_iter()
        .enumerate()
    {
        let model = format!("model-{index}.json");
        let report = format!("report-{index}.json");
        let output = Command::new(binary())
            .current_dir(directory.path())
            .args([
                "train",
                "--dataset",
                dataset.to_str().unwrap(),
                "--ood",
                ood.to_str().unwrap(),
                "--output",
                &model,
                "--report",
                &report,
                option,
                "0.50",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains(&format!("unknown train option `{option}`")));
        assert!(!directory.path().join(model).exists());
        assert!(!directory.path().join(report).exists());
    }
}

#[test]
fn inference_json_keeps_probabilities_contributions_and_hard_safety_boundary() {
    let model = project_path("models/eliza-intent-v1.json");
    let inferred = Command::new(binary())
        .args([
            "infer",
            "--model",
            model.to_str().unwrap(),
            "--json",
            "Today I feel calm",
        ])
        .output()
        .unwrap();
    assert!(inferred.status.success());
    let inference: Value = serde_json::from_slice(&inferred.stdout).unwrap();
    assert_eq!(inference["boundary"], "ml-intent");
    assert_eq!(inference["model"]["accepted"], true);
    assert_eq!(
        inference["model"]["probabilities"]
            .as_object()
            .unwrap()
            .len(),
        7
    );
    let contribution = &inference["model"]["top_features"][0];
    assert!(contribution["feature"].is_string());
    assert!(contribution["value"].is_number());
    assert!(contribution["weight"].is_number());
    assert!(contribution["contribution"].is_number());

    let safety = Command::new(binary())
        .args([
            "infer",
            "--model",
            model.to_str().unwrap(),
            "--json",
            "I want to die",
        ])
        .output()
        .unwrap();
    assert!(safety.status.success());
    let safety: Value = serde_json::from_slice(&safety.stdout).unwrap();
    assert_eq!(safety["boundary"], "safety-boundary");
    assert!(safety["model"].is_null());
}
