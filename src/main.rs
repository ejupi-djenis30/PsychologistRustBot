use eliza_lab::ml::{
    write_training_artifacts, Dataset, EvaluationMetrics, IntentModel, MlError, OodDataset,
    OodMetrics, TrainingConfig,
};
use eliza_lab::{ElizaEngine, Reply, MAX_INPUT_CHARS};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};

const DEFAULT_DATASET: &str = "fixtures/intents-v1.tsv";
const DEFAULT_OOD_DATASET: &str = "fixtures/ood-v1.tsv";
const DEFAULT_MODEL: &str = "models/eliza-intent-v1.json";
const DEFAULT_REPORT: &str = "reports/eliza-intent-v1.json";
const MAX_INPUT_BYTES: usize = MAX_INPUT_CHARS * 4;
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

enum InputLine {
    Text(String),
    TooLong,
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Serialize)]
struct InferenceOutput<'a> {
    response: &'a str,
    boundary: &'a str,
    turn: usize,
    model: Option<InferenceTrace<'a>>,
}

#[derive(Serialize)]
struct InferenceTrace<'a> {
    version: &'a str,
    label: &'a str,
    accepted: bool,
    confidence: f64,
    margin: f64,
    probabilities: &'a std::collections::BTreeMap<String, f64>,
    top_features: &'a [eliza_lab::ml::FeatureContribution],
}

#[derive(Serialize)]
struct EvaluationOutput {
    dataset_fingerprint: String,
    holdout: EvaluationMetrics,
    out_of_domain: OodMetrics,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("train") => train_command(&arguments[1..]),
        Some("evaluate") => evaluate_command(&arguments[1..]),
        Some("infer") => infer_command(&arguments[1..]),
        Some("chat") => chat_command(&arguments[1..]),
        Some("dataset") => dataset_command(&arguments[1..]),
        Some("--once") => legacy_once(&arguments[1..]),
        Some("--help" | "-h" | "help") => {
            print_help();
            Ok(())
        }
        Some(command) => Err(CliError(format!(
            "unknown command `{command}`; run `eliza-lab --help`"
        ))
        .into()),
        None => interactive(None),
    }
}

fn train_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut output_path = PathBuf::from(DEFAULT_MODEL);
    let mut report_path = PathBuf::from(DEFAULT_REPORT);
    let mut ood_path = None;
    let mut config = TrainingConfig::default();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--output" => output_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--report" => report_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--ood" => ood_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--seed" => config.seed = parse_option(arguments, &mut index, "seed")?,
            "--epochs" => config.epochs = parse_option(arguments, &mut index, "epochs")?,
            "--learning-rate" => {
                config.learning_rate = parse_option(arguments, &mut index, "learning rate")?
            }
            "--l2" => config.l2_penalty = parse_option(arguments, &mut index, "L2 penalty")?,
            "--holdout" => {
                config.holdout_fraction = parse_option(arguments, &mut index, "holdout fraction")?
            }
            "--max-features" => {
                config.vectorizer.max_features =
                    parse_option(arguments, &mut index, "max features")?
            }
            "--help" | "-h" => {
                print_train_help();
                return Ok(());
            }
            option => return Err(CliError(format!("unknown train option `{option}`")).into()),
        }
        index += 1;
    }

    let mut checked_paths = vec![
        ("model output", output_path.as_path()),
        ("report output", report_path.as_path()),
    ];
    if let Some(path) = dataset_path.as_deref() {
        checked_paths.push(("dataset", path));
    }
    if let Some(path) = ood_path.as_deref() {
        checked_paths.push(("OOD dataset", path));
    }
    ensure_distinct_paths(&checked_paths)?;

    let dataset = load_dataset(dataset_path.as_deref())?;
    let ood_dataset = load_ood_dataset(ood_path.as_deref())?;
    let (mut model, mut report) = IntentModel::train(&dataset, config)?;
    let split = dataset.stratified_split(
        model.training_config.holdout_fraction,
        model.training_config.seed,
    )?;
    let calibration = model.calibrate_thresholds(&split, &ood_dataset, 0.0, 0.98)?;
    report.training_metrics = model.evaluate(split.training_examples())?;
    report.holdout_metrics = model.evaluate(split.holdout_examples())?;
    report.calibration = Some(calibration);
    report.ood_metrics = Some(model.evaluate_ood(ood_dataset.examples())?);
    write_training_artifacts(&model, &output_path, &report, &report_path)?;

    println!("model       {}", output_path.display());
    println!("report      {}", report_path.display());
    println!("fingerprint {}", report.dataset_fingerprint);
    println!(
        "split        {} train / {} holdout (seed {})",
        report.training_example_ids.len(),
        report.holdout_example_ids.len(),
        report.seed
    );
    println!("features     {}", report.vocabulary_size);
    println!(
        "thresholds   confidence={:.2} margin={:.2} (calibrated without holdout)",
        model.training_config.thresholds.minimum_confidence,
        model.training_config.thresholds.minimum_margin,
    );
    print_metrics("train", &report.training_metrics);
    print_metrics("holdout", &report.holdout_metrics);
    print_ood_metrics(
        "out-of-domain",
        report
            .ood_metrics
            .as_ref()
            .expect("the CLI always records OOD evaluation"),
    );
    Ok(())
}

fn evaluate_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut model_path = None;
    let mut ood_path = None;
    let mut json = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--ood" => ood_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--json" => json = true,
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab evaluate [--dataset PATH] [--model PATH] [--ood PATH] [--json]\n\
                     Evaluates the deterministic holdout and a separate unlabeled OOD corpus."
                );
                return Ok(());
            }
            option => return Err(CliError(format!("unknown evaluate option `{option}`")).into()),
        }
        index += 1;
    }

    let dataset = load_dataset(dataset_path.as_deref())?;
    let model = load_model(model_path.as_deref())?;
    verify_fingerprint(&dataset, &model)?;
    let split = dataset.stratified_split(
        model.training_config.holdout_fraction,
        model.training_config.seed,
    )?;
    let metrics = model.evaluate(split.holdout_examples())?;
    let ood_dataset = load_ood_dataset(ood_path.as_deref())?;
    let ood_metrics = model.evaluate_ood(ood_dataset.examples())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&EvaluationOutput {
                dataset_fingerprint: model.dataset_fingerprint.clone(),
                holdout: metrics,
                out_of_domain: ood_metrics,
            })?
        );
    } else {
        println!(
            "model       {}",
            model_path
                .as_deref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<embedded eliza-intent-v1>".into())
        );
        println!("fingerprint {}", model.dataset_fingerprint);
        println!("holdout ids {}", split.holdout_examples().len());
        print_metrics("holdout", &metrics);
        print_confusion(&metrics.labels, &metrics.confusion_matrix);
        print_ood_metrics("out-of-domain", &ood_metrics);
    }
    Ok(())
}

fn infer_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut model_path = None;
    let mut json = false;
    let mut prompt = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--json" => json = true,
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab infer [--model PATH] [--json] <fictional prompt>\n\
                     Safety and input boundaries run before learned inference."
                );
                return Ok(());
            }
            value if value.starts_with('-') => {
                return Err(CliError(format!("unknown infer option `{value}`")).into())
            }
            value => prompt.push(value.to_owned()),
        }
        index += 1;
    }
    if prompt.is_empty() {
        return Err(CliError("infer requires a fictional prompt".into()).into());
    }
    let model = load_model(model_path.as_deref())?;
    let mut engine = ElizaEngine::new();
    let reply = engine.respond_with_model(&prompt.join(" "), &model);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&inference_output(&reply))?
        );
    } else {
        print_reply(&reply);
    }
    Ok(())
}

fn chat_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut model_path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--help" | "-h" => {
                println!("Usage: eliza-lab chat [--model PATH]");
                return Ok(());
            }
            option => return Err(CliError(format!("unknown chat option `{option}`")).into()),
        }
        index += 1;
    }
    interactive(Some(load_model(model_path.as_deref())?))
}

fn dataset_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "check" => {}
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--help" | "-h" => {
                println!("Usage: eliza-lab dataset check [--dataset PATH]");
                return Ok(());
            }
            option => return Err(CliError(format!("unknown dataset option `{option}`")).into()),
        }
        index += 1;
    }
    let dataset = load_dataset(dataset_path.as_deref())?;
    println!(
        "dataset     {}",
        dataset_path
            .as_deref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<embedded intents-v1>".into())
    );
    println!("fingerprint {}", dataset.fingerprint());
    println!("examples    {}", dataset.examples().len());
    for (label, count) in dataset.class_counts() {
        println!("class        {label}: {count}");
    }
    Ok(())
}

fn legacy_once(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut engine = ElizaEngine::new();
    let reply = engine.respond(&arguments.join(" "));
    println!("{}", reply.text);
    println!("rule={} turn={}", reply.rule_id, reply.turn);
    Ok(())
}

fn interactive(model: Option<IntentModel>) -> Result<(), Box<dyn Error>> {
    println!("ELIZA Lab — local dialogue classification you can inspect");
    println!("Educational software, not therapy or medical advice. No transcript is stored.");
    println!(
        "Mode: {}. Type /quit to leave.\n",
        if model.is_some() {
            "learned intent model with deterministic abstention"
        } else {
            "deterministic rule engine"
        }
    );

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut engine = ElizaEngine::new();
    let mut stdout = io::stdout().lock();
    write!(stdout, "you > ")?;
    stdout.flush()?;

    while let Some(line) = read_bounded_line(&mut stdin)? {
        let input = match line {
            InputLine::Text(input) => input,
            InputLine::TooLong => "x".repeat(MAX_INPUT_CHARS + 1),
        };
        if matches!(input.trim(), "/quit" | "/exit") {
            writeln!(stdout, "eliza > Goodbye.")?;
            break;
        }

        let response = match &model {
            Some(model) => engine.respond_with_model(&input, model),
            None => engine.respond(&input),
        };
        writeln!(stdout, "eliza > {}", response.text)?;
        write_trace(&mut stdout, &response)?;
        write!(stdout, "you > ")?;
        stdout.flush()?;
    }
    Ok(())
}

fn read_bounded_line(reader: &mut impl BufRead) -> io::Result<Option<InputLine>> {
    let mut bytes = Vec::with_capacity(MAX_INPUT_BYTES.min(256));
    let mut discarded = false;
    let mut saw_input = false;

    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !saw_input {
                return Ok(None);
            }
            break;
        }
        saw_input = true;
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_length = newline.unwrap_or(buffer.len());
        let remaining = (MAX_INPUT_BYTES + 2).saturating_sub(bytes.len());
        let copy_length = content_length.min(remaining);
        bytes.extend_from_slice(&buffer[..copy_length]);
        discarded |= content_length > remaining;

        let consumed = newline.map_or(buffer.len(), |position| position + 1);
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }

    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    if discarded || bytes.len() > MAX_INPUT_BYTES {
        return Ok(Some(InputLine::TooLong));
    }
    let text = String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(Some(InputLine::Text(text)))
}

fn inference_output(reply: &Reply) -> InferenceOutput<'_> {
    InferenceOutput {
        response: &reply.text,
        boundary: reply.rule_id,
        turn: reply.turn,
        model: reply.model_trace.as_ref().map(|trace| InferenceTrace {
            version: &trace.model_version,
            label: &trace.label,
            accepted: trace.accepted,
            confidence: trace.confidence,
            margin: trace.margin,
            probabilities: &trace.probabilities,
            top_features: &trace.top_features,
        }),
    }
}

fn print_reply(reply: &Reply) {
    println!("{}", reply.text);
    println!("boundary={} turn={}", reply.rule_id, reply.turn);
    if let Some(trace) = &reply.model_trace {
        println!(
            "model={} label={} accepted={} confidence={:.4} margin={:.4}",
            trace.model_version, trace.label, trace.accepted, trace.confidence, trace.margin
        );
        if !trace.top_features.is_empty() {
            println!(
                "features={}",
                trace
                    .top_features
                    .iter()
                    .map(|feature| format!("{}:{:.4}", feature.feature, feature.contribution))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}

fn write_trace(writer: &mut impl Write, reply: &Reply) -> io::Result<()> {
    writeln!(writer, "trace > {} (turn {})", reply.rule_id, reply.turn)?;
    if let Some(trace) = &reply.model_trace {
        writeln!(
            writer,
            "model > {} / accepted={} / confidence={:.3} / margin={:.3}",
            trace.label, trace.accepted, trace.confidence, trace.margin
        )?;
    }
    Ok(())
}

fn print_metrics(name: &str, metrics: &eliza_lab::ml::EvaluationMetrics) {
    println!(
        "{name:<12} n={} accuracy={:.4} macro_f1={:.4} log_loss={:.4} coverage={:.4} selective_accuracy={}",
        metrics.example_count,
        metrics.accuracy,
        metrics.macro_f1,
        metrics.log_loss,
        metrics.coverage,
        metrics
            .selective_accuracy
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".into())
    );
}

fn print_confusion(labels: &[String], matrix: &[Vec<usize>]) {
    println!("confusion matrix: rows=actual, columns=predicted");
    println!("labels: {}", labels.join(", "));
    for (label, row) in labels.iter().zip(matrix) {
        println!("{label:<12} {row:?}");
    }
}

fn print_ood_metrics(name: &str, metrics: &OodMetrics) {
    println!(
        "{name:<12} n={} accepted={} rejected={} coverage={:.4} abstention={:.4} mean_confidence={:.4}",
        metrics.example_count,
        metrics.accepted_examples,
        metrics.rejected_examples,
        metrics.coverage,
        metrics.abstention_rate,
        metrics.mean_confidence,
    );
}

fn verify_fingerprint(dataset: &Dataset, model: &IntentModel) -> Result<(), MlError> {
    let observed = dataset.fingerprint();
    if observed != model.dataset_fingerprint {
        return Err(MlError::InvalidDataset(format!(
            "fingerprint mismatch: model expects {}, dataset is {observed}",
            model.dataset_fingerprint
        )));
    }
    Ok(())
}

fn load_model(path: Option<&Path>) -> Result<IntentModel, MlError> {
    match path {
        Some(path) => IntentModel::read(path),
        None => IntentModel::bundled(),
    }
}

fn load_dataset(path: Option<&Path>) -> Result<Dataset, MlError> {
    match path {
        Some(path) => Dataset::read(path),
        None => Dataset::bundled(),
    }
}

fn load_ood_dataset(path: Option<&Path>) -> Result<OodDataset, MlError> {
    match path {
        Some(path) => OodDataset::read(path),
        None => OodDataset::bundled(),
    }
}

fn ensure_distinct_paths(paths: &[(&str, &Path)]) -> Result<(), CliError> {
    let mut seen = std::collections::HashMap::new();
    for (name, path) in paths {
        let key = collision_key(path)?;
        if let Some(previous) = seen.insert(key, *name) {
            return Err(CliError(format!(
                "{name} collides with {previous}; input and output paths must be distinct"
            )));
        }
    }
    Ok(())
}

fn collision_key(path: &Path) -> Result<String, CliError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| CliError(format!("cannot resolve current directory: {error}")))?
            .join(path)
    };

    let existing_ancestor = absolute
        .ancestors()
        .find(|ancestor| ancestor.exists())
        .ok_or_else(|| {
            CliError(format!(
                "cannot find an existing ancestor for {}",
                path.display()
            ))
        })?;
    let unresolved_suffix = absolute.strip_prefix(existing_ancestor).map_err(|error| {
        CliError(format!(
            "cannot resolve the suffix of {}: {error}",
            path.display()
        ))
    })?;
    let canonical_ancestor = existing_ancestor.canonicalize().map_err(|error| {
        CliError(format!(
            "cannot resolve existing ancestor {}: {error}",
            existing_ancestor.display()
        ))
    })?;
    let resolved = canonical_ancestor.join(unresolved_suffix);
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

fn option_value(arguments: &[String], index: &mut usize) -> Result<String, CliError> {
    let option = arguments[*index].clone();
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| CliError(format!("{option} requires a value")))
}

fn parse_option<T>(
    arguments: &[String],
    index: &mut usize,
    description: &str,
) -> Result<T, CliError>
where
    T: std::str::FromStr,
{
    let value = option_value(arguments, index)?;
    value
        .parse()
        .map_err(|_| CliError(format!("invalid {description} `{value}`")))
}

fn print_help() {
    println!(
        "ELIZA Lab {APP_VERSION}\n\
         Local, explainable intent classification with a non-clinical dialogue shell.\n\n\
         USAGE\n\
           eliza-lab train [options]\n\
           eliza-lab evaluate [options]\n\
           eliza-lab infer [options] <fictional prompt>\n\
           eliza-lab chat [--model PATH]\n\
           eliza-lab dataset check [--dataset PATH]\n\
           eliza-lab --once <prompt>       Legacy rule-only response\n\n\
         Run a command with --help for details. With no command, the rule-only shell starts."
    );
}

fn print_train_help() {
    println!(
        "Usage: eliza-lab train [options]\n\
         --dataset PATH              TSV input (default embedded {DEFAULT_DATASET})\n\
         --output PATH               Versioned model JSON (default {DEFAULT_MODEL})\n\
         --report PATH               Split and metric report (default {DEFAULT_REPORT})\n\
         --ood PATH                  OOD fixture (default embedded {DEFAULT_OOD_DATASET})\n\
         --seed INTEGER              Deterministic split seed\n\
         --epochs INTEGER            Full-batch gradient steps\n\
         --learning-rate FLOAT       Initial learning rate\n\
         --l2 FLOAT                  L2 penalty\n\
         --holdout FLOAT             Per-class holdout fraction\n\
         --max-features INTEGER      Vocabulary cap"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[cfg(unix)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(source, destination)
    }

    #[cfg(windows)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(source, destination)
    }

    #[test]
    fn bounded_reader_drains_an_oversized_line_before_the_next_prompt() {
        let input = format!("{}\nnext\n", "x".repeat(MAX_INPUT_BYTES + 20));
        let mut reader = Cursor::new(input.into_bytes());

        assert!(matches!(
            read_bounded_line(&mut reader).unwrap(),
            Some(InputLine::TooLong)
        ));
        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value, "next"),
            _ => panic!("the next bounded line should remain readable"),
        }
    }

    #[test]
    fn bounded_reader_accepts_a_maximum_size_utf8_line() {
        let input = format!("{}\r\n", "🙂".repeat(MAX_INPUT_CHARS));
        let mut reader = Cursor::new(input.into_bytes());

        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value.chars().count(), MAX_INPUT_CHARS),
            _ => panic!("a maximum-size UTF-8 line should be accepted"),
        }
    }

    #[test]
    fn fingerprint_guard_rejects_a_different_dataset() {
        let dataset = Dataset::read(std::path::Path::new(DEFAULT_DATASET)).unwrap();
        let (mut model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        model.dataset_fingerprint = "fnv1a64:0000000000000000".into();

        assert!(verify_fingerprint(&dataset, &model).is_err());
    }

    #[test]
    fn collision_keys_resolve_symlinked_parents_for_future_outputs() {
        let directory = std::env::temp_dir().join(format!(
            "eliza-collision-key-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let real_parent = directory.join("real");
        let alias_parent = directory.join("alias");
        std::fs::create_dir_all(&real_parent).unwrap();
        if let Err(error) = symlink_directory(&real_parent, &alias_parent) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                std::fs::remove_dir_all(directory).unwrap();
                return;
            }
            panic!("failed to create test directory symlink: {error}");
        }
        if std::fs::read_dir(&alias_parent).is_err() {
            #[cfg(unix)]
            std::fs::remove_file(&alias_parent).unwrap();
            #[cfg(windows)]
            std::fs::remove_dir(&alias_parent).unwrap();
            std::fs::remove_dir_all(directory).unwrap();
            return;
        }

        let real_output = real_parent.join("future/model.json");
        let aliased_output = alias_parent.join("future/model.json");
        assert_eq!(
            collision_key(&real_output).unwrap(),
            collision_key(&aliased_output).unwrap()
        );
        assert!(ensure_distinct_paths(&[
            ("model output", &real_output),
            ("report output", &aliased_output),
        ])
        .is_err());

        #[cfg(unix)]
        std::fs::remove_file(&alias_parent).unwrap();
        #[cfg(windows)]
        std::fs::remove_dir(&alias_parent).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }
}
