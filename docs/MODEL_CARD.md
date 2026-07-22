# ELIZA intent model card

## Artifact identity

| Field | Value |
| --- | --- |
| Model kind | `eliza-intent-softmax` |
| Model version | `1.0.0` |
| Serialization schema | `1` |
| Application release | `1.2.0` |
| Dataset fingerprint | `fnv1a64:e75750b1b0a83a78` |
| Split seed | `20260722` |
| Implementation | Rust, local CPU inference |

The model version and application version are separate on purpose. Model `1.0.0` is the first
stable weight format; ELIZA Lab `1.2.0` is the application release that introduces it. A future
CLI or site release can keep using the same model artifact without pretending that its weights
changed.

The FNV-1a fingerprint identifies a reproducible dataset snapshot. It is not a cryptographic
integrity proof. Release archives and source commits use SHA-256 and GitHub attestations instead.

## Intended use

This model demonstrates a complete, inspectable text-classification pipeline:

- validate a versioned TSV corpus;
- create a deterministic stratified split;
- fit a vocabulary on training text only;
- learn multiclass weights;
- serialize and reload the model;
- report class metrics and a confusion matrix;
- abstain when confidence or the top-two margin is too low;
- expose positive feature contributions for each prediction.

It classifies short, fictional English prompts into seven narrow interaction labels: `feeling`,
`goal`, `greeting`, `observation`, `ownership`, `question`, and `reason`.

## Prohibited and unsuitable uses

Do not use this model for therapy, diagnosis, triage, risk assessment, crisis detection, medical
advice, moderation, employment decisions, or any decision about a person. It was not trained on
real conversations, clinical language, demographic groups, dialect benchmarks, adversarial
examples, or multilingual data.

The explicit safety-phrase boundary in the dialogue shell runs before this model. That boundary
is a product stop condition, not an ML safety classifier, and it can have false positives and
false negatives.

## Data

The supervised corpus contains 112 purpose-written synthetic examples, exactly 16 per class. It
contains no imported chat logs, health records, names, account data, or other personal material.
The deterministic split produces 91 training rows and 21 holdout rows, with three holdout rows
per class.

A separate unlabeled OOD fixture contains 20 synthetic requests from unrelated domains such as
weather, travel, chemistry, finance, music, and code. It is used only when selecting abstention
thresholds. It has no target labels, so ELIZA Lab reports coverage and abstention—not OOD accuracy.

See [DATASET.md](DATASET.md) for the schema, class definitions, validation rules, and split
contract.

## Features and learning algorithm

The vectorizer extracts word unigrams and bigrams plus character 3-, 4-, and 5-grams. It computes
smoothed inverse document frequency from the training split, retains the 512 highest-document-
frequency features, applies logarithmic term frequency, and L2-normalizes each sparse vector.

The classifier is multinomial logistic regression trained with deterministic full-batch gradient
descent for 600 epochs. The initial learning rate is `0.8`, decays with the epoch count, and the
weight update applies an L2 penalty of `0.0005`. Parameters are quantized to twelve decimal places
after each update so identical inputs produce byte-identical model and report JSON.

## Abstention calibration

The operating point is selected by a deterministic grid search over confidence and margin. It uses
only the 91 training rows and the separate 20-row OOD fixture. Candidate thresholds must retain at
least `0.98` selective accuracy on training rows and accept zero rows in that OOD fixture. The
selected thresholds are:

- minimum confidence: `0.45`;
- minimum top-two margin: `0.20`.

The 21-row holdout is not used for vocabulary fitting, optimization, or threshold selection. The
checked-in report records `holdout_used_for_calibration: false` and lists every holdout prediction.

## Evaluation results

These values come from [`reports/eliza-intent-v1.json`](../reports/eliza-intent-v1.json), generated
by the checked-in training command.

| Set | Rows | Accuracy | Macro F1 | Coverage | Selective accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 91 | 1.000 | 1.000 | 79/91 (86.8%) | 1.000 |
| Holdout | 21 | 14/21 (66.7%) | 0.661 | 7/21 (33.3%) | 6/7 (85.7%) |

| Holdout class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| feeling | 1.000 | 1.000 | 1.000 | 3 |
| goal | 0.500 | 0.667 | 0.571 | 3 |
| greeting | 1.000 | 0.333 | 0.500 | 3 |
| observation | 0.400 | 0.667 | 0.500 | 3 |
| ownership | 0.750 | 1.000 | 0.857 | 3 |
| question | 0.500 | 0.333 | 0.400 | 3 |
| reason | 1.000 | 0.667 | 0.800 | 3 |

The OOD fixture produced 0 accepted and 20 rejected predictions. This only describes those 20
synthetic rows. It does not establish a general OOD-detection rate.

## Limitations

The holdout is very small: 21 rows. Raw holdout accuracy is about `0.667`, and the model abstains on
14 of those rows. These numbers are useful for checking the pipeline, not for claiming production
NLP quality or generalization. The examples share a concise written style, the labels are defined by
this demo, and feature contributions explain a linear score rather than human meaning.

Softmax confidence is not calibrated probability in the statistical sense. The thresholds are an
operating rule for this artifact, not a guarantee. Similar wording can move predictions sharply,
and unfamiliar inputs can still pass the gate.

## Reproduce and inspect

```bash
cargo run --locked -- train
cargo run --locked -- evaluate --json
cargo run --locked -- infer --json "Today I feel calm"
```

Two identical training runs are tested for byte-identical model and report files. Rust and browser
inference share a parity fixture, including accepted and abstained examples. Strict model loading
rejects unknown fields, unsupported schema versions, non-finite parameters, and non-rectangular
weights.
