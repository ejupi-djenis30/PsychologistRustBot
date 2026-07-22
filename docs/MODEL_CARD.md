# ELIZA Lab model card

## Open-set model v3.0.0

ELIZA Lab v3 is a small, deterministic intent classifier with an explicit abstention policy. It is
built to make an ML experiment inspectable, reproducible and difficult to overstate. It is not a
general language model and it is not a mental-health product.

| Field | Value |
| --- | --- |
| Model kind | `eliza-open-set-linear` |
| Model / bundle / schema | `3.0.0` / `3.0.0` / `3` |
| Implementation | Rust, deterministic local CPU training and inference |
| Frozen experiment seed | `4043100207104787`, derived from the four raw fixture hashes |
| Released artifacts | [`artifacts/eliza-open-set-v3`](../artifacts/eliza-open-set-v3) |
| Exact provenance | `manifest.json`, `model.json`, `policy.json`, `metrics.json`, `split-plan.json` |

The artifact JSON files are authoritative for the dataset hashes, seed, selected hyperparameters,
temperature, thresholds and every point estimate. The site verifies those files before displaying a
number or enabling inference.

## Intended use

Use this project to study or demonstrate:

- strict dataset contracts and leakage-resistant family splits;
- training-only TF-IDF and multinomial logistic regression;
- development-only candidate and operating-point selection;
- held-out probability calibration;
- open-set abstention and per-stratum OOD measurement;
- a held-out paired test for meaning-changing, lexically similar prompts;
- deterministic baseline and bootstrap reconstruction;
- cross-runtime, cryptographically linked model artifacts;
- local explanations for a linear classifier.

The seven narrow labels are `feeling`, `goal`, `greeting`, `observation`, `ownership`, `question`
and `reason`.

## Data and partitions

All prompts are synthetic and English-only. The supervised corpus has 525 rows arranged as 105
equal five-prompt families. The frozen split contains 315 training rows and 70 rows in each of
development, calibration and ID-test.

OOD-development and OOD-test each contain 36 rows, twelve three-prompt families and six broader
domains. Their domain groups are disjoint. Semantic, capability and noise strata each contribute
twelve rows per population. A separate contrast test contains fourteen two-prompt pairs and equal
support for all seven labels. It never participates in model or policy selection.

See [DATASET.md](DATASET.md) for schemas, similarity thresholds, workload bounds and the exact role
of each partition.

## Learning and selection

The model uses TF-IDF word uni- and bigrams plus character 3-, 4- and 5-grams. Full-batch
multinomial logistic regression runs deterministically. The source-declared model grid contains
three feature budgets and three L2 penalties. A `0.005` macro-F1 equivalence band prefers the
simpler candidate before auxiliary development metrics.

Temperature scaling sees only calibration rows. The abstention policy comes from a fixed 7 × 7
grid over confidence and top-two probability margin. It sees development and OOD-development, not
either final test.

## Evaluation contract

The checked-in `metrics.json` contains:

- the complete ID-test prediction ledger and confusion matrix;
- accuracy, macro F1, NLL, multiclass Brier, ECE, coverage and AURC;
- majority-class and Laplace unigram Naive Bayes baselines trained on training only;
- aggregate and per-stratum OOD coverage, AUROC, AUPR and FPR at 95% TPR;
- contrast row accuracy, macro F1, pair accuracy, prediction-flip rate and coverage;
- 1,000 deterministic 95% cluster-bootstrap intervals;
- learned-minus-unigram deltas and explicit limitations.

ID bootstrap samples held-out families within labels. OOD bootstrap samples broader domains. These
intervals describe uncertainty inside the synthetic fixtures; they do not create external validity.

## Frozen results

These values come from the one final 1,000-resample run made after the four fixtures, seed and
selection policy were frozen. Brackets show the cluster-bootstrap 95% interval where one is
available.

| Measure | Frozen result |
| --- | ---: |
| ID-test accuracy | `0.829` [`0.757`, `0.900`] |
| ID-test macro F1 | `0.823` [`0.746`, `0.893`] |
| ID decision coverage | `44 / 70` (`0.629`) |
| ID selective accuracy | `1.000` |
| ID negative log-likelihood | `0.567` [`0.390`, `0.750`] |
| Unigram Naive Bayes accuracy / macro F1 | `0.800` / `0.793` |
| Learned-minus-unigram accuracy / macro F1 | `+0.029` / `+0.030` |
| OOD-test AUROC | `0.803` [`0.689`, `0.906`] |
| OOD-test accepted coverage | `4 / 36` (`0.111`) |
| OOD-test FPR at 95% TPR | `0.778` [`0.361`, `0.972`] |
| Contrast row accuracy / macro F1 | `19 / 28` (`0.679`) / `0.640` |
| Contrast pair accuracy | `6 / 14` (`0.429`) |
| Contrast prediction-flip rate | `8 / 14` (`0.571`) |
| Contrast accepted coverage | `17 / 28` (`0.607`) |

Development selected a 2,048-feature model with L2 penalty `0.002`. Calibration selected
temperature `0.184848849699`; the fixed threshold grid selected confidence `0.70` and top-two
margin `0.40`.

The learned model beats the unigram baseline, but only modestly. The paired result is harder: fewer
than half the pairs are fully correct. OOD AUROC is useful as a ranking measure, while the high and
wide FPR-at-95%-TPR interval shows that high-recall separation is not reliable here. These are
limitations of the frozen result, not targets for post-test tuning.

## Audit history

An earlier prerelease v3 run was discarded before release. Review found that family size correlated
with partition role and that its OOD prompts were singleton families. The corpus and OOD protocol
were rebuilt before deriving a fresh seed and opening the final tests. No result from that discarded
run is presented as the current model.

On Windows, the first final CLI launch was rejected by local Application Control before the process
executed (`os error 4551`). The final evaluation therefore invoked the same public
`run_open_set_experiment` path once through Rust's permitted test harness with the frozen default
configuration and 1,000 resamples. No result existed before that invocation, and no fixture, seed,
grid or policy was changed afterward. Rust and browser verifiers then reproduced every ledger from
the resulting bundle.

## Limitations

- Synthetic English prompts cannot establish real-world generalization.
- Development is reused for model candidate and threshold selection, so selection optimism remains
  possible.
- OOD has only six broader test domains and three designed strata.
- A unigram baseline can reveal lexical shortcuts, but it cannot prove their absence.
- The paired contrast set is small, synthetic and source-authored; it is not an external benchmark.
- Feature contributions explain this linear model's score, not human meaning.
- Confidence and abstention thresholds are operating rules, not guarantees.

## Prohibited uses

Do not use ELIZA Lab for therapy, diagnosis, triage, risk assessment, crisis detection, medical
advice, moderation, employment decisions or any decision about a person. It was not trained on real
conversations, clinical language, demographic groups, dialect benchmarks or multilingual data.

The explicit safety-phrase boundary runs before learned inference. It can miss urgent wording and
match benign text. It is a product stop condition, not a safety classifier.

## Reproduce and inspect

```bash
cargo run --locked -- train-v3 --output target/open-set-v3
cargo run --locked -- bundle verify --bundle artifacts/eliza-open-set-v3
cargo run --locked -- bundle reproduce --bundle artifacts/eliza-open-set-v3
cargo run --locked -- infer --bundle artifacts/eliza-open-set-v3 --json "Today I feel calm"
```

Legacy model v1 remains readable only for compatibility:

```bash
cargo run --locked -- infer --legacy-v1 --model models/eliza-intent-v1.json --json "Today I feel calm"
```
