# Architecture

ELIZA Lab has five explicit layers:

1. `src/ml.rs` owns data validation, splitting, feature extraction, training, evaluation,
   calibration, inference, explanation, and strict JSON model loading.
2. `src/lib.rs` owns bounded dialogue behavior. Empty and oversized input plus the explicit safety
   stop run before ML inference. Rejected predictions use a deterministic fallback.
3. `src/main.rs` exposes dataset checks, training, evaluation, single inference, interactive chat,
   and the retained rule-only mode.
4. `site/` implements the same inference math in JavaScript and presents the prediction trace. It
   loads the checked-in model as a same-origin static asset. A load or validation failure is shown
   as `RULE FALLBACK`; it is never silently described as learned inference.
5. `src/open_set.rs` owns the version-two experimental path: grouped data, a four-way typed split,
   temperature scaling, development-only threshold selection, independent ID/OOD tests,
   deterministic bootstrap statistics, SHA-256-linked artifact bundles and compiled batch
   inference. It does not change the released v1 model contract.

## Data flow

```text
validated TSV
    → seeded stratified split
        → training-only TF-IDF vocabulary
            → multinomial logistic regression
                → versioned JSON model
                    → confidence + margin gate
                        → accepted intent or deterministic abstention
```

The browser sends no prompt to a server. Loading the page downloads the static model alongside its
CSS and JavaScript; subsequent feature extraction and prediction happen in the tab. The CLI needs
no network for training or inference once Rust dependencies are available locally.

## Open-set v2 data flow

```text
grouped supervised TSV
    → group-stratified SplitPlan
        ├── train → vocabulary + classifier weights
        ├── calibration → temperature only
        ├── development ─┐
        │                 ├── confidence + probability-margin policy
        └── ID-test       │       (never visible to selection)

OOD-development ─────────┘
OOD-test → final OOD metrics only

frozen model + frozen policy + untouched tests
    → ECE / Brier / NLL / risk-coverage / AURC
    → OOD AUROC / AUPR / FPR@95TPR
    → deterministic label-stratified row bootstrap 95% intervals
    → SHA-256-linked bundle manifest
```

The temperature-calibration function accepts only the calibration slice. The threshold-selection
function accepts only development plus OOD-development. This makes the no-test-leakage rule part of
the Rust type boundary rather than a comment or caller convention.

## Compatibility boundaries

- Application version: Cargo and release archives, currently `1.3.0`.
- Model version: learned weight semantics, currently `1.0.0`.
- Model schema: serialized field layout, currently `1`.
- Open-set bundle/model schema: experimental artifact layout `2`, model `2.0.0`.
- Dataset fingerprint: canonical supervised corpus content.

The loader rejects model artifacts outside these supported boundaries instead of guessing.

## Failure behavior

- Invalid or duplicate TSV rows stop training before an output is written.
- CLI output and report paths may not collide with each other or either input fixture.
- Model and report are serialized and synced before either destination changes. A recoverable pair
  transaction restores both previous artifacts if either install operation returns an error. This
  is failure recovery inside one process, not a claim of cross-file atomicity after power loss or
  process termination.
- Unknown JSON fields, versions, malformed fingerprints, divergent vectorizer configs, invalid
  feature names, duplicate vocabulary entries, non-finite or overflow-scale values, and non-rectangular parameters
  are rejected in both Rust and JavaScript.
- An empty feature vector always abstains.
- Safety language never reaches learned inference in the dialogue layer.
- V2 bundle loading rejects symlinks, oversized JSON, unknown fields, malformed shapes, provenance
  disagreement and any SHA-256 mismatch before constructing `CompiledModel`.
- `CompiledModel` builds the vocabulary index once. Its contrastive explanation is tested so bias
  delta plus all feature contributions reconstruct the exact top-two logit margin.
