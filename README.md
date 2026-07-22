<p align="center">
  <img src="site/assets/eliza-lab-lockup.svg" width="720" alt="ELIZA Lab — local machine learning you can inspect" />
</p>

# ELIZA Lab

[![CI](https://github.com/ejupi-djenis30/PsychologistRustBot/actions/workflows/ci.yml/badge.svg)](https://github.com/ejupi-djenis30/PsychologistRustBot/actions/workflows/ci.yml)

> Train a real intent classifier locally, reproduce its evaluation, and inspect every decision.

This repository began as a Telegram “psychologist” bot. That framing was misleading, and the
implementation stored sensitive conversations in an insecure database. The project has been
rebuilt as **ELIZA Lab**: an educational Rust machine-learning pipeline and browser lab with no
accounts, prompt submission, transcript storage, diagnosis, or therapeutic claims.

[Open the interactive demo](https://ejupi-djenis30.github.io/PsychologistRustBot/) ·
[Model card](docs/MODEL_CARD.md) · [Dataset contract](docs/DATASET.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Safety model](SECURITY.md)

## What makes it useful

- **A complete learning pipeline:** strict TSV validation, a deterministic stratified split,
  training-only vocabulary fitting, TF-IDF features, multinomial logistic regression, evaluation,
  uncertainty calibration, serialization, and inference live in the repository.
- **Inspectable predictions:** JSON and browser traces expose all class probabilities, confidence,
  top-two margin, and the strongest positive feature contributions.
- **Reproducible artifacts:** two identical training runs produce byte-identical model and report
  files. The report records every split ID and every holdout prediction.
- **Real abstention:** thresholds are selected from training plus a separate OOD fixture without
  looking at the holdout. Inputs with weak evidence use a deterministic fallback.
- **Private by construction:** CLI and browser inference are local. Prompts are not stored or sent
  to a model service.
- **Hard non-clinical boundary:** input limits and explicit safety-stop phrases run before ML
  inference. The stop is not presented as crisis detection.

## Install a verified build

Download the archive for your system from the
[latest release](https://github.com/ejupi-djenis30/PsychologistRustBot/releases/latest):

| Platform | Release asset |
| --- | --- |
| Linux x64 | `eliza-lab-v<version>-linux-x86_64.tar.gz` |
| Windows x64 | `eliza-lab-v<version>-windows-x86_64.zip` |
| macOS Apple Silicon | `eliza-lab-v<version>-macos-aarch64.tar.gz` |
| macOS Intel | `eliza-lab-v<version>-macos-x86_64.tar.gz` |

Compare it with the matching `.sha256` file or `SHA256SUMS`, then verify its GitHub attestation:

```bash
gh attestation verify <downloaded-archive> --repo ejupi-djenis30/PsychologistRustBot
```

Extract the archive and run the included `eliza-lab` executable. Model `1.0.0` and the synthetic
fixtures are embedded, so inference, evaluation, and retraining need no separate model download.
The application version (`1.2.0`) and model version are intentionally independent.

## Run it

Inspect one learned prediction:

```bash
cargo run --locked -- infer --json "Today I feel calm"
```

Start an uncertainty-aware local session:

```bash
cargo run --locked -- chat
```

Rebuild the checked-in model and evaluation report from the embedded synthetic fixtures:

```bash
cargo run --locked -- train
cargo run --locked -- evaluate
```

Use `--dataset`, `--ood`, `--model`, `--output`, or `--report` to supply explicit paths. The CLI
refuses collisions between inputs and outputs. The original deterministic rule mode remains
available for comparison:

```bash
cargo run --locked -- --once "I feel uncertain about my next step"
```

## Honest evaluation boundary

The checked-in corpus has 112 synthetic English examples. Seed `20260722` produces 91 training
rows and a 21-row holdout. On that small holdout the model records 14/21 raw accuracy, macro-F1
`0.661`, and 7/21 decision coverage; 6 of the 7 accepted predictions are correct. It accepts 0 of
20 rows in the separate synthetic OOD fixture used during threshold selection.

Those figures verify this pipeline. They do not demonstrate production NLP generalization. Read
the [model card](docs/MODEL_CARD.md) before interpreting them.

## Verify it

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all --locked
node --test site/tests/*.test.mjs
node site/scripts/validate-site.mjs
node --test scripts/tests/*.test.mjs
node scripts/release-contract.mjs verify
node scripts/release-contract.mjs audit-policy
```

## Release verification

The release workflow exercises the same contract on pull requests and manual runs before any tag
exists. It builds and smoke-tests locked Rust binaries for Linux x64, Windows x64, macOS Intel, and
macOS Apple Silicon. Unix binaries ship in `.tar.gz` archives that preserve their executable bit;
Windows ships as `.zip`. The workflow checks every archive name, SHA-256 digest, source commit,
dependency record, SPDX 2.3 entry, and RustSec result as one release set. The RustSec policy pins
both `cargo-audit` and the advisory database commit, denies every warning, and allows no ignored
advisories. Pull requests and manual runs cannot request OIDC attestations or publish a GitHub
Release, even when a manual run starts from a tag ref.

Workflow policy tests parse CI, Pages, and release YAML as explicit structures. They reject anchors,
aliases, tags, duplicate keys, hidden permission overrides, local actions, and any remote Action that
is not pinned to a full commit SHA.

GitHub release publication is authorized under the MIT License. The versioned policy in
`.github/release-policy.json`, the SPDX expression in `Cargo.toml`, and the repository-root
[`LICENSE`](LICENSE) file must agree before a tag run can call the release API. The crate remains
private to this repository (`publish = false`); the supported distribution channel is the verified
GitHub Release assembled by the workflow.

A release can only be published from a `v*` tag pushed for the version in `Cargo.toml`, with a dated
section for that version in `CHANGELOG.md` and no pending text under `Unreleased`. For example,
version `1.2.0` accepts `v1.2.0` and rejects every other tag. The workflow assembles all four native
archives from verified file-descriptor snapshots, creates a consolidated `SHA256SUMS` file covering
every release asset, and adds GitHub provenance attestations. The publish job independently verifies
each attestation against this repository, workflow, tag ref, and source commit before it can touch a
release.

The first authorization requires the pushed tag to resolve to the current remote default-branch tip.
The publisher then creates an exact contract-bearing draft and waits, for a bounded time, until
GitHub's paginated release listing exposes that same draft. It does not upload assets before the draft
is uniquely visible. If a run stops during upload, a later run finds the draft and resumes it without
rebuilding or guessing which commit it represents. Duplicate drafts or foreign contract metadata
stop the run before mutation. Recovery and immutable reruns use GitHub's compare API to prove that the
exact release commit is still identical to or an ancestor of the current default branch; a divergent
commit is rejected without touching the draft. The release `target_commitish` must be that exact commit.
The publisher rechecks the protected tag, branch ancestry, draft contract, and complete remote name,
size, and SHA-256 inventory before promotion. Publication is the irreversible boundary: GitHub must
report the result as both immutable and latest. A rerun only verifies an already-published release and
never rewrites it.

The repository protects `refs/tags/v*` against updates and deletions with a GitHub ruleset and has
immutable releases enabled. The workflow revalidates the tag and requires GitHub to mark the final
release immutable.
Prerelease and build-metadata versions remain rejected until a separate prerelease policy exists.

The RustSec database commit and its commit time are pinned together. CI enforces a 14-day freshness
window on every change and once a week, so a reproducible audit cannot quietly become an outdated
audit. Updating the pin requires reviewing the new official RustSec commit and recording its commit
epoch in `.github/rustsec-audit-policy.json`.

Verify downloaded files with:

```bash
sha256sum -c SHA256SUMS --ignore-missing
gh attestation verify <downloaded-file> -R ejupi-djenis30/PsychologistRustBot
```

To test a proposed tag without creating one, start the **Release** workflow manually and provide
the tag in `release_tag`, or run:

```bash
node scripts/release-contract.mjs verify --tag v1.2.0
```

## Architecture

```text
src/ml.rs                   data, vectorizer, training, metrics, model IO
src/lib.rs                  hard boundaries and dialogue routing
src/main.rs                 train / evaluate / infer / chat CLI
fixtures/                   supervised, OOD, and parity corpora
models/                     versioned learned artifact
reports/                    generated split, calibration, and metrics
site/ml-engine.mjs          browser implementation of model inference
docs/                       model card, dataset contract, architecture
```

Rust and JavaScript run the same versioned model against
[`fixtures/ml-parity.tsv`](fixtures/ml-parity.tsv). The original rule fallback retains its separate
[`fixtures/parity.tsv`](fixtures/parity.tsv) contract. See [the architecture document](docs/ARCHITECTURE.md)
for data flow and failure behavior.

The safety phrase list is only an exit condition for the experiment. It can miss urgent language
and can match benign discussion of a phrase. Never use this project to assess a person or decide
whether help is needed. See [the safety and privacy model](SECURITY.md).

## Provenance

ELIZA Lab grew out of a small 2023 experiment. Ejupi Labs and the project contributors rebuilt it
around a safer premise, removing Telegram, MySQL, OpenAI, and the PHP administration panel instead
of preserving unsafe behaviour for compatibility.

## License

ELIZA Lab is available under the [MIT License](LICENSE).
