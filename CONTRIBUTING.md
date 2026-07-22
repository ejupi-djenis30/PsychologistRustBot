# Contributing to ELIZA Lab

ELIZA Lab implements a small, reproducible intent-classification pipeline with a deterministic
dialogue fallback. It is not therapy, diagnosis, crisis detection, or a substitute for a person.
Contributions must keep that boundary obvious in code, copy, data, tests, and screenshots.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening an issue

Use harmless fictional prompts. Do not paste or attach real conversations, health information,
names, contact details, account data, or anything another person shared in confidence. A bug report
does not need a personal transcript; reduce it to the shortest invented phrase that shows the same
behavior.

Report vulnerabilities through [the security policy](SECURITY.md), not a public issue. If someone
may be in immediate danger, contact local emergency services or a trusted person; this repository
cannot assess or monitor that situation.

## Local setup

Install Rust 1.81 or newer and Node.js 20 or newer. Runtime inference is local; Serde is used only
for the strict, versioned model and report formats.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all --locked
cargo run --locked -- train --output target/model.json --report target/report.json
cmp models/eliza-intent-v1.json target/model.json
cmp reports/eliza-intent-v1.json target/report.json
cargo build --release --locked
cargo run --locked -- train-v2 --output target/open-set-v2
diff -r artifacts/eliza-open-set-v2 target/open-set-v2
cargo run --locked -- bundle verify --bundle artifacts/eliza-open-set-v2
cargo run --locked -- bundle reproduce --bundle artifacts/eliza-open-set-v2
node --test site/tests/*.test.mjs
node site/scripts/validate-site.mjs
cargo audit
```

If you edit a workflow, run [`actionlint`](https://github.com/rhysd/actionlint) when it is available.

## What a good change includes

- Add invented, non-sensitive test input for every behavior change.
- Update both Rust and JavaScript inference when their shared model contract changes.
- Add learned cases to `fixtures/ml-parity.tsv`; keep rule-only cases in `fixtures/parity.tsv`.
- Regenerate `models/eliza-intent-v1.json` and `reports/eliza-intent-v1.json` after any supervised
  corpus, OOD fixture, vectorizer, optimizer, or threshold-calibration change.
- Never use holdout rows or their results to fit the vocabulary, weights, or decision thresholds.
- In v2, preserve all six populations: train fits features and weights; calibration fits only the
  temperature; development plus OOD-development select thresholds; ID-test and OOD-test report
  final results only. Related `group_id` values may not cross ID partitions.
- Regenerate all five files in `artifacts/eliza-open-set-v2/` together. Never edit a digest or a
  generated metric by hand; the verify and reproduce commands must both pass.
- Describe the 112-row corpus and 21-row holdout as synthetic educational fixtures. Do not turn
  their metrics into a production-language claim.
- Keep the 512-code-point input limit, bounded CLI reader, saturating turn count, and 40-turn browser
  transcript unless the PR explains and tests a safer replacement.
- Describe safety phrase matching as a narrow exit condition. Do not call it detection, assessment,
  prevention, care, or clinical advice.
- Keep the browser demo local-only: no analytics, accounts, transcript storage, remote models, or
  prompt submission. The checked-in model may load only as a same-origin static asset.
- Avoid new dependencies unless a small standard-library implementation is clearly less safe.

Safety phrase matching has known false positives and false negatives. A contribution must not hide
that limitation or imply that a passing test makes the engine suitable for real-world care.

## Pull requests

Keep commits focused. In the PR, state what was wrong, what now enforces the behavior, and which
commands you ran. Check screenshots, fixtures, test names, and terminal output for personal or
sensitive conversation text before pushing.

By submitting a contribution, you confirm that you have the right to provide it and agree that it
may be distributed under the repository's [MIT License](LICENSE).
