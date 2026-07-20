# Contributing to ELIZA Lab

ELIZA Lab explains a small rule-based dialogue engine. It is not therapy, diagnosis, crisis
detection, or a substitute for a person. Contributions must keep that boundary obvious in code,
copy, tests, and screenshots.

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

Install Rust 1.81 or newer and Node.js 20 or newer. The production engine has no third-party Rust
dependencies.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all --locked
cargo build --release --locked
node --test site/tests/*.test.mjs
node site/scripts/validate-site.mjs
cargo audit
```

If you edit a workflow, run [`actionlint`](https://github.com/rhysd/actionlint) when it is available.

## What a good change includes

- Add invented, non-sensitive test input for every behavior change.
- Update both the Rust and JavaScript engines when their shared contract changes.
- Add the case to `fixtures/parity.tsv` when both engines should produce the same trace.
- Keep the 512-code-point input limit, bounded CLI reader, saturating turn count, and 40-turn browser
  transcript unless the PR explains and tests a safer replacement.
- Describe safety phrase matching as a narrow exit condition. Do not call it detection, assessment,
  prevention, care, or clinical advice.
- Keep the browser demo local-only: no analytics, accounts, transcript storage, remote models, or
  network submission.
- Avoid new dependencies unless a small standard-library implementation is clearly less safe.

Safety phrase matching has known false positives and false negatives. A contribution must not hide
that limitation or imply that a passing test makes the engine suitable for real-world care.

## Pull requests

Keep commits focused. In the PR, state what was wrong, what now enforces the behavior, and which
commands you ran. Check screenshots, fixtures, test names, and terminal output for personal or
sensitive conversation text before pushing.

This repository does not currently grant a reuse license. By submitting a contribution, you
confirm that you have the right to provide it; the repository's licensing status does not change.
