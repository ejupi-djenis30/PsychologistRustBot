<p align="center">
  <img src="site/assets/eliza-lab-lockup.svg" width="720" alt="ELIZA Lab — transparent dialogue, entirely local" />
</p>

# ELIZA Lab

[![CI](https://github.com/ejupi-djenis30/PsychologistRustBot/actions/workflows/ci.yml/badge.svg)](https://github.com/ejupi-djenis30/PsychologistRustBot/actions/workflows/ci.yml)

> A transparent, local-only conversation engine for learning how early rule-based dialogue systems work.

This repository began as a Telegram “psychologist” bot. That framing was misleading, and the
implementation stored sensitive conversations in an insecure database. The project has been
rebuilt as **ELIZA Lab**: an educational Rust engine and browser demo with no accounts, network
calls, transcript storage, diagnosis, or therapeutic claims.

[Open the interactive demo](https://ejupi-djenis30.github.io/PsychologistRustBot/) ·
[Watch the demo](site/assets/demo.mp4) · [Read the safety model](SECURITY.md)

## What makes it useful

- **Rule traces:** every response names the pattern that produced it.
- **Private by construction:** the Rust CLI and web demo run locally and do not store input.
- **Deterministic:** the same turn sequence produces the same fallback sequence.
- **Honest boundaries:** a small phrase list exits the experiment instead of imitating care. It
  is deliberately not presented as crisis detection.
- **Bounded sessions:** prompts stop at 512 Unicode code points, CLI lines are byte-bounded, and
  the browser retains at most 40 visible turns.
- **Cross-runtime contract:** Rust and JavaScript run against the same response corpus in CI.

## Run it

```bash
cargo run
```

For a single deterministic response:

```bash
cargo run -- --once "I feel uncertain about my next step"
```

## Verify it

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all
node --test site/tests/*.test.mjs
node site/scripts/validate-site.mjs
```

## Architecture

```text
src/lib.rs       pure rule engine and trace model
src/main.rs      local CLI adapter
site/            static explanatory interface for GitHub Pages
site/tests/      browser-engine regression tests
```

The production engine has no third-party runtime dependencies. It keeps only a saturating turn
counter in memory. The browser demo mirrors the same documented rule order, and both engines run
against [`fixtures/parity.tsv`](fixtures/parity.tsv) to prevent silent response drift.

The safety phrase list is only an exit condition for the experiment. It can miss urgent language
and can match benign discussion of a phrase. Never use this project to assess a person or decide
whether help is needed. See [the safety and privacy model](SECURITY.md).

## Provenance

The original 2023 experiment was created by Djenis Ejupi. The current refactor deliberately
removes Telegram, MySQL, OpenAI, and the PHP administration panel rather than preserving unsafe
behaviour for compatibility.

## License

No license file is currently provided. All rights remain with their respective authors.
