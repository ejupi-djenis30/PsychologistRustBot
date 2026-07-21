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
[Support](SUPPORT.md) · [Read the safety model](SECURITY.md)

## What makes it useful

- **Rule traces:** every response names the pattern that produced it.
- **Private by construction:** the Rust CLI and web demo run locally and do not store input.
- **Deterministic:** the same turn sequence produces the same fallback sequence.
- **Honest boundaries:** a small phrase list exits the experiment instead of imitating care. It
  is deliberately not presented as crisis detection.
- **Bounded sessions:** prompts stop at 512 Unicode code points, CLI lines are byte-bounded, and
  the browser retains at most 40 visible turns.
- **Cross-runtime contract:** Rust and JavaScript run against the same response corpus in CI.

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

Extract the archive and run the included `eliza-lab` executable. It needs no account, network
connection, model download, or database.

## Run it

To build and run from source instead:

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
version `1.1.2` accepts `v1.1.2` and rejects every other tag. The workflow assembles all four native
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
node scripts/release-contract.mjs verify --tag v1.1.2
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

ELIZA Lab is available under the [MIT License](LICENSE).
