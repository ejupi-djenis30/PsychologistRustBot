# Changelog

## Unreleased

## 1.1.2 — 2026-07-20

- Reissue the verified builds from the repository's privacy-safe history. The ELIZA engine and
  interface are unchanged; only commit attribution and release provenance changed.
- Keep the retired immutable tags unavailable instead of reusing them, so the release advances to
  `v1.1.2`.

## 1.1.1 — 2026-07-20

- Wait for GitHub's release listing to expose a newly created draft before uploading any asset.
- Keep delayed draft discovery bounded and fail closed without mutating an undiscoverable draft.
- Cover both eventual-consistency recovery and retry exhaustion with deterministic publisher tests.

## 1.1.0 — 2026-07-20

- Add a tag-gated release pipeline for Linux x64, Windows x64, macOS Intel, and Apple Silicon with
  native CLI smoke tests and platform-appropriate archives.
- Verify Cargo/tag/commit parity, local and remote artifact inventory, SHA-256 checksums, dependency
  evidence, SPDX 2.3 output, and GitHub build provenance before a release can be published.
- Resume only an exact contract-bearing draft, verify the uploaded remote bytes, and recheck both
  tag and inventory before the draft-to-published transition.
- Run the complete packaging contract on pull requests and manual workflow runs without publishing.
- Pin the RustSec scanner and advisory database, and carry its no-warning result into the verified
  release inventory.
- Require a pushed tag at the current default-branch tip for initial authorization, a dated
  changelog section, safe file snapshots, immutable releases, and idempotent rerun verification.
- Independently verify every GitHub attestation identity before publication and confirm the final
  release is both immutable and latest.
- Discover interrupted drafts through GitHub's authenticated paginated release listing and reject
  duplicate or foreign release state before mutation.
- License the project under MIT and authorize GitHub release publication only while the Cargo SPDX
  expression, repository license file, and versioned policy agree.
- Enforce a maximum age for the pinned RustSec database and check that its recorded commit time
  matches the fetched official commit.
- Pin CI and Pages runners and toolchains, and keep Cargo checks locked to the committed dependency
  graph.
- Replace substring safety checks with deterministic word-boundary phrase matching.
- Expand explicit safety exit phrases while documenting false positives and false negatives.
- Align Rust and browser tokenization, apostrophe handling and pronoun reflection.
- Add a shared Rust/JavaScript parity corpus.
- Bound CLI line reads, browser transcript growth and turn-counter overflow.
- Add a distinct accessible safety response and clearer input/session limits.
- Pin CI and Pages actions to reviewed commit SHAs and add weekly Dependabot coverage.

## 1.0.0 — 2026-07-19

- Rebuild the legacy Telegram experiment as a local-only Rust engine and static learning tool.
- Remove the database, remote model, accounts and therapeutic framing.
