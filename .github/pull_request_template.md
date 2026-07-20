## Outcome

<!-- What changed, why it matters, and what a reviewer can verify. -->

## Verification

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --locked -- -D warnings`
- [ ] `cargo test --all --locked`
- [ ] Pinned RustSec audit passes with no vulnerability or warning exceptions
- [ ] `node --test site/tests/*.test.mjs`
- [ ] `node site/scripts/validate-site.mjs`
- [ ] Keyboard, focus, and accessibility behavior checked when the UI changed

## Privacy and network boundary

- [ ] No analytics, accounts, transcript storage, remote models, or network submission added
- [ ] Test data and screenshots are fictional and contain no personal or health information
- [ ] Any intentional network behavior is described and justified below

Network and privacy notes:

## Release and repository contract

- [ ] Version, artifact, and fixture changes are documented and aligned across Rust and the browser demo
- [ ] The current Cargo version has a dated changelog section; release tags leave `Unreleased` empty
- [ ] Release or packaging changes pass `node --test scripts/tests/*.test.mjs`
- [ ] Licensing status is unchanged, or Cargo, the license file, and `.github/release-policy.json` record the same explicit approval

Artifact, version, and license notes:

## Safety and security boundary

- [ ] Copy does not present ELIZA Lab as therapy, diagnosis, crisis detection, or care
- [ ] Safety-exit limitations and possible false positives or negatives remain explicit
- [ ] Security-sensitive changes include a threat or abuse case and a regression test

Security and safety notes:
