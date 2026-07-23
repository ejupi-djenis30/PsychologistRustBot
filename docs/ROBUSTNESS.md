# Robustness audit

ELIZA Lab includes a local metamorphic audit for the verified open-set model. It answers a narrow
operational question: when the same bounded input is presented with harmless formatting changes or
small typographic damage, how stable are the model probabilities and routed decisions?

The audit is not another training or selection stage. It never changes the vocabulary, weights,
temperature or abstention thresholds. It can therefore run against the v3 bundle without changing
the bundle schema or invalidating existing artifacts.

## Input and privacy boundary

`robustness audit` reads JSONL from standard input:

```json
{"id":"fictional-01","text":"I intend to test one concrete next step"}
```

Each object must contain exactly `id` and `text`. IDs must be unique ASCII identifiers of at most
128 bytes. Text is non-empty and limited to the same 512 Unicode characters as inference. The
reader rejects oversized lines, unknown fields, duplicate IDs and more than 100,000 rows. It also
caps the complete stream at 64 MiB, counts blank lines toward a 100,000-physical-line limit and
allows at most 18,432 bytes per physical line. An oversized line stops the audit after the first
18,433 bytes; the reader does not consume an unbounded remainder before returning the error.

The report contains only aggregate counts and metrics. It never serializes an input ID, prompt,
per-row prediction or transformed text. Processing is local and bounded; the command does not
write a transcript or contact a service. JSON schema failures identify only the physical line and
a generic failure class. They do not echo prompt content, object fields or parser diagnostics.

Every report identifies schema `1`, perturbation suite `1.0.0`, the population mode, model and split
digests, and the exact temperature and abstention thresholds. Caller-provided prompt fingerprints
are deliberately omitted: short texts can be guessed from an unsalted hash, so such a digest would
weaken the aggregate-only privacy boundary.

Use invented, non-sensitive prompts anyway. A local aggregate report is not a reason to process
health information, private conversations or data you do not have the right to use.

## Perturbation families

The four formatting transformations are preprocessing invariants:

- swap ASCII letter case;
- replace horizontal spaces with equivalent whitespace;
- use Unicode fullwidth compatibility forms for ASCII letters and digits;
- add token-irrelevant terminal punctuation.

All four must normalize to the same feature vector. The default gate therefore requires `1.0`
top-label and routed-decision agreement plus exactly `0.0` normalized Jensen–Shannon divergence.
The gate evaluates the unrounded in-memory measurements. JSON output is quantized independently to
nine decimal places for deterministic reports, so a drift too small to appear in the report still
fails the invariant.

Three typographic transformations provide a harder stress test:

- delete one internal character from the longest eligible token;
- transpose one pair of unequal adjacent characters;
- duplicate one internal character.

These edits are controlled noise, not guaranteed paraphrases. There is no universal pass target.
Projects may opt into decision-agreement and divergence thresholds after declaring them before an
evaluation run.

## Metrics

For each perturbation, each family and the complete run, the report records:

- top-label, acceptance and routed-decision agreement;
- label and acceptance flip counts;
- accepted-to-abstained and abstained-to-accepted transitions;
- mean and maximum absolute confidence change;
- mean and maximum Jensen–Shannon divergence, normalized to `[0, 1]`.

`skipped_applications` counts transformation attempts that could not produce a distinct bounded
variant. In a family aggregate, the same input may contribute one skip for more than one
transformation.

A routed decision treats two abstentions as agreement. When both predictions are accepted, their
labels must also match. Label agreement remains separate so a hidden top-label change under two
abstentions is still visible.

## Run it

```bash
printf '%s\n' \
  '{"id":"fictional-01","text":"I intend to test one concrete next step"}' \
  '{"id":"fictional-02","text":"Today I feel calm about the outcome"}' \
  | cargo run --locked -- robustness audit
```

The embedded, semantically verified v3 bundle is the default. Use an external verified bundle with
`--bundle PATH`.

For a deterministic regression run over the verified bundle's frozen ID-test:

```bash
cargo run --locked -- robustness audit --bundle-id-test \
  > target/robustness-id-test-report.json
node scripts/verify-robustness-report.mjs \
  target/robustness-id-test-report.json
```

This mode is a post-training consistency diagnostic. It must not be used to tune the model,
temperature or abstention policy against the final test. CI enforces only the formatting
invariants; it does not turn the observed typo score into a model-selection target.

The ID-test provenance is a capability boundary, not a report label supplied by a caller. Bundle
verification returns a `VerifiedBundle` whose fields cannot be constructed or replaced outside the
verification module. The audit extracts the frozen ID-test from that value, checks its count
against the verified metrics and consumes the bundle to compile inference. The general parsed-case
API can report only `provided-cases`.

Optional CI gates are explicit:

```bash
cargo run --locked -- robustness audit \
  --minimum-typographic-decision-agreement 0.80 \
  --maximum-typographic-js-divergence 0.20 \
  < fictional-audit.jsonl
```

The JSON report is flushed before a selected gate failure, so CI can retain the evidence. A failing
gate exits non-zero. Raw gate evidence is intentionally not serialized. Library callers therefore
cannot deserialize a report and reapply a gate to rounded values; enforcement fails closed and
requires a fresh audit in the same process.

## Site evidence

The public robustness dashboard is checked against a report generated in the CI, Pages and release
quality jobs. `scripts/verify-robustness-report.mjs` binds the report to the checked manifest,
policy, split digest, frozen ID-test count and prediction ledger. It reconstructs counts,
agreements, weighted means and maxima for each family from the seven perturbation slices, then
compares every displayed value and graph width with the report. A stale or internally inconsistent
metric stops CI, deployment and publication.

## Interpretation limits

High stability on generated edits does not establish accuracy, fairness, safety or real-world
generalization. Typographic stability can even preserve the same wrong answer. The audit has no
labels and measures consistency, not correctness. Use it beside the frozen ID, OOD, contrast,
calibration and baseline reports—not as a replacement for them.
