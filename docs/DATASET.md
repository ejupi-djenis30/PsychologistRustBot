# Dataset contract

## Supervised corpus

[`fixtures/intents-v1.tsv`](../fixtures/intents-v1.tsv) is UTF-8 TSV with this exact header:

```text
id<TAB>label<TAB>text
```

Every row must have a non-empty stable ID, a lowercase label, and a fictional English prompt no
longer than 512 Unicode code points. IDs and normalized text must be unique. Each label needs at
least five rows so a stratified train/holdout split is possible. Rows must contain exactly the
documented number of tab-separated fields; tabs inside IDs, labels, or prompt text are rejected.

| Label | Narrow meaning in this dataset |
| --- | --- |
| `greeting` | An opening or salutation |
| `feeling` | A direct statement of an emotional state |
| `reason` | An explanation, cause, or consequence |
| `ownership` | A statement centered on the speaker's own object, plan, or context |
| `question` | A direct request for an answer or explanation |
| `goal` | An intention, objective, or next step |
| `observation` | A descriptive statement without another listed intent |

The corpus has 112 synthetic rows, 16 for each class. It was written for this repository and does
not contain real conversations or personal, clinical, account, or contact data.

## Reproducible split

ELIZA Lab groups rows by label, orders them using a stable FNV-1a hash of the seed, ID, label, and
normalized text, and places 20% of each class in holdout. Seed `20260722` yields 91 train rows and
21 holdout rows, three per class. IDs for both partitions are written to the model report.

The vectorizer is fit after the split and receives only training rows. A regression test constructs
holdout-only terms and verifies that none enter the vocabulary.

The dataset fingerprint sorts canonical rows before hashing, so reordering the TSV does not invent
a new dataset identity. FNV is used for deterministic identity and partitioning, not security.

## Out-of-domain fixture

[`fixtures/ood-v1.tsv`](../fixtures/ood-v1.tsv) uses this header:

```text
id<TAB>text
```

Its 20 synthetic rows intentionally ask for unrelated capabilities. They have no labels because
forcing them into one of the seven in-domain classes would create a false ground truth. Threshold
calibration can measure how many pass or abstain, but cannot report OOD accuracy.

Neither supervised holdout rows nor their metrics participate in threshold selection. Calibration
accepts the typed deterministic split, verifies its dataset fingerprint and seed, reconstructs the
train/holdout boundary, and rejects any OOD ID or normalized text that overlaps supervised data.

## Editing the corpus

A corpus change creates a new fingerprint and can move both the deterministic split and learned
weights. A pull request that changes data must therefore regenerate the model and report, explain
the class-level effect, and keep all examples fictional and non-clinical.
