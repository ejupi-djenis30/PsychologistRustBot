# Dataset contract

## Version-two grouped corpus

[`fixtures/intents-v2.tsv`](../fixtures/intents-v2.tsv) adds an explicit grouping boundary:

```text
id<TAB>group_id<TAB>label<TAB>text
```

`group_id` identifies related wording that must never cross partitions. A group belongs to exactly
one label, IDs and normalized text remain unique, and every label needs at least four groups. The
current synthetic fixture has eight two-row groups per label. These pairs are only an initial
leakage-control mechanism; they are not a substitute for a larger independently annotated corpus.

Seed `20260722` hashes whole groups within each label and produces four typed partitions. For each
label, the split assigns roughly 10% of groups to each non-training partition, with at least one
group in every partition and at least one left for training. The checked-in eight-group labels
therefore use one group for each evaluation role and five for training; larger datasets scale the
quotas instead of keeping fixed one-group tests. The strategy identifier is
`group-stratified-scaled-four-way-v2`.

| Partition | Rows | May influence |
| --- | ---: | --- |
| Train | 70 | TF-IDF vocabulary and classifier weights |
| Development | 14 | Confidence and probability-margin thresholds |
| Calibration | 14 | Temperature scaling only |
| ID-test | 14 | Final ID metrics only |

Every label appears in every partition. No ID or group appears in more than one partition. The
serialized split plan lists every assignment and is SHA-256 linked to the model, policy and metrics
through the v2 artifact manifest.

## Version-two OOD populations

[`fixtures/ood-dev-v2.tsv`](../fixtures/ood-dev-v2.tsv) and
[`fixtures/ood-test-v2.tsv`](../fixtures/ood-test-v2.tsv) use:

```text
id<TAB>group_id<TAB>text
```

OOD-development has 20 rows and may influence only the abstention thresholds. OOD-test has 20
different rows and is opened only after the temperature and thresholds are frozen. The loader
rejects reused IDs, groups or normalized text across supervised, OOD-development and OOD-test.

Both sets remain synthetic and small. AUROC, AUPR and FPR at 95% TPR describe only these fixtures;
the deterministic row-bootstrap intervals expose within-fixture sampling uncertainty but cannot
measure between-group variation or manufacture external validity.

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
