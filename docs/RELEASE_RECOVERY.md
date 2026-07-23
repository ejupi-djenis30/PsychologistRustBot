# Release draft recovery

Release recovery is a narrow, manual path for one failure mode: the protected tag-push workflow
created a contract-bearing draft, could not rediscover it through GitHub's release listing, and
stopped before uploading an asset. The recovery workflow never creates a release, rebuilds an
artifact, or chooses a draft on the operator's behalf.

## Evidence and trust boundary

An operator supplies four identifiers to the **Recover release draft** workflow:

- the signed release tag;
- the exact commit referenced by that tag;
- the failed tag-push **Release** workflow run ID;
- the exact empty draft release ID reported by that run.

These values are selectors, not proof. Before any release mutation, the recovery program retrieves
and cross-checks GitHub's API state:

1. The recovery run must be an active manual dispatch of the dedicated workflow at the current
   default-branch tip.
2. The source run must be the protected tag-push workflow for the same repository, tag and commit.
   Its current attempt must contain the exact release job topology. Every prerequisite must have
   succeeded, and only the final publication step may have failed.
3. The exact `Publish GitHub Release` job database ID is taken from that validated source
   run/attempt. The program downloads that job's Actions log through GitHub's authenticated API and
   host-pinned signed redirect. It requires exactly one complete line whose message is:

   ```text
   release-publish: New draft <ID> could not be uniquely rediscovered before asset mutation
   ```

   `<ID>` must equal the requested recovery draft ID. The log is accepted only as valid UTF-8,
   uncompressed `text/plain`, with a 2 MiB byte limit, 4,096-line limit, 16 KiB per-line limit and a
   terminating newline. Missing, malformed, duplicated, mismatched or truncated receipts stop
   recovery. Local log files are never accepted.
4. The original `verified-release-assets` artifact must be unique, unexpired and bound to that
   source run, repository, tag and commit. Its GitHub provenance attestation must cover every exact
   asset digest.
5. GitHub must still verify the annotated tag signature. The release commit must remain identical
   to or an ancestor of the current default branch.
6. The draft selected by ID must still match the release contract. Its remote assets may only be an
   exact subset of the attested inventory; conflicting names, sizes, digests, duplicate IDs or
   unexpected assets stop recovery.

`release.created_at` is not draft-creation evidence. GitHub documents and returns it as the
tag-or-commit creation time, so recovery validates it only as timestamp-shaped metadata. The causal
link between the failed source job and the draft comes from the exact job ID and the exact log
receipt above.

## Attempts and resumability

GitHub can rerun a workflow without replacing the original run ID. Recovery therefore treats two
attempt numbers separately:

- **source attempt** is the source run's current attempt. Its publication job must contain the
  matching draft receipt and be the only failed stage;
- **attestation attempt** is read from the verified provenance statement. It may be the current
  source attempt or an earlier successful attempt from the same run, but never a later or unrelated
  attempt.

If recovery itself stops after uploading some assets, the next manual dispatch revalidates the
entire chain and uploads only the missing members of the exact inventory. It never deletes a remote
asset during recovery and never uploads the same asset twice by design.

## Operational use

Run recovery only after inspecting the failed source run and confirming that its final log reports
the exact empty draft ID. Dispatch the workflow from the repository's current default branch and
enter all four identifiers exactly. Stop and investigate instead of retrying with different IDs if
any check fails.

The workflow is intentionally manual. It has no schedule, no automatic failure trigger and no path
from an ordinary `workflow_dispatch` in the release workflow. Publication remains the irreversible
boundary: immediately before promotion, the recovery job re-proves its own execution identity, the
signed tag, branch ancestry, draft contract and complete remote inventory. GitHub must then report
the release as immutable and latest.
