import assert from "node:assert/strict";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  SUPPORTED_TARGETS,
  artifactName,
  buildAuditEvidence,
  buildReleaseContract,
  generateSpdx,
  packageBinary,
  parseCargoPackage,
  parseChangelogRelease,
  verifyAuditDatabasePolicy,
  verifyAndAssembleAssets,
  verifyVersionTag,
} from "../release-contract.mjs";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const manifestPath = path.join(repositoryRoot, "Cargo.toml");
const lockPath = path.join(repositoryRoot, "Cargo.lock");
const auditPolicyPath = path.join(repositoryRoot, ".github", "rustsec-audit-policy.json");
const auditPolicy = JSON.parse(readFileSync(auditPolicyPath, "utf8"));

function createCompleteFixture(prefix) {
  const temporaryRoot = mkdtempSync(path.join(tmpdir(), prefix));
  const inputDirectory = path.join(temporaryRoot, "input");
  const evidenceDirectory = path.join(temporaryRoot, "evidence");
  const outputDirectory = path.join(temporaryRoot, "output");
  mkdirSync(inputDirectory);
  mkdirSync(evidenceDirectory);
  const commit = "a".repeat(40);

  for (const [index, target] of Object.keys(SUPPORTED_TARGETS).entries()) {
    const source = path.join(temporaryRoot, `source-${index}`);
    writeFileSync(source, `portable binary fixture ${target}\n`, "utf8");
    const targetOutput = path.join(temporaryRoot, `target-${index}`);
    packageBinary({ target, source, outputDirectory: targetOutput, commit, manifestPath });
    for (const entry of readdirSync(targetOutput)) {
      copyFileSync(path.join(targetOutput, entry), path.join(inputDirectory, entry));
    }
  }

  const contract = buildReleaseContract(manifestPath, "", commit);
  writeFileSync(path.join(evidenceDirectory, "release-contract.json"), `${JSON.stringify(contract, null, 2)}\n`);
  copyFileSync(lockPath, path.join(evidenceDirectory, "Cargo.lock"));
  writeFileSync(path.join(evidenceDirectory, "cargo-tree.txt"), `${contract.package} v${contract.version}\n`, "utf8");
  const lockPackages = readFileSync(lockPath, "utf8")
    .split("[[package]]")
    .slice(1)
    .map((block, index) => {
      const name = block.match(/^\s*name\s*=\s*"([^"]+)"/mu)?.[1];
      const version = block.match(/^\s*version\s*=\s*"([^"]+)"/mu)?.[1];
      assert.ok(name && version, "test fixture must parse every Cargo.lock package");
      return { id: `fixture:${index}:${name}@${version}`, name, version, source: null, license: null };
    });
  const rootPackage = lockPackages.find(
    (entry) => entry.name === contract.package && entry.version === contract.version,
  );
  assert.ok(rootPackage, "test fixture must contain the workspace root package");
  const rootId = rootPackage.id;
  const metadata = {
    packages: lockPackages,
    resolve: {
      root: rootId,
      nodes: lockPackages.map((entry) => ({ id: entry.id, dependencies: [] })),
    },
  };
  writeFileSync(path.join(evidenceDirectory, "cargo-metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
  const sbom = generateSpdx(metadata, { manifestPath, lockPath });
  writeFileSync(path.join(evidenceDirectory, `${contract.package}-v${contract.version}.spdx.json`), `${JSON.stringify(sbom, null, 2)}\n`, "utf8");
  const auditReport = {
    database: { "advisory-count": 1166, "last-commit": null, "last-updated": null },
    lockfile: { "dependency-count": lockPackages.length },
    settings: { target_arch: [], target_os: [], severity: null, ignore: [], informational_warnings: ["unmaintained", "unsound", "notice"] },
    vulnerabilities: { found: false, count: 0, list: [] },
    warnings: {},
  };
  const auditEvidence = buildAuditEvidence({
    report: auditReport,
    toolVersion: auditPolicy.tool.version,
    databaseCommit: auditPolicy.database.commit,
    databaseCommitEpoch: auditPolicy.database.commitEpoch,
    policyPath: auditPolicyPath,
    lockPath,
  });
  copyFileSync(auditPolicyPath, path.join(evidenceDirectory, "rustsec-audit-policy.json"));
  writeFileSync(path.join(evidenceDirectory, "rustsec-audit.json"), `${JSON.stringify(auditEvidence, null, 2)}\n`, "utf8");

  return { temporaryRoot, inputDirectory, evidenceDirectory, outputDirectory, contract, commit };
}

test("reads the package version only from the Cargo package table", () => {
  const parsed = parseCargoPackage(`
[package]
name = "eliza-lab"
version = "2.3.4"

[dependencies]
version = "99.0.0"
`);
  assert.deepEqual(parsed, { name: "eliza-lab", version: "2.3.4", license: null });
});

test("rejects prerelease and build metadata until prerelease publishing is defined", () => {
  for (const version of ["2.3.4-rc.1", "2.3.4+build.5"]) {
    assert.throws(
      () => parseCargoPackage(`[package]\nname = "eliza-lab"\nversion = "${version}"\n`),
      /stable SemVer/u,
    );
    assert.throws(() => verifyVersionTag(version), /non-stable SemVer/u);
  }
});

test("requires an exact v-prefixed tag for the Cargo version", () => {
  assert.equal(verifyVersionTag("1.1.0", "v1.1.0"), "v1.1.0");
  assert.equal(verifyVersionTag("1.1.0"), "v1.1.0");
  assert.throws(() => verifyVersionTag("1.1.0", "1.1.0"), /does not match/u);
  assert.throws(() => verifyVersionTag("1.1.0", "v1.1.1"), /does not match/u);
});

test("reads a dated release section only from visible Markdown", () => {
  const changelog = `# Changelog

## Unreleased

<!--
## 1.1.0 — 1999-01-01
- Hidden comment.
-->

\`\`\`markdown
## 1.1.0 — 1999-01-02
- Hidden fence.
\`\`\`

## 1.1.0 — 2026-07-20

- Visible release note.
`;
  const parsed = parseChangelogRelease(changelog, "1.1.0", { requireUnreleasedEmpty: true });
  assert.equal(parsed.releaseDate, "2026-07-20");
  assert.deepEqual(parsed.noteLines, ["- Visible release note."]);
});

test("rejects hidden, undated, invalid, or unreleased changelog entries", () => {
  assert.throws(
    () => parseChangelogRelease("## Unreleased\n\n<!-- ## 1.1.0 — 2026-07-20 -->\n", "1.1.0"),
    /exactly one visible, dated section/u,
  );
  assert.throws(
    () => parseChangelogRelease("## Unreleased\n\n## 1.1.0\n\n- Note.\n", "1.1.0"),
    /must include an ISO date/u,
  );
  assert.throws(
    () => parseChangelogRelease("## Unreleased\n\n## 1.1.0 — 2026-02-30\n\n- Note.\n", "1.1.0"),
    /invalid date/u,
  );
  assert.throws(
    () => parseChangelogRelease("## Unreleased\n\n- Not released.\n\n## 1.1.0 — 2026-07-20\n\n- Note.\n", "1.1.0", { requireUnreleasedEmpty: true }),
    /still contains unreleased changes/u,
  );
});

test("defines collision-free names for every supported platform", () => {
  const names = Object.keys(SUPPORTED_TARGETS).map((target) => artifactName("1.1.0", target));
  assert.equal(new Set(names).size, 4);
  assert.deepEqual(names.sort(), [
    "eliza-lab-v1.1.0-linux-x86_64.tar.gz",
    "eliza-lab-v1.1.0-macos-aarch64.tar.gz",
    "eliza-lab-v1.1.0-macos-x86_64.tar.gz",
    "eliza-lab-v1.1.0-windows-x86_64.zip",
  ]);
});

test("packages and re-verifies the complete release inventory", () => {
  const { inputDirectory, evidenceDirectory, outputDirectory, contract, commit } = createCompleteFixture("eliza-release-contract-");
  const inventory = verifyAndAssembleAssets({
    inputDirectory,
    evidenceDirectory,
    outputDirectory,
    expectedCommit: commit,
    manifestPath,
  });
  assert.equal(inventory.sourceCommit, commit);
  assert.equal(inventory.files.length, 20);
  assert.match(readFileSync(path.join(outputDirectory, "SHA256SUMS"), "utf8"), new RegExp(`${contract.package}-v${contract.version}-linux-x86_64`, "u"));
});

test("fails closed when a packaged binary no longer matches its checksum", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, contract, commit } = createCompleteFixture("eliza-release-tamper-");
  const artifact = artifactName(contract.version, "x86_64-unknown-linux-gnu");
  writeFileSync(path.join(inputDirectory, `${artifact}.sha256`), `${"0".repeat(64)}  ${artifact}\n`, "utf8");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "tampered-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    /Checksum file does not match/u,
  );
});

test("rejects artifacts built from a commit other than the workflow commit", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory } = createCompleteFixture("eliza-release-commit-");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "wrong-commit-output"),
      expectedCommit: "b".repeat(40),
      manifestPath,
    }),
    /Artifact manifest does not exactly match/u,
  );
});

test("rejects an SBOM altered after generation", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, contract, commit } = createCompleteFixture("eliza-release-sbom-");
  const sbomPath = path.join(evidenceDirectory, `${contract.package}-v${contract.version}.spdx.json`);
  const sbom = JSON.parse(readFileSync(sbomPath, "utf8"));
  sbom.packages[0].name = "altered-package";
  writeFileSync(sbomPath, `${JSON.stringify(sbom, null, 2)}\n`, "utf8");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "altered-sbom-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    /SBOM content does not match/u,
  );
});

test("rejects dependency evidence altered after the quality gate", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, commit } = createCompleteFixture("eliza-release-evidence-");
  writeFileSync(path.join(evidenceDirectory, "Cargo.lock"), "version = 3\n", "utf8");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "altered-evidence-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    /Cargo\.lock evidence differs/u,
  );
});

test("rejects vulnerability evidence altered after the RustSec gate", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, commit } = createCompleteFixture("eliza-release-audit-");
  const auditPath = path.join(evidenceDirectory, "rustsec-audit.json");
  const audit = JSON.parse(readFileSync(auditPath, "utf8"));
  audit.report.vulnerabilities = { found: true, count: 1, list: [{ advisory: { id: "RUSTSEC-TEST-0001" } }] };
  writeFileSync(auditPath, `${JSON.stringify(audit, null, 2)}\n`, "utf8");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "altered-audit-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    /vulnerable dependencies/u,
  );
});

test("rejects a substituted RustSec policy", () => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, commit } = createCompleteFixture("eliza-release-policy-");
  const policyPath = path.join(evidenceDirectory, "rustsec-audit-policy.json");
  const policy = JSON.parse(readFileSync(policyPath, "utf8"));
  policy.ignoredAdvisories = ["RUSTSEC-TEST-0001"];
  writeFileSync(policyPath, `${JSON.stringify(policy, null, 2)}\n`, "utf8");
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "altered-policy-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    /policy evidence differs/u,
  );
});

test("enforces the pinned RustSec commit time and maximum database age", () => {
  const deadline = auditPolicy.database.commitEpoch + auditPolicy.database.maximumAgeDays * 86_400;
  assert.equal(
    verifyAuditDatabasePolicy({
      policyPath: auditPolicyPath,
      databaseCommit: auditPolicy.database.commit,
      databaseCommitEpoch: auditPolicy.database.commitEpoch,
      nowEpochSeconds: deadline,
    }).database.commit,
    auditPolicy.database.commit,
  );
  assert.throws(
    () => verifyAuditDatabasePolicy({
      policyPath: auditPolicyPath,
      databaseCommit: auditPolicy.database.commit,
      databaseCommitEpoch: auditPolicy.database.commitEpoch,
      nowEpochSeconds: deadline + 1,
    }),
    /older than the allowed/u,
  );
  assert.throws(
    () => verifyAuditDatabasePolicy({
      policyPath: auditPolicyPath,
      databaseCommit: auditPolicy.database.commit,
      databaseCommitEpoch: auditPolicy.database.commitEpoch - 1,
      nowEpochSeconds: deadline,
    }),
    /commit time does not match policy/u,
  );
});

test("rejects symlinked supply-chain evidence", (context) => {
  const { temporaryRoot, inputDirectory, evidenceDirectory, commit } = createCompleteFixture("eliza-release-symlink-");
  const evidencePath = path.join(evidenceDirectory, "cargo-tree.txt");
  const replacementPath = path.join(temporaryRoot, "outside-cargo-tree.txt");
  writeFileSync(replacementPath, readFileSync(evidencePath));
  unlinkSync(evidencePath);
  try {
    symlinkSync(replacementPath, evidencePath, "file");
  } catch (error) {
    if (error.code === "EPERM") {
      context.skip("Creating file symlinks requires Windows Developer Mode");
      return;
    }
    throw error;
  }
  assert.throws(
    () => verifyAndAssembleAssets({
      inputDirectory,
      evidenceDirectory,
      outputDirectory: path.join(temporaryRoot, "symlink-output"),
      expectedCommit: commit,
      manifestPath,
    }),
    process.platform === "win32" ? /symbolic link|unknown error/iu : /symbolic link/u,
  );
});
