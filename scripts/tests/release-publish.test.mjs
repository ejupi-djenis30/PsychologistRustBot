import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { expectedReleaseFileNames } from "../release-contract.mjs";
import {
  buildLocalInventory,
  GitHubApiClient,
  GitHubApiError,
  publishRelease,
  recoverEmptyDraftRelease,
  releaseContractBody,
} from "../release-publish.mjs";

const repository = "ejupi-djenis30/PsychologistRustBot";
const tag = "v1.4.0";
const expectedCommit = "a".repeat(40);
const recoveryWorkflowCommit = "b".repeat(40);
const signedTagObject = "c".repeat(40);
const sourceRunId = 101;
const recoveryRunId = 202;
const repositoryId = 594_442_371;
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function createAuthorizedRepositoryFixture() {
  const directory = mkdtempSync(path.join(tmpdir(), "eliza-authorized-release-"));
  const manifestPath = path.join(directory, "Cargo.toml");
  const manifest = readFileSync(path.join(repositoryRoot, "Cargo.toml"), "utf8");
  writeFileSync(manifestPath, manifest, "utf8");
  writeFileSync(path.join(directory, "CHANGELOG.md"), readFileSync(path.join(repositoryRoot, "CHANGELOG.md")));
  writeFileSync(path.join(directory, "LICENSE"), readFileSync(path.join(repositoryRoot, "LICENSE")));
  const policyPath = path.join(directory, "release-policy.json");
  writeFileSync(policyPath, `${JSON.stringify({
    schemaVersion: 1,
    publicationEnabled: true,
    licenseFile: "LICENSE",
    spdxExpression: "MIT",
  }, null, 2)}\n`, "utf8");
  return Object.freeze({ manifestPath, policyPath });
}

const authorizedRepository = createAuthorizedRepositoryFixture();

function createDisabledRepositoryFixture() {
  const directory = mkdtempSync(path.join(tmpdir(), "eliza-disabled-release-"));
  const manifestPath = path.join(directory, "Cargo.toml");
  writeFileSync(manifestPath, readFileSync(path.join(repositoryRoot, "Cargo.toml")));
  const policyPath = path.join(directory, "release-policy.json");
  writeFileSync(policyPath, `${JSON.stringify({
    schemaVersion: 1,
    publicationEnabled: false,
    licenseFile: null,
    spdxExpression: null,
  }, null, 2)}\n`, "utf8");
  return Object.freeze({ manifestPath, policyPath });
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function createReleaseAssets() {
  const directory = mkdtempSync(path.join(tmpdir(), "eliza-publish-assets-"));
  const names = expectedReleaseFileNames();
  const checksums = [];
  for (const name of names.filter((entry) => entry !== "SHA256SUMS")) {
    const bytes = Buffer.from(`verified release asset: ${name}\n`);
    writeFileSync(path.join(directory, name), bytes);
    checksums.push(`${sha256(bytes)}  ${name}`);
  }
  writeFileSync(path.join(directory, "SHA256SUMS"), `${checksums.sort().join("\n")}\n`, "utf8");
  return directory;
}

class FakeGitHubApi {
  constructor({
    tagCommit = expectedCommit,
    tagCommits,
    defaultBranch = "master",
    defaultBranchCommits = [expectedCommit],
    releaseState = "draft",
    injectExtraAsset = false,
    injectPostPublishExtraAsset = false,
    refuseTransition = false,
    throwAfterTransition = false,
    draftTarget = expectedCommit,
    beforeFirstTagRead,
    publishedImmutable = true,
    latestReleaseId = 7,
    extraReleases = [],
    comparisonStatuses = [],
    createdDraftVisibilityDelay = 0,
  } = {}) {
    this.tagCommit = tagCommit;
    this.tagCommits = tagCommits || [tagCommit];
    this.tagReadCount = 0;
    this.defaultBranch = defaultBranch;
    this.defaultBranchCommits = defaultBranchCommits;
    this.defaultBranchReadCount = 0;
    this.injectExtraAsset = injectExtraAsset;
    this.injectPostPublishExtraAsset = injectPostPublishExtraAsset;
    this.refuseTransition = refuseTransition;
    this.throwAfterTransition = throwAfterTransition;
    this.publishedImmutable = publishedImmutable;
    this.latestReleaseId = latestReleaseId;
    this.injected = false;
    this.beforeFirstTagRead = beforeFirstTagRead;
    this.tagRead = false;
    this.calls = [];
    this.nextAssetId = 100;
    this.extraReleases = structuredClone(extraReleases);
    this.comparisonStatuses = [...comparisonStatuses];
    this.comparisonReadCount = 0;
    this.createdDraftVisibilityDelay = createdDraftVisibilityDelay;
    this.createdDraftVisibilityReadsRemaining = 0;
    this.release = releaseState === "missing" ? null : {
      id: 7,
      tag_name: tag,
      target_commitish: draftTarget,
      name: `ELIZA Lab ${tag}`,
      body: releaseContractBody(tag, draftTarget),
      draft: releaseState === "draft",
      immutable: releaseState === "published" ? publishedImmutable : false,
      prerelease: false,
      upload_url: `https://uploads.github.test/repos/${repository}/releases/7/assets{?name,label}`,
      html_url: `https://github.test/${repository}/releases/tag/${tag}`,
      assets: releaseState === "draft" ? [{ id: 4, name: "unverified-extra.txt", size: 5, state: "uploaded", digest: `sha256:${"f".repeat(64)}` }] : [],
    };
  }

  async request(endpoint, options = {}) {
    const method = options.method || "GET";
    this.calls.push({ endpoint, method, json: options.json });
    if (endpoint === `repos/${repository}`) {
      return { default_branch: this.defaultBranch };
    }
    if (endpoint === `repos/${repository}/git/ref/heads/${encodeURIComponent(this.defaultBranch)}`) {
      const index = Math.min(this.defaultBranchReadCount, this.defaultBranchCommits.length - 1);
      this.defaultBranchReadCount += 1;
      return { object: { type: "commit", sha: this.defaultBranchCommits[index] } };
    }
    if (endpoint === `repos/${repository}/git/ref/tags/${tag}`) {
      if (!this.tagRead) {
        this.tagRead = true;
        this.beforeFirstTagRead?.();
      }
      const index = Math.min(this.tagReadCount, this.tagCommits.length - 1);
      this.tagReadCount += 1;
      return { object: { type: "commit", sha: this.tagCommits[index] } };
    }
    const comparison = endpoint.match(new RegExp(`^repos/${repository}/compare/([0-9a-f]{40})\\.\\.\\.([0-9a-f]{40})$`, "u"));
    if (comparison && method === "GET") {
      const [, baseCommit, headCommit] = comparison;
      const configuredIndex = Math.min(this.comparisonReadCount, Math.max(this.comparisonStatuses.length - 1, 0));
      const status = this.comparisonStatuses[configuredIndex]
        || (baseCommit === headCommit ? "identical" : "ahead");
      this.comparisonReadCount += 1;
      const contained = ["ahead", "identical"].includes(status);
      return {
        status,
        base_commit: { sha: baseCommit },
        merge_base_commit: { sha: contained ? baseCommit : "d".repeat(40) },
      };
    }
    const releaseList = endpoint.match(new RegExp(`^repos/${repository}/releases\\?per_page=100&page=(\\d+)$`, "u"));
    if (releaseList && method === "GET") {
      const page = Number(releaseList[1]);
      let visibleRelease = this.release;
      if (page === 1 && this.createdDraftVisibilityReadsRemaining > 0) {
        this.createdDraftVisibilityReadsRemaining -= 1;
        visibleRelease = null;
      }
      const releases = [...this.extraReleases, ...(visibleRelease ? [visibleRelease] : [])];
      return structuredClone(releases.slice((page - 1) * 100, page * 100));
    }
    if (endpoint === `repos/${repository}/releases/latest`) {
      return { id: this.latestReleaseId, tag_name: this.latestReleaseId === 7 ? tag : "v1.0.0" };
    }
    if (endpoint === `repos/${repository}/releases` && method === "POST") {
      this.release = {
        id: 7,
        tag_name: options.json.tag_name,
        target_commitish: options.json.target_commitish,
        name: options.json.name,
        body: options.json.body,
        draft: true,
        immutable: false,
        prerelease: options.json.prerelease,
        upload_url: `https://uploads.github.test/repos/${repository}/releases/7/assets{?name,label}`,
        html_url: `https://github.test/${repository}/releases/tag/${tag}`,
        assets: [],
      };
      this.createdDraftVisibilityReadsRemaining = this.createdDraftVisibilityDelay;
      return structuredClone(this.release);
    }
    if (endpoint.startsWith(`repos/${repository}/releases/assets/`) && method === "DELETE") {
      const id = Number(endpoint.split("/").at(-1));
      this.release.assets = this.release.assets.filter((asset) => asset.id !== id);
      return null;
    }
    if (endpoint === `repos/${repository}/releases/7` && method === "GET") {
      if (this.injectExtraAsset && this.release.assets.length === expectedReleaseFileNames().length && !this.injected) {
        this.injected = true;
        const bytes = Buffer.from("unexpected remote asset");
        this.release.assets.push({
          id: this.nextAssetId++,
          name: "unexpected.txt",
          size: bytes.length,
          state: "uploaded",
          digest: `sha256:${sha256(bytes)}`,
        });
      }
      if (this.injectPostPublishExtraAsset && this.release.draft === false && !this.injected) {
        this.injected = true;
        const bytes = Buffer.from("unexpected post-publication asset");
        this.release.assets.push({
          id: this.nextAssetId++,
          name: "unexpected-after-publish.txt",
          size: bytes.length,
          state: "uploaded",
          digest: `sha256:${sha256(bytes)}`,
        });
      }
      return structuredClone(this.release);
    }
    if (endpoint === `repos/${repository}/releases/7` && method === "PATCH") {
      if (options.json.draft === false && !this.refuseTransition) {
        this.release.draft = false;
        this.release.immutable = this.publishedImmutable;
        if (this.throwAfterTransition) throw new GitHubApiError(502, "Ambiguous publish response");
      }
      return structuredClone(this.release);
    }
    throw new Error(`Unexpected fake GitHub API call: ${method} ${endpoint}`);
  }

  async uploadReleaseAsset(rawUrl, suppliedRepository, releaseId, asset) {
    assert.equal(suppliedRepository, repository);
    assert.equal(releaseId, 7);
    const uploadUrl = new URL(rawUrl.replace(/\{.*$/u, ""));
    uploadUrl.searchParams.set("name", asset.name);
    this.calls.push({ endpoint: uploadUrl.toString(), method: "POST" });
    const bytes = Buffer.from(asset.bytes);
    this.release.assets.push({
      id: this.nextAssetId++,
      name: asset.name,
      size: bytes.length,
      state: "uploaded",
      digest: `sha256:${sha256(bytes)}`,
    });
    return structuredClone(this.release.assets.at(-1));
  }
}

function sourceRunJobs() {
  const jobs = [
    ["Quality and supply-chain gates", "success"],
    ["Build Linux x64", "success"],
    ["Build Windows x64", "success"],
    ["Build macOS Intel", "success"],
    ["Build macOS Apple Silicon", "success"],
    ["Verify and assemble release inventory", "success"],
    ["Release candidate gate", "success"],
    ["Attest verified release inventory", "success"],
    ["Publish GitHub Release", "failure"],
  ].map(([name, conclusion], index) => ({
    id: 1_000 + index,
    run_id: sourceRunId,
    run_attempt: 1,
    name,
    status: "completed",
    conclusion,
    steps: [],
  }));
  jobs.find((job) => job.name === "Attest verified release inventory").steps = [
    { number: 2, name: "Download verified release inventory", status: "completed", conclusion: "success" },
    { number: 3, name: "Attest release assets", status: "completed", conclusion: "success" },
  ];
  jobs.find((job) => job.name === "Publish GitHub Release").steps = [
    { number: 2, name: "Check out source", status: "completed", conclusion: "success" },
    { number: 3, name: "Install Node.js 22.23.1", status: "completed", conclusion: "success" },
    { number: 4, name: "Install verified GitHub CLI", status: "completed", conclusion: "success" },
    { number: 5, name: "Download verified release inventory", status: "completed", conclusion: "success" },
    { number: 6, name: "Verify release attestations before publication", status: "completed", conclusion: "success" },
    { number: 7, name: "Publish only an exact verified remote inventory", status: "completed", conclusion: "failure" },
  ];
  jobs.find((job) => job.name === "Publish GitHub Release").started_at = "2026-07-23T08:33:07Z";
  jobs.find((job) => job.name === "Publish GitHub Release").completed_at = "2026-07-23T08:35:06Z";
  return jobs;
}

class RecoveryFakeGitHubApi extends FakeGitHubApi {
  constructor({
    listDraft = false,
    listedConflict,
  } = {}) {
    super({
      releaseState: "draft",
      defaultBranchCommits: [recoveryWorkflowCommit],
      comparisonStatuses: ["ahead"],
    });
    this.listDraft = listDraft;
    this.listedConflict = listedConflict;
    this.release.assets = [];
    this.release.immutable = false;
    this.release.published_at = null;
    this.release.updated_at = "2026-07-23T08:34:24Z";
    this.release.author = { login: "github-actions[bot]", id: 41_898_282, type: "Bot" };
    this.sourceRun = {
      id: sourceRunId,
      name: "Release",
      path: ".github/workflows/release.yml",
      event: "push",
      status: "completed",
      conclusion: "failure",
      head_branch: tag,
      head_sha: expectedCommit,
      run_attempt: 1,
      repository: { id: repositoryId, full_name: repository },
      head_repository: { id: repositoryId, full_name: repository },
      created_at: "2026-07-23T08:18:10Z",
      updated_at: "2026-07-23T08:35:07Z",
    };
    this.recoveryRun = {
      id: recoveryRunId,
      name: "Recover release draft",
      path: ".github/workflows/release-recovery.yml",
      event: "workflow_dispatch",
      status: "in_progress",
      conclusion: null,
      head_branch: "master",
      head_sha: recoveryWorkflowCommit,
      run_attempt: 1,
      repository: { id: repositoryId, full_name: repository },
      created_at: "2026-07-23T12:00:00Z",
    };
    this.jobs = sourceRunJobs();
    this.artifact = {
      id: 3_003,
      name: "verified-release-assets",
      expired: false,
      size_in_bytes: 4_500_000,
      workflow_run: {
        id: sourceRunId,
        head_branch: tag,
        head_sha: expectedCommit,
        repository_id: repositoryId,
        head_repository_id: repositoryId,
      },
    };
  }

  async request(endpoint, options = {}) {
    const method = options.method || "GET";
    const record = () => this.calls.push({ endpoint, method, json: options.json });
    if (endpoint === `repos/${repository}/actions/runs/${recoveryRunId}`) {
      record();
      return structuredClone(this.recoveryRun);
    }
    if (endpoint === `repos/${repository}/actions/runs/${sourceRunId}`) {
      record();
      return structuredClone(this.sourceRun);
    }
    if (endpoint === `repos/${repository}/actions/runs/${sourceRunId}/jobs?filter=all&per_page=100&page=1`) {
      record();
      return { total_count: this.jobs.length, jobs: structuredClone(this.jobs) };
    }
    if (endpoint === `repos/${repository}/actions/runs/${sourceRunId}/artifacts?per_page=100&page=1`) {
      record();
      return { total_count: 1, artifacts: [structuredClone(this.artifact)] };
    }
    if (endpoint === `repos/${repository}/git/ref/tags/${tag}`) {
      record();
      return {
        ref: `refs/tags/${tag}`,
        object: { type: "tag", sha: signedTagObject },
      };
    }
    if (endpoint === `repos/${repository}/git/tags/${signedTagObject}`) {
      record();
      return {
        sha: signedTagObject,
        tag,
        object: { type: "commit", sha: expectedCommit },
        verification: {
          verified: true,
          reason: "valid",
          signature: "signed-tag-signature",
          payload: "signed-tag-payload",
        },
      };
    }
    const releaseList = endpoint.match(new RegExp(`^repos/${repository}/releases\\?per_page=100&page=(\\d+)$`, "u"));
    if (releaseList && method === "GET") {
      record();
      if (Number(releaseList[1]) !== 1) return [];
      if (this.listedConflict) return [structuredClone(this.listedConflict)];
      return this.listDraft ? [structuredClone(this.release)] : [];
    }
    return super.request(endpoint, options);
  }
}

function createRecoveryAttestation(assetDirectory, mutate) {
  const inventory = buildLocalInventory(assetDirectory, authorizedRepository.manifestPath);
  const repositoryUrl = `https://github.com/${repository}`;
  const tagRef = `refs/tags/${tag}`;
  const workflowIdentity = `${repositoryUrl}/.github/workflows/release.yml@${tagRef}`;
  const invocationId = `${repositoryUrl}/actions/runs/${sourceRunId}/attempts/1`;
  const statement = {
    _type: "https://in-toto.io/Statement/v1",
    subject: inventory.map(({ name, sha256: digest }) => ({ name, digest: { sha256: digest } })),
    predicateType: "https://slsa.dev/provenance/v1",
    predicate: {
      buildDefinition: {
        buildType: "https://actions.github.io/buildtypes/workflow/v1",
        externalParameters: {
          workflow: {
            path: ".github/workflows/release.yml",
            ref: tagRef,
            repository: repositoryUrl,
          },
        },
        internalParameters: {
          github: {
            event_name: "push",
            repository_id: String(repositoryId),
            repository_owner_id: "69587167",
            runner_environment: "github-hosted",
          },
        },
        resolvedDependencies: [{
          uri: `git+${repositoryUrl}@${tagRef}`,
          digest: { gitCommit: expectedCommit },
        }],
      },
      runDetails: {
        builder: { id: workflowIdentity },
        metadata: { invocationId },
      },
    },
  };
  const verificationResult = {
    mediaType: "application/vnd.dev.sigstore.verificationresult+json;version=0.1",
    signature: {
      certificate: {
        issuer: "https://token.actions.githubusercontent.com",
        subjectAlternativeName: workflowIdentity,
        githubWorkflowTrigger: "push",
        githubWorkflowSHA: expectedCommit,
        githubWorkflowName: "Release",
        githubWorkflowRepository: repository,
        githubWorkflowRef: tagRef,
        buildSignerURI: workflowIdentity,
        buildSignerDigest: expectedCommit,
        buildConfigURI: workflowIdentity,
        buildConfigDigest: expectedCommit,
        runnerEnvironment: "github-hosted",
        sourceRepositoryURI: repositoryUrl,
        sourceRepositoryDigest: expectedCommit,
        sourceRepositoryRef: tagRef,
        sourceRepositoryIdentifier: String(repositoryId),
        buildTrigger: "push",
        runInvocationURI: invocationId,
        sourceRepositoryVisibilityAtSigning: "public",
      },
    },
    verifiedTimestamps: [{ type: "Tlog" }],
    verifiedIdentity: { runnerEnvironment: "github-hosted" },
    statement,
  };
  const result = {
    attestation: {
      bundle: {
        verificationMaterial: { tlogEntries: [{}] },
        dsseEnvelope: {
          payloadType: "application/vnd.in-toto+json",
          payload: "",
          signatures: [{ sig: "verified-signature" }],
        },
      },
    },
    verificationResult,
  };
  mutate?.(result);
  result.attestation.bundle.dsseEnvelope.payload = Buffer
    .from(JSON.stringify(result.verificationResult.statement), "utf8")
    .toString("base64");
  const directory = mkdtempSync(path.join(tmpdir(), "eliza-recovery-attestation-"));
  const attestationPath = path.join(directory, "attestation.json");
  writeFileSync(attestationPath, `${JSON.stringify([result])}\n`, "utf8");
  return attestationPath;
}

function recoverWith(api, assetDirectory = createReleaseAssets(), overrides = {}) {
  const attestationPath = overrides.attestationPath || createRecoveryAttestation(assetDirectory);
  return recoverEmptyDraftRelease({
    api,
    repository,
    tag,
    expectedCommit,
    assetDirectory,
    releaseId: 7,
    sourceRunId,
    recoveryRunId,
    workflowCommit: recoveryWorkflowCommit,
    workflowRef: "master",
    attestationPath,
    eventName: "workflow_dispatch",
    refType: "branch",
    manifestPath: authorizedRepository.manifestPath,
    publicationPolicyPath: authorizedRepository.policyPath,
    pause: async () => {},
    ...overrides,
  });
}

function publishWith(api, assetDirectory = createReleaseAssets(), overrides = {}) {
  return publishRelease({
    api,
    repository,
    tag,
    expectedCommit,
    assetDirectory,
    eventName: "push",
    refType: "tag",
    manifestPath: authorizedRepository.manifestPath,
    publicationPolicyPath: authorizedRepository.policyPath,
    pause: async () => {},
    ...overrides,
  });
}

test("pins the API and release upload hosts", async () => {
  assert.throws(
    () => new GitHubApiClient({ token: "test-token", apiBase: "https://example.test" }),
    /API base/u,
  );
  const client = new GitHubApiClient({ token: "test-token" });
  await assert.rejects(
    client.uploadReleaseAsset(
      `https://example.test/repos/${repository}/releases/7/assets{?name,label}`,
      repository,
      7,
      { name: "asset.txt", size: 1, sha256: sha256(Buffer.from("x")), bytes: Buffer.from("x") },
    ),
    /upload host/u,
  );
});

test("refuses workflow_dispatch even when it targets a tag ref", async () => {
  const api = new FakeGitHubApi();
  await assert.rejects(
    publishWith(api, createReleaseAssets(), { eventName: "workflow_dispatch", refType: "tag" }),
    /requires a push event for a tag/u,
  );
  assert.equal(api.calls.length, 0);
});

test("fails closed before any GitHub API call while publication has no approved license", async () => {
  const api = new FakeGitHubApi();
  const disabledRepository = createDisabledRepositoryFixture();
  await assert.rejects(
    publishWith(api, createReleaseAssets(), {
      manifestPath: disabledRepository.manifestPath,
      publicationPolicyPath: disabledRepository.policyPath,
    }),
    /publication is disabled until a license is selected/u,
  );
  assert.equal(api.calls.length, 0);
});

test("discovers an authorized draft through the authenticated paginated release listing", async () => {
  const unrelated = Array.from({ length: 100 }, (_, index) => ({
    id: 1_000 + index,
    tag_name: `v0.0.${index}`,
    draft: index % 2 === 0,
  }));
  const api = new FakeGitHubApi({ extraReleases: unrelated });
  const result = await publishWith(api);
  assert.equal(result.tag, tag);
  assert.ok(api.calls.some((call) => call.endpoint === `repos/${repository}/releases?per_page=100&page=2`));
  assert.ok(api.calls.every((call) => !call.endpoint.includes("/releases/tags/")));
});

test("refuses duplicate drafts for one protected tag before mutation", async () => {
  const api = new FakeGitHubApi({ extraReleases: [{ id: 8, tag_name: tag, draft: true }] });
  await assert.rejects(publishWith(api), /multiple releases or drafts/u);
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("removes a dirty draft and publishes only the exact verified inventory", async () => {
  const api = new FakeGitHubApi();
  const result = await publishWith(api);
  assert.equal(result.assetCount, expectedReleaseFileNames().length);
  assert.equal(api.release.draft, false);
  assert.deepEqual(api.release.assets.map((asset) => asset.name).sort(), expectedReleaseFileNames());
  const deleteIndex = api.calls.findIndex((call) => call.method === "DELETE");
  const firstUploadIndex = api.calls.findIndex((call) => call.endpoint.startsWith("https://uploads.github.test/"));
  const publishIndex = api.calls.findIndex((call) => call.method === "PATCH");
  const lastUploadIndex = api.calls.map((call, index) => ({ call, index })).filter(({ call }) => call.endpoint.startsWith("https://uploads.github.test/")).at(-1).index;
  assert.ok(deleteIndex >= 0 && deleteIndex < firstUploadIndex);
  assert.ok(publishIndex > lastUploadIndex);
  assert.ok(api.calls.slice(lastUploadIndex + 1, publishIndex).some((call) => call.method === "GET"));
  assert.ok(api.calls.slice(lastUploadIndex + 1, publishIndex).some((call) => call.endpoint.includes("/compare/")));
  assert.ok(api.defaultBranchReadCount > 0);
});

test("uploads the verified byte snapshot when an asset path changes", async () => {
  const assetDirectory = createReleaseAssets();
  const assetName = expectedReleaseFileNames().find((name) => name !== "SHA256SUMS");
  const assetPath = path.join(assetDirectory, assetName);
  const originalBytes = readFileSync(assetPath);
  const api = new FakeGitHubApi({
    beforeFirstTagRead: () => writeFileSync(assetPath, "replacement that was not verified\n", "utf8"),
  });

  await publishWith(api, assetDirectory);

  const uploaded = api.release.assets.find((asset) => asset.name === assetName);
  assert.equal(uploaded.digest, `sha256:${sha256(originalBytes)}`);
  assert.notEqual(uploaded.digest, `sha256:${sha256(readFileSync(assetPath))}`);
});

test("refuses a tag that does not resolve to the workflow commit before mutation", async () => {
  const api = new FakeGitHubApi({ tagCommit: "b".repeat(40) });
  await assert.rejects(
    publishWith(api),
    /changed during draft recovery/u,
  );
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("refuses a tag that is not the current default-branch tip before mutation", async () => {
  const api = new FakeGitHubApi({ releaseState: "missing", defaultBranchCommits: ["b".repeat(40)] });
  await assert.rejects(
    publishWith(api),
    /not tied to the current master tip/u,
  );
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("does not authorize a draft when the default branch moves during authorization", async () => {
  const api = new FakeGitHubApi({
    releaseState: "missing",
    defaultBranchCommits: [expectedCommit, "b".repeat(40)],
  });
  await assert.rejects(
    publishWith(api),
    /not tied to the current master tip/u,
  );
  assert.equal(api.release, null);
  assert.ok(api.calls.every((call) => call.method !== "PATCH"));
});

test("resumes an exact authorized draft after the default branch advances", async () => {
  const api = new FakeGitHubApi({ defaultBranchCommits: ["b".repeat(40)] });
  const result = await publishWith(api);
  assert.equal(result.tag, tag);
  assert.equal(api.release.draft, false);
  assert.ok(api.defaultBranchReadCount > 0);
  assert.ok(api.comparisonReadCount > 0);
});

test("refuses a contract-shaped draft outside the current default branch before mutation", async () => {
  const api = new FakeGitHubApi({
    defaultBranchCommits: ["b".repeat(40)],
    comparisonStatuses: ["diverged"],
  });
  await assert.rejects(
    publishWith(api),
    /not identical to or an ancestor of current master during draft recovery/u,
  );
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("keeps the release in draft when the remote inventory gains an extra asset", async () => {
  const api = new FakeGitHubApi({ injectExtraAsset: true });
  await assert.rejects(
    publishWith(api),
    /not byte-for-byte identical/u,
  );
  assert.equal(api.release.draft, true);
  assert.ok(api.calls.every((call) => call.method !== "PATCH"));
});

test("refuses an already-published release whose inventory is not exact", async () => {
  const api = new FakeGitHubApi({ releaseState: "published" });
  await assert.rejects(
    publishWith(api),
    /not byte-for-byte identical/u,
  );
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("refuses to reset a draft created for a different source commit", async () => {
  const api = new FakeGitHubApi({ draftTarget: "c".repeat(40) });
  await assert.rejects(
    publishWith(api),
    /target_commitish must be exactly/u,
  );
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
});

test("refuses a contract-shaped draft whose target_commitish is not the release commit", async () => {
  const api = new FakeGitHubApi();
  api.release.target_commitish = "master";
  await assert.rejects(publishWith(api), /target_commitish must be exactly/u);
  assert.ok(api.calls.every((call) => !["POST", "PATCH", "DELETE"].includes(call.method)));
});

test("creates a missing draft before uploading and verifying assets", async () => {
  const api = new FakeGitHubApi({ releaseState: "missing" });
  const result = await publishWith(api);
  assert.equal(result.tag, tag);
  assert.ok(api.calls.some((call) => call.endpoint === `repos/${repository}/releases` && call.method === "POST"));
  assert.equal(api.release.draft, false);
});

test("waits for a newly created draft to become visible before mutating assets", async () => {
  const pauses = [];
  const api = new FakeGitHubApi({
    releaseState: "missing",
    createdDraftVisibilityDelay: 2,
  });
  const result = await publishWith(api, createReleaseAssets(), {
    pause: async (milliseconds) => pauses.push(milliseconds),
  });

  assert.equal(result.tag, tag);
  assert.deepEqual(pauses, [1000, 2000]);
  assert.equal(api.release.draft, false);
  const creationIndex = api.calls.findIndex((call) => call.endpoint === `repos/${repository}/releases` && call.method === "POST");
  const firstUploadIndex = api.calls.findIndex((call) => call.endpoint.startsWith("https://uploads.github.test/"));
  assert.ok(creationIndex >= 0 && firstUploadIndex > creationIndex);
});

test("keeps an undiscoverable newly created draft untouched after bounded retries", async () => {
  const pauses = [];
  const api = new FakeGitHubApi({
    releaseState: "missing",
    createdDraftVisibilityDelay: 10,
  });

  await assert.rejects(
    publishWith(api, createReleaseAssets(), {
      pause: async (milliseconds) => pauses.push(milliseconds),
    }),
    /could not be uniquely rediscovered before asset mutation/u,
  );

  assert.deepEqual(pauses, [1000, 2000, 4000, 5000, 5000, 5000, 5000, 5000, 5000]);
  assert.equal(api.release.draft, true);
  assert.equal(api.release.assets.length, 0);
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
  assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
});

test("reruns verify an exact immutable release without mutating it", async () => {
  const assetDirectory = createReleaseAssets();
  const api = new FakeGitHubApi();
  await publishWith(api, assetDirectory);
  const mutationsBefore = api.calls.filter((call) => ["POST", "PATCH", "DELETE"].includes(call.method)).length;
  const comparisonsBefore = api.comparisonReadCount;
  api.defaultBranchCommits = ["b".repeat(40)];
  const result = await publishWith(api, assetDirectory);
  assert.equal(result.releaseId, 7);
  assert.ok(api.comparisonReadCount > comparisonsBefore);
  assert.equal(
    api.calls.filter((call) => ["POST", "PATCH", "DELETE"].includes(call.method)).length,
    mutationsBefore,
  );
});

test("refuses an immutable rerun outside the current default branch without mutation", async () => {
  const assetDirectory = createReleaseAssets();
  const api = new FakeGitHubApi();
  await publishWith(api, assetDirectory);
  const mutationsBefore = api.calls.filter((call) => ["POST", "PATCH", "DELETE"].includes(call.method)).length;
  api.defaultBranchCommits = ["b".repeat(40)];
  api.comparisonStatuses = ["diverged"];
  api.comparisonReadCount = 0;
  await assert.rejects(
    publishWith(api, assetDirectory),
    /not identical to or an ancestor of current master during published-release verification/u,
  );
  assert.equal(
    api.calls.filter((call) => ["POST", "PATCH", "DELETE"].includes(call.method)).length,
    mutationsBefore,
  );
});

test("refuses a draft with a modified authorization body", async () => {
  const api = new FakeGitHubApi();
  api.release.body = "manually changed";
  await assert.rejects(publishWith(api), /invalid authorization body/u);
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
});

test("fails if GitHub does not complete the draft-to-published transition", async () => {
  const api = new FakeGitHubApi({ refuseTransition: true });
  await assert.rejects(
    publishWith(api),
    /did not publish/u,
  );
  assert.equal(api.release.draft, true);
});

test("never attempts to revert an immutable release after final verification fails", async () => {
  const api = new FakeGitHubApi({ injectPostPublishExtraAsset: true });
  await assert.rejects(
    publishWith(api),
    /Immutable release .* final verification failed/u,
  );
  assert.equal(api.release.draft, false);
  const transitions = api.calls.filter((call) => call.method === "PATCH");
  assert.equal(transitions.length, 1);
  assert.deepEqual(transitions[0].json, { draft: false, make_latest: "true" });
  assert.equal(api.calls.filter((call) => call.method === "DELETE").length, 1);
});

test("keeps the authorized draft when the tag moves before publication", async () => {
  const api = new FakeGitHubApi({ tagCommits: [expectedCommit, expectedCommit, "b".repeat(40)] });
  await assert.rejects(
    publishWith(api),
    /changed during final pre-publication verification/u,
  );
  assert.equal(api.release.draft, true);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 0);
});

test("reconciles an ambiguous publish response to the exact immutable release", async () => {
  const api = new FakeGitHubApi({ throwAfterTransition: true });
  const result = await publishWith(api);
  assert.equal(result.releaseId, 7);
  assert.equal(api.release.draft, false);
  assert.equal(api.release.immutable, true);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 1);
});

test("reports immutable publication if the protected tag drifts after promotion", async () => {
  const api = new FakeGitHubApi({
    tagCommits: [expectedCommit, expectedCommit, expectedCommit, "b".repeat(40)],
  });
  await assert.rejects(
    publishWith(api),
    /Immutable release .*Protected tag .* changed during published-release verification/u,
  );
  assert.equal(api.release.draft, false);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 1);
});

test("fails explicitly when GitHub does not mark the published release immutable", async () => {
  const api = new FakeGitHubApi({ publishedImmutable: false });
  await assert.rejects(
    publishWith(api),
    /Published GitHub release .* is not immutable/u,
  );
  assert.equal(api.release.draft, false);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 1);
});

test("fails explicitly when the newly published release is not latest", async () => {
  const api = new FakeGitHubApi({ latestReleaseId: 99 });
  await assert.rejects(publishWith(api), /not the latest release/u);
  assert.equal(api.release.draft, false);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 1);
});

test("recovers an exact REST-invisible empty draft from the original run artifact", async () => {
  const api = new RecoveryFakeGitHubApi();
  const result = await recoverWith(api);

  assert.equal(result.releaseId, 7);
  assert.equal(result.sourceRunId, sourceRunId);
  assert.equal(result.sourceArtifactId, api.artifact.id);
  assert.equal(result.assetCount, expectedReleaseFileNames().length);
  assert.equal(api.release.draft, false);
  assert.equal(api.release.immutable, true);
  assert.deepEqual(api.release.assets.map((asset) => asset.name).sort(), expectedReleaseFileNames());
  assert.ok(api.calls.some((call) => call.endpoint.includes(`/actions/runs/${sourceRunId}/artifacts`)));
  assert.ok(api.calls.some((call) => call.endpoint === `repos/${repository}/releases/7`));
  assert.equal(api.calls.filter((call) => call.method === "DELETE").length, 0);
  assert.equal(
    api.calls.filter((call) => call.endpoint === `repos/${repository}/releases` && call.method === "POST").length,
    0,
  );
  const publishIndex = api.calls.findIndex((call) => call.method === "PATCH");
  assert.equal(api.calls[publishIndex - 1].endpoint, `repos/${repository}/releases/7`);
  assert.equal(api.calls[publishIndex - 1].method, "GET");
});

test("recovery refuses non-dispatch execution before reading GitHub state", async () => {
  const api = new RecoveryFakeGitHubApi();
  await assert.rejects(
    recoverWith(api, createReleaseAssets(), { eventName: "push", refType: "tag" }),
    /requires an explicit workflow_dispatch event/u,
  );
  assert.equal(api.calls.length, 0);
});

test("recovery refuses a dirty exact draft without deleting or uploading anything", async () => {
  const api = new RecoveryFakeGitHubApi();
  api.release.assets.push({
    id: 99,
    name: "partial.txt",
    size: 1,
    state: "uploaded",
    digest: `sha256:${"f".repeat(64)}`,
  });
  await assert.rejects(recoverWith(api), /must be exactly empty/u);
  assert.ok(api.calls.every((call) => call.method !== "PATCH" && call.method !== "DELETE"));
  assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
});

test("recovery refuses any failed prerequisite job in the source run before release mutation", async () => {
  const api = new RecoveryFakeGitHubApi();
  api.jobs.find((job) => job.name === "Quality and supply-chain gates").conclusion = "failure";
  await assert.rejects(recoverWith(api), /Quality and supply-chain gates must conclude success/u);
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
  assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
});

test("recovery refuses an expired or mismatched original artifact before release mutation", async () => {
  for (const mutate of [
    (artifact) => { artifact.expired = true; },
    (artifact) => { artifact.workflow_run.id = sourceRunId + 1; },
  ]) {
    const api = new RecoveryFakeGitHubApi();
    mutate(api.artifact);
    await assert.rejects(recoverWith(api), /artifact (?:is expired|identity does not match)/u);
    assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
    assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
  }
});

test("recovery binds attestation subjects and invocation to the exact source run", async () => {
  const assetDirectory = createReleaseAssets();
  for (const [mutate, expected] of [
    [
      (result) => {
        result.verificationResult.statement.predicate.runDetails.metadata.invocationId =
          `https://github.com/${repository}/actions/runs/999/attempts/1`;
      },
      /run details do not match/u,
    ],
    [
      (result) => {
        result.verificationResult.statement.subject[0].digest.sha256 = "f".repeat(64);
      },
      /subjects do not exactly match/u,
    ],
  ]) {
    const api = new RecoveryFakeGitHubApi();
    const attestationPath = createRecoveryAttestation(assetDirectory, mutate);
    await assert.rejects(recoverWith(api, assetDirectory, { attestationPath }), expected);
    assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
    assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
  }
});

test("recovery requires the exact one-entry JSON array emitted by gh attestation verify", async () => {
  const assetDirectory = createReleaseAssets();
  const canonicalPath = createRecoveryAttestation(assetDirectory);
  const canonical = JSON.parse(readFileSync(canonicalPath, "utf8"));
  for (const value of [canonical[0], [canonical[0], structuredClone(canonical[0])]]) {
    const directory = mkdtempSync(path.join(tmpdir(), "eliza-recovery-attestation-shape-"));
    const attestationPath = path.join(directory, "attestation.json");
    writeFileSync(attestationPath, `${JSON.stringify(value)}\n`, "utf8");
    const api = new RecoveryFakeGitHubApi();
    await assert.rejects(
      recoverWith(api, assetDirectory, { attestationPath }),
      /must contain exactly one JSON-array result/u,
    );
    assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
    assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
  }
});

test("recovery requires the same GitHub-verified annotated tag object throughout", async () => {
  const api = new RecoveryFakeGitHubApi();
  const originalRequest = api.request.bind(api);
  let annotatedReads = 0;
  api.request = async (endpoint, options) => {
    const response = await originalRequest(endpoint, options);
    if (endpoint === `repos/${repository}/git/tags/${signedTagObject}`) {
      annotatedReads += 1;
      if (annotatedReads >= 2) response.verification.verified = false;
    }
    return response;
  };
  await assert.rejects(recoverWith(api), /does not have a valid GitHub-verified signature/u);
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
  assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
});

test("recovery refuses publication when the trusted recovery commit stops being the default tip during upload", async () => {
  const api = new RecoveryFakeGitHubApi();
  const originalUpload = api.uploadReleaseAsset.bind(api);
  let uploads = 0;
  api.uploadReleaseAsset = async (...argumentsList) => {
    const uploaded = await originalUpload(...argumentsList);
    uploads += 1;
    if (uploads === expectedReleaseFileNames().length) {
      api.defaultBranchCommits = ["d".repeat(40)];
      api.defaultBranchReadCount = 0;
    }
    return uploaded;
  };

  await assert.rejects(recoverWith(api), /is not the current master tip/u);
  assert.equal(api.release.draft, true);
  assert.equal(api.calls.filter((call) => call.method === "PATCH").length, 0);
});

test("recovery rejects a conflicting listed release even when the exact draft is addressable by ID", async () => {
  const conflict = {
    id: 8,
    tag_name: tag,
    target_commitish: expectedCommit,
    name: `ELIZA Lab ${tag}`,
    body: releaseContractBody(tag, expectedCommit),
    draft: true,
    immutable: false,
    prerelease: false,
    published_at: null,
    upload_url: `https://uploads.github.test/repos/${repository}/releases/8/assets{?name,label}`,
    author: { login: "github-actions[bot]", id: 41_898_282, type: "Bot" },
    assets: [],
  };
  const api = new RecoveryFakeGitHubApi({ listedConflict: conflict });
  await assert.rejects(recoverWith(api), /not exact recovery draft/u);
  assert.ok(api.calls.every((call) => !["PATCH", "DELETE"].includes(call.method)));
  assert.ok(api.calls.every((call) => !call.endpoint.startsWith("https://uploads.github.test/")));
});
