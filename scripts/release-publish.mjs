import { createHash } from "node:crypto";
import { readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

import {
  buildReleaseContract,
  expectedReleaseFileNames,
  readRegularFileSnapshot,
} from "./release-contract.mjs";
import { assertPublicationAuthorized } from "./release-policy.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "..");
const defaultManifestPath = path.join(repositoryRoot, "Cargo.toml");
const gitCommitPattern = /^[0-9a-f]{40}$/u;
const githubApiVersion = "2026-03-10";
const releaseWorkflowPath = ".github/workflows/release.yml";
const recoveryWorkflowPath = ".github/workflows/release-recovery.yml";
const githubActionsOidcIssuer = "https://token.actions.githubusercontent.com";
const slsaProvenancePredicate = "https://slsa.dev/provenance/v1";
const recoveryJobContract = Object.freeze([
  Object.freeze({ name: "Quality and supply-chain gates", conclusion: "success" }),
  Object.freeze({ name: "Build Linux x64", conclusion: "success" }),
  Object.freeze({ name: "Build Windows x64", conclusion: "success" }),
  Object.freeze({ name: "Build macOS Intel", conclusion: "success" }),
  Object.freeze({ name: "Build macOS Apple Silicon", conclusion: "success" }),
  Object.freeze({ name: "Verify and assemble release inventory", conclusion: "success" }),
  Object.freeze({ name: "Release candidate gate", conclusion: "success" }),
  Object.freeze({ name: "Attest verified release inventory", conclusion: "success" }),
  Object.freeze({ name: "Publish GitHub Release", conclusion: "failure" }),
]);

function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function sha256Buffer(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

export function releaseContractBody(tag, expectedCommit) {
  return [
    "Verified cross-platform build of ELIZA Lab.",
    "",
    `Source commit: \`${expectedCommit}\``,
    "",
    `<!-- eliza-release-contract:v1:${tag}:${expectedCommit} -->`,
  ].join("\n");
}

function contentType(fileName) {
  if (fileName.endsWith(".tar.gz")) return "application/gzip";
  if (fileName.endsWith(".zip")) return "application/zip";
  if (fileName.endsWith(".json")) return "application/json";
  return "text/plain; charset=utf-8";
}

function parseChecksums(contents) {
  const entries = contents.trimEnd().split(/\r?\n/u).map((line) => {
    const match = line.match(/^([0-9a-f]{64})  ([^/\\\r\n]+)$/u);
    invariant(match, `Invalid SHA256SUMS entry: ${JSON.stringify(line)}`);
    return Object.freeze({ sha256: match[1], name: match[2] });
  });
  invariant(entries.length > 0, "SHA256SUMS is empty");
  const names = entries.map((entry) => entry.name);
  invariant(new Set(names).size === names.length, "SHA256SUMS contains duplicate filenames");
  return entries.sort((left, right) => left.name.localeCompare(right.name));
}

export function buildLocalInventory(directory, manifestPath = defaultManifestPath) {
  const root = path.resolve(directory);
  invariant(statSync(root).isDirectory(), `Release asset path is not a directory: ${root}`);
  const expectedNames = expectedReleaseFileNames(manifestPath);
  const names = readdirSync(root).sort();
  invariant(isDeepStrictEqual(names, expectedNames), `Local release inventory differs from the contract. Expected ${expectedNames.join(", ")}; found ${names.join(", ")}`);

  const inventory = names.map((name) => {
    const filePath = path.join(root, name);
    const bytes = readRegularFileSnapshot(filePath, `Release asset ${name}`);
    return Object.freeze({ name, size: bytes.length, sha256: sha256Buffer(bytes), bytes });
  });

  const checksumAsset = inventory.find((entry) => entry.name === "SHA256SUMS");
  invariant(checksumAsset, "Local release inventory is missing SHA256SUMS");
  const checksumEntries = parseChecksums(checksumAsset.bytes.toString("utf8"));
  const expectedChecksums = inventory
    .filter((entry) => entry.name !== "SHA256SUMS")
    .map((entry) => ({ name: entry.name, sha256: entry.sha256 }))
    .sort((left, right) => left.name.localeCompare(right.name));
  invariant(isDeepStrictEqual(checksumEntries, expectedChecksums), "SHA256SUMS does not exactly cover every other release asset");
  return inventory;
}

export class GitHubApiError extends Error {
  constructor(status, message) {
    super(`GitHub API ${status}: ${message}`);
    this.name = "GitHubApiError";
    this.status = status;
  }
}

export class GitHubApiClient {
  constructor({ token, apiBase = "https://api.github.com" }) {
    invariant(token, "GITHUB_TOKEN is required");
    const baseUrl = new URL(apiBase);
    invariant(baseUrl.protocol === "https:" && baseUrl.hostname === "api.github.com" && baseUrl.pathname === "/", "GitHub API base must be https://api.github.com");
    this.token = token;
    this.apiBase = baseUrl.origin;
  }

  assertUploadUrl(rawUrl, repository, releaseId) {
    const uploadUrl = new URL(rawUrl.replace(/\{.*$/u, ""));
    invariant(uploadUrl.protocol === "https:", "GitHub release upload URL must use HTTPS");
    invariant(uploadUrl.hostname === "uploads.github.com", `Unexpected GitHub release upload host: ${uploadUrl.hostname}`);
    invariant(uploadUrl.pathname === `/repos/${repository}/releases/${releaseId}/assets`, "GitHub release upload URL does not match the expected repository and draft");
  }

  async request(endpoint, {
    method = "GET",
    json,
    raw = false,
  } = {}) {
    invariant(!/^https?:/u.test(endpoint), "Absolute URLs are not allowed in general GitHub API requests");
    const url = `${this.apiBase}/${endpoint.replace(/^\//u, "")}`;
    const headers = {
      Accept: raw ? "application/octet-stream" : "application/vnd.github+json",
      Authorization: `Bearer ${this.token}`,
      "User-Agent": "ELIZA-Lab-release-publisher",
      "X-GitHub-Api-Version": githubApiVersion,
    };
    let requestBody;
    if (json !== undefined) {
      headers["Content-Type"] = "application/json";
      requestBody = JSON.stringify(json);
    }

    const response = await fetch(url, { method, headers, body: requestBody, redirect: "error" });
    if (!response.ok) {
      let message = response.statusText;
      try {
        message = (await response.json()).message || message;
      } catch {
        // Keep the HTTP status text when GitHub did not return JSON.
      }
      throw new GitHubApiError(response.status, message);
    }
    if (response.status === 204) {
      return null;
    }
    if (raw) {
      return Buffer.from(await response.arrayBuffer());
    }
    return response.json();
  }

  async uploadReleaseAsset(rawUrl, repository, releaseId, asset) {
    this.assertUploadUrl(rawUrl, repository, releaseId);
    invariant(asset && /^[^/\\\r\n]+$/u.test(asset.name), "Release asset has an invalid filename");
    invariant(Buffer.isBuffer(asset.bytes), `Release asset ${asset.name} is missing its verified byte snapshot`);
    invariant(asset.bytes.length === asset.size, `Release asset ${asset.name} changed size before upload`);
    invariant(sha256Buffer(asset.bytes) === asset.sha256, `Release asset ${asset.name} changed after verification`);

    const uploadUrl = new URL(rawUrl.replace(/\{.*$/u, ""));
    uploadUrl.searchParams.set("name", asset.name);
    const headers = {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${this.token}`,
      "Content-Type": contentType(asset.name),
      "Content-Length": String(asset.bytes.length),
      "User-Agent": "ELIZA-Lab-release-publisher",
      "X-GitHub-Api-Version": githubApiVersion,
    };

    // This is the intended release boundary: only the exact checksummed inventory is sent to GitHub's pinned upload host.
    // codeql[js/file-access-to-http]
    const response = await fetch(uploadUrl, { method: "POST", headers, body: asset.bytes, redirect: "error" });
    if (!response.ok) {
      let message = response.statusText;
      try {
        message = (await response.json()).message || message;
      } catch {
        // Keep the HTTP status text when GitHub did not return JSON.
      }
      throw new GitHubApiError(response.status, message);
    }
    return response.json();
  }
}

function validateRepository(repository) {
  invariant(/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(repository), `Invalid GitHub repository: ${repository}`);
}

function validatePositiveSafeInteger(value, label) {
  invariant(Number.isSafeInteger(value) && value > 0, `${label} must be a positive safe integer`);
  return value;
}

function parsePositiveSafeInteger(value, label) {
  invariant(typeof value === "string" && /^[1-9][0-9]*$/u.test(value), `${label} must be a canonical positive integer`);
  return validatePositiveSafeInteger(Number(value), label);
}

async function readReleaseById(api, repository, releaseId) {
  validatePositiveSafeInteger(releaseId, "Recovery release ID");
  const release = await api.request(`repos/${repository}/releases/${releaseId}`);
  invariant(release?.id === releaseId, `GitHub returned release ${release?.id} for requested recovery release ${releaseId}`);
  return release;
}

async function findReleaseForTag(api, repository, tag) {
  const matches = [];
  const pageSize = 100;
  const maximumPages = 100;

  for (let page = 1; page <= maximumPages; page += 1) {
    const releases = await api.request(`repos/${repository}/releases?per_page=${pageSize}&page=${page}`);
    invariant(Array.isArray(releases), "GitHub release listing did not return an array");
    for (const release of releases) {
      if (release?.tag_name === tag) {
        matches.push(release);
      }
    }
    if (releases.length < pageSize) {
      invariant(matches.length <= 1, `GitHub contains multiple releases or drafts for protected tag ${tag}`);
      return matches[0];
    }
  }

  throw new Error(`GitHub release listing exceeded ${maximumPages * pageSize} entries; refusing incomplete draft discovery`);
}

export function assertTrustedTagEvent(eventName, refType) {
  invariant(
    eventName === "push" && refType === "tag",
    `Release publication requires a push event for a tag; received ${eventName || "missing event"}/${refType || "missing ref type"}`,
  );
}

export function assertTrustedRecoveryDispatch(eventName, refType) {
  invariant(
    eventName === "workflow_dispatch" && refType === "branch",
    `Release recovery requires an explicit workflow_dispatch event for the default branch; received ${eventName || "missing event"}/${refType || "missing ref type"}`,
  );
}

async function resolveTagCommit(api, repository, tag) {
  const ref = await api.request(`repos/${repository}/git/ref/tags/${encodeURIComponent(tag)}`);
  let object = ref?.object;
  for (let depth = 0; depth < 5; depth += 1) {
    invariant(object && gitCommitPattern.test(object.sha), "GitHub tag reference returned an invalid object SHA");
    if (object.type === "commit") {
      return object.sha;
    }
    invariant(object.type === "tag", `GitHub tag reference points to unsupported object type: ${object.type}`);
    const annotatedTag = await api.request(`repos/${repository}/git/tags/${object.sha}`);
    object = annotatedTag.object;
  }
  throw new Error(`GitHub tag ${tag} exceeds the supported annotation depth`);
}

async function resolveVerifiedTagCommit(api, repository, tag, expectedCommit, expectedTagObjectSha) {
  const ref = await api.request(`repos/${repository}/git/ref/tags/${encodeURIComponent(tag)}`);
  invariant(ref?.ref === `refs/tags/${tag}`, `GitHub returned an unexpected reference for protected tag ${tag}`);
  invariant(
    ref?.object?.type === "tag" && gitCommitPattern.test(ref.object.sha),
    `Recovery requires protected tag ${tag} to be a signed annotated tag`,
  );
  if (expectedTagObjectSha !== undefined) {
    invariant(ref.object.sha === expectedTagObjectSha, `Signed tag object for ${tag} changed during recovery`);
  }

  const annotatedTag = await api.request(`repos/${repository}/git/tags/${ref.object.sha}`);
  invariant(annotatedTag?.sha === ref.object.sha, `GitHub returned the wrong annotated tag object for ${tag}`);
  invariant(annotatedTag.tag === tag, `Annotated tag object ${ref.object.sha} does not identify ${tag}`);
  invariant(
    annotatedTag?.object?.type === "commit" && annotatedTag.object.sha === expectedCommit,
    `Signed tag ${tag} does not point directly to release commit ${expectedCommit}`,
  );
  invariant(
    annotatedTag?.verification?.verified === true
      && annotatedTag.verification.reason === "valid"
      && typeof annotatedTag.verification.signature === "string"
      && annotatedTag.verification.signature.length > 0
      && typeof annotatedTag.verification.payload === "string"
      && annotatedTag.verification.payload.length > 0,
    `Annotated tag ${tag} does not have a valid GitHub-verified signature`,
  );
  return Object.freeze({ commit: expectedCommit, tagObjectSha: ref.object.sha });
}

async function resolveDefaultBranchTip(api, repository) {
  const repositoryState = await api.request(`repos/${repository}`);
  const branch = repositoryState?.default_branch;
  invariant(
    typeof branch === "string"
      && branch.length > 0
      && branch.length <= 255
      && /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/-]+$/u.test(branch),
    "GitHub repository returned an invalid default branch",
  );
  const ref = await api.request(`repos/${repository}/git/ref/heads/${encodeURIComponent(branch)}`);
  invariant(
    ref?.object?.type === "commit" && gitCommitPattern.test(ref.object.sha),
    `GitHub default branch ${branch} does not resolve directly to a commit`,
  );
  return Object.freeze({ branch, commit: ref.object.sha });
}

function validateRunRepository(run, repository, label) {
  invariant(run?.repository?.full_name === repository, `${label} belongs to an unexpected repository`);
  return validatePositiveSafeInteger(run.repository.id, `${label} repository ID`);
}

function validateIsoTimestamp(value, label) {
  invariant(typeof value === "string" && Number.isFinite(Date.parse(value)), `${label} must be an ISO timestamp`);
  return Date.parse(value);
}

async function verifyRecoveryExecution({
  api,
  repository,
  recoveryRunId,
  workflowCommit,
  workflowRef,
}) {
  validatePositiveSafeInteger(recoveryRunId, "Recovery workflow run ID");
  invariant(gitCommitPattern.test(workflowCommit), "Recovery workflow commit must be a lowercase 40-character Git SHA");
  invariant(
    typeof workflowRef === "string" && workflowRef.length > 0 && workflowRef.length <= 255,
    "Recovery workflow ref must be a branch name",
  );

  const recoveryRun = await api.request(`repos/${repository}/actions/runs/${recoveryRunId}`);
  invariant(recoveryRun?.id === recoveryRunId, `GitHub returned the wrong recovery workflow run`);
  invariant(recoveryRun.name === "Recover release draft", "Recovery must execute from the dedicated recovery workflow");
  invariant(recoveryRun.path === recoveryWorkflowPath, "Recovery workflow path does not match the dedicated recovery workflow");
  invariant(recoveryRun.event === "workflow_dispatch", "Recovery workflow run was not explicitly dispatched");
  invariant(recoveryRun.status === "in_progress" && recoveryRun.conclusion === null, "Recovery workflow run is not actively executing");
  invariant(recoveryRun.head_branch === workflowRef, "Recovery workflow run branch does not match its execution ref");
  invariant(recoveryRun.head_sha === workflowCommit, "Recovery workflow run commit does not match the checked-out recovery code");
  validatePositiveSafeInteger(recoveryRun.run_attempt, "Recovery workflow run attempt");
  const repositoryId = validateRunRepository(recoveryRun, repository, "Recovery workflow run");

  const defaultTip = await resolveDefaultBranchTip(api, repository);
  invariant(workflowRef === defaultTip.branch, `Recovery workflow must be dispatched from default branch ${defaultTip.branch}`);
  invariant(
    workflowCommit === defaultTip.commit,
    `Recovery workflow commit ${workflowCommit} is not the current ${defaultTip.branch} tip ${defaultTip.commit}`,
  );
  const createdAt = validateIsoTimestamp(recoveryRun.created_at, "Recovery workflow creation time");
  return Object.freeze({ createdAt, defaultBranch: defaultTip.branch, repositoryId });
}

function requireRunStep(job, name, conclusion) {
  invariant(Array.isArray(job?.steps), `Workflow job ${job?.name || "unknown"} is missing its step evidence`);
  const matches = job.steps.filter((step) => step?.name === name);
  invariant(matches.length === 1, `Workflow job ${job.name} must contain exactly one ${name} step`);
  const [step] = matches;
  invariant(
    step.status === "completed" && step.conclusion === conclusion,
    `Workflow step ${job.name}/${name} must be completed with conclusion ${conclusion}`,
  );
  invariant(Number.isSafeInteger(step.number) && step.number > 0, `Workflow step ${job.name}/${name} has an invalid number`);
  return step;
}

async function readSourceRunArtifacts(api, repository, sourceRunId) {
  const artifacts = [];
  const artifactIds = new Set();
  const pageSize = 100;
  const maximumPages = 100;
  let expectedTotal;

  for (let page = 1; page <= maximumPages; page += 1) {
    const response = await api.request(
      `repos/${repository}/actions/runs/${sourceRunId}/artifacts?per_page=${pageSize}&page=${page}`,
    );
    invariant(
      Number.isSafeInteger(response?.total_count) && response.total_count >= 0,
      "Source workflow artifact listing returned an invalid total",
    );
    if (expectedTotal === undefined) expectedTotal = response.total_count;
    invariant(response.total_count === expectedTotal, "Source workflow artifact total changed during verification");
    invariant(Array.isArray(response.artifacts), "Source workflow artifact listing did not return an array");

    for (const artifact of response.artifacts) {
      validatePositiveSafeInteger(artifact?.id, "Source workflow artifact ID");
      invariant(!artifactIds.has(artifact.id), `Source workflow artifact ${artifact.id} was returned more than once`);
      artifactIds.add(artifact.id);
      artifacts.push(artifact);
    }
    invariant(artifacts.length <= expectedTotal, "Source workflow returned more artifacts than its declared total");
    if (response.artifacts.length < pageSize) {
      invariant(artifacts.length === expectedTotal, "Source workflow artifact listing was incomplete");
      return artifacts;
    }
  }

  throw new Error(`Source workflow artifact listing exceeded ${maximumPages * pageSize} entries`);
}

async function verifyRecoverySourceRun({
  api,
  repository,
  repositoryId,
  tag,
  expectedCommit,
  sourceRunId,
  recoveryCreatedAt,
}) {
  validatePositiveSafeInteger(sourceRunId, "Source workflow run ID");
  const sourceRun = await api.request(`repos/${repository}/actions/runs/${sourceRunId}`);
  invariant(sourceRun?.id === sourceRunId, "GitHub returned the wrong source workflow run");
  invariant(sourceRun.name === "Release" && sourceRun.path === releaseWorkflowPath, "Source run is not the protected release workflow");
  invariant(sourceRun.event === "push", "Source release workflow was not triggered by a tag push");
  invariant(sourceRun.status === "completed" && sourceRun.conclusion === "failure", "Source release workflow must be completed with only publication failing");
  invariant(sourceRun.head_branch === tag, `Source release workflow tag does not match ${tag}`);
  invariant(sourceRun.head_sha === expectedCommit, "Source release workflow commit does not match the requested release commit");
  const sourceAttempt = validatePositiveSafeInteger(sourceRun.run_attempt, "Source workflow run attempt");
  invariant(validateRunRepository(sourceRun, repository, "Source workflow run") === repositoryId, "Source and recovery runs use different repository identities");
  invariant(
    sourceRun?.head_repository?.id === repositoryId && sourceRun.head_repository.full_name === repository,
    "Source workflow head repository does not match the protected repository",
  );
  const sourceCreatedAt = validateIsoTimestamp(sourceRun.created_at, "Source workflow creation time");
  const sourceUpdatedAt = validateIsoTimestamp(sourceRun.updated_at, "Source workflow update time");
  invariant(sourceCreatedAt <= sourceUpdatedAt, "Source workflow timestamps are inconsistent");
  invariant(sourceCreatedAt < recoveryCreatedAt, "Source release workflow is not older than the recovery workflow");

  const jobsResponse = await api.request(
    `repos/${repository}/actions/runs/${sourceRunId}/jobs?filter=all&per_page=100&page=1`,
  );
  invariant(
    jobsResponse?.total_count === recoveryJobContract.length
      && Array.isArray(jobsResponse.jobs)
      && jobsResponse.jobs.length === recoveryJobContract.length,
    `Source release workflow must contain exactly ${recoveryJobContract.length} jobs`,
  );
  const jobsByName = new Map();
  for (const job of jobsResponse.jobs) {
    invariant(typeof job?.name === "string" && !jobsByName.has(job.name), "Source release workflow contains duplicate or invalid job names");
    invariant(job.run_id === sourceRunId && job.run_attempt === sourceAttempt, `Workflow job ${job.name} belongs to the wrong run attempt`);
    invariant(job.status === "completed", `Workflow job ${job.name} is not completed`);
    jobsByName.set(job.name, job);
  }
  for (const expectedJob of recoveryJobContract) {
    const job = jobsByName.get(expectedJob.name);
    invariant(job, `Source release workflow is missing job ${expectedJob.name}`);
    invariant(job.conclusion === expectedJob.conclusion, `Workflow job ${expectedJob.name} must conclude ${expectedJob.conclusion}`);
  }
  invariant(jobsByName.size === recoveryJobContract.length, "Source release workflow contains an unexpected job");

  const attestJob = jobsByName.get("Attest verified release inventory");
  const attestDownload = requireRunStep(attestJob, "Download verified release inventory", "success");
  const attestStep = requireRunStep(attestJob, "Attest release assets", "success");
  invariant(attestDownload.number < attestStep.number, "Source workflow attested assets before downloading the verified inventory");

  const publishJob = jobsByName.get("Publish GitHub Release");
  const requiredPublishSteps = [
    ["Check out source", "success"],
    ["Install Node.js 22.23.1", "success"],
    ["Install verified GitHub CLI", "success"],
    ["Download verified release inventory", "success"],
    ["Verify release attestations before publication", "success"],
    ["Publish only an exact verified remote inventory", "failure"],
  ].map(([name, conclusion]) => requireRunStep(publishJob, name, conclusion));
  invariant(
    requiredPublishSteps.every((step, index) => index === 0 || requiredPublishSteps[index - 1].number < step.number),
    "Source publication steps did not execute in the protected order",
  );
  invariant(
    publishJob.steps.filter((step) => step?.conclusion === "failure").length === 1,
    "Source publication job must have exactly one failed step",
  );
  const publishStartedAt = validateIsoTimestamp(publishJob.started_at, "Source publication job start time");
  const publishCompletedAt = validateIsoTimestamp(publishJob.completed_at, "Source publication job completion time");
  invariant(
    sourceCreatedAt <= publishStartedAt
      && publishStartedAt <= publishCompletedAt
      && publishCompletedAt <= sourceUpdatedAt,
    "Source publication job timestamps fall outside the source workflow",
  );

  const artifacts = await readSourceRunArtifacts(api, repository, sourceRunId);
  const verifiedArtifacts = artifacts.filter((artifact) => artifact?.name === "verified-release-assets");
  invariant(verifiedArtifacts.length === 1, "Source workflow must contain exactly one verified-release-assets artifact");
  const [artifact] = verifiedArtifacts;
  invariant(artifact.expired === false, "Source verified-release-assets artifact is expired");
  invariant(
    Number.isSafeInteger(artifact.size_in_bytes) && artifact.size_in_bytes > 0,
    "Source verified-release-assets artifact has an invalid size",
  );
  invariant(
    artifact?.workflow_run?.id === sourceRunId
      && artifact.workflow_run.head_branch === tag
      && artifact.workflow_run.head_sha === expectedCommit
      && artifact.workflow_run.repository_id === repositoryId
      && artifact.workflow_run.head_repository_id === repositoryId,
    "Source verified-release-assets artifact identity does not match the protected source run",
  );

  return Object.freeze({
    artifactId: artifact.id,
    publishCompletedAt,
    publishStartedAt,
    runAttempt: sourceAttempt,
  });
}

async function verifyReleaseSourceAtDefaultTip(api, repository, tag, expectedCommit, phase) {
  const before = await resolveDefaultBranchTip(api, repository);
  const tagCommit = await resolveTagCommit(api, repository, tag);
  const after = await resolveDefaultBranchTip(api, repository);
  invariant(before.branch === after.branch, `Default branch changed during ${phase} verification`);
  invariant(tagCommit === expectedCommit, `Protected tag ${tag} resolves to ${tagCommit}, not ${expectedCommit}`);
  invariant(
    before.commit === expectedCommit && after.commit === expectedCommit,
    `Protected tag ${tag} is not tied to the current ${after.branch} tip during ${phase} verification`,
  );
  return after.branch;
}

async function verifyReleaseSourceContainedInDefaultBranch(api, repository, tag, expectedCommit, phase) {
  const before = await resolveDefaultBranchTip(api, repository);
  const tagCommit = await resolveTagCommit(api, repository, tag);
  invariant(tagCommit === expectedCommit, `Protected tag ${tag} changed during ${phase}`);
  const comparison = await api.request(
    `repos/${repository}/compare/${expectedCommit}...${before.commit}`,
  );
  const after = await resolveDefaultBranchTip(api, repository);
  invariant(
    before.branch === after.branch && before.commit === after.commit,
    `Default branch changed during ${phase} verification`,
  );
  invariant(
    comparison?.base_commit?.sha === expectedCommit
      && comparison?.merge_base_commit?.sha === expectedCommit
      && ["ahead", "identical"].includes(comparison?.status),
    `Release commit ${expectedCommit} is not identical to or an ancestor of current ${after.branch} during ${phase}`,
  );
  return after.branch;
}

async function verifyTagCommit(api, repository, tag, expectedCommit, phase) {
  const tagCommit = await resolveTagCommit(api, repository, tag);
  invariant(tagCommit === expectedCommit, `Protected tag ${tag} changed during ${phase}`);
}

async function verifySignedReleaseSourceContainedInDefaultBranch(
  api,
  repository,
  tag,
  expectedCommit,
  tagObjectSha,
  phase,
) {
  await resolveVerifiedTagCommit(api, repository, tag, expectedCommit, tagObjectSha);
  await verifyReleaseSourceContainedInDefaultBranch(api, repository, tag, expectedCommit, phase);
  await resolveVerifiedTagCommit(api, repository, tag, expectedCommit, tagObjectSha);
}

function readAttestationVerification(attestationPath) {
  const bytes = readRegularFileSnapshot(path.resolve(attestationPath), "Recovery attestation verification");
  invariant(bytes.length > 0 && bytes.length <= 32 * 1024 * 1024, "Recovery attestation verification has an invalid size");
  let result;
  try {
    result = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`Recovery attestation verification is not valid JSON: ${error.message}`);
  }
  invariant(
    Array.isArray(result)
      && result.length === 1
      && result[0]
      && typeof result[0] === "object"
      && !Array.isArray(result[0]),
    "Recovery attestation verification must contain exactly one JSON-array result",
  );
  return result[0];
}

function decodeAttestationPayload(result) {
  const envelope = result?.attestation?.bundle?.dsseEnvelope;
  invariant(envelope?.payloadType === "application/vnd.in-toto+json", "Recovery attestation has an unexpected DSSE payload type");
  invariant(
    typeof envelope.payload === "string"
      && envelope.payload.length > 0
      && /^[A-Za-z0-9+/]+={0,2}$/u.test(envelope.payload),
    "Recovery attestation has an invalid DSSE payload",
  );
  invariant(
    Array.isArray(envelope.signatures)
      && envelope.signatures.length > 0
      && envelope.signatures.every((signature) => typeof signature?.sig === "string" && signature.sig.length > 0),
    "Recovery attestation does not contain a DSSE signature",
  );
  invariant(
    Array.isArray(result?.attestation?.bundle?.verificationMaterial?.tlogEntries)
      && result.attestation.bundle.verificationMaterial.tlogEntries.length > 0,
    "Recovery attestation does not contain transparency-log evidence",
  );
  try {
    return JSON.parse(Buffer.from(envelope.payload, "base64").toString("utf8"));
  } catch (error) {
    throw new Error(`Recovery attestation DSSE payload is not valid JSON: ${error.message}`);
  }
}

function verifyRecoveryAttestation({
  attestationPath,
  repository,
  repositoryId,
  tag,
  expectedCommit,
  sourceRunId,
  sourceRunAttempt,
  localInventory,
}) {
  const result = readAttestationVerification(attestationPath);
  const verification = result.verificationResult;
  invariant(
    verification?.mediaType === "application/vnd.dev.sigstore.verificationresult+json;version=0.1",
    "Recovery attestation is not a GitHub CLI verification result",
  );
  invariant(
    Array.isArray(verification.verifiedTimestamps) && verification.verifiedTimestamps.length > 0,
    "Recovery attestation has no verified transparency-log timestamp",
  );
  invariant(
    verification?.verifiedIdentity?.runnerEnvironment === "github-hosted",
    "Recovery attestation identity is not bound to a GitHub-hosted runner",
  );

  const repositoryUrl = `https://github.com/${repository}`;
  const tagRef = `refs/tags/${tag}`;
  const workflowIdentity = `${repositoryUrl}/${releaseWorkflowPath}@${tagRef}`;
  const invocationId = `${repositoryUrl}/actions/runs/${sourceRunId}/attempts/${sourceRunAttempt}`;
  const certificate = verification?.signature?.certificate;
  invariant(certificate?.issuer === githubActionsOidcIssuer, "Recovery attestation certificate has an unexpected OIDC issuer");
  invariant(certificate.subjectAlternativeName === workflowIdentity, "Recovery attestation certificate has an unexpected workflow identity");
  invariant(certificate.githubWorkflowTrigger === "push" && certificate.buildTrigger === "push", "Recovery attestation was not produced by a tag push");
  invariant(certificate.githubWorkflowSHA === expectedCommit, "Recovery attestation workflow SHA does not match the release commit");
  invariant(certificate.githubWorkflowName === "Release", "Recovery attestation has an unexpected workflow name");
  invariant(certificate.githubWorkflowRepository === repository, "Recovery attestation has an unexpected workflow repository");
  invariant(certificate.githubWorkflowRef === tagRef, "Recovery attestation has an unexpected workflow ref");
  invariant(
    certificate.buildSignerURI === workflowIdentity
      && certificate.buildSignerDigest === expectedCommit
      && certificate.buildConfigURI === workflowIdentity
      && certificate.buildConfigDigest === expectedCommit,
    "Recovery attestation signer identity does not match the protected release workflow",
  );
  invariant(certificate.runnerEnvironment === "github-hosted", "Recovery attestation certificate identifies a self-hosted runner");
  invariant(
    certificate.sourceRepositoryURI === repositoryUrl
      && certificate.sourceRepositoryDigest === expectedCommit
      && certificate.sourceRepositoryRef === tagRef
      && certificate.sourceRepositoryIdentifier === String(repositoryId),
    "Recovery attestation source identity does not match the protected repository",
  );
  invariant(certificate.runInvocationURI === invocationId, "Recovery attestation certificate belongs to a different workflow run");
  invariant(certificate.sourceRepositoryVisibilityAtSigning === "public", "Recovery attestation was not signed for the public source repository");

  const statement = verification.statement;
  const signedStatement = decodeAttestationPayload(result);
  invariant(isDeepStrictEqual(signedStatement, statement), "Verified attestation statement differs from its signed DSSE payload");
  invariant(statement?._type === "https://in-toto.io/Statement/v1", "Recovery attestation has an unexpected statement type");
  invariant(statement.predicateType === slsaProvenancePredicate, "Recovery attestation has an unexpected predicate type");
  invariant(Array.isArray(statement.subject), "Recovery attestation subject is not an array");
  const actualSubjects = statement.subject.map((subject) => {
    invariant(
      subject
        && typeof subject.name === "string"
        && /^[^/\\\r\n]+$/u.test(subject.name)
        && subject.digest
        && isDeepStrictEqual(Object.keys(subject.digest), ["sha256"])
        && /^[0-9a-f]{64}$/u.test(subject.digest.sha256),
      "Recovery attestation contains an invalid subject",
    );
    return { name: subject.name, sha256: subject.digest.sha256 };
  }).sort((left, right) => left.name.localeCompare(right.name));
  invariant(
    new Set(actualSubjects.map((subject) => subject.name)).size === actualSubjects.length,
    "Recovery attestation contains duplicate subjects",
  );
  const expectedSubjects = localInventory
    .map(({ name, sha256 }) => ({ name, sha256 }))
    .sort((left, right) => left.name.localeCompare(right.name));
  invariant(
    isDeepStrictEqual(actualSubjects, expectedSubjects),
    "Recovery attestation subjects do not exactly match the verified local release inventory",
  );

  const buildDefinition = statement?.predicate?.buildDefinition;
  invariant(
    buildDefinition?.buildType === "https://actions.github.io/buildtypes/workflow/v1",
    "Recovery attestation has an unexpected build type",
  );
  invariant(
    isDeepStrictEqual(
      buildDefinition.externalParameters?.workflow,
      { path: releaseWorkflowPath, ref: tagRef, repository: repositoryUrl },
    ),
    "Recovery attestation external workflow identity does not match the release workflow",
  );
  invariant(
    buildDefinition.internalParameters?.github?.event_name === "push"
      && buildDefinition.internalParameters.github.repository_id === String(repositoryId)
      && buildDefinition.internalParameters.github.runner_environment === "github-hosted",
    "Recovery attestation internal workflow identity does not match the source run",
  );
  invariant(
    isDeepStrictEqual(
      buildDefinition.resolvedDependencies,
      [{
        uri: `git+${repositoryUrl}@${tagRef}`,
        digest: { gitCommit: expectedCommit },
      }],
    ),
    "Recovery attestation resolved source does not match the signed release tag",
  );
  invariant(
    statement?.predicate?.runDetails?.builder?.id === workflowIdentity
      && statement.predicate.runDetails.metadata?.invocationId === invocationId,
    "Recovery attestation run details do not match the exact source workflow attempt",
  );
}

function validateReleaseMetadata(release, tag, expectedCommit) {
  invariant(Number.isSafeInteger(release?.id) && release.id > 0, "GitHub release has an invalid ID");
  invariant(release.tag_name === tag, `GitHub release tag ${release.tag_name} does not match ${tag}`);
  invariant(release.prerelease === false, `GitHub draft ${tag} is unexpectedly marked as a prerelease`);
  invariant(release.name === `ELIZA Lab ${tag}`, `GitHub draft ${tag} has unexpected release metadata`);
  invariant(
    release.target_commitish === expectedCommit,
    `GitHub release ${tag} target_commitish must be exactly ${expectedCommit}`,
  );
  invariant(release.body === releaseContractBody(tag, expectedCommit), `GitHub release ${tag} has an invalid authorization body`);
  invariant(Array.isArray(release.assets), "GitHub release does not contain an asset inventory");
}

function validateDraft(release, tag, expectedCommit) {
  validateReleaseMetadata(release, tag, expectedCommit);
  invariant(release.draft === true, `Refusing to modify already-published release ${tag}`);
  invariant(typeof release.upload_url === "string", "GitHub draft is missing its upload URL");
}

function validateEmptyRecoveryDraft(release, tag, expectedCommit, releaseId, sourcePublication) {
  validateDraft(release, tag, expectedCommit);
  invariant(release.id === releaseId, `Recovery draft ID does not match ${releaseId}`);
  invariant(release.immutable === false, `Recovery draft ${tag} is unexpectedly immutable`);
  invariant(release.published_at === null, `Recovery draft ${tag} has publication history`);
  invariant(
    release?.author?.login === "github-actions[bot]"
      && release.author.id === 41_898_282
      && release.author.type === "Bot",
    `Recovery draft ${tag} was not created by GitHub Actions`,
  );
  invariant(release.assets.length === 0, `Recovery draft ${tag} must be exactly empty`);
  const updatedAt = validateIsoTimestamp(release.updated_at, `Recovery draft ${tag} update time`);
  invariant(
    updatedAt >= sourcePublication.publishStartedAt && updatedAt <= sourcePublication.publishCompletedAt,
    `Recovery draft ${tag} was not created or last updated by the exact failed source publication job`,
  );
}

function validatePublishedRelease(release, tag, expectedCommit) {
  validateReleaseMetadata(release, tag, expectedCommit);
  invariant(release.draft === false, `GitHub release ${tag} remained a draft`);
  invariant(release.immutable === true, `Published GitHub release ${tag} is not immutable`);
  invariant(typeof release.html_url === "string" && release.html_url.length > 0, "Published release is missing its URL");
}

async function resetDraftAssets(api, repository, release, tag, expectedCommit) {
  validateDraft(release, tag, expectedCommit);
  const assetIds = new Set();
  for (const asset of release.assets) {
    invariant(Number.isSafeInteger(asset.id) && asset.id > 0, "GitHub draft contains an asset with an invalid ID");
    invariant(!assetIds.has(asset.id), "GitHub draft contains duplicate asset IDs");
    assetIds.add(asset.id);
    await api.request(`repos/${repository}/releases/assets/${asset.id}`, { method: "DELETE" });
  }
  const cleanDraft = await api.request(`repos/${repository}/releases/${release.id}`);
  validateDraft(cleanDraft, tag, expectedCommit);
  invariant(cleanDraft.assets.length === 0, `GitHub draft ${tag} still has assets after its controlled reset`);
  return cleanDraft;
}

async function remoteAssetDigest(api, repository, asset) {
  if (typeof asset.digest === "string" && /^sha256:[0-9a-f]{64}$/u.test(asset.digest)) {
    return asset.digest.slice("sha256:".length);
  }
  invariant(Number.isSafeInteger(asset.id) && asset.id > 0, "Cannot download a remote asset with an invalid ID");
  const bytes = await api.request(`repos/${repository}/releases/assets/${asset.id}`, { raw: true });
  invariant(Buffer.isBuffer(bytes), `GitHub asset download did not return bytes for ${asset.name}`);
  return sha256Buffer(bytes);
}

export async function verifyRemoteRelease({ api, repository, release, tag, expectedCommit, localInventory, expectedDraft }) {
  if (expectedDraft) validateDraft(release, tag, expectedCommit);
  else validatePublishedRelease(release, tag, expectedCommit);
  const remoteNames = release.assets.map((asset) => asset.name);
  invariant(new Set(remoteNames).size === remoteNames.length, "Remote release contains duplicate asset names");

  const remoteInventory = [];
  for (const asset of release.assets) {
    invariant(asset.state === "uploaded", `Remote release asset is not fully uploaded: ${asset.name}`);
    invariant(Number.isSafeInteger(asset.size) && asset.size > 0, `Remote release asset has an invalid size: ${asset.name}`);
    remoteInventory.push({
      name: asset.name,
      size: asset.size,
      sha256: await remoteAssetDigest(api, repository, asset),
    });
  }
  remoteInventory.sort((left, right) => left.name.localeCompare(right.name));
  const expected = localInventory
    .map(({ name, size, sha256 }) => ({ name, size, sha256 }))
    .sort((left, right) => left.name.localeCompare(right.name));
  invariant(
    isDeepStrictEqual(remoteInventory, expected),
    `Remote release inventory is not byte-for-byte identical to the verified local inventory`,
  );
}

async function verifyPublishedState({
  api,
  repository,
  releaseId,
  tag,
  expectedCommit,
  localInventory,
  requireLatest,
  verifyPublishedSource,
}) {
  const published = await api.request(`repos/${repository}/releases/${releaseId}`);
  validatePublishedRelease(published, tag, expectedCommit);
  if (verifyPublishedSource) {
    await verifyPublishedSource();
  } else {
    await verifyReleaseSourceContainedInDefaultBranch(
      api,
      repository,
      tag,
      expectedCommit,
      "published-release verification",
    );
  }
  await verifyRemoteRelease({
    api,
    repository,
    release: published,
    tag,
    expectedCommit,
    localInventory,
    expectedDraft: false,
  });
  if (requireLatest) {
    const latest = await api.request(`repos/${repository}/releases/latest`);
    invariant(latest?.id === releaseId && latest.tag_name === tag, `Published release ${tag} is not the latest release`);
  }
  return published;
}

async function waitForPublishedState({
  api,
  repository,
  releaseId,
  tag,
  expectedCommit,
  localInventory,
  pause,
  verifyPublishedSource,
}) {
  let lastError;
  for (let attempt = 0; attempt < 10; attempt += 1) {
    try {
      return await verifyPublishedState({
        api,
        repository,
        releaseId,
        tag,
        expectedCommit,
        localInventory,
        requireLatest: true,
        verifyPublishedSource,
      });
    } catch (error) {
      lastError = error;
      if (attempt < 9) await pause(Math.min(2 ** attempt, 10) * 1000);
    }
  }
  throw lastError;
}

async function waitForCreatedDraft({
  api,
  repository,
  tag,
  createdReleaseId,
  pause,
}) {
  const maximumAttempts = 10;

  for (let attempt = 0; attempt < maximumAttempts; attempt += 1) {
    const discoveredDraft = await findReleaseForTag(api, repository, tag);
    if (discoveredDraft) {
      invariant(
        discoveredDraft.id === createdReleaseId,
        `GitHub returned release ${discoveredDraft.id} while rediscovering newly created draft ${createdReleaseId}`,
      );
      return discoveredDraft;
    }

    if (attempt < maximumAttempts - 1) {
      await pause(Math.min(2 ** attempt, 5) * 1000);
    }
  }

  throw new Error(`New draft ${createdReleaseId} could not be uniquely rediscovered before asset mutation`);
}

async function uploadAndPublishVerifiedRelease({
  api,
  repository,
  release,
  tag,
  expectedCommit,
  localInventory,
  verifyBeforePublication,
  verifyImmediatelyBeforePublication,
  verifyPublishedSource,
  pause,
}) {
  invariant(typeof api.uploadReleaseAsset === "function", "GitHub API client cannot upload verified release assets");
  for (const asset of localInventory) {
    await api.uploadReleaseAsset(release.upload_url, repository, release.id, asset);
  }

  const uploadedDraft = await api.request(`repos/${repository}/releases/${release.id}`);
  await verifyRemoteRelease({
    api,
    repository,
    release: uploadedDraft,
    tag,
    expectedCommit,
    localInventory,
    expectedDraft: true,
  });
  await verifyBeforePublication();
  await verifyImmediatelyBeforePublication();

  // Keep the exact remote inventory as the final read before the irreversible draft transition.
  const confirmedDraft = await api.request(`repos/${repository}/releases/${release.id}`);
  await verifyRemoteRelease({
    api,
    repository,
    release: confirmedDraft,
    tag,
    expectedCommit,
    localInventory,
    expectedDraft: true,
  });

  let transition;
  try {
    transition = await api.request(`repos/${repository}/releases/${release.id}`, {
      method: "PATCH",
      json: { draft: false, make_latest: "true" },
    });
  } catch (transitionError) {
    try {
      const reconciled = await api.request(`repos/${repository}/releases/${release.id}`);
      if (reconciled.draft !== false) throw transitionError;
      transition = reconciled;
    } catch (reconciliationError) {
      if (reconciliationError === transitionError) throw transitionError;
      throw new AggregateError(
        [transitionError, reconciliationError],
        `Release ${tag} publication returned an ambiguous response and could not be reconciled`,
      );
    }
  }
  invariant(transition.draft === false, `GitHub did not publish release ${tag}`);

  try {
    return await waitForPublishedState({
      api,
      repository,
      releaseId: release.id,
      tag,
      expectedCommit,
      localInventory,
      pause,
      verifyPublishedSource,
    });
  } catch (error) {
    throw new Error(
      `Immutable release ${tag} was published but final verification failed; manual review is required: ${error.message}`,
      { cause: error },
    );
  }
}

export async function publishRelease({
  api,
  repository,
  tag,
  expectedCommit,
  assetDirectory,
  eventName,
  refType,
  manifestPath = defaultManifestPath,
  publicationPolicyPath,
  pause = (milliseconds) => new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds)),
}) {
  validateRepository(repository);
  assertTrustedTagEvent(eventName, refType);
  invariant(gitCommitPattern.test(expectedCommit), "Expected release commit must be a lowercase 40-character Git SHA");
  assertPublicationAuthorized({ policyPath: publicationPolicyPath, manifestPath });
  const contract = buildReleaseContract(manifestPath, tag, expectedCommit);
  const localInventory = buildLocalInventory(assetDirectory, manifestPath);

  let release = await findReleaseForTag(api, repository, tag);

  if (release?.draft === false) {
    const published = await verifyPublishedState({
      api,
      repository,
      releaseId: release.id,
      tag,
      expectedCommit,
      localInventory,
      requireLatest: false,
    });
    return Object.freeze({
      releaseId: published.id,
      tag: contract.expectedTag,
      commit: expectedCommit,
      assetCount: localInventory.length,
      htmlUrl: published.html_url,
    });
  }

  const recoveringDraft = Boolean(release);
  if (release) {
    validateDraft(release, tag, expectedCommit);
  } else {
    await verifyReleaseSourceAtDefaultTip(api, repository, tag, expectedCommit, "draft authorization");
    release = await api.request(`repos/${repository}/releases`, {
      method: "POST",
      json: {
        tag_name: tag,
        target_commitish: expectedCommit,
        name: `ELIZA Lab ${tag}`,
        body: releaseContractBody(tag, expectedCommit),
        draft: true,
        prerelease: false,
        generate_release_notes: false,
      },
    });
    validateDraft(release, tag, expectedCommit);
    const createdReleaseId = release.id;
    const discoveredDraft = await waitForCreatedDraft({
      api,
      repository,
      tag,
      createdReleaseId,
      pause,
    });
    validateDraft(discoveredDraft, tag, expectedCommit);
    release = discoveredDraft;
  }

  await verifyReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    recoveringDraft ? "draft recovery" : "new-draft verification",
  );

  release = await resetDraftAssets(api, repository, release, tag, expectedCommit);
  const published = await uploadAndPublishVerifiedRelease({
    api,
    repository,
    release,
    tag,
    expectedCommit,
    localInventory,
    verifyBeforePublication: () => verifyTagCommit(
      api,
      repository,
      tag,
      expectedCommit,
      "pre-publication verification",
    ),
    verifyImmediatelyBeforePublication: () => verifyReleaseSourceContainedInDefaultBranch(
      api,
      repository,
      tag,
      expectedCommit,
      "final pre-publication verification",
    ),
    pause,
  });

  return Object.freeze({
    releaseId: release.id,
    tag: contract.expectedTag,
    commit: expectedCommit,
    assetCount: localInventory.length,
    htmlUrl: published.html_url,
  });
}

export async function recoverEmptyDraftRelease({
  api,
  repository,
  tag,
  expectedCommit,
  assetDirectory,
  releaseId,
  sourceRunId,
  recoveryRunId,
  workflowCommit,
  workflowRef,
  attestationPath,
  eventName,
  refType,
  manifestPath = defaultManifestPath,
  publicationPolicyPath,
  pause = (milliseconds) => new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds)),
}) {
  validateRepository(repository);
  assertTrustedRecoveryDispatch(eventName, refType);
  invariant(gitCommitPattern.test(expectedCommit), "Expected release commit must be a lowercase 40-character Git SHA");
  validatePositiveSafeInteger(releaseId, "Recovery release ID");
  validatePositiveSafeInteger(sourceRunId, "Source workflow run ID");
  validatePositiveSafeInteger(recoveryRunId, "Recovery workflow run ID");
  invariant(sourceRunId !== recoveryRunId, "Source and recovery workflow run IDs must differ");
  assertPublicationAuthorized({ policyPath: publicationPolicyPath, manifestPath });
  const contract = buildReleaseContract(manifestPath, tag, expectedCommit);
  const localInventory = buildLocalInventory(assetDirectory, manifestPath);

  const execution = await verifyRecoveryExecution({
    api,
    repository,
    recoveryRunId,
    workflowCommit,
    workflowRef,
  });
  const source = await verifyRecoverySourceRun({
    api,
    repository,
    repositoryId: execution.repositoryId,
    tag,
    expectedCommit,
    sourceRunId,
    recoveryCreatedAt: execution.createdAt,
  });
  verifyRecoveryAttestation({
    attestationPath,
    repository,
    repositoryId: execution.repositoryId,
    tag,
    expectedCommit,
    sourceRunId,
    sourceRunAttempt: source.runAttempt,
    localInventory,
  });

  const signedTag = await resolveVerifiedTagCommit(api, repository, tag, expectedCommit);
  await verifySignedReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    signedTag.tagObjectSha,
    "recovery authorization",
  );

  const selectedDraft = await readReleaseById(api, repository, releaseId);
  validateEmptyRecoveryDraft(selectedDraft, tag, expectedCommit, releaseId, source);
  const listedRelease = await findReleaseForTag(api, repository, tag);
  if (listedRelease) {
    invariant(
      listedRelease.id === releaseId,
      `GitHub release listing identifies release ${listedRelease.id}, not exact recovery draft ${releaseId}`,
    );
    validateEmptyRecoveryDraft(listedRelease, tag, expectedCommit, releaseId, source);
  }

  // Re-prove trusted execution and the immutable source immediately before the first release mutation.
  await verifyRecoveryExecution({
    api,
    repository,
    recoveryRunId,
    workflowCommit,
    workflowRef,
  });
  await verifySignedReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    signedTag.tagObjectSha,
    "final recovery authorization",
  );
  const emptyDraft = await readReleaseById(api, repository, releaseId);
  validateEmptyRecoveryDraft(emptyDraft, tag, expectedCommit, releaseId, source);

  const verifyExactSignedSource = (phase) => verifySignedReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    signedTag.tagObjectSha,
    phase,
  );
  const published = await uploadAndPublishVerifiedRelease({
    api,
    repository,
    release: emptyDraft,
    tag,
    expectedCommit,
    localInventory,
    verifyBeforePublication: () => resolveVerifiedTagCommit(
      api,
      repository,
      tag,
      expectedCommit,
      signedTag.tagObjectSha,
    ),
    verifyImmediatelyBeforePublication: async () => {
      await verifyExactSignedSource("final recovery pre-publication verification");
      await verifyRecoveryExecution({
        api,
        repository,
        recoveryRunId,
        workflowCommit,
        workflowRef,
      });
    },
    verifyPublishedSource: () => verifyExactSignedSource("recovered published-release verification"),
    pause,
  });

  return Object.freeze({
    releaseId: published.id,
    sourceRunId,
    sourceArtifactId: source.artifactId,
    tag: contract.expectedTag,
    commit: expectedCommit,
    assetCount: localInventory.length,
    htmlUrl: published.html_url,
  });
}

function parseOptions(argumentsList) {
  const options = {};
  for (let index = 0; index < argumentsList.length; index += 2) {
    const option = argumentsList[index];
    invariant(option?.startsWith("--") && index + 1 < argumentsList.length, `Invalid command option: ${option}`);
    const name = option.slice(2);
    invariant(name.length > 0 && !Object.hasOwn(options, name), `Duplicate or empty command option: ${option}`);
    options[name] = argumentsList[index + 1];
  }
  return options;
}

async function runCli() {
  const [command, ...rawOptions] = process.argv.slice(2);
  const options = parseOptions(rawOptions);
  const api = new GitHubApiClient({ token: process.env.GITHUB_TOKEN, apiBase: process.env.GITHUB_API_URL });
  if (command === "publish") {
    for (const required of ["repo", "tag", "commit", "assets"]) {
      invariant(options[required], `Missing required option --${required}`);
    }
    const allowed = new Set(["repo", "tag", "commit", "assets"]);
    invariant(Object.keys(options).every((name) => allowed.has(name)), "Unknown release publish option");
    const result = await publishRelease({
      api,
      repository: options.repo,
      tag: options.tag,
      expectedCommit: options.commit,
      assetDirectory: options.assets,
      eventName: process.env.GITHUB_EVENT_NAME,
      refType: process.env.GITHUB_REF_TYPE,
    });
    console.log(`Published ${result.tag} from ${result.commit} with ${result.assetCount} verified assets: ${result.htmlUrl}`);
    return;
  }

  invariant(
    command === "recover-empty-draft",
    "Usage: release-publish.mjs recover-empty-draft --repo OWNER/REPO --tag vX.Y.Z --commit SHA --assets DIRECTORY --release-id ID --source-run-id ID --recovery-run-id ID --workflow-commit SHA --workflow-ref BRANCH --attestation FILE --manifest FILE --policy FILE",
  );
  const requiredOptions = [
    "repo",
    "tag",
    "commit",
    "assets",
    "release-id",
    "source-run-id",
    "recovery-run-id",
    "workflow-commit",
    "workflow-ref",
    "attestation",
    "manifest",
    "policy",
  ];
  for (const required of requiredOptions) {
    invariant(options[required], `Missing required option --${required}`);
  }
  invariant(
    Object.keys(options).every((name) => requiredOptions.includes(name)),
    "Unknown release recovery option",
  );
  const result = await recoverEmptyDraftRelease({
    api,
    repository: options.repo,
    tag: options.tag,
    expectedCommit: options.commit,
    assetDirectory: options.assets,
    releaseId: parsePositiveSafeInteger(options["release-id"], "Recovery release ID"),
    sourceRunId: parsePositiveSafeInteger(options["source-run-id"], "Source workflow run ID"),
    recoveryRunId: parsePositiveSafeInteger(options["recovery-run-id"], "Recovery workflow run ID"),
    workflowCommit: options["workflow-commit"],
    workflowRef: options["workflow-ref"],
    attestationPath: options.attestation,
    eventName: process.env.GITHUB_EVENT_NAME,
    refType: process.env.GITHUB_REF_TYPE,
    manifestPath: options.manifest,
    publicationPolicyPath: options.policy,
  });
  console.log(
    `Recovered draft ${result.releaseId} for ${result.tag} from source run ${result.sourceRunId} artifact ${result.sourceArtifactId}; published ${result.assetCount} verified assets: ${result.htmlUrl}`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  runCli().catch((error) => {
    console.error(`release-publish: ${error.message}`);
    process.exitCode = 1;
  });
}
