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

async function verifyPublishedState({ api, repository, releaseId, tag, expectedCommit, localInventory, requireLatest }) {
  const published = await api.request(`repos/${repository}/releases/${releaseId}`);
  validatePublishedRelease(published, tag, expectedCommit);
  await verifyReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    "published-release verification",
  );
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
      });
    } catch (error) {
      lastError = error;
      if (attempt < 9) await pause(Math.min(2 ** attempt, 10) * 1000);
    }
  }
  throw lastError;
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
    const discoveredDraft = await findReleaseForTag(api, repository, tag);
    invariant(
      discoveredDraft?.id === createdReleaseId,
      `New draft ${createdReleaseId} could not be uniquely rediscovered before asset mutation`,
    );
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
  invariant(typeof api.uploadReleaseAsset === "function", "GitHub API client cannot upload verified release assets");
  for (const asset of localInventory) {
    await api.uploadReleaseAsset(release.upload_url, repository, release.id, asset);
  }

  const uploadedDraft = await api.request(`repos/${repository}/releases/${release.id}`);
  await verifyRemoteRelease({ api, repository, release: uploadedDraft, tag, expectedCommit, localInventory, expectedDraft: true });
  await verifyTagCommit(api, repository, tag, expectedCommit, "pre-publication verification");
  const confirmedDraft = await api.request(`repos/${repository}/releases/${release.id}`);
  await verifyRemoteRelease({ api, repository, release: confirmedDraft, tag, expectedCommit, localInventory, expectedDraft: true });
  await verifyReleaseSourceContainedInDefaultBranch(
    api,
    repository,
    tag,
    expectedCommit,
    "final pre-publication verification",
  );

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

  let published;
  try {
    published = await waitForPublishedState({
      api,
      repository,
      releaseId: release.id,
      tag,
      expectedCommit,
      localInventory,
      pause,
    });
  } catch (error) {
    throw new Error(
      `Immutable release ${tag} was published but final verification failed; manual review is required: ${error.message}`,
      { cause: error },
    );
  }

  return Object.freeze({
    releaseId: release.id,
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
    options[option.slice(2)] = argumentsList[index + 1];
  }
  return options;
}

async function runCli() {
  const [command, ...rawOptions] = process.argv.slice(2);
  invariant(command === "publish", "Usage: release-publish.mjs publish --repo OWNER/REPO --tag vX.Y.Z --commit SHA --assets DIRECTORY");
  const options = parseOptions(rawOptions);
  for (const required of ["repo", "tag", "commit", "assets"]) {
    invariant(options[required], `Missing required option --${required}`);
  }
  const allowed = new Set(["repo", "tag", "commit", "assets"]);
  invariant(Object.keys(options).every((name) => allowed.has(name)), "Unknown release publish option");
  const api = new GitHubApiClient({ token: process.env.GITHUB_TOKEN, apiBase: process.env.GITHUB_API_URL });
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
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  runCli().catch((error) => {
    console.error(`release-publish: ${error.message}`);
    process.exitCode = 1;
  });
}
