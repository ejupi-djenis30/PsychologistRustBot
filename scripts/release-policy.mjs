import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

import { parseCargoPackage, readRegularFileSnapshot } from "./release-contract.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "..");
const defaultManifestPath = path.join(repositoryRoot, "Cargo.toml");
const defaultPolicyPath = path.join(repositoryRoot, ".github", "release-policy.json");
const supportedLicenseFiles = new Set(["LICENSE", "LICENSE.md", "LICENSE.txt"]);
const spdxExpressionPattern = /^[A-Za-z0-9.+-]+(?: (?:AND|OR) [A-Za-z0-9.+-]+)*(?: WITH [A-Za-z0-9.+-]+)?$/u;

function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function readJsonSnapshot(filePath, label) {
  const bytes = readRegularFileSnapshot(filePath, label);
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

export function readPublicationPolicy(policyPath = defaultPolicyPath) {
  const policy = readJsonSnapshot(path.resolve(policyPath), "Release publication policy");
  invariant(
    isDeepStrictEqual(
      Object.keys(policy || {}).sort(),
      ["licenseFile", "publicationEnabled", "schemaVersion", "spdxExpression"],
    ),
    "Release publication policy contains unexpected or missing fields",
  );
  invariant(policy.schemaVersion === 1, "Release publication policy has an unsupported schema version");
  invariant(typeof policy.publicationEnabled === "boolean", "Release publication policy must explicitly enable or disable publication");

  if (!policy.publicationEnabled) {
    invariant(policy.licenseFile === null && policy.spdxExpression === null, "A disabled release policy must not claim a license");
    return Object.freeze({ ...policy });
  }

  invariant(
    typeof policy.licenseFile === "string" && supportedLicenseFiles.has(policy.licenseFile),
    "An enabled release policy must name a supported repository-root license file",
  );
  invariant(
    typeof policy.spdxExpression === "string"
      && policy.spdxExpression.length <= 128
      && spdxExpressionPattern.test(policy.spdxExpression),
    "An enabled release policy must contain a conservative SPDX license expression",
  );
  return Object.freeze({ ...policy });
}

export function assertPublicationAuthorized({
  policyPath = defaultPolicyPath,
  manifestPath = defaultManifestPath,
} = {}) {
  const resolvedPolicyPath = path.resolve(policyPath);
  const policy = readPublicationPolicy(resolvedPolicyPath);
  invariant(
    policy.publicationEnabled,
    "GitHub release publication is disabled until a license is selected and explicitly approved in .github/release-policy.json",
  );

  const resolvedManifestPath = path.resolve(manifestPath);
  const cargoPackage = parseCargoPackage(
    readRegularFileSnapshot(resolvedManifestPath, "Cargo.toml").toString("utf8"),
  );
  invariant(
    cargoPackage.license === policy.spdxExpression,
    `Cargo.toml license ${JSON.stringify(cargoPackage.license)} does not match the approved SPDX expression ${policy.spdxExpression}`,
  );

  const repositoryDirectory = path.dirname(resolvedManifestPath);
  const licensePath = path.join(repositoryDirectory, policy.licenseFile);
  invariant(
    path.dirname(licensePath) === repositoryDirectory,
    "Approved license file must stay at the repository root",
  );
  const licenseBytes = readRegularFileSnapshot(licensePath, `Approved license file ${policy.licenseFile}`);
  invariant(licenseBytes.length >= 64, `Approved license file ${policy.licenseFile} is unexpectedly short`);

  return Object.freeze({
    licenseFile: policy.licenseFile,
    spdxExpression: policy.spdxExpression,
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

function runCli() {
  const [command, ...rawOptions] = process.argv.slice(2);
  invariant(command === "verify", "Usage: release-policy.mjs verify [--policy PATH] [--manifest PATH]");
  const options = parseOptions(rawOptions);
  invariant(Object.keys(options).every((name) => ["policy", "manifest"].includes(name)), "Unknown release policy option");
  const authorization = assertPublicationAuthorized({
    policyPath: options.policy ? path.resolve(options.policy) : defaultPolicyPath,
    manifestPath: options.manifest ? path.resolve(options.manifest) : defaultManifestPath,
  });
  console.log(`Release publication authorized under ${authorization.spdxExpression} (${authorization.licenseFile}).`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    runCli();
  } catch (error) {
    console.error(`release-policy: ${error.message}`);
    process.exitCode = 1;
  }
}
