import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { assertPublicationAuthorized, readPublicationPolicy } from "../release-policy.mjs";

test("keeps the repository license, manifest, and release policy aligned", () => {
  assert.deepEqual(assertPublicationAuthorized(), {
    licenseFile: "LICENSE",
    spdxExpression: "MIT",
  });
});

function createFixture({ enabled = true, cargoLicense = "MIT", policyLicense = "MIT", licenseFile = "LICENSE" } = {}) {
  const directory = mkdtempSync(path.join(tmpdir(), "eliza-release-policy-"));
  const manifestPath = path.join(directory, "Cargo.toml");
  writeFileSync(
    manifestPath,
    `[package]\nname = "eliza-lab"\nversion = "1.1.0"\nlicense = "${cargoLicense}"\n`,
    "utf8",
  );
  if (licenseFile) {
    writeFileSync(
      path.join(directory, licenseFile),
      "MIT License\n\nPermission is hereby granted for this deterministic automated test fixture.\n",
      "utf8",
    );
  }
  const policyPath = path.join(directory, "release-policy.json");
  writeFileSync(policyPath, `${JSON.stringify({
    schemaVersion: 1,
    publicationEnabled: enabled,
    licenseFile: enabled ? licenseFile : null,
    spdxExpression: enabled ? policyLicense : null,
  }, null, 2)}\n`, "utf8");
  return { manifestPath, policyPath };
}

test("keeps publication explicitly disabled without claiming a license", () => {
  const fixture = createFixture({ enabled: false, licenseFile: null });
  assert.equal(readPublicationPolicy(fixture.policyPath).publicationEnabled, false);
  assert.throws(
    () => assertPublicationAuthorized(fixture),
    /publication is disabled until a license is selected/u,
  );
});

test("authorizes publication only when policy, manifest, and license file agree", () => {
  const fixture = createFixture();
  assert.deepEqual(assertPublicationAuthorized(fixture), {
    licenseFile: "LICENSE",
    spdxExpression: "MIT",
  });
});

test("rejects a policy whose SPDX expression differs from Cargo.toml", () => {
  const fixture = createFixture({ cargoLicense: "Apache-2.0" });
  assert.throws(() => assertPublicationAuthorized(fixture), /does not match the approved SPDX expression/u);
});

test("rejects missing, short, or unexpected license files", () => {
  const missing = createFixture();
  unlinkSync(path.join(path.dirname(missing.manifestPath), "LICENSE"));
  assert.throws(() => assertPublicationAuthorized(missing), /ENOENT|no such file/iu);

  const short = createFixture({ licenseFile: "LICENSE.txt" });
  writeFileSync(path.join(path.dirname(short.manifestPath), "LICENSE.txt"), "short\n", "utf8");
  assert.throws(() => assertPublicationAuthorized(short), /unexpectedly short/u);

  const unsupported = createFixture();
  const policy = JSON.parse(readFileSync(unsupported.policyPath, "utf8"));
  policy.licenseFile = "COPYING";
  writeFileSync(unsupported.policyPath, `${JSON.stringify(policy)}\n`, "utf8");
  assert.throws(() => readPublicationPolicy(unsupported.policyPath), /supported repository-root license file/u);
});

test("rejects extra policy fields instead of silently widening authorization", () => {
  const fixture = createFixture();
  const policy = {
    ...readPublicationPolicy(fixture.policyPath),
    bypass: true,
  };
  writeFileSync(fixture.policyPath, `${JSON.stringify(policy)}\n`, "utf8");
  assert.throws(() => readPublicationPolicy(fixture.policyPath), /unexpected or missing fields/u);
});
