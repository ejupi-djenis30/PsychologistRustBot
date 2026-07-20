const mappingEntryPattern = /^([A-Za-z_][A-Za-z0-9_-]*):(?:\s*(.*))?$/u;
const blockScalarPattern = /^[>|][+-]?\d?$/u;
const yamlNodeMetadataPattern = /^(?:[&*][^\s[\]{},]*|!<[^>\r\n]+>|![^\s[\]{},]*)(?:\s|$)/u;
const yamlLineBreakPattern = /\r\n|[\r\u0085\u2028\u2029]/gu;
const blockScalarMarker = Symbol("workflow block scalar");

function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function hasOwn(record, key) {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function stripInlineComment(value) {
  let singleQuoted = false;
  let doubleQuoted = false;

  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === "'" && !doubleQuoted) {
      if (singleQuoted && value[index + 1] === "'") {
        index += 1;
      } else {
        singleQuoted = !singleQuoted;
      }
      continue;
    }
    if (character === '"' && !singleQuoted && value[index - 1] !== "\\") {
      doubleQuoted = !doubleQuoted;
      continue;
    }
    if (character === "#" && !singleQuoted && !doubleQuoted && (index === 0 || /\s/u.test(value[index - 1]))) {
      return value.slice(0, index).trimEnd();
    }
  }

  invariant(!singleQuoted && !doubleQuoted, "Workflow contains an unterminated quoted scalar");
  return value.trimEnd();
}

function decodeScalar(value, label, lineNumber) {
  const scalar = stripInlineComment(value).trim();
  invariant(scalar.length > 0, `${label}:${lineNumber} contains an empty scalar`);
  let syntaxOnly = "";
  let singleQuoted = false;
  let doubleQuoted = false;
  for (let index = 0; index < scalar.length; index += 1) {
    if (!singleQuoted && !doubleQuoted && scalar.startsWith("${{", index)) {
      const expressionEnd = scalar.indexOf("}}", index + 3);
      invariant(expressionEnd >= 0, `${label}:${lineNumber} contains an unterminated GitHub expression`);
      syntaxOnly += " ".repeat(expressionEnd + 2 - index);
      index = expressionEnd + 1;
      continue;
    }
    const character = scalar[index];
    if (character === "'" && !doubleQuoted) {
      if (singleQuoted && scalar[index + 1] === "'") {
        syntaxOnly += "  ";
        index += 1;
      } else {
        singleQuoted = !singleQuoted;
        syntaxOnly += " ";
      }
      continue;
    }
    if (character === '"' && !singleQuoted && scalar[index - 1] !== "\\") {
      doubleQuoted = !doubleQuoted;
      syntaxOnly += " ";
      continue;
    }
    syntaxOnly += singleQuoted || doubleQuoted ? " " : character;
  }
  invariant(
    !yamlNodeMetadataPattern.test(scalar)
      && !/(?:^|[\[,{:])\s*[&*!]/u.test(syntaxOnly),
    `${label}:${lineNumber} uses a YAML anchor, alias, or tag; policy-managed workflows must spell out every node`,
  );
  invariant(
    !scalar.startsWith("{") && !(scalar.startsWith("[") && scalar.includes(":")),
    `${label}:${lineNumber} uses an unsupported flow mapping; security-sensitive workflow mappings must use block style`,
  );
  if (scalar.startsWith('"')) {
    invariant(scalar.endsWith('"'), `${label}:${lineNumber} contains an unterminated double-quoted scalar`);
    try {
      return JSON.parse(scalar);
    } catch (error) {
      throw new Error(`${label}:${lineNumber} contains an invalid double-quoted scalar: ${error.message}`);
    }
  }
  if (scalar.startsWith("'")) {
    invariant(scalar.endsWith("'"), `${label}:${lineNumber} contains an unterminated single-quoted scalar`);
    return scalar.slice(1, -1).replaceAll("''", "'");
  }
  return scalar;
}

function parseMappingEntry(source, label, lineNumber) {
  const match = stripInlineComment(source).match(mappingEntryPattern);
  invariant(match, `${label}:${lineNumber} is not a supported YAML mapping entry`);
  return { key: match[1], value: (match[2] || "").trim() };
}

function mergeMappings(target, source, label, lineNumber) {
  for (const [key, value] of Object.entries(source)) {
    invariant(!hasOwn(target, key), `${label}:${lineNumber} contains duplicate mapping key ${key}`);
    target[key] = value;
  }
}

export function parseWorkflowYaml(source, label = "workflow") {
  invariant(typeof source === "string", `${label} source must be text`);
  const lines = source.replace(yamlLineBreakPattern, "\n").split("\n");

  function inspectLine(index) {
    const raw = lines[index];
    invariant(!raw.includes("\t"), `${label}:${index + 1} uses a tab; workflow indentation must use spaces`);
    const indentation = raw.match(/^ */u)[0].length;
    const content = raw.slice(indentation);
    return {
      content,
      indentation,
      lineNumber: index + 1,
    };
  }

  function nextSignificant(startIndex) {
    for (let index = startIndex; index < lines.length; index += 1) {
      const inspected = inspectLine(index);
      const trimmed = inspected.content.trim();
      if (trimmed && !trimmed.startsWith("#")) {
        return { ...inspected, index };
      }
    }
    return null;
  }

  function parseBlockScalar(rawValue, startIndex, parentIndentation, lineNumber) {
    let index = startIndex;
    while (index < lines.length) {
      const inspected = inspectLine(index);
      if (!inspected.content.trim() || inspected.indentation > parentIndentation) {
        index += 1;
        continue;
      }
      break;
    }

    const contentEnd = index;
    const explicitIndentation = rawValue.match(/\d$/u)?.[0];
    let contentIndentation = explicitIndentation
      ? parentIndentation + Number(explicitIndentation)
      : null;

    if (contentIndentation === null) {
      for (let contentIndex = startIndex; contentIndex < contentEnd; contentIndex += 1) {
        const inspected = inspectLine(contentIndex);
        if (inspected.content.trim()) {
          contentIndentation = inspected.indentation;
          break;
        }
      }
    }

    invariant(contentIndentation !== null, `${label}:${lineNumber} contains an empty block scalar`);
    invariant(
      contentIndentation > parentIndentation,
      `${label}:${lineNumber} block scalar content must be indented below its mapping key`,
    );

    const contentLines = [];
    for (let contentIndex = startIndex; contentIndex < contentEnd; contentIndex += 1) {
      const inspected = inspectLine(contentIndex);
      if (!inspected.content.trim()) {
        contentLines.push("");
        continue;
      }
      invariant(
        inspected.indentation >= contentIndentation,
        `${label}:${inspected.lineNumber} block scalar content has inconsistent indentation`,
      );
      contentLines.push(lines[contentIndex].slice(contentIndentation));
    }
    while (contentLines.at(-1) === "") contentLines.pop();

    return {
      index: contentEnd,
      value: Object.freeze({
        [blockScalarMarker]: true,
        header: rawValue,
        lines: Object.freeze(contentLines),
      }),
    };
  }

  function parseValue(rawValue, parentIndentation, startIndex, lineNumber) {
    if (blockScalarPattern.test(rawValue)) {
      return parseBlockScalar(rawValue, startIndex, parentIndentation, lineNumber);
    }
    if (rawValue) {
      return { index: startIndex, value: decodeScalar(rawValue, label, lineNumber) };
    }

    const child = nextSignificant(startIndex);
    if (!child || child.indentation <= parentIndentation) {
      return { index: startIndex, value: null };
    }
    const parsed = parseNode(child.index, child.indentation);
    return { index: parsed.index, value: parsed.value };
  }

  function parseMapping(startIndex, indentation) {
    const result = Object.create(null);
    let index = startIndex;

    while (index < lines.length) {
      const current = nextSignificant(index);
      if (!current || current.indentation < indentation) break;
      invariant(current.indentation === indentation, `${label}:${current.lineNumber} has unexpected indentation`);
      if (/^-(?:\s|$)/u.test(current.content)) break;

      const entry = parseMappingEntry(current.content, label, current.lineNumber);
      invariant(!hasOwn(result, entry.key), `${label}:${current.lineNumber} contains duplicate mapping key ${entry.key}`);
      const parsedValue = parseValue(entry.value, indentation, current.index + 1, current.lineNumber);
      result[entry.key] = parsedValue.value;
      index = parsedValue.index;
    }

    return { index, value: result };
  }

  function parseSequence(startIndex, indentation) {
    const result = [];
    let index = startIndex;

    while (index < lines.length) {
      const current = nextSignificant(index);
      if (!current || current.indentation < indentation) break;
      invariant(current.indentation === indentation, `${label}:${current.lineNumber} has unexpected sequence indentation`);
      const sequenceMatch = current.content.match(/^-\s*(.*)$/u);
      if (!sequenceMatch) break;

      const itemSource = stripInlineComment(sequenceMatch[1]).trim();
      index = current.index + 1;
      if (!itemSource) {
        const child = nextSignificant(index);
        invariant(child && child.indentation > indentation, `${label}:${current.lineNumber} contains an empty sequence item`);
        const parsed = parseNode(child.index, child.indentation);
        result.push(parsed.value);
        index = parsed.index;
        continue;
      }

      const entryMatch = itemSource.match(mappingEntryPattern);
      if (!entryMatch) {
        result.push(decodeScalar(itemSource, label, current.lineNumber));
        continue;
      }

      const item = Object.create(null);
      const logicalIndentation = indentation + 2;
      const entry = parseMappingEntry(itemSource, label, current.lineNumber);
      const parsedValue = parseValue(entry.value, logicalIndentation, index, current.lineNumber);
      item[entry.key] = parsedValue.value;
      index = parsedValue.index;

      const continuation = nextSignificant(index);
      if (continuation && continuation.indentation > indentation) {
        invariant(
          continuation.indentation === logicalIndentation,
          `${label}:${continuation.lineNumber} has unexpected sequence mapping indentation`,
        );
        const parsedContinuation = parseMapping(continuation.index, logicalIndentation);
        mergeMappings(item, parsedContinuation.value, label, continuation.lineNumber);
        index = parsedContinuation.index;
      }
      result.push(item);
    }

    return { index, value: result };
  }

  function parseNode(startIndex, indentation) {
    const current = nextSignificant(startIndex);
    invariant(current && current.indentation === indentation, `${label} is missing a YAML node`);
    return /^-(?:\s|$)/u.test(current.content)
      ? parseSequence(current.index, indentation)
      : parseMapping(current.index, indentation);
  }

  const first = nextSignificant(0);
  invariant(first, `${label} is empty`);
  invariant(first.indentation === 0, `${label}:${first.lineNumber} root must start at indentation zero`);
  const parsed = parseNode(first.index, 0);
  const trailing = nextSignificant(parsed.index);
  invariant(!trailing, `${label}:${trailing?.lineNumber} contains an unparsed YAML node`);
  invariant(parsed.value && !Array.isArray(parsed.value), `${label} root must be a mapping`);
  return parsed.value;
}

function assertExactMapping(value, expected, label) {
  invariant(value && typeof value === "object" && !Array.isArray(value), `${label} must be a mapping`);
  const actualKeys = Object.keys(value).sort();
  const expectedKeys = Object.keys(expected).sort();
  invariant(
    JSON.stringify(actualKeys) === JSON.stringify(expectedKeys),
    `${label} must contain exactly ${expectedKeys.join(", ")}; found ${actualKeys.join(", ") || "none"}`,
  );
  for (const [key, expectedValue] of Object.entries(expected)) {
    invariant(value[key] === expectedValue, `${label}.${key} must be ${expectedValue}`);
  }
}

function assertExactKeys(value, expectedKeys, label) {
  invariant(value && typeof value === "object" && !Array.isArray(value), `${label} must be a mapping`);
  const actualKeys = Object.keys(value).sort();
  const sortedExpectedKeys = [...expectedKeys].sort();
  invariant(
    JSON.stringify(actualKeys) === JSON.stringify(sortedExpectedKeys),
    `${label} must contain exactly ${sortedExpectedKeys.join(", ")}; found ${actualKeys.join(", ") || "none"}`,
  );
}

function isMapping(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function assertWorkflowStructure(workflow, label) {
  invariant(isMapping(workflow), `${label} root must be a mapping`);
  invariant(isMapping(workflow.jobs), `${label} jobs must be a mapping`);
  if (hasOwn(workflow, "permissions")) {
    invariant(isMapping(workflow.permissions), `${label} top-level permissions must be a mapping`);
  }

  for (const [jobName, job] of Object.entries(workflow.jobs)) {
    invariant(isMapping(job), `${label} job ${jobName} must be a mapping`);
    if (hasOwn(job, "permissions")) {
      invariant(isMapping(job.permissions), `${label} job ${jobName} permissions must be a mapping`);
    }
    if (hasOwn(job, "steps")) {
      invariant(Array.isArray(job.steps), `${label} job ${jobName} steps must be a sequence`);
      for (const [index, step] of job.steps.entries()) {
        invariant(isMapping(step), `${label} job ${jobName} step ${index + 1} must be a mapping`);
      }
    }
  }
  return workflow;
}

export function assertReleasePermissions(source, label = "release workflow") {
  const workflow = assertWorkflowStructure(parseWorkflowYaml(source, label), label);
  assertExactMapping(workflow.permissions, { contents: "read" }, `${label} top-level permissions`);

  for (const jobName of ["quality", "build", "assemble", "release_candidate_gate", "attest", "publish"]) {
    invariant(
      workflow.jobs[jobName] && typeof workflow.jobs[jobName] === "object" && !Array.isArray(workflow.jobs[jobName]),
      `${label} is missing the ${jobName} job`,
    );
  }
  for (const [jobName, job] of Object.entries(workflow.jobs)) {
    if (!["attest", "publish"].includes(jobName)) {
      invariant(!hasOwn(job, "permissions"), `${label} job ${jobName} must inherit the read-only top-level permissions`);
    }
  }
  assertExactMapping(
    workflow.jobs.attest.permissions,
    { attestations: "write", contents: "read", "id-token": "write" },
    `${label} attest permissions`,
  );
  assertExactMapping(
    workflow.jobs.publish.permissions,
    { attestations: "read", contents: "write" },
    `${label} publish permissions`,
  );
  return workflow;
}

const releaseCandidateGateRunLines = Object.freeze([
  "set -euo pipefail",
  '[[ "$QUALITY_RESULT" == "success" ]]',
  '[[ "$BUILD_RESULT" == "success" ]]',
  '[[ "$ASSEMBLE_RESULT" == "success" ]]',
]);

export function assertReleaseCandidateGate(source, label = "release workflow") {
  const workflow = assertWorkflowStructure(parseWorkflowYaml(source, label), label);
  const job = workflow.jobs.release_candidate_gate;
  invariant(isMapping(job), `${label} is missing the release_candidate_gate job`);
  invariant(
    !hasOwn(job, "continue-on-error"),
    `${label} release_candidate_gate must not define continue-on-error`,
  );
  assertExactKeys(
    job,
    ["if", "name", "needs", "runs-on", "steps", "timeout-minutes"],
    `${label} release_candidate_gate`,
  );
  invariant(job.name === "Release candidate gate", `${label} release_candidate_gate name must stay exact`);
  invariant(job.if === "always()", `${label} release_candidate_gate if must be always()`);
  invariant(
    job.needs === "[quality, build, assemble]",
    `${label} release_candidate_gate needs must be exactly [quality, build, assemble]`,
  );
  invariant(job["runs-on"] === "ubuntu-22.04", `${label} release_candidate_gate runner must be ubuntu-22.04`);
  invariant(job["timeout-minutes"] === "2", `${label} release_candidate_gate timeout must be 2 minutes`);
  invariant(job.steps.length === 1, `${label} release_candidate_gate must contain exactly one step`);

  const [step] = job.steps;
  invariant(
    !hasOwn(step, "continue-on-error"),
    `${label} release_candidate_gate step must not define continue-on-error`,
  );
  assertExactKeys(step, ["env", "name", "run", "shell"], `${label} release_candidate_gate step`);
  invariant(
    step.name === "Require every release candidate stage to pass",
    `${label} release_candidate_gate step name must stay exact`,
  );
  invariant(step.shell === "bash", `${label} release_candidate_gate step shell must be bash`);
  assertExactMapping(
    step.env,
    {
      ASSEMBLE_RESULT: "${{ needs.assemble.result }}",
      BUILD_RESULT: "${{ needs.build.result }}",
      QUALITY_RESULT: "${{ needs.quality.result }}",
    },
    `${label} release_candidate_gate step env`,
  );
  invariant(
    step.run?.[blockScalarMarker] === true && step.run.header === "|",
    `${label} release_candidate_gate run must be an exact literal block scalar`,
  );
  invariant(
    JSON.stringify(step.run.lines) === JSON.stringify(releaseCandidateGateRunLines),
    `${label} release_candidate_gate run body must stay exact`,
  );
  return job;
}

function collectActionReferences(value, location, references) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => collectActionReferences(entry, `${location}[${index}]`, references));
    return;
  }
  if (!value || typeof value !== "object") return;

  for (const [key, entry] of Object.entries(value)) {
    const entryLocation = `${location}.${key}`;
    if (key === "uses") {
      invariant(typeof entry === "string", `${entryLocation} must be a scalar action reference`);
      references.push({ location: entryLocation, reference: entry });
    } else {
      collectActionReferences(entry, entryLocation, references);
    }
  }
}

function isPinnedRemoteActionReference(reference) {
  const match = reference.match(/^([^/@]+)\/([^/@]+)@([0-9a-f]{40})$/u);
  if (!match) return false;
  const [, owner, repository] = match;
  return owner.length <= 39
    && /^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$/u.test(owner)
    && !owner.includes("--")
    && repository.length <= 100
    && /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/u.test(repository);
}

export function assertPinnedActionReferences(source, label = "workflow") {
  const workflow = assertWorkflowStructure(parseWorkflowYaml(source, label), label);
  const references = [];
  collectActionReferences(workflow, label, references);
  invariant(references.length > 0, `${label} does not contain any active action references`);
  for (const { location, reference } of references) {
    invariant(
      isPinnedRemoteActionReference(reference),
      `${location} must use a valid remote GitHub owner/repository@lowercase-40-character-commit; local actions are forbidden`,
    );
  }
  return references.map(({ reference }) => reference);
}
