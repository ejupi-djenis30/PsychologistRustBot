import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ElizaEngine } from "../engine.mjs";

test("explains a feeling reflection", () => {
  const engine = new ElizaEngine();
  const reply = engine.respond("I feel uncertain about my next step.");

  assert.equal(reply.rule, "feeling-reflection");
  assert.equal(reply.transformed, "uncertain about your next step");
  assert.equal(reply.text, "What makes you feel uncertain about your next step?");
});

test("stops the simulation for urgent-safety language", () => {
  const engine = new ElizaEngine();
  const reply = engine.respond("I might hurt myself");

  assert.equal(reply.rule, "safety-boundary");
  assert.match(reply.text, /emergency number/);
  assert.doesNotMatch(reply.text, /diagnos/i);
});

test("uses phrase boundaries and recognizes explicit safety variants", () => {
  const engine = new ElizaEngine();
  assert.notEqual(engine.respond("I want to skill myself").rule, "safety-boundary");
  assert.equal(engine.respond("I don’t want to live").rule, "safety-boundary");
  assert.equal(engine.respond("I am suicidal").rule, "safety-boundary");
});

test("uses deterministic fallbacks", () => {
  const engine = new ElizaEngine();
  const first = engine.respond("A statement");
  const second = engine.respond("Another statement");

  assert.equal(first.rule, "fallback");
  assert.notEqual(first.text, second.text);
});

test("does not expose transcript state", () => {
  const engine = new ElizaEngine();
  engine.respond("A private fictional sentence");

  assert.deepEqual(Object.keys(engine), []);
  assert.equal(engine.turn, 1);
});

test("matches the shared Rust/browser parity corpus", async () => {
  const corpus = await readFile(new URL("../../fixtures/parity.tsv", import.meta.url), "utf8");
  for (const [index, line] of corpus.split(/\r?\n/u).entries()) {
    if (!line || line.startsWith("#")) continue;
    const [input, rule, response] = line.split("\t", 3);
    assert.ok(input && rule && response, `invalid corpus row ${index + 1}`);
    const reply = new ElizaEngine().respond(input);
    assert.equal(reply.rule, rule, `rule mismatch on row ${index + 1}`);
    assert.equal(reply.text, response, `text mismatch on row ${index + 1}`);
  }
});

test("bounds oversized input without allocating a code-point array", () => {
  const engine = new ElizaEngine();
  assert.equal(engine.respond("🙂".repeat(513)).rule, "input-boundary");
});
