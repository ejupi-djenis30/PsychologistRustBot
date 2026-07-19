import assert from "node:assert/strict";
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
  assert.match(reply.text, /emergency services/);
  assert.doesNotMatch(reply.text, /diagnos/i);
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
