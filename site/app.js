import { ElizaEngine, MAX_INPUT_CHARS } from "./engine.mjs";

const MAX_TRANSCRIPT_MESSAGES = 80;
const form = document.querySelector("[data-form]");
const input = form?.elements.namedItem("message");
const transcript = document.querySelector("[data-transcript]");
const resetButton = document.querySelector("[data-reset]");
const ruleOutput = document.querySelector("[data-rule]");
const keywordOutput = document.querySelector("[data-keyword]");
const transformOutput = document.querySelector("[data-transform]");
const traceStatus = document.querySelector("[data-trace-status]");

let engine = new ElizaEngine();

const boundedCharacters = (value) => {
  let output = "";
  let count = 0;
  for (const character of value) {
    if (count === MAX_INPUT_CHARS) return { text: output, truncated: true };
    output += character;
    count += 1;
  }
  return { text: output, truncated: false };
};

const displayBounded = (value) => {
  const bounded = boundedCharacters(value);
  return bounded.truncated ? `${bounded.text}…` : bounded.text;
};

const trimTranscript = () => {
  while (transcript && transcript.childElementCount > MAX_TRANSCRIPT_MESSAGES) {
    transcript.firstElementChild?.remove();
  }
};

const appendMessage = (author, text, turn, className, alert = false) => {
  const article = document.createElement("article");
  article.className = `message ${className}`;
  article.dataset.turn = String(turn);
  if (alert) article.setAttribute("role", "alert");
  const label = document.createElement("span");
  label.textContent = `${author} / ${String(turn).padStart(2, "0")}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(label, paragraph);
  transcript?.append(article);
};

const scrollTranscript = () => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  transcript?.scrollTo({ top: transcript.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
};

const runPrompt = (value) => {
  const prompt = String(value ?? "").trim();
  if (!prompt) return;

  const reply = engine.respond(prompt);
  appendMessage("YOU", displayBounded(prompt), reply.turn, "message-user");
  appendMessage(
    "ELIZA",
    reply.text,
    reply.turn,
    reply.rule === "safety-boundary" ? "message-system message-safety" : "message-system",
    reply.rule === "safety-boundary",
  );
  trimTranscript();
  scrollTranscript();
  if (ruleOutput) ruleOutput.textContent = reply.rule;
  if (keywordOutput) keywordOutput.textContent = reply.keyword ?? "—";
  if (transformOutput) transformOutput.textContent = reply.transformed ?? "—";
  if (traceStatus) traceStatus.textContent = `TURN ${String(reply.turn).padStart(2, "0")}`;
};

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!(input instanceof HTMLInputElement)) return;
  runPrompt(input.value);
  input.value = "";
  input.focus();
});

input?.addEventListener("input", () => {
  if (!(input instanceof HTMLInputElement)) return;
  const bounded = boundedCharacters(input.value);
  if (!bounded.truncated) return;
  input.value = bounded.text;
  if (traceStatus) traceStatus.textContent = "INPUT LIMIT";
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!(button instanceof HTMLButtonElement)) return;
    runPrompt(button.dataset.sample ?? "");
    input?.focus();
  });
});

resetButton?.addEventListener("click", () => {
  engine = new ElizaEngine();
  transcript?.replaceChildren();
  appendMessage("ELIZA", "Hello. What is on your mind today?", 0, "message-system");
  if (ruleOutput) ruleOutput.textContent = "waiting-for-input";
  if (keywordOutput) keywordOutput.textContent = "—";
  if (transformOutput) transformOutput.textContent = "—";
  if (traceStatus) traceStatus.textContent = "CLEARED";
  input?.focus();
});
