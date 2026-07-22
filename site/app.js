import { ElizaEngine, MAX_INPUT_CHARS } from "./engine.mjs?v=1.2.0-2";
import { IntentModel } from "./ml-engine.mjs?v=1.2.0-2";

const MAX_TRANSCRIPT_MESSAGES = 80;
const form = document.querySelector("[data-form]");
const input = form?.elements.namedItem("message");
const transcript = document.querySelector("[data-transcript]");
const resetButton = document.querySelector("[data-reset]");
const ruleOutput = document.querySelector("[data-rule]");
const decisionOutput = document.querySelector("[data-decision]");
const confidenceOutput = document.querySelector("[data-confidence]");
const marginOutput = document.querySelector("[data-margin]");
const evidenceOutput = document.querySelector("[data-evidence]");
const modelStatus = document.querySelector("[data-model-status]");
const traceStatus = document.querySelector("[data-trace-status]");
const labShell = document.querySelector(".lab-shell");
const gatedControls = document.querySelectorAll("[data-model-gated]");

let engine = new ElizaEngine();
let activeModel = null;

const setModelReady = () => {
  labShell?.setAttribute("aria-busy", "false");
  gatedControls.forEach((control) => {
    if (control instanceof HTMLButtonElement || control instanceof HTMLInputElement) {
      control.disabled = false;
    }
  });
};

const initializeModel = async () => {
  try {
    const response = await fetch("./data/eliza-intent-v1.json", {
      cache: "no-cache",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error(`model request failed with ${response.status}`);
    activeModel = new IntentModel(await response.json());
    engine = new ElizaEngine(activeModel);
    if (modelStatus) modelStatus.textContent = `ML ${activeModel.version} READY`;
  } catch {
    activeModel = null;
    engine = new ElizaEngine();
    if (modelStatus) modelStatus.textContent = "RULE FALLBACK";
  } finally {
    setModelReady();
  }
};

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
  if (decisionOutput) {
    decisionOutput.textContent = reply.model
      ? `${reply.model.label} / ${reply.model.accepted ? "accepted" : "abstained"}`
      : reply.keyword ?? "hard boundary";
  }
  if (confidenceOutput) {
    confidenceOutput.textContent = reply.model
      ? `${(reply.model.confidence * 100).toFixed(1)}%`
      : "—";
  }
  if (marginOutput) {
    marginOutput.textContent = reply.model ? reply.model.margin.toFixed(3) : "—";
  }
  if (evidenceOutput) {
    evidenceOutput.textContent = reply.model?.topFeatures.length
      ? reply.model.topFeatures
          .slice(0, 3)
          .map(({ feature, contribution }) => `${feature} (${contribution.toFixed(3)})`)
          .join(" · ")
      : "—";
  }
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
  engine = new ElizaEngine(activeModel);
  transcript?.replaceChildren();
  appendMessage("ELIZA", "Hello. What is on your mind today?", 0, "message-system");
  if (ruleOutput) ruleOutput.textContent = "waiting-for-input";
  if (decisionOutput) decisionOutput.textContent = "—";
  if (confidenceOutput) confidenceOutput.textContent = "—";
  if (marginOutput) marginOutput.textContent = "—";
  if (evidenceOutput) evidenceOutput.textContent = "—";
  if (traceStatus) traceStatus.textContent = "CLEARED";
  input?.focus();
});

void initializeModel();
