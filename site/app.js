import { ElizaEngine } from "./engine.mjs";

const form = document.querySelector("[data-form]");
const input = form?.elements.namedItem("message");
const transcript = document.querySelector("[data-transcript]");
const resetButton = document.querySelector("[data-reset]");
const ruleOutput = document.querySelector("[data-rule]");
const keywordOutput = document.querySelector("[data-keyword]");
const transformOutput = document.querySelector("[data-transform]");
const traceStatus = document.querySelector("[data-trace-status]");

let engine = new ElizaEngine();

const appendMessage = (author, text, turn, className) => {
  const article = document.createElement("article");
  article.className = `message ${className}`;
  const label = document.createElement("span");
  label.textContent = `${author} / ${String(turn).padStart(2, "0")}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(label, paragraph);
  transcript?.append(article);
  transcript?.scrollTo({ top: transcript.scrollHeight, behavior: "smooth" });
};

const runPrompt = (value) => {
  const prompt = value.trim();
  if (!prompt) return;

  const reply = engine.respond(prompt);
  appendMessage("YOU", prompt, reply.turn, "message-user");
  appendMessage("ELIZA", reply.text, reply.turn, "message-system");
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

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!(button instanceof HTMLButtonElement)) return;
    runPrompt(button.dataset.sample ?? "");
  });
});

resetButton?.addEventListener("click", () => {
  engine = new ElizaEngine();
  if (transcript) {
    transcript.innerHTML = `
      <article class="message message-system">
        <span>ELIZA / 00</span>
        <p>Hello. What is on your mind today?</p>
      </article>`;
  }
  if (ruleOutput) ruleOutput.textContent = "waiting-for-input";
  if (keywordOutput) keywordOutput.textContent = "—";
  if (transformOutput) transformOutput.textContent = "—";
  if (traceStatus) traceStatus.textContent = "IDLE";
  input?.focus();
});
