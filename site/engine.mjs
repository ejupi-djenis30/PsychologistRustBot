const MAX_INPUT_CHARS = 512;

const normalize = (value) => value.toLowerCase().trim().replace(/\s+/g, " ");

const reflect = (value) => {
  const map = new Map([
    ["i", "you"],
    ["me", "you"],
    ["my", "your"],
    ["mine", "yours"],
    ["am", "are"],
    ["you", "I"],
    ["your", "my"],
    ["yours", "mine"],
    ["are", "am"],
  ]);

  return value
    .replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "")
    .split(/\s+/)
    .map((word) => map.get(word) ?? word)
    .join(" ");
};

const containsWord = (value, expected) =>
  value.split(/[^a-z0-9']+/i).some((word) => word === expected);

const createReply = (text, rule, turn, keyword = null, transformed = null) => ({
  text,
  rule,
  turn,
  keyword,
  transformed,
});

export class ElizaEngine {
  #turn = 0;

  get turn() {
    return this.#turn;
  }

  respond(input) {
    this.#turn += 1;
    const turn = this.#turn;
    const trimmed = String(input ?? "").trim();

    if (!trimmed) {
      return createReply(
        "Take your time. What would you like to examine?",
        "empty-input",
        turn,
      );
    }

    if ([...trimmed].length > MAX_INPUT_CHARS) {
      return createReply(
        "That is more text than this small teaching demo can inspect at once. Try one short thought.",
        "input-boundary",
        turn,
      );
    }

    const value = normalize(trimmed);
    const safetyPhrases = [
      "suicide",
      "kill myself",
      "self harm",
      "hurt myself",
      "immediate danger",
    ];

    if (safetyPhrases.some((phrase) => value.includes(phrase))) {
      return createReply(
        "This educational demo cannot help with urgent safety needs. If you may be in immediate danger, contact local emergency services or a trusted person now.",
        "safety-boundary",
        turn,
        "urgent safety language",
      );
    }

    if (containsWord(value, "hello") || containsWord(value, "hi") || value === "hey") {
      return createReply("Hello. What is on your mind today?", "greeting", turn, "hello");
    }

    for (const prefix of ["i feel ", "i am ", "i'm "]) {
      if (value.startsWith(prefix)) {
        const transformed = reflect(value.slice(prefix.length));
        return createReply(
          `What makes you feel ${transformed}?`,
          "feeling-reflection",
          turn,
          prefix.trim(),
          transformed,
        );
      }
    }

    if (value.includes(" because ")) {
      const transformed = reflect(value.split(" because ", 2)[1]);
      return createReply(
        `What makes ${transformed} an important reason for you?`,
        "because-probe",
        turn,
        "because",
        transformed,
      );
    }

    if (value.startsWith("my ")) {
      const transformed = reflect(value.slice(3));
      return createReply(
        `How does your ${transformed} affect you?`,
        "ownership-reflection",
        turn,
        "my",
        transformed,
      );
    }

    if (value.endsWith("?")) {
      return createReply(
        "What answer would feel most useful to explore?",
        "question-return",
        turn,
        "question",
      );
    }

    const fallbacks = [
      "Tell me a little more about that.",
      "What part of that stands out most to you?",
      "How did you arrive at that thought?",
      "What would change if you looked at it another way?",
    ];
    return createReply(fallbacks[(turn - 1) % fallbacks.length], "fallback", turn);
  }
}
