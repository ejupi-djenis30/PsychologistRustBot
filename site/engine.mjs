export const MAX_INPUT_CHARS = 512;

const SAFETY_PHRASES = [
  "suicide",
  "suicidal",
  "kill myself",
  "end my life",
  "take my life",
  "hurt myself",
  "harm myself",
  "self harm",
  "want to die",
  "don't want to live",
  "do not want to live",
  "can't go on",
  "cannot go on",
  "immediate danger",
];

const normalize = (value) =>
  value.toLowerCase().replace(/[’‘]/gu, "'").trim().replace(/\s+/gu, " ");

const words = (value) =>
  value.match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) ?? [];

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

  return words(value)
    .map((word) => map.get(word) ?? word)
    .join(" ");
};

const containsPhrase = (value, phrase) => {
  const valueWords = words(value);
  const phraseWords = words(phrase);
  if (!phraseWords.length || phraseWords.length > valueWords.length) return false;
  return valueWords.some((_, start) =>
    phraseWords.every((word, offset) => valueWords[start + offset] === word),
  );
};

const containsWord = (value, expected) => containsPhrase(value, expected);

const exceedsCharacterLimit = (value, limit) => {
  let count = 0;
  for (const _character of value) {
    count += 1;
    if (count > limit) return true;
  }
  return false;
};

const createReply = (text, rule, turn, keyword = null, transformed = null, model = null) => ({
  text,
  rule,
  turn,
  keyword,
  transformed,
  model,
});

export class ElizaEngine {
  #turn = 0;
  #model;

  constructor(model = null) {
    this.#model = model;
  }

  get turn() {
    return this.#turn;
  }

  respond(input) {
    this.#turn = Math.min(this.#turn + 1, Number.MAX_SAFE_INTEGER);
    const turn = this.#turn;
    const trimmed = String(input ?? "").trim();

    if (!trimmed) {
      return createReply(
        "Take your time. What would you like to examine?",
        "empty-input",
        turn,
      );
    }

    if (exceedsCharacterLimit(trimmed, MAX_INPUT_CHARS)) {
      return createReply(
        "That is more text than this small teaching demo can inspect at once. Try one short thought.",
        "input-boundary",
        turn,
      );
    }

    const value = normalize(trimmed);
    if (SAFETY_PHRASES.some((phrase) => containsPhrase(value, phrase))) {
      return createReply(
        "This demo cannot assess or support an emergency. If you might act on thoughts of suicide or self-harm, call your local emergency number now or reach a trusted person who can stay with you.",
        "safety-boundary",
        turn,
        "matched safety phrase",
      );
    }

    if (this.#model) {
      const prediction = this.#model.predict(trimmed);
      const responses = new Map([
        ["greeting", "Hello. What would you like to examine today?"],
        ["feeling", "Which part of that feeling would be useful to examine?"],
        ["reason", "Which part of that explanation matters most here?"],
        ["ownership", "How does that situation affect what you can do next?"],
        ["question", "What answer would feel most useful to explore?"],
        ["goal", "What is the smallest concrete step you could test next?"],
        ["observation", "What part of that observation stands out most to you?"],
      ]);
      const response = responses.get(prediction.label);
      const accepted = prediction.accepted && Boolean(response);
      const trace = {
        version: this.#model.version,
        label: prediction.label,
        accepted,
        confidence: prediction.confidence,
        margin: prediction.margin,
        probabilities: prediction.probabilities,
        topFeatures: prediction.topFeatures,
      };
      if (accepted) {
        return createReply(response, "ml-intent", turn, prediction.label, null, trace);
      }
      const fallbacks = [
        "Tell me a little more about that.",
        "What part of that stands out most to you?",
        "How did you arrive at that thought?",
        "What would change if you looked at it another way?",
      ];
      return createReply(
        fallbacks[(turn - 1) % fallbacks.length],
        "ml-abstain",
        turn,
        prediction.label,
        null,
        trace,
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
