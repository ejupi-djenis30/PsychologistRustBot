//! A small, transparent ELIZA-style engine.
//!
//! The engine is intentionally local and deterministic. It keeps only a turn counter,
//! never writes conversation content, and exposes the rule used for every response.

const MAX_INPUT_CHARS: usize = 512;

/// A response together with the rule trace shown by the learning interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub text: String,
    pub rule_id: &'static str,
    pub keyword: Option<String>,
    pub transformed_fragment: Option<String>,
    pub turn: usize,
}

/// Deterministic conversation engine with no transcript storage.
#[derive(Debug, Clone, Default)]
pub struct ElizaEngine {
    turn: usize,
}

impl ElizaEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn(&self) -> usize {
        self.turn
    }

    pub fn respond(&mut self, input: &str) -> Reply {
        self.turn += 1;
        let turn = self.turn;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return reply(
                "Take your time. What would you like to examine?",
                "empty-input",
                None,
                None,
                turn,
            );
        }

        if trimmed.chars().count() > MAX_INPUT_CHARS {
            return reply(
                "That is more text than this small teaching demo can inspect at once. Try one short thought.",
                "input-boundary",
                None,
                None,
                turn,
            );
        }

        let normalized = normalize(trimmed);

        if contains_any(
            &normalized,
            &[
                "suicide",
                "kill myself",
                "self harm",
                "hurt myself",
                "immediate danger",
            ],
        ) {
            return reply(
                "This educational demo cannot help with urgent safety needs. If you may be in immediate danger, contact local emergency services or a trusted person now.",
                "safety-boundary",
                Some("urgent safety language"),
                None,
                turn,
            );
        }

        if contains_word(&normalized, "hello")
            || contains_word(&normalized, "hi")
            || normalized == "hey"
        {
            return reply(
                "Hello. What is on your mind today?",
                "greeting",
                Some("hello"),
                None,
                turn,
            );
        }

        for prefix in ["i feel ", "i am ", "i'm "] {
            if let Some(fragment) = normalized.strip_prefix(prefix) {
                let reflected = reflect(fragment);
                return reply(
                    format!("What makes you feel {reflected}?"),
                    "feeling-reflection",
                    Some(prefix.trim()),
                    Some(reflected),
                    turn,
                );
            }
        }

        if let Some((_, reason)) = normalized.split_once(" because ") {
            let reflected = reflect(reason);
            return reply(
                format!("What makes {reflected} an important reason for you?"),
                "because-probe",
                Some("because"),
                Some(reflected),
                turn,
            );
        }

        if let Some(fragment) = normalized.strip_prefix("my ") {
            let reflected = reflect(fragment);
            return reply(
                format!("How does your {reflected} affect you?"),
                "ownership-reflection",
                Some("my"),
                Some(reflected),
                turn,
            );
        }

        if normalized.ends_with('?') {
            return reply(
                "What answer would feel most useful to explore?",
                "question-return",
                Some("question"),
                None,
                turn,
            );
        }

        const FALLBACKS: [&str; 4] = [
            "Tell me a little more about that.",
            "What part of that stands out most to you?",
            "How did you arrive at that thought?",
            "What would change if you looked at it another way?",
        ];
        reply(
            FALLBACKS[(turn - 1) % FALLBACKS.len()],
            "fallback",
            None,
            None,
            turn,
        )
    }
}

fn reply(
    text: impl Into<String>,
    rule_id: &'static str,
    keyword: Option<&str>,
    transformed_fragment: Option<String>,
    turn: usize,
) -> Reply {
    Reply {
        text: text.into(),
        rule_id,
        keyword: keyword.map(str::to_string),
        transformed_fragment,
        turn,
    }
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .any(|word| word == needle)
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn reflect(value: &str) -> String {
    value
        .trim_matches(|character: char| character.is_ascii_punctuation())
        .split_whitespace()
        .map(|word| match word {
            "i" => "you",
            "me" => "you",
            "my" => "your",
            "mine" => "yours",
            "am" => "are",
            "you" => "I",
            "your" => "my",
            "yours" => "mine",
            "are" => "am",
            _ => word,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets_without_storing_the_message() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("Hello");

        assert_eq!(response.rule_id, "greeting");
        assert_eq!(engine.turn(), 1);
        assert_eq!(
            std::mem::size_of::<ElizaEngine>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn reflects_a_feeling() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I feel uncertain about my next step.");

        assert_eq!(response.rule_id, "feeling-reflection");
        assert_eq!(
            response.transformed_fragment.as_deref(),
            Some("uncertain about your next step")
        );
        assert_eq!(
            response.text,
            "What makes you feel uncertain about your next step?"
        );
    }

    #[test]
    fn explains_a_because_rule() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I paused because my plan changed");

        assert_eq!(response.rule_id, "because-probe");
        assert_eq!(response.keyword.as_deref(), Some("because"));
        assert!(response.text.contains("your plan changed"));
    }

    #[test]
    fn routes_urgent_language_to_a_clear_boundary() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I might hurt myself");

        assert_eq!(response.rule_id, "safety-boundary");
        assert!(response.text.contains("emergency services"));
        assert!(!response.text.to_lowercase().contains("diagnos"));
    }

    #[test]
    fn rejects_oversized_input() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond(&"x".repeat(MAX_INPUT_CHARS + 1));

        assert_eq!(response.rule_id, "input-boundary");
    }

    #[test]
    fn cycles_deterministic_fallbacks() {
        let mut engine = ElizaEngine::new();
        let first = engine.respond("A statement");
        let second = engine.respond("Another statement");

        assert_eq!(first.rule_id, "fallback");
        assert_eq!(second.rule_id, "fallback");
        assert_ne!(first.text, second.text);
    }
}
