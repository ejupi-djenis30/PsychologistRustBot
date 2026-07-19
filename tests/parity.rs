use eliza_lab::ElizaEngine;

#[test]
fn matches_the_shared_browser_corpus() {
    for (index, line) in include_str!("../fixtures/parity.tsv").lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields = line.splitn(3, '\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 3, "invalid corpus row {}", index + 1);

        let mut engine = ElizaEngine::new();
        let reply = engine.respond(fields[0]);
        assert_eq!(
            reply.rule_id,
            fields[1],
            "rule mismatch on row {}",
            index + 1
        );
        assert_eq!(reply.text, fields[2], "text mismatch on row {}", index + 1);
    }
}
