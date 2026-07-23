use eliza_lab::ml::IntentModel;
use serde_json::Value;

const TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{context}: expected {expected:.16}, observed {actual:.16}"
    );
}

#[test]
fn matches_the_shared_rust_browser_model_corpus() {
    let model = IntentModel::from_json(include_str!("../models/eliza-intent-v1.json")).unwrap();
    for (index, line) in include_str!("../fixtures/ml-parity.tsv")
        .lines()
        .enumerate()
    {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields = line.splitn(2, '\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 2, "invalid ML corpus row {}", index + 1);
        let expected: Value = serde_json::from_str(fields[1]).unwrap();
        let prediction = model.predict(fields[0]);
        assert_eq!(prediction.label, expected["label"], "row {}", index + 1);
        assert_eq!(
            prediction.accepted,
            expected["accepted"].as_bool().unwrap(),
            "row {}",
            index + 1
        );
        close(
            prediction.confidence,
            expected["confidence"].as_f64().unwrap(),
            &format!("row {} confidence", index + 1),
        );
        close(
            prediction.margin,
            expected["margin"].as_f64().unwrap(),
            &format!("row {} margin", index + 1),
        );
        for (label, expected_probability) in expected["probabilities"].as_object().unwrap() {
            close(
                prediction.probabilities[label],
                expected_probability.as_f64().unwrap(),
                &format!("row {} probability {label}", index + 1),
            );
        }
        for (feature_index, expected_feature) in expected["top_features"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let actual = &prediction.top_features[feature_index];
            assert_eq!(actual.feature, expected_feature["feature"]);
            close(
                actual.value,
                expected_feature["value"].as_f64().unwrap(),
                &format!("row {} feature {feature_index} value", index + 1),
            );
            close(
                actual.weight,
                expected_feature["weight"].as_f64().unwrap(),
                &format!("row {} feature {feature_index} weight", index + 1),
            );
            close(
                actual.contribution,
                expected_feature["contribution"].as_f64().unwrap(),
                &format!("row {} feature {feature_index} contribution", index + 1),
            );
        }
    }
}
