pub mod example;
pub mod expect;
pub mod inspect;
pub mod measure;
pub mod probs;
pub mod run;
pub mod simulate;
pub mod toeinsum;
#[cfg(feature = "typst")]
pub mod visualize;

use anyhow::{Context, anyhow};
use num_complex::Complex64;
use std::collections::HashMap;
use std::io::Read;
use yao_rs::Circuit;

pub fn load_circuit(path: &str) -> anyhow::Result<Circuit> {
    let json = if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read circuit from stdin")?;
        buf
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read circuit from '{path}'"))?
    };

    yao_rs::circuit_from_json(&json).map_err(|e| anyhow!("Failed to parse circuit: {e}"))
}

pub fn format_measurement(
    outcomes: &[Vec<usize>],
    shots: usize,
    locs: Option<&[usize]>,
    num_qubits: usize,
) -> (String, serde_json::Value) {
    let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for outcome in outcomes {
        *counts.entry(outcome.clone()).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let counts_map: serde_json::Map<String, serde_json::Value> = sorted
        .iter()
        .map(|(outcome, count)| {
            let key = outcome
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("");
            (key, serde_json::Value::from(*count))
        })
        .collect();

    let json_value = serde_json::json!({
        "num_qubits": num_qubits,
        "shots": shots,
        "locs": locs,
        "counts": counts_map,
        "outcomes": outcomes,
    });

    let mut human = format!("Measurement results ({shots} shots):\n");
    for (outcome, count) in &sorted {
        let pct = (*count as f64 / shots as f64) * 100.0;
        human.push_str(&format!(
            "  |{}> : {} ({pct:.1}%)\n",
            outcome
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(""),
            count,
        ));
    }

    (human, json_value)
}

pub fn format_expectation(op_str: &str, value: Complex64) -> (String, serde_json::Value) {
    let json_value = serde_json::json!({
        "operator": op_str,
        "expectation_value": {
            "re": value.re,
            "im": value.im,
        },
    });

    let human = if value.im.abs() < 1e-10 {
        format!("<op> = {:.10}", value.re)
    } else {
        format!("<op> = {:.10} + {:.10}i", value.re, value.im)
    };

    (human, json_value)
}
