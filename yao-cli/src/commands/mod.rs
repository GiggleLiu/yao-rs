#[cfg(feature = "omeinsum")]
pub mod contract;
pub mod example;
pub mod expect;
pub mod fetch;
#[cfg(feature = "qasm")]
pub mod fromqasm;
pub mod inspect;
pub mod measure;
#[cfg(feature = "omeinsum")]
pub mod optimize;
pub mod probs;
pub mod run;
pub mod simulate;
pub mod toeinsum;
#[cfg(feature = "qasm")]
pub mod toqasm;
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

#[cfg(feature = "omeinsum")]
pub fn load_stdin_or_file(path: &str) -> anyhow::Result<String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read from stdin")?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("Failed to read '{path}'"))
    }
}

/// Validate user-provided locations before entering the infallible library API.
pub fn validate_locs(nqubits: usize, locs: Option<&[usize]>) -> anyhow::Result<()> {
    let mut seen = std::collections::HashSet::new();
    for &loc in locs.unwrap_or_default() {
        anyhow::ensure!(
            loc < nqubits,
            "Measurement location {loc} is out of range for {nqubits} qubits"
        );
        anyhow::ensure!(seen.insert(loc), "Duplicate measurement location {loc}");
    }
    Ok(())
}

pub fn simulation_input(
    circuit: &Circuit,
    input_path: Option<&str>,
) -> anyhow::Result<crate::state_io::State> {
    let n = circuit.nbits;
    let noisy = circuit
        .elements
        .iter()
        .any(|e| matches!(e, yao_rs::CircuitElement::Channel(_)));
    crate::state_io::checked_elements(n, noisy)?;
    if let Some(path) = input_path {
        let reg = crate::state_io::read_state(path)?;
        anyhow::ensure!(
            reg.nqubits() == n,
            "Input state has {} qubits but circuit has {n}",
            reg.nqubits()
        );
        Ok(reg)
    } else {
        Ok(crate::state_io::State::Pure(yao_rs::ArrayReg::zero_state(
            n,
        )))
    }
}

pub fn sample_state(
    state: &crate::state_io::State,
    shots: usize,
    locs: Option<&[usize]>,
    seed: Option<u64>,
) -> anyhow::Result<Vec<Vec<usize>>> {
    use rand::{
        SeedableRng,
        distr::{Distribution, weighted::WeightedIndex},
        rngs::StdRng,
    };
    validate_locs(state.nqubits(), locs)?;
    let probabilities = state.probs(locs);
    let distribution = WeightedIndex::new(&probabilities)
        .context("State has no valid probability distribution")?;
    let mut rng = seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);
    let n = locs.map_or(state.nqubits(), <[usize]>::len);
    Ok((0..shots)
        .map(|_| {
            let outcome = distribution.sample(&mut rng);
            (0..n).map(|i| (outcome >> (n - 1 - i)) & 1).collect()
        })
        .collect())
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
        format!("<{op_str}> = {:.10}", value.re)
    } else {
        format!("<{op_str}> = {:.10} + {:.10}i", value.re, value.im)
    };

    (human, json_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_format_expectation_shows_operator_name() {
        let (human, _json) = format_expectation("Z(0)Z(1)", Complex64::new(1.0, 0.0));
        assert!(
            human.contains("Z(0)Z(1)"),
            "Expected operator name in output, got: {human}"
        );
        assert!(!human.contains("<op>"), "Should not contain literal <op>");
    }

    #[test]
    fn test_format_expectation_complex_value() {
        let (human, _json) = format_expectation("X(0)", Complex64::new(0.5, 0.3));
        assert!(human.contains("X(0)"));
        assert!(human.contains("0.3")); // imaginary part shown
    }
}
