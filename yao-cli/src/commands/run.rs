use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::collections::HashMap;
use std::io::BufWriter;
use yao_rs::{State, apply};

pub fn run(
    circuit_path: &str,
    input_path: Option<&str>,
    shots: Option<usize>,
    op: Option<&str>,
    locs: Option<&[usize]>,
    out: &OutputConfig,
) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let input_state = if let Some(path) = input_path {
        state_io::read_state(path)?
    } else {
        State::zero_state(&circuit.dims)
    };

    let result = apply(&circuit, &input_state);

    if let Some(nshots) = shots {
        let mut rng = rand::thread_rng();
        let outcomes = yao_rs::measure(&result, locs, nshots, &mut rng);

        let json_value = serde_json::json!({
            "num_qubits": result.dims.len(),
            "shots": nshots,
            "locs": locs,
            "outcomes": &outcomes,
        });

        let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
        for outcome in &outcomes {
            *counts.entry(outcome.clone()).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let mut human = format!("Measurement results ({nshots} shots):\n");
        for (outcome, count) in &sorted {
            let pct = (*count as f64 / nshots as f64) * 100.0;
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

        out.emit(&human, &json_value)
    } else if let Some(op_str) = op {
        let operator = crate::operator_parser::parse_operator(op_str)?;
        let value = crate::commands::expect::compute_expectation(&result, &operator);

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

        out.emit(&human, &json_value)
    } else {
        if let Some(ref path) = out.output {
            state_io::write_state(&result, path)?;
            out.info(&format!("State written to {}", path.display()));
        } else {
            let stdout = std::io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            state_io::write_state_to_writer(&result, &mut writer)?;
        }
        Ok(())
    }
}
