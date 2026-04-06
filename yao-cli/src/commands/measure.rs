use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::collections::HashMap;

pub fn measure(
    input: &str,
    shots: usize,
    locs: Option<&[usize]>,
    out: &OutputConfig,
) -> Result<()> {
    let state = state_io::read_state(input)?;
    let mut rng = rand::thread_rng();
    let outcomes = yao_rs::measure(&state, locs, shots, &mut rng);

    let json_value = serde_json::json!({
        "num_qubits": state.dims.len(),
        "shots": shots,
        "locs": locs,
        "outcomes": &outcomes,
    });

    let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for outcome in &outcomes {
        *counts.entry(outcome.clone()).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

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

    out.emit(&human, &json_value)
}
