use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;

pub fn probs(input: &str, locs: Option<&[usize]>, out: &OutputConfig) -> Result<()> {
    let state = state_io::read_state(input)?;
    let probabilities = yao_rs::probs(&state, locs);

    let json_value = serde_json::json!({
        "num_qubits": state.dims.len(),
        "locs": locs,
        "probabilities": &probabilities,
    });

    let mut human = String::from("Probabilities:\n");
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability > 1e-10 {
            let num_bits = locs.map_or(state.dims.len(), |selected| selected.len());
            let label = format!("{index:0>width$b}", width = num_bits);
            human.push_str(&format!("  |{label}> : {probability:.6}\n"));
        }
    }

    out.emit(&human, &json_value)
}
