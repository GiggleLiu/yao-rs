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

    let marginal_dims: Vec<usize> = locs.map_or_else(
        || state.dims.clone(),
        |selected| selected.iter().map(|&loc| state.dims[loc]).collect(),
    );

    let mut human = String::from("Probabilities:\n");
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability > 1e-10 {
            let indices = yao_rs::linear_to_indices(index, &marginal_dims);
            let label: String = indices.iter().map(|d| d.to_string()).collect();
            human.push_str(&format!("  |{label}> : {probability:.6}\n"));
        }
    }

    out.emit(&human, &json_value)
}
