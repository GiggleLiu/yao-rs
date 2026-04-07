use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;

pub fn probs(input: &str, locs: Option<&[usize]>, out: &OutputConfig) -> Result<()> {
    let reg = state_io::read_state(input)?;
    let probabilities = yao_rs::probs(&reg, locs);

    let nbits = reg.nqubits();
    let json_value = serde_json::json!({
        "num_qubits": nbits,
        "locs": locs,
        "probabilities": &probabilities,
    });

    let measured_bits = locs.map_or(nbits, |l| l.len());

    let mut human = String::from("Probabilities:\n");
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability > 1e-10 {
            let label: String = (0..measured_bits)
                .map(|bit| {
                    let b = (index >> (measured_bits - 1 - bit)) & 1;
                    char::from(b'0' + b as u8)
                })
                .collect();
            human.push_str(&format!("  |{label}> : {probability:.6}\n"));
        }
    }

    out.emit(&human, &json_value)
}
