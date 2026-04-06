use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::{Result, bail};

pub fn toeinsum(circuit_path: &str, mode: &str, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let dto = match mode {
        "pure" => {
            let tn = yao_rs::circuit_to_einsum(&circuit);
            TensorNetworkDto::from_pure(&tn)
        }
        "dm" => {
            let tn = yao_rs::circuit_to_einsum_dm(&circuit);
            TensorNetworkDto::from_dm(&tn)
        }
        _ => bail!("Unknown mode '{mode}'. Use 'pure' or 'dm'."),
    };

    let json_value = serde_json::to_value(&dto)?;
    let human = format!(
        "Tensor Network (mode={}):\n  Tensors: {}\n  Labels: {}\n",
        mode,
        dto.tensors.len(),
        dto.size_dict.len(),
    );

    out.emit(&human, &json_value)
}
