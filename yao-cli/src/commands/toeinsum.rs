use crate::cli::TnMode;
use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::Result;

pub fn toeinsum(
    circuit_path: &str,
    mode: TnMode,
    op: Option<&str>,
    out: &OutputConfig,
) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let dto = if let Some(op_str) = op {
        let operator = crate::operator_parser::parse_operator_for_qubits(op_str, circuit.nbits)?;
        anyhow::ensure!(
            operator
                .opstrings()
                .iter()
                .flat_map(|s| s.ops())
                .all(|(site, op)| *op == yao_rs::Op::I || circuit.dims[*site] == 2),
            "Nonidentity operators require qubit sites"
        );
        if matches!(mode, TnMode::Dm)
            || circuit
                .elements
                .iter()
                .any(|e| matches!(e, yao_rs::CircuitElement::Channel(_)))
        {
            TensorNetworkDto::from_dm(&yao_rs::circuit_to_expectation_dm(&circuit, &operator))
        } else {
            TensorNetworkDto::from_pure(&yao_rs::circuit_to_expectation(&circuit, &operator))
        }
    } else {
        match mode {
            TnMode::Pure => {
                let tn = yao_rs::circuit_to_einsum(&circuit);
                TensorNetworkDto::from_pure(&tn)
            }
            TnMode::Dm => {
                let tn = yao_rs::circuit_to_einsum_dm(&circuit);
                TensorNetworkDto::from_dm(&tn)
            }
            TnMode::Overlap => {
                let tn = yao_rs::circuit_to_overlap(&circuit);
                TensorNetworkDto::from_pure(&tn)
            }
            TnMode::State => {
                let tn = yao_rs::circuit_to_einsum_with_boundary(&circuit, &[]);
                TensorNetworkDto::from_pure(&tn)
            }
        }
    };

    let json_value = serde_json::to_value(&dto)?;
    let human = format!(
        "Tensor Network (mode={}):\n  Tensors: {}\n  Labels: {}\n",
        dto.mode,
        dto.tensors.len(),
        dto.size_dict.len(),
    );

    out.emit(&human, &json_value)
}
