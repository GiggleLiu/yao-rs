use crate::output::OutputConfig;
use anyhow::Result;
use yao_rs::qasm;

pub fn toqasm(input: &str, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(input)?;
    let qasm_str = qasm::to_qasm(&circuit)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let json_value = serde_json::json!({ "qasm": &qasm_str });
    out.emit(&qasm_str, &json_value)
}
