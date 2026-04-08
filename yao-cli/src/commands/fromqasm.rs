use crate::output::OutputConfig;
use anyhow::{Context, Result};
use std::io::Read;
use yao_rs::qasm;

pub fn fromqasm(input: &str, out: &OutputConfig) -> Result<()> {
    let result = if input == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read QASM from stdin")?;
        qasm::from_qasm(&buf)
    } else {
        qasm::from_qasm_file(input)
    }
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    if !result.measurements.is_empty() {
        out.info(&format!(
            "note: {} measurement instruction(s) collected (not part of circuit)",
            result.measurements.len()
        ));
    }

    let json_str = yao_rs::circuit_to_json(&result.circuit);
    let json_value: serde_json::Value = serde_json::from_str(&json_str)?;
    let human = format!("{}", result.circuit);

    out.emit(&human, &json_value)
}
