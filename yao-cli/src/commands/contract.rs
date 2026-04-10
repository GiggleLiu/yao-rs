use crate::cli::ContractMode;
use crate::output::OutputConfig;
use anyhow::Result;
use yao_rs::contractor::contract;

pub fn contract_cmd(
    circuit_path: &str,
    mode: ContractMode,
    op: Option<&str>,
    out: &OutputConfig,
) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    if let Some(op_str) = op {
        let operator = crate::operator_parser::parse_operator(op_str)?;
        let tn = yao_rs::circuit_to_expectation(&circuit, &operator);
        let result = contract(&tn);
        let val = result.iter().next().unwrap();
        let (human, json_value) = super::format_expectation(op_str, *val);
        out.emit(&human, &json_value)
    } else {
        match mode {
            ContractMode::Overlap => {
                let tn = yao_rs::circuit_to_overlap(&circuit);
                let result = contract(&tn);
                let val = result.iter().next().unwrap();
                let human = format!("⟨0|U|0⟩ = {:.10} + {:.10}i\n", val.re, val.im);
                let json_value = serde_json::json!({
                    "re": val.re,
                    "im": val.im,
                });
                out.emit(&human, &json_value)
            }
            ContractMode::State => {
                let tn = yao_rs::circuit_to_einsum_with_boundary(&circuit, &[]);
                let result = contract(&tn);
                let data: Vec<_> = result
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.norm() > 1e-15)
                    .map(|(i, c)| {
                        serde_json::json!({
                            "index": i,
                            "re": c.re,
                            "im": c.im,
                            "prob": c.norm_sqr(),
                        })
                    })
                    .collect();
                let human = data
                    .iter()
                    .map(|e| {
                        format!(
                            "  |{}⟩: {:.6} + {:.6}i  (p={:.6})",
                            e["index"], e["re"], e["im"], e["prob"]
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let json_value = serde_json::json!(data);
                out.emit(&format!("State vector:\n{human}\n"), &json_value)
            }
        }
    }
}
