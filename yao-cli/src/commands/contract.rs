use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::Result;
use omeco::NestedEinsum;
use omeco::json::NestedEinsumTree;
use yao_rs::contractor::contract_with_tree;

pub fn contract_cmd(input_path: &str, out: &OutputConfig) -> Result<()> {
    let json = super::load_stdin_or_file(input_path)?;
    let dto: TensorNetworkDto =
        serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("Failed to parse TN JSON: {e}"))?;

    let tree_json: NestedEinsumTree<usize> = dto
        .contraction_order
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Tensor network has no contraction order. \
                 Run `yao optimize` first, e.g.:\n  \
                 yao toeinsum circuit.json | yao optimize - | yao contract -"
            )
        })?
        .clone();

    let tree: NestedEinsum<usize> = tree_json.into();
    let tn = dto.to_tensor_network()?;
    let result = contract_with_tree(&tn, tree);

    let num_open_legs = dto.eincode.output_indices.len();

    if result.ndim() == 0 || result.len() == 1 {
        // Scalar result (overlap or expectation)
        let val = result.iter().next().unwrap();
        let human = if val.im.abs() < 1e-10 {
            format!("Result = {:.10}", val.re)
        } else {
            format!("Result = {:.10} + {:.10}i", val.re, val.im)
        };
        let json_value = serde_json::json!({
            "re": val.re,
            "im": val.im,
        });
        out.emit(&human, &json_value)
    } else {
        // Vector result (state)
        let data: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm() > 1e-15)
            .map(|(i, c)| {
                let bitstr = format!("{:0>width$b}", i, width = num_open_legs);
                serde_json::json!({
                    "index": i,
                    "bitstring": bitstr,
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
                    e["bitstring"].as_str().unwrap(),
                    e["re"],
                    e["im"],
                    e["prob"]
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let json_value = serde_json::json!(data);
        out.emit(&format!("State vector:\n{human}\n"), &json_value)
    }
}
