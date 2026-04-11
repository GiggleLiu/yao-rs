use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::Result;
use omeco::NestedEinsum;
use omeco::json::NestedEinsumTree;
use yao_rs::contractor::contract_with_tree;

fn index_to_mixed_radix(idx: usize, dims: &[usize]) -> String {
    let mut idx = idx;
    let mut digits = vec![0usize; dims.len()];
    for d in (0..dims.len()).rev() {
        digits[d] = idx % dims[d];
        idx /= dims[d];
    }
    digits
        .iter()
        .map(|digit| digit.to_string())
        .collect::<Vec<_>>()
        .join("")
}

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
    let dims: Vec<usize> = dto
        .eincode
        .output_indices
        .iter()
        .map(|label| {
            dto.size_dict.get(label).copied().ok_or_else(|| {
                anyhow::anyhow!("Missing size_dict entry for output label '{label}'")
            })
        })
        .collect::<Result<_>>()?;
    let all_binary = dims.iter().all(|&d| d == 2);

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
                let bitstr = if all_binary {
                    format!("{:0>width$b}", i, width = num_open_legs)
                } else {
                    index_to_mixed_radix(i, &dims)
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_to_mixed_radix_formats_non_binary_digits() {
        assert_eq!(index_to_mixed_radix(5, &[2, 3]), "12");
        assert_eq!(index_to_mixed_radix(1, &[3, 4]), "01");
    }
}
