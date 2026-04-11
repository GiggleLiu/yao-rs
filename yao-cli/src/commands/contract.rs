use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::Result;
use omeco::NestedEinsum;
use omeco::json::NestedEinsumTree;
use yao_rs::contractor::contract_dm_with_tree;

fn index_to_mixed_radix(idx: usize, dims: &[usize]) -> String {
    let mut digits = vec![0usize; dims.len()];
    let mut value = idx;

    for (digit, dim) in digits.iter_mut().rev().zip(dims.iter().rev()) {
        *digit = value % dim;
        value /= dim;
    }

    digits
        .into_iter()
        .map(|digit| digit.to_string())
        .collect::<String>()
}

pub fn contract_cmd(input_path: &str, out: &OutputConfig) -> Result<()> {
    let json = super::load_stdin_or_file(input_path)?;
    let dto: TensorNetworkDto =
        serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("Failed to parse TN JSON: {e}"))?;

    let tree_json: NestedEinsumTree<i32> = dto
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

    let tree: NestedEinsum<i32> = tree_json.into();
    let tn = dto.to_tensor_network()?;
    let output_dims: Vec<usize> = tn.code.iy.iter().map(|label| tn.size_dict[label]).collect();
    let result = contract_dm_with_tree(&tn, tree);

    if result.ndim() == 0 || result.len() == 1 {
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
        // Convert to standard (C/row-major) layout so flat enumeration matches
        // the row-major mixed-radix decomposition used for bitstring formatting.
        let result_c = result.as_standard_layout();
        let data: Vec<_> = result_c
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm() > 1e-15)
            .map(|(i, c)| {
                let bitstring = index_to_mixed_radix(i, &output_dims);
                serde_json::json!({
                    "index": i,
                    "bitstring": bitstring,
                    "re": c.re,
                    "im": c.im,
                    "prob": c.norm_sqr(),
                })
            })
            .collect();
        let human = data
            .iter()
            .map(|entry| {
                format!(
                    "  |{}⟩: {:.6} + {:.6}i  (p={:.6})",
                    entry["bitstring"].as_str().unwrap(),
                    entry["re"],
                    entry["im"],
                    entry["prob"]
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let json_value = serde_json::json!(data);
        out.emit(&format!("Tensor entries:\n{human}\n"), &json_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, ShapeBuilder};
    use num_complex::Complex64;

    #[test]
    fn test_index_to_mixed_radix_formats_non_binary_digits() {
        assert_eq!(index_to_mixed_radix(5, &[2, 3]), "12");
        assert_eq!(index_to_mixed_radix(1, &[3, 4]), "01");
    }

    #[test]
    fn test_as_standard_layout_fixes_column_major_iteration() {
        // A 2x2 column-major array: logical [0,0]=s, [0,1]=0, [1,0]=0, [1,1]=s
        // but stored in Fortran order: data = [s, 0, 0, s] maps to
        // memory positions [0,0], [1,0], [0,1], [1,1]
        let s = 1.0 / 2.0_f64.sqrt();
        let data = vec![
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(s, 0.0),
        ];
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]).f(), data).unwrap();

        // Without as_standard_layout, iter().enumerate() gives column-major order
        // which would produce wrong bitstring indices
        let result_c = arr.as_standard_layout();
        let entries: Vec<(usize, Complex64)> = result_c
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm() > 1e-15)
            .map(|(i, c)| (i, *c))
            .collect();
        assert_eq!(entries.len(), 2);
        // Row-major index 0 = [0,0] -> bitstring "00"
        assert_eq!(entries[0].0, 0);
        // Row-major index 3 = [1,1] -> bitstring "11"
        assert_eq!(entries[1].0, 3);
    }

    #[test]
    fn test_mixed_radix_qudit_state() {
        // 3x2 standard-layout array with single entry at [2,1]
        let mut data = vec![Complex64::new(0.0, 0.0); 6];
        data[5] = Complex64::new(1.0, 0.0); // row-major index 5 = [2,1]
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&[3, 2]), data).unwrap();

        let result_c = arr.as_standard_layout();
        let entries: Vec<(usize, String)> = result_c
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm() > 1e-15)
            .map(|(i, _)| (i, index_to_mixed_radix(i, &[3, 2])))
            .collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, "21");
    }
}
