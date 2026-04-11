use ndarray::ArrayD;
use num_complex::Complex64;
use omeco::json::NestedEinsumTree;
use omeco::EinCode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use yao_rs::{TensorNetwork, TensorNetworkDM};

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorDto {
    pub shape: Vec<usize>,
    pub data_re: Vec<f64>,
    pub data_im: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EinCodeDto {
    pub input_indices: Vec<Vec<String>>,
    pub output_indices: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorNetworkDto {
    pub format: String,
    pub mode: String,
    pub eincode: EinCodeDto,
    pub tensors: Vec<TensorDto>,
    pub size_dict: HashMap<String, usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contraction_order: Option<NestedEinsumTree<usize>>,
}

impl TensorNetworkDto {
    pub fn from_pure(tn: &TensorNetwork) -> Self {
        Self {
            format: "yao-tn-v1".to_string(),
            mode: "pure".to_string(),
            eincode: eincode_from_pure(tn),
            tensors: tensors_from_network(&tn.tensors),
            size_dict: tn
                .size_dict
                .iter()
                .map(|(label, size)| (label.to_string(), *size))
                .collect(),
            contraction_order: None,
        }
    }

    pub fn from_dm(tn: &TensorNetworkDM) -> Self {
        Self {
            format: "yao-tn-v1".to_string(),
            mode: "dm".to_string(),
            eincode: EinCodeDto {
                input_indices: tn
                    .code
                    .ixs
                    .iter()
                    .map(|legs| legs.iter().map(|label| label.to_string()).collect())
                    .collect(),
                output_indices: tn.code.iy.iter().map(|label| label.to_string()).collect(),
            },
            tensors: tensors_from_network(&tn.tensors),
            size_dict: tn
                .size_dict
                .iter()
                .map(|(label, size)| (label.to_string(), *size))
                .collect(),
            contraction_order: None,
        }
    }

    /// Reconstruct a `TensorNetwork` from this DTO.
    ///
    /// Parses string labels back to `usize`, rebuilds `EinCode` and tensors.
    #[allow(dead_code)]
    pub fn to_tensor_network(&self) -> anyhow::Result<TensorNetwork> {
        // Parse input indices: Vec<Vec<String>> -> Vec<Vec<usize>>
        let ixs: Vec<Vec<usize>> = self
            .eincode
            .input_indices
            .iter()
            .map(|legs| {
                legs.iter()
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Parse output indices: Vec<String> -> Vec<usize>
        let iy: Vec<usize> = self
            .eincode
            .output_indices
            .iter()
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;

        let code = EinCode::new(ixs, iy);

        // Parse size_dict: HashMap<String, usize> -> HashMap<usize, usize>
        let size_dict: HashMap<usize, usize> = self
            .size_dict
            .iter()
            .map(|(k, v)| Ok((k.parse::<usize>()?, *v)))
            .collect::<Result<HashMap<_, _>, std::num::ParseIntError>>()?;

        // Reconstruct tensors
        let tensors: Vec<ArrayD<Complex64>> = self
            .tensors
            .iter()
            .map(|t| {
                let data: Vec<Complex64> = t
                    .data_re
                    .iter()
                    .zip(t.data_im.iter())
                    .map(|(&re, &im)| Complex64::new(re, im))
                    .collect();
                ArrayD::from_shape_vec(ndarray::IxDyn(&t.shape), data)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TensorNetwork {
            code,
            tensors,
            size_dict,
        })
    }
}

fn tensors_from_network(tensors: &[ndarray::ArrayD<num_complex::Complex64>]) -> Vec<TensorDto> {
    tensors
        .iter()
        .map(|tensor| TensorDto {
            shape: tensor.shape().to_vec(),
            data_re: tensor.iter().map(|value| value.re).collect(),
            data_im: tensor.iter().map(|value| value.im).collect(),
        })
        .collect()
}

fn eincode_from_pure(tn: &TensorNetwork) -> EinCodeDto {
    EinCodeDto {
        input_indices: tn
            .code
            .ixs
            .iter()
            .map(|legs| legs.iter().map(|label| label.to_string()).collect())
            .collect(),
        output_indices: tn.code.iy.iter().map(|label| label.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yao_rs::{Circuit, Gate, circuit_to_einsum, put};

    #[test]
    fn test_tn_dto_round_trip() {
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = circuit_to_einsum(&circuit);
        let dto = TensorNetworkDto::from_pure(&tn);

        assert_eq!(dto.format, "yao-tn-v1");
        assert_eq!(dto.mode, "pure");
        assert!(!dto.tensors.is_empty());

        let json = serde_json::to_string_pretty(&dto).unwrap();
        let parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.format, dto.format);
        assert_eq!(parsed.mode, dto.mode);
        assert_eq!(parsed.tensors.len(), dto.tensors.len());
        assert_eq!(parsed.size_dict, dto.size_dict);
        assert_eq!(parsed.eincode.output_indices, dto.eincode.output_indices);
        for (orig, rt) in dto.tensors.iter().zip(parsed.tensors.iter()) {
            assert_eq!(orig.shape, rt.shape);
            assert_eq!(orig.data_re, rt.data_re);
            assert_eq!(orig.data_im, rt.data_im);
        }
    }

    #[test]
    fn test_tn_dto_to_tensor_network_round_trip() {
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = circuit_to_einsum(&circuit);
        let dto = TensorNetworkDto::from_pure(&tn);

        let json = serde_json::to_string(&dto).unwrap();
        let parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        let tn2 = parsed.to_tensor_network().unwrap();

        assert_eq!(tn2.code.ixs, tn.code.ixs);
        assert_eq!(tn2.code.iy, tn.code.iy);
        assert_eq!(tn2.size_dict, tn.size_dict);
        assert_eq!(tn2.tensors.len(), tn.tensors.len());
        for (a, b) in tn2.tensors.iter().zip(tn.tensors.iter()) {
            assert_eq!(a.shape(), b.shape());
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).norm() < 1e-15);
            }
        }
    }

    #[test]
    fn test_tn_dto_contraction_order_serialization() {
        use omeco::json::NestedEinsumTree;

        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = circuit_to_einsum(&circuit);
        let mut dto = TensorNetworkDto::from_pure(&tn);

        assert!(dto.contraction_order.is_none());

        let tree: NestedEinsumTree<usize> = NestedEinsumTree::Leaf {
            isleaf: true,
            tensor_index: 0,
        };
        dto.contraction_order = Some(tree);

        let json = serde_json::to_string(&dto).unwrap();
        assert!(json.contains("contraction_order"));

        let parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        assert!(parsed.contraction_order.is_some());
    }
}
