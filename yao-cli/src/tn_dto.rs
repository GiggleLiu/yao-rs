use ndarray::ArrayD;
use num_complex::Complex64;
use omeco::EinCode;
use omeco::json::NestedEinsumTree;
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
    pub contraction_order: Option<NestedEinsumTree<i32>>,
}

impl TensorNetworkDto {
    pub fn from_pure(tn: &TensorNetwork) -> Self {
        Self {
            format: "yao-tn-v1".to_string(),
            mode: "pure".to_string(),
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

    /// Reconstruct the tensor network using signed labels.
    ///
    /// Pure-mode labels remain positive; density-matrix mode uses negative labels
    /// for bra legs.
    pub fn to_tensor_network(&self) -> anyhow::Result<TensorNetworkDM> {
        let ixs: Vec<Vec<i32>> = self
            .eincode
            .input_indices
            .iter()
            .map(|legs| {
                legs.iter()
                    .map(|label| label.parse::<i32>())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let iy: Vec<i32> = self
            .eincode
            .output_indices
            .iter()
            .map(|label| label.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()?;

        let size_dict: HashMap<i32, usize> = self
            .size_dict
            .iter()
            .map(|(label, size)| Ok((label.parse::<i32>()?, *size)))
            .collect::<Result<HashMap<_, _>, std::num::ParseIntError>>()?;

        Ok(TensorNetworkDM {
            code: EinCode::new(ixs, iy),
            tensors: reconstruct_tensors(&self.tensors)?,
            size_dict,
        })
    }
}

fn reconstruct_tensors(tensor_dtos: &[TensorDto]) -> anyhow::Result<Vec<ArrayD<Complex64>>> {
    tensor_dtos
        .iter()
        .map(|tensor| {
            let data: Vec<Complex64> = tensor
                .data_re
                .iter()
                .zip(tensor.data_im.iter())
                .map(|(&re, &im)| Complex64::new(re, im))
                .collect();

            ArrayD::from_shape_vec(ndarray::IxDyn(&tensor.shape), data)
                .map_err(|e| anyhow::anyhow!("Failed to reconstruct tensor: {e}"))
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use yao_rs::{Circuit, Gate, circuit_to_einsum, put};

    fn pure_labels_as_i32(tn: &TensorNetwork) -> (Vec<Vec<i32>>, Vec<i32>, HashMap<i32, usize>) {
        let ixs = tn
            .code
            .ixs
            .iter()
            .map(|legs| legs.iter().map(|&label| label as i32).collect())
            .collect();
        let iy = tn.code.iy.iter().map(|&label| label as i32).collect();
        let size_dict = tn
            .size_dict
            .iter()
            .map(|(&label, &size)| (label as i32, size))
            .collect();

        (ixs, iy, size_dict)
    }

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
        let (expected_ixs, expected_iy, expected_size_dict) = pure_labels_as_i32(&tn);

        assert_eq!(tn2.code.ixs, expected_ixs);
        assert_eq!(tn2.code.iy, expected_iy);
        assert_eq!(tn2.size_dict, expected_size_dict);
        assert_eq!(tn2.tensors.len(), tn.tensors.len());
        for (a, b) in tn2.tensors.iter().zip(tn.tensors.iter()) {
            assert_eq!(a.shape(), b.shape());
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).norm() < 1e-15);
            }
        }
    }

    #[test]
    fn test_tn_dto_to_tensor_network_round_trip_density_matrix_labels() {
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = yao_rs::circuit_to_einsum_dm(&circuit);
        let dto = TensorNetworkDto::from_dm(&tn);

        let json = serde_json::to_string(&dto).unwrap();
        let parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        let tn2 = parsed.to_tensor_network().unwrap();

        assert_eq!(tn2.code.ixs, tn.code.ixs);
        assert_eq!(tn2.code.iy, tn.code.iy);
        assert_eq!(tn2.size_dict, tn.size_dict);
        assert!(tn2.code.ixs.iter().flatten().any(|&label| label < 0));
        assert!(tn2.code.iy.iter().any(|&label| label < 0));
    }

    #[test]
    fn test_tn_dto_contraction_order_serialization() {
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = circuit_to_einsum(&circuit);
        let mut dto = TensorNetworkDto::from_pure(&tn);

        assert!(dto.contraction_order.is_none());

        dto.contraction_order = Some(NestedEinsumTree::Leaf {
            isleaf: true,
            tensor_index: 0,
        });

        let json = serde_json::to_string(&dto).unwrap();
        assert!(json.contains("contraction_order"));

        let parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        assert!(parsed.contraction_order.is_some());
    }
}
