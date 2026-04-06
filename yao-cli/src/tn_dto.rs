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
        }
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
}
