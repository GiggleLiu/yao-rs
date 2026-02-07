//! Feature-gated libtorch-based tensor network contractor.
//!
//! Enable with the `torch` feature flag:
//! ```toml
//! [dependencies]
//! yao-rs = { version = "0.1", features = ["torch"] }
//! ```

use std::collections::HashMap;

use ndarray::ArrayD;
use num_complex::Complex64;
use omeco::{EinCode, GreedyMethod, NestedEinsum};
use tch::{Device, Kind, Tensor};

use crate::einsum::TensorNetwork;

/// Contract a tensor network on the specified device using libtorch.
///
/// Uses omeco's greedy optimizer to find a contraction order, then
/// executes pairwise contractions using `tch::Tensor::einsum`.
///
/// # Arguments
/// * `tn` - The tensor network to contract
/// * `device` - `Device::Cpu` or `Device::Cuda(n)` for GPU
///
/// # Returns
/// The contracted result as a `tch::Tensor` (complex64).
///
/// # Panics
/// Panics if the contraction order optimization fails.
pub fn contract(tn: &TensorNetwork, device: Device) -> Tensor {
    // Convert ndarray tensors to tch tensors
    let tch_tensors: Vec<Tensor> = tn
        .tensors
        .iter()
        .map(|t| ndarray_to_tch(t, device))
        .collect();

    // Optimize contraction order
    let tree = omeco::optimize_greedy(&tn.code, &tn.size_dict, &GreedyMethod::default())
        .expect("Contraction order optimization failed");

    // Execute the contraction tree
    execute_tree(&tree, &tch_tensors, &tn.code.ixs)
}

/// Convert an ndarray complex64 tensor to a tch::Tensor on the given device.
fn ndarray_to_tch(arr: &ArrayD<Complex64>, device: Device) -> Tensor {
    let shape: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();

    // Interleave real and imaginary parts for ComplexFloat64
    let data: Vec<f64> = arr.iter().flat_map(|c| [c.re, c.im]).collect();

    // Create as real tensor with doubled last dimension, then view as complex
    let numel: i64 = shape.iter().product();
    let real_tensor = Tensor::from_slice(&data)
        .to_device(device)
        .reshape([numel, 2]);

    // Convert to complex
    Tensor::view_as_complex(&real_tensor).reshape(shape)
}

/// Recursively execute a NestedEinsum contraction tree.
fn execute_tree(
    tree: &NestedEinsum<usize>,
    tensors: &[Tensor],
    original_ixs: &[Vec<usize>],
) -> Tensor {
    match tree {
        NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].shallow_clone(),
        NestedEinsum::Node { args, eins } => {
            // Recursively compute children
            let child_results: Vec<Tensor> = args
                .iter()
                .map(|child| execute_tree(child, tensors, original_ixs))
                .collect();

            // Build einsum string from the node's EinCode
            let einsum_str = eincode_to_string(eins);

            // Execute the contraction
            Tensor::einsum(&einsum_str, &child_results, None::<&[i64]>)
        }
    }
}

/// Convert an EinCode<usize> to an einsum notation string.
/// Maps usize labels to single characters (a, b, c, ...).
fn eincode_to_string(code: &EinCode<usize>) -> String {
    // Collect all unique labels
    let mut all_labels: Vec<usize> = Vec::new();
    for ixs in &code.ixs {
        for &l in ixs {
            if !all_labels.contains(&l) {
                all_labels.push(l);
            }
        }
    }
    for &l in &code.iy {
        if !all_labels.contains(&l) {
            all_labels.push(l);
        }
    }

    // Map labels to characters
    let label_to_char: HashMap<usize, char> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, index_to_char(i)))
        .collect();

    // Build the string: "input1,input2,...->output"
    let inputs: Vec<String> = code
        .ixs
        .iter()
        .map(|ixs| ixs.iter().map(|l| label_to_char[l]).collect())
        .collect();

    let output: String = code.iy.iter().map(|l| label_to_char[l]).collect();

    format!("{}->{}", inputs.join(","), output)
}

/// Map an index to a character for einsum notation.
/// Uses a-z (26), then A-Z (26), for up to 52 unique labels.
fn index_to_char(i: usize) -> char {
    if i < 26 {
        (b'a' + i as u8) as char
    } else if i < 52 {
        (b'A' + (i - 26) as u8) as char
    } else {
        panic!("Too many unique labels (max 52) for einsum string notation")
    }
}
