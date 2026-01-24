use std::collections::HashMap;

use ndarray::ArrayD;
use num_complex::Complex64;
use omeco::EinCode;

use crate::circuit::Circuit;
use crate::tensors::gate_to_tensor;

/// A tensor network representation of a quantum circuit.
///
/// Contains the einsum contraction code, the tensor data, and
/// a size dictionary mapping labels to their dimensions.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    pub code: EinCode<usize>,
    pub tensors: Vec<ArrayD<Complex64>>,
    pub size_dict: HashMap<usize, usize>,
}

/// Convert a quantum circuit into a tensor network (einsum) representation.
///
/// The algorithm assigns integer labels to tensor legs:
/// - Labels 0..n-1 are initial state indices for each site
/// - Non-diagonal gates (or gates with controls) allocate new output labels
/// - Diagonal gates without controls reuse current labels (no new allocation)
///
/// # Arguments
/// * `circuit` - The quantum circuit to convert
///
/// # Returns
/// A `TensorNetwork` containing the EinCode, tensors, and size dictionary.
pub fn circuit_to_einsum(circuit: &Circuit) -> TensorNetwork {
    let n = circuit.num_sites();

    // Labels 0..n-1 are initial state indices for each site
    let mut current_labels: Vec<usize> = (0..n).collect();
    let mut next_label: usize = n;

    // Initialize size_dict: label -> dimension for initial labels
    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        size_dict.insert(i, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<usize>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    for pg in &circuit.gates {
        // Get the tensor for this gate
        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

        // Determine all_locs = control_locs ++ target_locs
        let all_locs = pg.all_locs();

        // Check if gate is diagonal and has no controls
        let has_controls = !pg.control_locs.is_empty();
        let is_diagonal = pg.gate.is_diagonal() && !has_controls;

        if is_diagonal {
            // Diagonal (no controls): tensor legs are just current labels of target sites.
            // Labels don't change.
            let tensor_ixs: Vec<usize> = pg.target_locs.iter()
                .map(|&loc| current_labels[loc])
                .collect();
            all_ixs.push(tensor_ixs);
        } else {
            // Non-diagonal (or has controls): allocate new output labels for all involved sites.
            // Tensor legs are [new_labels..., current_input_labels...]
            let mut tensor_ixs: Vec<usize> = Vec::new();

            // Allocate new output labels for all involved sites
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }

            // Tensor indices: [new_labels..., current_input_labels...]
            tensor_ixs.extend(&new_labels);
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }

            // Update current_labels for involved sites
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }

            all_ixs.push(tensor_ixs);
        }

        all_tensors.push(tensor);
    }

    // Output labels = final current_labels
    let output_labels = current_labels;

    TensorNetwork {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}
