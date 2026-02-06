use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use omeco::EinCode;

use crate::circuit::{Circuit, CircuitElement};
use crate::operator::{OperatorPolynomial, op_matrix};
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

    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) => continue, // Skip annotations
        };

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
            let tensor_ixs: Vec<usize> = pg
                .target_locs
                .iter()
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

/// Convert circuit to tensor network for ⟨0|U|0⟩ (overlap with zero state)
///
/// This computes the amplitude of the circuit applied to |0...0⟩ and projected
/// onto ⟨0...0|. All qubits are pinned to |0⟩ in both initial and final states,
/// resulting in a scalar output.
///
/// # Arguments
/// * `circuit` - The quantum circuit to convert
///
/// # Returns
/// A `TensorNetwork` representing ⟨0|U|0⟩ with empty output indices (scalar result).
pub fn circuit_to_overlap(circuit: &Circuit) -> TensorNetwork {
    // Use existing circuit_to_einsum_with_boundary with all qubits pinned to |0⟩
    let final_state: Vec<usize> = (0..circuit.num_sites()).collect();
    circuit_to_einsum_with_boundary(circuit, &final_state)
}

/// Convert a quantum circuit into a tensor network with boundary conditions.
///
/// The initial state is always |0...0⟩. Each qubit gets a rank-1 tensor
/// `[1, 0, ..., 0]` (length = dims[i]) attached to its input leg.
///
/// Qubits listed in `final_state` are pinned to |0⟩ in the output,
/// receiving a similar rank-1 tensor on their output leg.
/// Unpinned qubits remain as open indices in the result.
///
/// - All qubits pinned → scalar result (amplitude ⟨0|C|0⟩)
/// - No qubits pinned → full output state tensor
///
/// # Arguments
/// * `circuit` - The quantum circuit to convert
/// * `final_state` - Qubit indices to pin to |0⟩ in the output
///
/// # Returns
/// A `TensorNetwork` with boundary tensors included.
pub fn circuit_to_einsum_with_boundary(circuit: &Circuit, final_state: &[usize]) -> TensorNetwork {
    let n = circuit.num_sites();

    // Labels 0..n-1 are initial state indices for each site
    let mut current_labels: Vec<usize> = (0..n).collect();
    let mut next_label: usize = n;

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        size_dict.insert(i, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<usize>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    // Add initial state boundary tensors: |0⟩ on each qubit's input leg
    for i in 0..n {
        let d = circuit.dims[i];
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[d]), data).unwrap();
        all_ixs.push(vec![i]); // input label for qubit i
        all_tensors.push(tensor);
    }

    // Process gates (same as circuit_to_einsum)
    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) => continue, // Skip annotations
        };

        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);
        let all_locs = pg.all_locs();
        let has_controls = !pg.control_locs.is_empty();
        let is_diagonal = pg.gate.is_diagonal() && !has_controls;

        if is_diagonal {
            let tensor_ixs: Vec<usize> = pg
                .target_locs
                .iter()
                .map(|&loc| current_labels[loc])
                .collect();
            all_ixs.push(tensor_ixs);
        } else {
            let mut tensor_ixs: Vec<usize> = Vec::new();
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }
            tensor_ixs.extend(&new_labels);
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }
            all_ixs.push(tensor_ixs);
        }

        all_tensors.push(tensor);
    }

    // Add final state boundary tensors for pinned qubits
    for &qubit in final_state {
        let d = circuit.dims[qubit];
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[d]), data).unwrap();
        all_ixs.push(vec![current_labels[qubit]]); // output label for this qubit
        all_tensors.push(tensor);
    }

    // Output indices = final labels of unpinned qubits only
    let pinned: std::collections::HashSet<usize> = final_state.iter().copied().collect();
    let output_labels: Vec<usize> = (0..n)
        .filter(|i| !pinned.contains(i))
        .map(|i| current_labels[i])
        .collect();

    TensorNetwork {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}

/// Convert circuit to tensor network for computing expectation value ⟨0|U†OU|0⟩
///
/// This creates a tensor network representing the expectation value of an operator
/// O (given as an OperatorPolynomial) with respect to the circuit applied to |0...0⟩.
///
/// The structure is:
/// ```text
///    ⟨0|  ⟨0|  ⟨0|
///     │    │    │
///    ┌┴────┴────┴┐
///    │    U†     │
///    └┬────┬────┬┘
///     │    │    │
///    [  O = Σ cᵢ·Oᵢ  ]
///     │    │    │
///    ┌┴────┴────┴┐
///    │     U     │
///    └┬────┬────┬┘
///     │    │    │
///    |0⟩  |0⟩  |0⟩
/// ```
///
/// # Arguments
/// * `circuit` - The quantum circuit U
/// * `operator` - The operator polynomial O = Σ cᵢ·Oᵢ
///
/// # Returns
/// A `TensorNetwork` representing ⟨0|U†OU|0⟩ with empty output indices (scalar result).
///
/// # Note
/// For simplicity, this implementation handles single-term operators where each term
/// consists of single-site operators. The operator polynomial coefficients are multiplied
/// into the first operator tensor.
pub fn circuit_to_expectation(circuit: &Circuit, operator: &OperatorPolynomial) -> TensorNetwork {
    let n = circuit.num_sites();

    // Labels 0..n-1 are initial state indices for each site
    let mut current_labels: Vec<usize> = (0..n).collect();
    let mut next_label: usize = n;

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        size_dict.insert(i, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<usize>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    // ===== Part 1: Initial state boundary tensors |0⟩ on each qubit =====
    for i in 0..n {
        let d = circuit.dims[i];
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(IxDyn(&[d]), data).unwrap();
        all_ixs.push(vec![i]); // input label for qubit i
        all_tensors.push(tensor);
    }

    // ===== Part 2: U circuit tensors =====
    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) => continue, // Skip annotations
        };

        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);
        let all_locs = pg.all_locs();
        let has_controls = !pg.control_locs.is_empty();
        let is_diagonal = pg.gate.is_diagonal() && !has_controls;

        if is_diagonal {
            let tensor_ixs: Vec<usize> = pg
                .target_locs
                .iter()
                .map(|&loc| current_labels[loc])
                .collect();
            all_ixs.push(tensor_ixs);
        } else {
            let mut tensor_ixs: Vec<usize> = Vec::new();
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }
            tensor_ixs.extend(&new_labels);
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }
            all_ixs.push(tensor_ixs);
        }

        all_tensors.push(tensor);
    }

    // Labels after U - these are where we insert the operator
    let u_output_labels = current_labels.clone();

    // ===== Part 3: Operator tensors =====
    // Handle operator polynomial - we create tensors for each term and combine
    // For now, we handle single-site operators in each term

    // We need to insert operator tensors that connect U output to U† input
    // Each site gets an operator (Identity if not explicitly specified in the term)

    // For simplicity, we handle a single-term operator polynomial first
    // TODO: Support multi-term polynomials by either:
    // 1. Creating separate tensor networks and summing results
    // 2. Using a more sophisticated contraction strategy

    if operator.is_empty() {
        // Zero operator - return a network that gives 0
        // We can do this by adding a zero tensor
        let zero_tensor =
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![Complex64::new(0.0, 0.0)]).unwrap();
        all_ixs.push(vec![]);
        all_tensors.push(zero_tensor);
    } else {
        // For each site, collect which operator is applied
        let mut site_ops: Vec<Option<(crate::operator::Op, Complex64)>> = vec![None; n];

        // Use the first term (simplification - full implementation would handle all terms)
        let (coeff, opstring) = operator.iter().next().unwrap();

        for (site, op) in opstring.ops() {
            site_ops[*site] = Some((*op, *coeff));
        }

        // Insert operator tensors for each site
        for i in 0..n {
            let d = circuit.dims[i];

            // Get the operator matrix (or identity if none specified)
            let op_mat = if let Some((op, coeff_for_site)) = &site_ops[i] {
                let mut mat = op_matrix(op);
                // Apply coefficient to first operator only to avoid multiplying multiple times
                if i == opstring.ops().first().map(|(s, _)| *s).unwrap_or(i) {
                    for elem in mat.iter_mut() {
                        *elem *= coeff_for_site;
                    }
                }
                mat
            } else {
                op_matrix(&crate::operator::Op::I)
            };

            // Create tensor: shape (d, d) for operator connecting U output to U† input
            let input_label = u_output_labels[i];
            let output_label = next_label;
            next_label += 1;
            size_dict.insert(output_label, d);

            // Tensor data: op_mat[out, in] -> shape [d, d] with legs [output_label, input_label]
            let mut data = Vec::with_capacity(d * d);
            for out_idx in 0..d {
                for in_idx in 0..d {
                    data.push(op_mat[[out_idx, in_idx]]);
                }
            }
            let tensor = ArrayD::from_shape_vec(IxDyn(&[d, d]), data).unwrap();
            all_ixs.push(vec![output_label, input_label]);
            all_tensors.push(tensor);

            current_labels[i] = output_label;
        }
    }

    // ===== Part 4: U† circuit tensors (conjugate transpose, reverse order) =====
    // For U†, we process gates in reverse order and conjugate the matrices
    // We need to filter to only gates and reverse
    let gates_only: Vec<_> = circuit
        .elements
        .iter()
        .filter_map(|e| match e {
            CircuitElement::Gate(pg) => Some(pg),
            CircuitElement::Annotation(_) => None,
        })
        .collect();

    for pg in gates_only.iter().rev() {
        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

        // Conjugate the tensor for U†
        let conj_tensor = tensor.mapv(|c| c.conj());

        let all_locs = pg.all_locs();
        let has_controls = !pg.control_locs.is_empty();
        let is_diagonal = pg.gate.is_diagonal() && !has_controls;

        if is_diagonal {
            // For diagonal U†: still diagonal, just conjugated values
            let tensor_ixs: Vec<usize> = pg
                .target_locs
                .iter()
                .map(|&loc| current_labels[loc])
                .collect();
            all_ixs.push(tensor_ixs);
            all_tensors.push(conj_tensor);
        } else {
            // For non-diagonal U†: need to transpose (swap in/out legs)
            // Original: [out0, out1, ..., in0, in1, ...]
            // Adjoint: [in0, in1, ..., out0, out1, ...] with conjugate values

            let n_sites = all_locs.len();

            // Transpose the tensor: swap first half and second half of axes
            let mut axes: Vec<usize> = (n_sites..2 * n_sites).collect();
            axes.extend(0..n_sites);
            let transposed = conj_tensor.permuted_axes(axes.as_slice());

            let mut tensor_ixs: Vec<usize> = Vec::new();
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }
            // For adjoint: new output labels first, then current input labels
            tensor_ixs.extend(&new_labels);
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }
            all_ixs.push(tensor_ixs);
            all_tensors.push(transposed.into_owned());
        }
    }

    // ===== Part 5: Final state boundary tensors ⟨0| on each qubit =====
    for (&d, &label) in circuit.dims.iter().zip(current_labels.iter()).take(n) {
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(IxDyn(&[d]), data).unwrap();
        all_ixs.push(vec![label]); // output label for this qubit
        all_tensors.push(tensor);
    }

    // Output is empty (scalar result)
    let output_labels: Vec<usize> = vec![];

    TensorNetwork {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}
