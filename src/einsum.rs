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
/// - Non-diagonal gates allocate new output labels
/// - Diagonal gates reuse current labels (no new allocation). A controlled
///   gate whose target is diagonal has a diagonal full matrix, so it reuses
///   labels too (YaoToEinsum `isdiag` semantics); the contraction value is
///   identical to the dense form, on a strictly easier label hypergraph.
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
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => continue,
        };

        // Get the tensor for this gate
        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

        // Determine all_locs = control_locs ++ target_locs
        let all_locs = pg.all_locs();

        // Diagonal (with or without controls): full matrix is diagonal
        let is_diagonal = pg.gate.is_diagonal();

        if is_diagonal {
            // Diagonal: tensor legs are the current labels of all involved
            // sites (controls ++ targets). Labels don't change.
            let tensor_ixs: Vec<usize> = all_locs.iter().map(|&loc| current_labels[loc]).collect();
            all_ixs.push(tensor_ixs);
        } else {
            // Non-diagonal: allocate new output labels for all involved sites.
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
/// `[1, 0, ..., 0]` (length = `dims[i]`) attached to its input leg.
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
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => continue,
        };

        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);
        let all_locs = pg.all_locs();
        let is_diagonal = pg.gate.is_diagonal();

        if is_diagonal {
            let tensor_ixs: Vec<usize> = all_locs.iter().map(|&loc| current_labels[loc]).collect();
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
/// Terms share the circuit tensors and a summed term-selection index. Each
/// coefficient appears once, including identity terms. Identity factors support
/// arbitrary site dimensions; nonidentity operators require qubits.
///
/// # Panics
/// Panics for channels (use the density-matrix variant), invalid operator sites,
/// duplicate sites within a term, nonfinite coefficients, or nonqubit operators.
pub fn circuit_to_expectation(circuit: &Circuit, operator: &OperatorPolynomial) -> TensorNetwork {
    assert!(
        !circuit
            .elements
            .iter()
            .any(|e| matches!(e, CircuitElement::Channel(_))),
        "Pure expectation export does not support channels; use circuit_to_expectation_dm"
    );
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
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => continue,
        };

        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);
        let all_locs = pg.all_locs();
        let is_diagonal = pg.gate.is_diagonal();

        if is_diagonal {
            let tensor_ixs: Vec<usize> = all_locs.iter().map(|&loc| current_labels[loc]).collect();
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

    // Share U and U† across all polynomial terms. The selector is a summed
    // hyperedge; slicing it also provides a bounded term-by-term execution path.
    let output_labels: Vec<_> = (0..n)
        .map(|i| {
            let label = next_label;
            next_label += 1;
            size_dict.insert(label, circuit.dims[i]);
            label
        })
        .collect();
    append_observable(
        &circuit.dims,
        operator,
        &current_labels,
        &output_labels,
        next_label,
        &mut all_ixs,
        &mut all_tensors,
        &mut size_dict,
    );
    if operator.len() > 1 && n > 0 {
        next_label += 1;
    }
    current_labels = output_labels;

    // ===== Part 4: U† circuit tensors (conjugate transpose, reverse order) =====
    // For U†, we process gates in reverse order and conjugate the matrices
    // We need to filter to only gates and reverse
    let gates_only: Vec<_> = circuit
        .elements
        .iter()
        .filter_map(|e| match e {
            CircuitElement::Gate(pg) => Some(pg),
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => None,
        })
        .collect();

    for pg in gates_only.iter().rev() {
        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

        // Conjugate the tensor for U†
        let conj_tensor = tensor.mapv(|c| c.conj());

        let all_locs = pg.all_locs();
        let is_diagonal = pg.gate.is_diagonal();

        if is_diagonal {
            // For diagonal U†: still diagonal, just conjugated values
            let tensor_ixs: Vec<usize> = all_locs.iter().map(|&loc| current_labels[loc]).collect();
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

/// Tensor network with i32 labels for density matrix mode.
/// Positive labels = ket (forward), negative = bra (conjugate).
///
/// Julia ref: YaoToEinsum/src/Core.jl TensorNetwork
#[derive(Debug, Clone)]
pub struct TensorNetworkDM {
    pub code: EinCode<i32>,
    pub tensors: Vec<ArrayD<Complex64>>,
    pub size_dict: HashMap<i32, usize>,
}

/// Convert a circuit to a density matrix tensor network.
///
/// Pure gates are doubled (ket + bra copies). Noise channels are
/// converted to superoperator tensors.
///
/// Initial state: |0...0><0...0|
///
/// Julia ref: YaoToEinsum/src/circuitmap.jl:353-381 yao2einsum(; mode=DensityMatrixMode())
pub fn circuit_to_einsum_dm(circuit: &Circuit) -> TensorNetworkDM {
    let n = circuit.num_sites();

    // Labels 1..n for ket, -1..-n for bra (matching Yao.jl)
    let mut slots: Vec<i32> = (1..=n as i32).collect();
    let mut next_label: i32 = n as i32 + 1;

    let mut size_dict: HashMap<i32, usize> = HashMap::new();
    for i in 0..n {
        let label = (i + 1) as i32;
        size_dict.insert(label, circuit.dims[i]);
        size_dict.insert(-label, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<i32>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    // Initial state: |0><0| boundary tensors on each qubit
    for (&d, &slot) in circuit.dims.iter().zip(slots.iter()).take(n) {
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(IxDyn(&[d]), data.clone()).unwrap();
        let tensor_conj = tensor.clone();

        // Ket boundary
        all_ixs.push(vec![slot]);
        all_tensors.push(tensor);
        // Bra boundary
        all_ixs.push(vec![-slot]);
        all_tensors.push(tensor_conj);
    }

    for element in &circuit.elements {
        match element {
            CircuitElement::Gate(pg) => {
                let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

                let all_locs = pg.all_locs();
                let is_diag = pg.gate.is_diagonal();

                if is_diag {
                    // Diagonal: reuse labels, add ket and bra copies
                    let ket_ixs: Vec<i32> = all_locs.iter().map(|&loc| slots[loc]).collect();
                    all_ixs.push(ket_ixs.clone());
                    all_tensors.push(tensor.clone());

                    // Bra copy: conj tensor, negated labels
                    let bra_ixs: Vec<i32> = ket_ixs.iter().map(|&l| -l).collect();
                    all_ixs.push(bra_ixs);
                    all_tensors.push(tensor.mapv(|c| c.conj()));
                } else {
                    // Non-diagonal: allocate new output labels
                    let mut new_labels: Vec<i32> = Vec::new();
                    for &loc in &all_locs {
                        let nl = next_label;
                        next_label += 1;
                        size_dict.insert(nl, circuit.dims[loc]);
                        size_dict.insert(-nl, circuit.dims[loc]);
                        new_labels.push(nl);
                    }

                    // Ket tensor: [new_out..., current_in...]
                    let mut ket_ixs: Vec<i32> = new_labels.clone();
                    for &loc in &all_locs {
                        ket_ixs.push(slots[loc]);
                    }
                    all_ixs.push(ket_ixs);
                    all_tensors.push(tensor.clone());

                    // Bra tensor: conj, negated labels
                    let mut bra_ixs: Vec<i32> = new_labels.iter().map(|&l| -l).collect();
                    for &loc in &all_locs {
                        bra_ixs.push(-slots[loc]);
                    }
                    all_ixs.push(bra_ixs);
                    all_tensors.push(tensor.mapv(|c| c.conj()));

                    // Update slots
                    for (i, &loc) in all_locs.iter().enumerate() {
                        slots[loc] = new_labels[i];
                    }
                }
            }
            CircuitElement::Channel(pc) => {
                // Convert to superoperator tensor
                let superop = pc.channel.superop();
                let k = pc.locs.len();
                let d = circuit.dims[pc.locs[0]];

                // Reshape to D^(4k) tensor
                let shape: Vec<usize> = vec![d; 4 * k];
                let tensor =
                    ArrayD::from_shape_vec(IxDyn(&shape), superop.into_raw_vec_and_offset().0)
                        .unwrap();

                // Allocate new output labels
                let mut new_labels: Vec<i32> = Vec::new();
                for &loc in &pc.locs {
                    let nl = next_label;
                    next_label += 1;
                    size_dict.insert(nl, circuit.dims[loc]);
                    size_dict.insert(-nl, circuit.dims[loc]);
                    new_labels.push(nl);
                }

                // Superoperator S maps rho_in to rho_out:
                // S = sum_i kron(K_i^*, K_i)
                // As matrix: S[bra_out * d + ket_out, bra_in * d + ket_in]
                // Reshaped to tensor: S[bra_out, ket_out, bra_in, ket_in]
                // Labels: [-out, out, -in, in]
                let mut ixs: Vec<i32> = Vec::new();
                for &l in &new_labels {
                    ixs.push(-l); // out bra
                }
                for &l in &new_labels {
                    ixs.push(l); // out ket
                }
                for &loc in &pc.locs {
                    ixs.push(-slots[loc]); // in bra
                }
                for &loc in &pc.locs {
                    ixs.push(slots[loc]); // in ket
                }

                all_ixs.push(ixs);
                all_tensors.push(tensor);

                // Update slots
                for (i, &loc) in pc.locs.iter().enumerate() {
                    slots[loc] = new_labels[i];
                }
            }
            CircuitElement::Annotation(_) => continue,
        }
    }

    // Output: [ket_slots, bra_slots] for full density matrix
    let mut output_labels: Vec<i32> = slots.clone();
    output_labels.extend(slots.iter().map(|&l| -l));

    TensorNetworkDM {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}

/// Compute expectation value tr(O * rho) in density matrix mode.
///
/// Builds the DM tensor network, inserts the operator on the ket side,
/// and traces ket with bra indices to produce a scalar.
///
/// Julia ref: circuitmap.jl:252-258 eat_observable!
pub fn circuit_to_expectation_dm(
    circuit: &Circuit,
    operator: &OperatorPolynomial,
) -> TensorNetworkDM {
    let n = circuit.num_sites();

    // Build the DM tensor network
    let mut tn = circuit_to_einsum_dm(circuit);

    // The current output labels are [ket_slots, bra_slots]
    let ket_labels: Vec<i32> = tn.code.iy[..n].to_vec();
    let bra_labels: Vec<i32> = tn.code.iy[n..].to_vec();

    let selector = tn
        .size_dict
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .expect("Observable label overflow");
    append_observable(
        &circuit.dims,
        operator,
        &ket_labels,
        &bra_labels,
        selector,
        &mut tn.code.ixs,
        &mut tn.tensors,
        &mut tn.size_dict,
    );
    tn.code.iy.clear();
    tn
}

// Factor a polynomial into local operator tensors joined by a term selector.
// The singleton case keeps its original rank-two representation.
#[allow(clippy::too_many_arguments)]
fn append_observable<L: omeco::Label>(
    dims: &[usize],
    operator: &OperatorPolynomial,
    inputs: &[L],
    outputs: &[L],
    selector: L,
    ixs: &mut Vec<Vec<L>>,
    tensors: &mut Vec<ArrayD<Complex64>>,
    sizes: &mut HashMap<L, usize>,
) {
    use crate::operator::Op;
    assert_eq!(
        operator.coeffs().len(),
        operator.opstrings().len(),
        "Invalid polynomial lengths"
    );
    for (coefficient, word) in operator.iter() {
        assert!(
            coefficient.re.is_finite() && coefficient.im.is_finite(),
            "Observable coefficients must be finite"
        );
        let mut sites = std::collections::HashSet::new();
        for &(site, op) in word.ops() {
            assert!(site < dims.len(), "Observable site out of range");
            assert!(sites.insert(site), "Duplicate observable site");
            assert!(
                op == Op::I || dims[site] == 2,
                "Nonidentity observable operators require qubits"
            );
        }
    }
    if dims.is_empty() {
        ixs.push(vec![]);
        tensors.push(ArrayD::from_elem(
            IxDyn(&[]),
            operator.coeffs().iter().sum(),
        ));
        return;
    }
    let terms = operator.len().max(1);
    let multiple = terms > 1;
    if multiple {
        sizes.insert(selector.clone(), terms);
    }
    for (site, &d) in dims.iter().enumerate() {
        let mut shape = vec![d, d];
        let mut legs = vec![outputs[site].clone(), inputs[site].clone()];
        if multiple {
            shape.insert(0, terms);
            legs.insert(0, selector.clone());
        }
        let mut tensor = ArrayD::zeros(IxDyn(&shape));
        for k in 0..terms {
            let (coefficient, op) = operator
                .coeffs()
                .get(k)
                .zip(operator.opstrings().get(k))
                .map_or((Complex64::new(0., 0.), Op::I), |(c, word)| {
                    (
                        *c,
                        word.ops()
                            .iter()
                            .find(|(s, _)| *s == site)
                            .map_or(Op::I, |(_, op)| *op),
                    )
                });
            let mut matrix = if op == Op::I {
                ndarray::Array2::eye(d)
            } else {
                op_matrix(&op)
            };
            let coefficient = if site == 0 {
                coefficient
            } else {
                Complex64::new(1., 0.)
            };
            matrix.mapv_inplace(|value| coefficient * value);
            if multiple {
                tensor
                    .index_axis_mut(ndarray::Axis(0), k)
                    .assign(&matrix.into_dyn());
            } else {
                tensor.assign(&matrix.into_dyn());
            }
        }
        ixs.push(legs);
        tensors.push(tensor);
    }
}

#[cfg(test)]
#[path = "unit_tests/observables.rs"]
mod observable_tests;
