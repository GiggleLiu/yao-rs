use std::collections::HashMap;

use ndarray::{Array1, ArrayD, IxDyn};
use num_complex::Complex64;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate, put, control};

// Helper to wrap PositionedGate in CircuitElement::Gate
fn gate(pg: PositionedGate) -> CircuitElement {
    CircuitElement::Gate(pg)
}
use yao_rs::einsum::{circuit_to_einsum, TensorNetwork};
use yao_rs::gate::Gate;
use yao_rs::state::State;

const ATOL: f64 = 1e-10;

// ==================== Helper Functions ====================

/// Decompose a flat index into a multi-index given dimensions (row-major order).
fn flat_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % shape[i];
        index /= shape[i];
    }
    multi
}

/// Compose a multi-index into a flat index given dimensions (row-major order).
fn multi_to_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = shape[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

/// Contract two tensors along shared (contracted) indices.
///
/// a: tensor A with indices a_indices
/// b: tensor B with indices b_indices
/// contracted: the set of index labels that are summed over and removed
/// size_dict: maps index labels to their dimensions
///
/// Indices that appear in both a_indices and b_indices but NOT in contracted
/// are treated as "batch" dimensions (element-wise, kept in result once).
///
/// Returns (result_tensor, result_indices)
fn contract_tensors(
    a: &ArrayD<Complex64>,
    a_indices: &[usize],
    b: &ArrayD<Complex64>,
    b_indices: &[usize],
    contracted: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> (ArrayD<Complex64>, Vec<usize>) {
    // Batch indices: appear in both a and b but are NOT contracted
    let batch: Vec<usize> = a_indices
        .iter()
        .filter(|idx| b_indices.contains(idx) && !contracted.contains(idx))
        .copied()
        .collect();

    // Result indices: all indices from a and b except contracted ones,
    // with batch indices appearing only once
    let mut result_indices: Vec<usize> = Vec::new();
    for &idx in a_indices {
        if !contracted.contains(&idx) {
            result_indices.push(idx);
        }
    }
    for &idx in b_indices {
        if !contracted.contains(&idx) && !batch.contains(&idx) {
            // Only add b's non-batch, non-contracted indices
            if !result_indices.contains(&idx) {
                result_indices.push(idx);
            }
        }
    }

    // Build result shape
    let result_shape: Vec<usize> = result_indices.iter().map(|idx| size_dict[idx]).collect();
    let result_total: usize = if result_shape.is_empty() { 1 } else { result_shape.iter().product() };

    // Build shapes for a and b
    let a_shape: Vec<usize> = a_indices.iter().map(|idx| size_dict[idx]).collect();
    let b_shape: Vec<usize> = b_indices.iter().map(|idx| size_dict[idx]).collect();
    let a_total: usize = if a_shape.is_empty() { 1 } else { a_shape.iter().product() };
    let b_total: usize = if b_shape.is_empty() { 1 } else { b_shape.iter().product() };

    let mut result_data = vec![Complex64::new(0.0, 0.0); result_total];

    // Iterate over all index combinations of a and b
    for ai in 0..a_total {
        let a_multi = flat_to_multi(ai, &a_shape);
        let a_val = a[IxDyn(&a_multi)];

        if a_val == Complex64::new(0.0, 0.0) {
            continue;
        }

        for bi in 0..b_total {
            let b_multi = flat_to_multi(bi, &b_shape);

            // Check that contracted indices match between a and b
            let mut match_ok = true;
            for &s in contracted {
                let a_pos = a_indices.iter().position(|&x| x == s);
                let b_pos = b_indices.iter().position(|&x| x == s);
                match (a_pos, b_pos) {
                    (Some(ap), Some(bp)) => {
                        if a_multi[ap] != b_multi[bp] {
                            match_ok = false;
                            break;
                        }
                    }
                    _ => {
                        match_ok = false;
                        break;
                    }
                }
            }
            if !match_ok {
                continue;
            }

            // Check that batch indices match between a and b
            for &s in &batch {
                let a_pos = a_indices.iter().position(|&x| x == s);
                let b_pos = b_indices.iter().position(|&x| x == s);
                match (a_pos, b_pos) {
                    (Some(ap), Some(bp)) => {
                        if a_multi[ap] != b_multi[bp] {
                            match_ok = false;
                            break;
                        }
                    }
                    _ => {
                        match_ok = false;
                        break;
                    }
                }
            }
            if !match_ok {
                continue;
            }

            let b_val = b[IxDyn(&b_multi)];
            let product = a_val * b_val;

            if product == Complex64::new(0.0, 0.0) {
                continue;
            }

            // Determine result multi-index
            let mut res_multi = vec![0usize; result_indices.len()];
            for (ri, &ridx) in result_indices.iter().enumerate() {
                if let Some(pos) = a_indices.iter().position(|&x| x == ridx) {
                    res_multi[ri] = a_multi[pos];
                } else if let Some(pos) = b_indices.iter().position(|&x| x == ridx) {
                    res_multi[ri] = b_multi[pos];
                }
            }

            let flat_idx = if result_shape.is_empty() {
                0
            } else {
                multi_to_flat(&res_multi, &result_shape)
            };
            result_data[flat_idx] += product;
        }
    }

    let result_tensor = if result_shape.is_empty() {
        ArrayD::from_shape_vec(IxDyn(&[]), result_data).unwrap()
    } else {
        ArrayD::from_shape_vec(IxDyn(&result_shape), result_data).unwrap()
    };

    (result_tensor, result_indices)
}

/// Naive contraction of a tensor network with an input state.
///
/// 1. Convert state to a tensor with shape (d0, d1, ..., d_{n-1}) and indices [0, 1, ..., n-1]
/// 2. For each gate tensor, contract with the current result
/// 3. Permute the result to match tn.code.iy order
/// 4. Flatten to Array1
fn naive_contract(tn: &TensorNetwork, state: &State) -> Array1<Complex64> {
    let n = state.dims.len();

    // Convert state to a multi-dimensional tensor with initial indices 0..n-1
    let state_shape: Vec<usize> = state.dims.clone();
    let state_tensor = ArrayD::from_shape_vec(
        IxDyn(&state_shape),
        state.data.to_vec(),
    ).unwrap();

    let mut current_tensor = state_tensor;
    let mut current_indices: Vec<usize> = (0..n).collect();

    // Collect all indices that appear in the output or in future gate tensors
    // to determine which shared indices should be contracted at each step.
    let output_set: std::collections::HashSet<usize> = tn.code.iy.iter().copied().collect();

    // Contract each gate tensor sequentially
    for (i, gate_tensor) in tn.tensors.iter().enumerate() {
        let gate_indices = &tn.code.ixs[i];

        // Determine which indices are needed in the future (output or later gates)
        let mut future_needed: std::collections::HashSet<usize> = output_set.clone();
        for j in (i + 1)..tn.tensors.len() {
            for &idx in &tn.code.ixs[j] {
                future_needed.insert(idx);
            }
        }

        // Shared indices are those appearing in both current tensor and gate tensor
        // Only contract (sum over) shared indices that are NOT needed in the future
        let shared: Vec<usize> = current_indices
            .iter()
            .filter(|idx| gate_indices.contains(idx) && !future_needed.contains(idx))
            .copied()
            .collect();

        let (result, result_indices) = contract_tensors(
            &current_tensor,
            &current_indices,
            gate_tensor,
            gate_indices,
            &shared,
            &tn.size_dict,
        );

        current_tensor = result;
        current_indices = result_indices;
    }

    // Permute result to match tn.code.iy order
    let output_indices = &tn.code.iy;

    // Build permutation: for each position in iy, find its position in current_indices
    let perm: Vec<usize> = output_indices
        .iter()
        .map(|idx| {
            current_indices
                .iter()
                .position(|x| x == idx)
                .unwrap_or_else(|| panic!("Output index {} not found in current indices {:?}", idx, current_indices))
        })
        .collect();

    // Permute by building a new tensor
    let output_shape: Vec<usize> = output_indices
        .iter()
        .map(|idx| tn.size_dict[idx])
        .collect();
    let total: usize = output_shape.iter().product();
    let mut result_data = vec![Complex64::new(0.0, 0.0); total];

    let current_shape: Vec<usize> = current_indices
        .iter()
        .map(|idx| tn.size_dict[idx])
        .collect();
    let current_total: usize = current_shape.iter().product();

    for flat in 0..current_total {
        let multi = flat_to_multi(flat, &current_shape);
        let val = current_tensor[IxDyn(&multi)];
        if val == Complex64::new(0.0, 0.0) {
            continue;
        }

        // Apply permutation
        let mut out_multi = vec![0usize; output_indices.len()];
        for (out_pos, &src_pos) in perm.iter().enumerate() {
            out_multi[out_pos] = multi[src_pos];
        }

        let out_flat = multi_to_flat(&out_multi, &output_shape);
        result_data[out_flat] = val;
    }

    Array1::from_vec(result_data)
}

/// Assert that two state vectors are close element-wise.
fn assert_states_close(a: &Array1<Complex64>, b: &Array1<Complex64>) {
    assert_eq!(
        a.len(),
        b.len(),
        "State vectors have different lengths: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).norm();
        assert!(
            diff < ATOL,
            "States differ at index {}: got {:?}, expected {:?}, diff = {}",
            i,
            av,
            bv,
            diff
        );
    }
}

// ==================== Test Cases ====================

#[test]
fn test_integration_x_gate_single_qubit() {
    // X|0> = |1>
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(Gate::X, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_h_gate_single_qubit() {
    // H|0> = (|0> + |1>) / sqrt(2)
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_cnot_on_00() {
    // CNOT|00> = |00>
    let dims = vec![2, 2];
    let state = State::product_state(&dims, &[0, 0]);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![0],
            vec![true],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_cnot_on_01() {
    // CNOT|01> = |01>
    let dims = vec![2, 2];
    let state = State::product_state(&dims, &[0, 1]);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![0],
            vec![true],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_cnot_on_10() {
    // CNOT|10> = |11>
    let dims = vec![2, 2];
    let state = State::product_state(&dims, &[1, 0]);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![0],
            vec![true],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_cnot_on_11() {
    // CNOT|11> = |10>
    let dims = vec![2, 2];
    let state = State::product_state(&dims, &[1, 1]);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![0],
            vec![true],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_bell_state() {
    // H on qubit 0, then CNOT(0->1) on |00> gives Bell state (|00> + |11>) / sqrt(2)
    let dims = vec![2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_rz_diagonal_gate() {
    // Rz(pi/4) on |0> - diagonal gate
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let theta = std::f64::consts::FRAC_PI_4;
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::Rz(theta),
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_rz_on_superposition() {
    // Rz(pi/3) on H|0> = Rz on (|0> + |1>)/sqrt(2)
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let theta = std::f64::consts::FRAC_PI_3;
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::Rz(theta), vec![0], vec![], vec![]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_3_qubit_circuit() {
    // 3-qubit circuit: H on qubit 0, CNOT(0,1), Ry(pi/5) on qubit 2, CNOT(2,1)
    let dims = vec![2, 2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
            PositionedGate::new(
                Gate::Ry(std::f64::consts::FRAC_PI_4),
                vec![2],
                vec![],
                vec![],
            ),
            PositionedGate::new(Gate::X, vec![1], vec![2], vec![true]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_qutrit_cyclic_permutation() {
    // Cyclic permutation on a qutrit: |0> -> |1>, |1> -> |2>, |2> -> |0>
    let dims = vec![3];
    let state = State::zero_state(&dims);

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    // P|i> = |i+1 mod 3>
    // P = [[0,0,1],[1,0,0],[0,1,0]]
    let perm_matrix = ndarray::Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::Custom {
                matrix: perm_matrix,
                is_diagonal: false,
                label: "qutrit_cyclic_perm".to_string(),
            },
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_qutrit_on_state_2() {
    // Cyclic permutation on a qutrit starting from |2>: |2> -> |0>
    let dims = vec![3];
    let state = State::product_state(&dims, &[2]);

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let perm_matrix = ndarray::Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::Custom {
                matrix: perm_matrix,
                is_diagonal: false,
                label: "qutrit_cyclic_perm".to_string(),
            },
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_mixed_dimensions() {
    // Mixed dimensions: qubit (d=2) + qutrit (d=3) + qubit (d=2)
    // Total dimension: 2 * 3 * 2 = 12
    let dims = vec![2, 3, 2];
    let state = State::zero_state(&dims);

    // Apply H on qubit 0
    // Apply cyclic permutation on qutrit (site 1)
    // Apply X on qubit 2
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let perm_matrix = ndarray::Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(
                Gate::Custom {
                    matrix: perm_matrix,
                    is_diagonal: false,
                    label: "qutrit_perm".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            ),
            PositionedGate::new(Gate::X, vec![2], vec![], vec![]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_mixed_dimensions_with_diagonal() {
    // Mixed dimensions with a diagonal custom gate on qutrit
    let dims = vec![2, 3, 2];
    let state = State::zero_state(&dims);

    // Diagonal qutrit gate: phase gate with different phases for each level
    let qutrit_diag_matrix = ndarray::Array2::from_diag(&ndarray::Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
    ]));

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let perm_matrix = ndarray::Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let circuit = Circuit::new(
        dims.clone(),
        vec![
            // H on qubit 0
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            // Non-diagonal custom gate on qutrit (site 1)
            PositionedGate::new(
                Gate::Custom {
                    matrix: perm_matrix,
                    is_diagonal: false,
                    label: "qutrit_perm".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            ),
            // Diagonal custom gate on qutrit (site 1)
            PositionedGate::new(
                Gate::Custom {
                    matrix: qutrit_diag_matrix,
                    is_diagonal: true,
                    label: "qutrit_diagonal_phase".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            ),
            // X on qubit 2
            PositionedGate::new(Gate::X, vec![2], vec![], vec![]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_multiple_diagonal_gates() {
    // Multiple diagonal gates: Z, S, T on same qubit
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::Z, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::S, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::T, vec![0], vec![], vec![]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_h_gate_on_one_state() {
    // H|1> = (|0> - |1>) / sqrt(2)
    let dims = vec![2];
    let state = State::product_state(&dims, &[1]);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_ry_gate() {
    // Ry(pi/3) on |0>
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![gate(PositionedGate::new(
            Gate::Ry(std::f64::consts::FRAC_PI_3),
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_two_qubit_h_on_both() {
    // H on qubit 0 and H on qubit 1 of a 2-qubit circuit
    let dims = vec![2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::H, vec![1], vec![], vec![]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_3_qubit_ghz_like() {
    // Create GHZ-like state: H on qubit 0, CNOT(0,1), CNOT(0,2)
    let dims = vec![2, 2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
            PositionedGate::new(Gate::X, vec![2], vec![0], vec![true]),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);

    assert_states_close(&apply_result.data, &contract_result);
}

// === QFT integration tests ===

fn build_qft_circuit(n: usize) -> Circuit {
    use std::f64::consts::PI;
    let mut gates: Vec<PositionedGate> = Vec::new();
    for i in 0..n {
        gates.push(put(vec![i], Gate::H));
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1 << (j + 1)) as f64;
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }
    for i in 0..(n / 2) {
        gates.push(PositionedGate::new(Gate::SWAP, vec![i, n - 1 - i], vec![], vec![]));
    }
    Circuit::new(vec![2; n], gates).unwrap()
}

#[test]
fn test_integration_qft_zero_state() {
    // QFT|0⟩ = uniform superposition: (1/√N) Σ |j⟩
    let n = 3;
    let circuit = build_qft_circuit(n);
    let state = State::zero_state(&vec![2; n]);
    let result = apply(&circuit, &state);
    let total_dim = 1 << n;
    let expected_amp = 1.0 / (total_dim as f64).sqrt();
    for i in 0..total_dim {
        assert!((result.data[i].re - expected_amp).abs() < 1e-10);
        assert!(result.data[i].im.abs() < 1e-10);
    }
}

#[test]
fn test_integration_qft_basis_state() {
    // QFT|k⟩ = (1/√N) Σ_j e^(2πi jk/N) |j⟩
    use std::f64::consts::PI;
    let n = 3;
    let total_dim: usize = 1 << n;
    let circuit = build_qft_circuit(n);

    // Test |1⟩ = |001⟩
    let state = State::product_state(&vec![2; n], &[0, 0, 1]);
    let result = apply(&circuit, &state);
    let norm = 1.0 / (total_dim as f64).sqrt();
    for j in 0..total_dim {
        let expected = Complex64::from_polar(norm, 2.0 * PI * (j as f64) / (total_dim as f64));
        assert!((result.data[j] - expected).norm() < 1e-10,
            "Mismatch at j={}: got {:?}, expected {:?}", j, result.data[j], expected);
    }
}

#[test]
fn test_integration_qft_apply_vs_einsum() {
    // Verify apply() matches tensor network contraction for QFT
    let n = 4;
    let circuit = build_qft_circuit(n);
    let state = State::product_state(&vec![2; n], &[0, 1, 1, 0]);
    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = naive_contract(&tn, &state);
    assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_qft_norm_preservation() {
    // QFT should preserve norm for any input
    let n = 4;
    let circuit = build_qft_circuit(n);
    for k in 0..(1usize << n) {
        let mut levels = vec![0usize; n];
        for bit in 0..n {
            levels[n - 1 - bit] = (k >> bit) & 1;
        }
        let state = State::product_state(&vec![2; n], &levels);
        let result = apply(&circuit, &state);
        assert!((result.norm() - 1.0).abs() < 1e-10,
            "Norm not preserved for input k={}: got {}", k, result.norm());
    }
}
