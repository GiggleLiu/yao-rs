mod common;

use std::collections::HashMap;

use ndarray::Array2;
use num_complex::Complex64;
use omeco::{GreedyMethod, contraction_complexity, optimize_code};

use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate};
use yao_rs::einsum::{circuit_to_einsum, circuit_to_expectation, circuit_to_overlap};
use yao_rs::gate::Gate;
use yao_rs::operator::{Op, OperatorPolynomial};
use yao_rs::{
    apply, circuit::control, circuit::put, circuit_to_einsum_with_boundary, easybuild::qft_circuit,
    state::State,
};

/// Helper to create a simple CircuitElement gate with no controls.
fn simple_gate(gate: Gate, target_locs: Vec<usize>) -> CircuitElement {
    CircuitElement::Gate(PositionedGate::new(gate, target_locs, vec![], vec![]))
}

/// Helper to create a controlled gate as a CircuitElement.
fn controlled_gate(
    gate: Gate,
    target_locs: Vec<usize>,
    control_locs: Vec<usize>,
    control_configs: Vec<bool>,
) -> CircuitElement {
    CircuitElement::Gate(PositionedGate::new(
        gate,
        target_locs,
        control_locs,
        control_configs,
    ))
}

// Helper function to wrap a PositionedGate in CircuitElement::Gate
fn gate(g: PositionedGate) -> CircuitElement {
    CircuitElement::Gate(g)
}

#[test]
fn test_single_h_gate_on_2_qubit_circuit() {
    // H gate on qubit 0 of a 2-qubit circuit
    // H is non-diagonal, so it allocates a new label
    // all_locs = [0], new_label = 2
    // tensor ixs = [2, 0] (new output label, then current input label)
    // output = [2, 1] (qubit 0 updated to label 2, qubit 1 stays at label 1)
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::H, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![2, 0]]);
    assert_eq!(tn.code.iy, vec![2, 1]);
    assert_eq!(tn.tensors.len(), 1);
}

#[test]
fn test_diagonal_z_gate() {
    // Z gate is diagonal, so no new labels are allocated
    // tensor ixs = [0] (just the current label of the target site)
    // output = [0, 1] (unchanged)
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::Z, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![0]]);
    assert_eq!(tn.code.iy, vec![0, 1]);
    assert_eq!(tn.tensors.len(), 1);
}

#[test]
fn test_cnot_gate() {
    // CNOT = X gate with control on qubit 0, target on qubit 1
    // Has controls, so not diagonal treatment
    // all_locs = [0, 1] (control_locs ++ target_locs)
    // new_labels = [2, 3]
    // tensor ixs = [2, 3, 0, 1]
    // output = [2, 3]
    let circuit = Circuit::new(
        vec![2, 2],
        vec![controlled_gate(Gate::X, vec![1], vec![0], vec![true])],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![2, 3, 0, 1]]);
    assert_eq!(tn.code.iy, vec![2, 3]);
    assert_eq!(tn.tensors.len(), 1);
}

#[test]
fn test_h_then_cnot() {
    // H on qubit 0 (non-diagonal): allocates label 2 for qubit 0
    //   tensor ixs = [2, 0], current_labels = [2, 1]
    // CNOT (control 0, target 1): all_locs = [0, 1]
    //   allocates labels 3, 4 for qubits 0 and 1
    //   tensor ixs = [3, 4, 2, 1]
    //   current_labels = [3, 4]
    // output = [3, 4]
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            simple_gate(Gate::H, vec![0]),
            controlled_gate(Gate::X, vec![1], vec![0], vec![true]),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![2, 0], vec![3, 4, 2, 1]]);
    assert_eq!(tn.code.iy, vec![3, 4]);
    assert_eq!(tn.tensors.len(), 2);
}

#[test]
fn test_empty_circuit() {
    // No gates: ixs is empty, output is initial labels [0, 1]
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, Vec::<Vec<usize>>::new());
    assert_eq!(tn.code.iy, vec![0, 1]);
    assert_eq!(tn.tensors.len(), 0);
}

#[test]
fn test_rz_then_h_on_same_qubit() {
    // Rz is diagonal: uses current label, no new allocation
    //   tensor ixs = [0], current_labels = [0, 1]
    // H is non-diagonal: allocates new label 2
    //   tensor ixs = [2, 0], current_labels = [2, 1]
    // output = [2, 1]
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            simple_gate(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]),
            simple_gate(Gate::H, vec![0]),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![0], vec![2, 0]]);
    assert_eq!(tn.code.iy, vec![2, 1]);
    assert_eq!(tn.tensors.len(), 2);
}

#[test]
fn test_size_dict_qubit_circuit() {
    // 2-qubit circuit with H then CNOT
    // Initial labels: 0 -> dim 2, 1 -> dim 2
    // H on qubit 0: new label 2 -> dim 2
    // CNOT: new labels 3 -> dim 2, 4 -> dim 2
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            simple_gate(Gate::H, vec![0]),
            controlled_gate(Gate::X, vec![1], vec![0], vec![true]),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    let mut expected_size_dict = HashMap::new();
    expected_size_dict.insert(0, 2);
    expected_size_dict.insert(1, 2);
    expected_size_dict.insert(2, 2);
    expected_size_dict.insert(3, 2);
    expected_size_dict.insert(4, 2);

    assert_eq!(tn.size_dict, expected_size_dict);
}

#[test]
fn test_size_dict_mixed_dimension_circuit() {
    // Mixed dimension circuit: qubit (d=2) and qutrit (d=3)
    // Use a custom gate for the qutrit site
    let qutrit_gate_matrix = Array2::from_diag(&ndarray::Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
    ]));
    let qutrit_diag_gate = Gate::Custom {
        matrix: qutrit_gate_matrix,
        is_diagonal: true,
        label: "qutrit_diagonal_phase".to_string(),
    };

    // Non-diagonal custom gate for qutrit
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let qutrit_non_diag_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![zero, one, zero, zero, zero, one, one, zero, zero],
    )
    .unwrap();
    let qutrit_non_diag_gate = Gate::Custom {
        matrix: qutrit_non_diag_matrix,
        is_diagonal: false,
        label: "qutrit_permutation".to_string(),
    };

    // Circuit: site 0 is qubit (d=2), site 1 is qutrit (d=3)
    // Gate 1: diagonal gate on site 1 (no new label)
    // Gate 2: non-diagonal gate on site 1 (new label allocated)
    let circuit = Circuit::new(
        vec![2, 3],
        vec![
            CircuitElement::Gate(PositionedGate::new(
                qutrit_diag_gate,
                vec![1],
                vec![],
                vec![],
            )),
            CircuitElement::Gate(PositionedGate::new(
                qutrit_non_diag_gate,
                vec![1],
                vec![],
                vec![],
            )),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    // Initial labels: 0 -> dim 2, 1 -> dim 3
    // Diagonal gate on site 1: ixs = [1], no new label
    // Non-diagonal gate on site 1: new label 2 -> dim 3, ixs = [2, 1]
    assert_eq!(tn.code.ixs, vec![vec![1], vec![2, 1]]);
    assert_eq!(tn.code.iy, vec![0, 2]);

    let mut expected_size_dict = HashMap::new();
    expected_size_dict.insert(0, 2);
    expected_size_dict.insert(1, 3);
    expected_size_dict.insert(2, 3);

    assert_eq!(tn.size_dict, expected_size_dict);
}

#[test]
fn test_diagonal_s_gate() {
    // S gate is diagonal, similar to Z
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::S, vec![1])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    // S on qubit 1: diagonal, ixs = [1], output = [0, 1]
    assert_eq!(tn.code.ixs, vec![vec![1]]);
    assert_eq!(tn.code.iy, vec![0, 1]);
}

#[test]
fn test_diagonal_t_gate() {
    // T gate is diagonal
    let circuit = Circuit::new(vec![2], vec![simple_gate(Gate::T, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![0]]);
    assert_eq!(tn.code.iy, vec![0]);
}

#[test]
fn test_multiple_non_diagonal_gates() {
    // H on qubit 0, then H on qubit 1 of a 2-qubit circuit
    // H on qubit 0: new label 2, ixs = [2, 0], current = [2, 1]
    // H on qubit 1: new label 3, ixs = [3, 1], current = [2, 3]
    // output = [2, 3]
    let circuit = Circuit::new(
        vec![2, 2],
        vec![simple_gate(Gate::H, vec![0]), simple_gate(Gate::H, vec![1])],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![2, 0], vec![3, 1]]);
    assert_eq!(tn.code.iy, vec![2, 3]);
}

#[test]
fn test_x_gate_non_diagonal() {
    // X gate is non-diagonal
    let circuit = Circuit::new(vec![2], vec![simple_gate(Gate::X, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![1, 0]]);
    assert_eq!(tn.code.iy, vec![1]);
}

#[test]
fn test_tensor_shapes_diagonal() {
    // Diagonal Z gate on qubit 0: tensor shape should be (2,)
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::Z, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.tensors[0].shape(), &[2]);
}

#[test]
fn test_tensor_shapes_non_diagonal() {
    // H gate on qubit 0: tensor shape should be (2, 2) = (out, in)
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::H, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.tensors[0].shape(), &[2, 2]);
}

#[test]
fn test_tensor_shapes_cnot() {
    // CNOT: 2 sites involved, shape = (2, 2, 2, 2) = (ctrl_out, tgt_out, ctrl_in, tgt_in)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![controlled_gate(Gate::X, vec![1], vec![0], vec![true])],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.tensors[0].shape(), &[2, 2, 2, 2]);
}

#[test]
fn test_single_qubit_circuit() {
    // Single qubit with X gate
    let circuit = Circuit::new(vec![2], vec![simple_gate(Gate::X, vec![0])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![1, 0]]);
    assert_eq!(tn.code.iy, vec![1]);
    assert_eq!(tn.size_dict.get(&0), Some(&2));
    assert_eq!(tn.size_dict.get(&1), Some(&2));
}

#[test]
fn test_three_qubit_circuit_with_mixed_gates() {
    // 3-qubit circuit: Z on qubit 0 (diagonal), H on qubit 1, CNOT(1,2)
    // Z on qubit 0: diagonal, ixs = [0], current = [0, 1, 2]
    // H on qubit 1: non-diagonal, new label 3, ixs = [3, 1], current = [0, 3, 2]
    // CNOT(1,2): control=1, target=2, all_locs=[1,2], new labels 4,5
    //   ixs = [4, 5, 3, 2], current = [0, 4, 5]
    // output = [0, 4, 5]
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            simple_gate(Gate::Z, vec![0]),
            simple_gate(Gate::H, vec![1]),
            controlled_gate(Gate::X, vec![2], vec![1], vec![true]),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![0], vec![3, 1], vec![4, 5, 3, 2]]);
    assert_eq!(tn.code.iy, vec![0, 4, 5]);
}

#[test]
fn test_eincode_validity() {
    // All output indices should appear in at least one input
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            simple_gate(Gate::H, vec![0]),
            controlled_gate(Gate::X, vec![1], vec![0], vec![true]),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert!(tn.code.is_valid());
}

#[test]
fn test_empty_single_site_circuit() {
    // Empty circuit with 1 site
    let circuit = Circuit::new(vec![2], vec![]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, Vec::<Vec<usize>>::new());
    assert_eq!(tn.code.iy, vec![0]);
    assert_eq!(tn.size_dict.len(), 1);
    assert_eq!(tn.size_dict.get(&0), Some(&2));
}

#[test]
fn test_swap_gate() {
    // SWAP acts on 2 qubits, is non-diagonal
    // all_locs = target_locs = [0, 1], new labels = [2, 3]
    // ixs = [2, 3, 0, 1]
    // output = [2, 3]
    let circuit = Circuit::new(vec![2, 2], vec![simple_gate(Gate::SWAP, vec![0, 1])]).unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![2, 3, 0, 1]]);
    assert_eq!(tn.code.iy, vec![2, 3]);
}

#[test]
fn test_diagonal_gate_preserves_labels_across_multiple_uses() {
    // Two diagonal gates on the same qubit: labels should not change
    // Z on qubit 0: ixs = [0]
    // S on qubit 0: ixs = [0]
    // output = [0]
    let circuit = Circuit::new(
        vec![2],
        vec![simple_gate(Gate::Z, vec![0]), simple_gate(Gate::S, vec![0])],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);

    assert_eq!(tn.code.ixs, vec![vec![0], vec![0]]);
    assert_eq!(tn.code.iy, vec![0]);
    // size_dict should only have initial label
    assert_eq!(tn.size_dict.len(), 1);
}

#[test]
fn test_circuit_to_overlap_simple() {
    // |0> -> H -> measure <0|
    // This computes <0|H|0> = 1/sqrt(2)
    let circuit = Circuit::new(vec![2], vec![simple_gate(Gate::H, vec![0])]).unwrap();

    let tn = circuit_to_overlap(&circuit);
    // Should have: initial |0>, H gate tensor, final <0|
    // 1 initial boundary tensor + 1 gate tensor + 1 final boundary tensor = 3
    assert_eq!(tn.tensors.len(), 3);
    // Output should be empty (scalar result since all qubits pinned)
    assert!(tn.code.iy.is_empty());
}

#[test]
fn test_circuit_to_expectation_z() {
    // <0|U^dag Z(0) U|0> where U = H
    // Expected structure: <0|H^dag Z H|0>
    // H is self-adjoint (H^dag = H), so this is <0|H Z H|0> = <0|X|0> = 0
    let circuit = Circuit::new(vec![2], vec![simple_gate(Gate::H, vec![0])]).unwrap();

    let z_op = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));
    let tn = circuit_to_expectation(&circuit, &z_op);

    // Should have tensors for the expectation structure:
    // |0> boundary + U gates + operator tensor + U^dag gates + <0| boundary
    // For 1 qubit with H: |0> + H + Z + H^dag + <0| = 5 tensors
    assert!(
        tn.tensors.len() >= 3,
        "Expected at least 3 tensors, got {}",
        tn.tensors.len()
    );

    // Output should be empty (scalar result)
    assert!(
        tn.code.iy.is_empty(),
        "Expected empty output for scalar expectation value"
    );
}

// ==================== Boundary tests (from test_boundary.rs) ====================

#[test]
fn test_boundary_all_pinned_identity() {
    // Empty circuit (identity): <00|I|00> = 1
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = common::contract_tn(&tn);
    assert_eq!(result.ndim(), 0);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(1.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_x_gate_amplitude() {
    // X on qubit 0: <0|X|0> = 0
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::X)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = common::contract_tn(&tn);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(0.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_h_gate_amplitude() {
    // H on qubit 0: <0|H|0> = 1/sqrt(2)
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = common::contract_tn(&tn);
    let expected = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    assert!(common::approx_eq(result[[]], expected, 1e-10));
}

#[test]
fn test_boundary_cnot_amplitude() {
    // CNOT on |00>: output is |00>, so <00|CNOT|00> = 1
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::X)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = common::contract_tn(&tn);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(1.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_matches_apply() {
    // H on qubit 0, CNOT(0->1): Bell state
    // <00|C|00> = 1/sqrt(2)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2]);
    let result_state = apply(&circuit, &state);

    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = common::contract_tn(&tn);

    // <00|Bell> = amplitude of |00> = 1/sqrt(2)
    let expected = result_state.data[0]; // first element of flat state
    assert!(common::approx_eq(result[[]], expected, 1e-10));
}

#[test]
fn test_boundary_open_legs() {
    // H on qubit 0 only, no pinning -> full state vector
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();

    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = common::contract_tn(&tn);

    let state = State::zero_state(&[2, 2]);
    let expected = apply(&circuit, &state);

    // result shape is [2, 2], expected.data is flat [4]
    assert_eq!(result.len(), expected.data.len());
    for flat_idx in 0..4 {
        let i = flat_idx / 2;
        let j = flat_idx % 2;
        assert!(
            common::approx_eq(result[[i, j]], expected.data[flat_idx], 1e-10),
            "Mismatch at flat_idx {}: got {:?}, expected {:?}",
            flat_idx,
            result[[i, j]],
            expected.data[flat_idx]
        );
    }
}

#[test]
fn test_boundary_partial_pinning() {
    // H on qubit 0, CNOT(0->1): Bell state
    // Pin qubit 0 only -> result over qubit 1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();

    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = common::contract_tn(&tn);

    // Bell = (|00> + |11>)/sqrt(2)
    // Pin qubit 0 to |0>: project -> 1/sqrt(2) * |0>_1
    assert_eq!(result.shape(), &[2]);
    assert!(common::approx_eq(
        result[[0]],
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        1e-10
    ));
    assert!(common::approx_eq(
        result[[1]],
        Complex64::new(0.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_qft_amplitude() {
    // QFT on |0>: <0|QFT|0> = 1/sqrt(2^n)
    let n = 3;
    let circuit = qft_circuit(n);
    let pinned: Vec<usize> = (0..n).collect();
    let tn = circuit_to_einsum_with_boundary(&circuit, &pinned);
    let result = common::contract_tn(&tn);

    let expected = Complex64::new(1.0 / (2.0_f64.powi(n as i32)).sqrt(), 0.0);
    assert!(
        common::approx_eq(result[[]], expected, 1e-10),
        "Got {:?}, expected {:?}",
        result[[]],
        expected
    );
}

#[test]
fn test_boundary_diagonal_gate() {
    // Z gate (diagonal) on |0>: Z|0> = |0>, so <0|Z|0> = 1
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Z)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = common::contract_tn(&tn);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(1.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_phase_gate() {
    // Phase(pi) ~ Z on |0>: <0|Z|0> = 1
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::Phase(std::f64::consts::PI))],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = common::contract_tn(&tn);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(1.0, 0.0),
        1e-10
    ));
}

#[test]
fn test_boundary_controlled_phase() {
    // Controlled-Phase(pi/2) on |00>: no effect since control is |0>
    // <00|C-P(pi/2)|00> = 1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![control(
            vec![0],
            vec![1],
            Gate::Phase(std::f64::consts::FRAC_PI_2),
        )],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = common::contract_tn(&tn);
    assert!(common::approx_eq(
        result[[]],
        Complex64::new(1.0, 0.0),
        1e-10
    ));
}

// ==================== Omeco optimization tests (from test_omeco.rs) ====================

#[test]
fn test_optimize_bell_circuit() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    let complexity = contraction_complexity(&nested, &tn.size_dict, &tn.code.ixs);
    assert!(complexity.tc > 0.0);
}

#[test]
fn test_optimize_ghz_circuit() {
    // 5-qubit GHZ preparation: H on 0, then CNOT chain
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    assert!(nested.is_binary());
}

#[test]
fn test_optimize_larger_circuit() {
    // 10-qubit circuit with H layer, CNOT chain, Rz layer
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
}

#[test]
fn test_single_gate_circuit_produces_valid_optimization() {
    // A single H gate on 1 qubit: produces 1 tensor, optimizer should return a leaf
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    // Single tensor produces a leaf in the contraction tree
    assert!(nested.is_leaf());
}

#[test]
fn test_single_cnot_gate_produces_valid_optimization() {
    // A single CNOT gate on 2 qubits: produces 1 tensor, optimizer should return a leaf
    let circuit = Circuit::new(
        vec![2, 2],
        vec![gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![0],
            vec![true],
        ))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    assert!(nested.is_leaf());
}

#[test]
fn test_eincode_is_always_valid_single_gate() {
    // Single H gate
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_bell_circuit() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_ghz_circuit() {
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_larger_circuit() {
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_optimize_complexity_ghz_circuit() {
    // Verify contraction complexity is computable for the GHZ circuit
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();
    let complexity = contraction_complexity(&optimized, &tn.size_dict, &tn.code.ixs);

    assert!(complexity.tc > 0.0);
    assert!(complexity.sc > 0.0);
}

#[test]
fn test_optimize_complexity_larger_circuit() {
    // Verify contraction complexity for the 10-qubit circuit
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();
    let complexity = contraction_complexity(&optimized, &tn.size_dict, &tn.code.ixs);

    assert!(complexity.tc > 0.0);
    assert!(complexity.sc > 0.0);
}

#[test]
fn test_single_diagonal_gate_valid_eincode() {
    // Single diagonal gate (Z) produces a valid EinCode
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::Z, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
    assert!(optimized.unwrap().is_leaf());
}

#[test]
fn test_single_rz_gate_valid_eincode() {
    // Single Rz gate (diagonal parametric) produces a valid EinCode
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(
            Gate::Rz(1.0),
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
    assert!(optimized.unwrap().is_leaf());
}

// ============================================================
// Ground truth validation against Yao.jl
// ============================================================

#[test]
fn test_einsum_ground_truth() {
    let data = common::load_einsum_data();
    let mut tested = 0;
    let mut skipped = 0;
    for case in &data.cases {
        // Skip cases that would be too large for naive contraction
        let total_dim: usize = case.dims.iter().product();
        if total_dim > 256 {
            skipped += 1;
            continue;
        }
        let circuit = common::circuit_from_case(case);
        let input_state = State::zero_state(&case.dims);
        let tn = circuit_to_einsum(&circuit);
        let result = common::naive_contract(&tn, &input_state);
        let expected = common::state_from_json(&case.output_state_re, &case.output_state_im);
        common::assert_states_close(&result, &expected);
        tested += 1;
    }
    assert!(
        tested > 0,
        "Expected to test at least some einsum cases (tested: {}, skipped: {})",
        tested,
        skipped
    );
}
