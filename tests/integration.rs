mod common;

use num_complex::Complex64;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate, control, put};

// Helper to wrap PositionedGate in CircuitElement::Gate
fn gate(pg: PositionedGate) -> CircuitElement {
    CircuitElement::Gate(pg)
}
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;
use yao_rs::state::State;

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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_bell_state() {
    // H on qubit 0, then CNOT(0->1) on |00> gives Bell state (|00> + |11>) / sqrt(2)
    let dims = vec![2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(
                Gate::Rz(theta),
                vec![0],
                vec![],
                vec![],
            )),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_3_qubit_circuit() {
    // 3-qubit circuit: H on qubit 0, CNOT(0,1), Ry(pi/5) on qubit 2, CNOT(2,1)
    let dims = vec![2, 2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
            gate(PositionedGate::new(
                Gate::Ry(std::f64::consts::FRAC_PI_4),
                vec![2],
                vec![],
                vec![],
            )),
            gate(PositionedGate::new(Gate::X, vec![1], vec![2], vec![true])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(
                Gate::Custom {
                    matrix: perm_matrix,
                    is_diagonal: false,
                    label: "qutrit_perm".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            )),
            gate(PositionedGate::new(Gate::X, vec![2], vec![], vec![])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            // Non-diagonal custom gate on qutrit (site 1)
            gate(PositionedGate::new(
                Gate::Custom {
                    matrix: perm_matrix,
                    is_diagonal: false,
                    label: "qutrit_perm".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            )),
            // Diagonal custom gate on qutrit (site 1)
            gate(PositionedGate::new(
                Gate::Custom {
                    matrix: qutrit_diag_matrix,
                    is_diagonal: true,
                    label: "qutrit_diagonal_phase".to_string(),
                },
                vec![1],
                vec![],
                vec![],
            )),
            // X on qubit 2
            gate(PositionedGate::new(Gate::X, vec![2], vec![], vec![])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_multiple_diagonal_gates() {
    // Multiple diagonal gates: Z, S, T on same qubit
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::Z, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::S, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::T, vec![0], vec![], vec![])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
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
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_two_qubit_h_on_both() {
    // H on qubit 0 and H on qubit 1 of a 2-qubit circuit
    let dims = vec![2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::H, vec![1], vec![], vec![])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

#[test]
fn test_integration_3_qubit_ghz_like() {
    // Create GHZ-like state: H on qubit 0, CNOT(0,1), CNOT(0,2)
    let dims = vec![2, 2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
            gate(PositionedGate::new(Gate::X, vec![2], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(&apply_result.data, &contract_result);
}

// === QFT integration tests ===

fn build_qft_circuit(n: usize) -> Circuit {
    use std::f64::consts::PI;
    let mut elements: Vec<CircuitElement> = Vec::new();
    for i in 0..n {
        elements.push(put(vec![i], Gate::H));
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1 << (j + 1)) as f64;
            elements.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }
    for i in 0..(n / 2) {
        elements.push(gate(PositionedGate::new(
            Gate::SWAP,
            vec![i, n - 1 - i],
            vec![],
            vec![],
        )));
    }
    Circuit::new(vec![2; n], elements).unwrap()
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
        assert!(
            (result.data[j] - expected).norm() < 1e-10,
            "Mismatch at j={}: got {:?}, expected {:?}",
            j,
            result.data[j],
            expected
        );
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
    let contract_result = common::naive_contract(&tn, &state);
    common::assert_states_close(&apply_result.data, &contract_result);
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
        assert!(
            (result.norm() - 1.0).abs() < 1e-10,
            "Norm not preserved for input k={}: got {}",
            k,
            result.norm()
        );
    }
}
