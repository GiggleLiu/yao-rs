use crate::common;

use bitbasis::BitStr;
use num_complex::Complex64;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate, control, put};

// Helper to wrap PositionedGate in CircuitElement::Gate
fn gate(pg: PositionedGate) -> CircuitElement {
    CircuitElement::Gate(pg)
}
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;
use yao_rs::register::ArrayReg;

// ==================== Test Cases ====================

#[test]
fn test_integration_x_gate_single_qubit() {
    // X|0> = |1>
    let state = ArrayReg::zero_state(1);
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::X, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_h_gate_single_qubit() {
    // H|0> = (|0> + |1>) / sqrt(2)
    let state = ArrayReg::zero_state(1);
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_cnot_on_00() {
    // CNOT|00> = |00>
    let state = ArrayReg::product_state(BitStr::<2>::new(0b00));
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

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_cnot_on_01() {
    // CNOT|01> = |01>
    // Old: State::product_state(&[2, 2], &[0, 1]) => flat index = 0*2+1 = 1
    let state = ArrayReg::product_state(BitStr::<2>::new(1));
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

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_cnot_on_10() {
    // CNOT|10> = |11>
    // Old: State::product_state(&[2, 2], &[1, 0]) => flat index = 1*2+0 = 2
    let state = ArrayReg::product_state(BitStr::<2>::new(2));
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

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_cnot_on_11() {
    // CNOT|11> = |10>
    // Old: State::product_state(&[2, 2], &[1, 1]) => flat index = 1*2+1 = 3
    let state = ArrayReg::product_state(BitStr::<2>::new(3));
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

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_bell_state() {
    // H on qubit 0, then CNOT(0->1) on |00> gives Bell state (|00> + |11>) / sqrt(2)
    let state = ArrayReg::zero_state(2);
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_rz_diagonal_gate() {
    // Rz(pi/4) on |0> - diagonal gate
    let state = ArrayReg::zero_state(1);
    let theta = std::f64::consts::FRAC_PI_4;
    let circuit = Circuit::new(
        vec![2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_rz_on_superposition() {
    // Rz(pi/3) on H|0> = Rz on (|0> + |1>)/sqrt(2)
    let state = ArrayReg::zero_state(1);
    let theta = std::f64::consts::FRAC_PI_3;
    let circuit = Circuit::new(
        vec![2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_3_qubit_circuit() {
    // 3-qubit circuit: H on qubit 0, CNOT(0,1), Ry(pi/5) on qubit 2, CNOT(2,1)
    let state = ArrayReg::zero_state(3);
    let circuit = Circuit::new(
        vec![2, 2, 2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

// Deleted: test_integration_qutrit_cyclic_permutation (qudit d=3)
// Deleted: test_integration_qutrit_on_state_2 (qudit d=3)
// Deleted: test_integration_mixed_dimensions (qudit d=3)
// Deleted: test_integration_mixed_dimensions_with_diagonal (qudit d=3)

#[test]
fn test_integration_multiple_diagonal_gates() {
    // Multiple diagonal gates: Z, S, T on same qubit
    let state = ArrayReg::zero_state(1);
    let circuit = Circuit::new(
        vec![2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_h_gate_on_one_state() {
    // H|1> = (|0> - |1>) / sqrt(2)
    let state = ArrayReg::product_state(BitStr::<1>::new(1));
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_ry_gate() {
    // Ry(pi/3) on |0>
    let state = ArrayReg::zero_state(1);
    let circuit = Circuit::new(
        vec![2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_two_qubit_h_on_both() {
    // H on qubit 0 and H on qubit 1 of a 2-qubit circuit
    let state = ArrayReg::zero_state(2);
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::H, vec![1], vec![], vec![])),
        ],
    )
    .unwrap();

    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);

    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_3_qubit_ghz_like() {
    // Create GHZ-like state: H on qubit 0, CNOT(0,1), CNOT(0,2)
    let state = ArrayReg::zero_state(3);
    let circuit = Circuit::new(
        vec![2, 2, 2],
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

    common::assert_states_close(apply_result.state_vec(), &contract_result);
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
    let state = ArrayReg::zero_state(n);
    let result = apply(&circuit, &state);
    let total_dim = 1 << n;
    let expected_amp = 1.0 / (total_dim as f64).sqrt();
    for i in 0..total_dim {
        assert!((result.state_vec()[i].re - expected_amp).abs() < 1e-10);
        assert!(result.state_vec()[i].im.abs() < 1e-10);
    }
}

#[test]
fn test_integration_qft_basis_state() {
    // QFT|k⟩ = (1/√N) Σ_j e^(2πi jk/N) |j⟩
    use std::f64::consts::PI;
    let n = 3;
    let total_dim: usize = 1 << n;
    let circuit = build_qft_circuit(n);

    // Test |1⟩ = |001⟩ => flat index = 0*4+0*2+1 = 1
    let state = ArrayReg::product_state(BitStr::<3>::new(1));
    let result = apply(&circuit, &state);
    let norm = 1.0 / (total_dim as f64).sqrt();
    for j in 0..total_dim {
        let expected = Complex64::from_polar(norm, 2.0 * PI * (j as f64) / (total_dim as f64));
        assert!(
            (result.state_vec()[j] - expected).norm() < 1e-10,
            "Mismatch at j={}: got {:?}, expected {:?}",
            j,
            result.state_vec()[j],
            expected
        );
    }
}

#[test]
fn test_integration_qft_apply_vs_einsum() {
    // Verify apply() matches tensor network contraction for QFT
    let n = 4;
    let circuit = build_qft_circuit(n);
    // Old: State::product_state(&vec![2; 4], &[0, 1, 1, 0]) => flat index = 0*8+1*4+1*2+0 = 6
    let state = ArrayReg::product_state(BitStr::<4>::new(6));
    let apply_result = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let contract_result = common::naive_contract(&tn, &state);
    common::assert_states_close(apply_result.state_vec(), &contract_result);
}

#[test]
fn test_integration_qft_norm_preservation() {
    // QFT should preserve norm for any input
    let n = 4;
    let circuit = build_qft_circuit(n);
    for k in 0..(1usize << n) {
        let state = ArrayReg::product_state(BitStr::<4>::new(k as u64));
        let result = apply(&circuit, &state);
        assert!(
            (result.norm() - 1.0).abs() < 1e-10,
            "Norm not preserved for input k={}: got {}",
            k,
            result.norm()
        );
    }
}
