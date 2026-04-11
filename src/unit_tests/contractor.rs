use crate::apply::apply;
use crate::circuit::{Circuit, control, put};
use crate::contractor::contract;
use crate::einsum::{circuit_to_einsum_with_boundary, circuit_to_overlap};
use crate::gate::Gate;
use crate::register::ArrayReg;
use num_complex::Complex64;

fn assert_scalar_close(result: &ndarray::ArrayD<Complex64>, expected: Complex64) {
    let val = result.iter().next().unwrap();
    assert!(
        (val - expected).norm() < 1e-10,
        "expected {expected}, got {val}"
    );
}

/// Cross-validate: contract the TN state vector and compare against apply().
fn cross_validate(circuit: &Circuit) {
    let tn = circuit_to_einsum_with_boundary(circuit, &[]);
    let tn_result = contract(&tn);
    let apply_result = apply(circuit, &ArrayReg::zero_state(circuit.nbits));
    let state = apply_result.state_vec();
    // Flatten TN result and compare element-wise
    for (i, (tn_val, apply_val)) in tn_result.iter().zip(state.iter()).enumerate() {
        assert!(
            (tn_val - apply_val).norm() < 1e-10,
            "mismatch at index {i}: tn={tn_val}, apply={apply_val}"
        );
    }
}

#[test]
fn test_contract_identity() {
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn);
    assert_scalar_close(&result, Complex64::new(1.0, 0.0));
}

#[test]
fn test_contract_h_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_bell_state() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_overlap() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_overlap(&circuit);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_state_vector() {
    // H|0⟩ = (|0⟩+|1⟩)/√2 — use boundary with no pinned outputs
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    let s = 1.0 / 2.0_f64.sqrt();
    assert!((result[[0]] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((result[[1]] - Complex64::new(s, 0.0)).norm() < 1e-10);
}

#[test]
fn test_contract_two_qubit_state() {
    // Bell state (|00⟩+|11⟩)/√2 — use boundary with no pinned outputs
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    let s = 1.0 / 2.0_f64.sqrt();
    assert!((result[[0, 0]] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((result[[0, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 1]] - Complex64::new(s, 0.0)).norm() < 1e-10);
}

// --- Asymmetric gate tests (catch transpose bugs) ---

#[test]
fn test_contract_y_gate() {
    // Y|0⟩ = i|1⟩ — asymmetric matrix, catches column-major bugs
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Y)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_rx_gate() {
    // Rx(π/3)|0⟩ — asymmetric complex entries
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::Rx(std::f64::consts::FRAC_PI_3))],
    )
    .unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_ry_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Ry(1.23))]).unwrap();
    cross_validate(&circuit);
}

// --- Diagonal gate tests ---

#[test]
fn test_contract_z_gate() {
    // Z is diagonal — exercises the diagonal gate path in einsum
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::Z)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_t_gate() {
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::T)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_s_gate() {
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::S)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_phase_gate() {
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::H), put(vec![0], Gate::Phase(0.7))],
    )
    .unwrap();
    cross_validate(&circuit);
}

// --- Multi-gate / deeper circuits ---

#[test]
fn test_contract_qft_3qubit() {
    let circuit = crate::easybuild::qft_circuit(3);
    cross_validate(&circuit);
}

#[test]
fn test_contract_ghz_4qubit() {
    let n = 4;
    let mut elements = vec![put(vec![0], Gate::H)];
    for i in 0..n - 1 {
        elements.push(control(vec![i], vec![i + 1], Gate::X));
    }
    let circuit = Circuit::new(vec![2; n], elements).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_mixed_gates() {
    // Circuit with both diagonal and non-diagonal gates
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![0], Gate::T), // diagonal
            control(vec![0], vec![1], Gate::X),
            put(vec![1], Gate::S), // diagonal
            put(vec![1], Gate::Ry(0.5)),
        ],
    )
    .unwrap();
    cross_validate(&circuit);
}

// --- Layout correctness test ---

#[test]
fn test_contract_result_is_standard_layout() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    assert!(
        result.is_standard_layout(),
        "result must be row-major (C layout)"
    );
}
