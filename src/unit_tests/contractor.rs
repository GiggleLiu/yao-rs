use crate::circuit::{Circuit, control, put};
use crate::contractor::contract;
use crate::einsum::{circuit_to_einsum_with_boundary, circuit_to_overlap};
use crate::gate::Gate;
use num_complex::Complex64;

fn assert_scalar_close(result: &ndarray::ArrayD<Complex64>, expected: Complex64) {
    let val = result.iter().next().unwrap();
    assert!(
        (val - expected).norm() < 1e-10,
        "expected {expected}, got {val}"
    );
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
