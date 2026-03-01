use num_complex::Complex64;
use yao_rs::circuit::{Circuit, put};
use yao_rs::einsum::circuit_to_einsum_dm;
use yao_rs::gate::Gate;

mod common;
use common::contract_tn_dm;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_dm_single_hadamard() {
    // H|0⟩ = (|0⟩+|1⟩)/sqrt(2)
    // rho = |psi><psi| = [[0.5, 0.5], [0.5, 0.5]]
    let elements = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], elements).unwrap();

    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    // Expected: 2x2 density matrix
    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[0, 1]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.5).abs() < 1e-10);

    // All imaginary parts should be 0
    assert!(rho[[0, 0]].im.abs() < 1e-10);
    assert!(rho[[0, 1]].im.abs() < 1e-10);
    assert!(rho[[1, 0]].im.abs() < 1e-10);
    assert!(rho[[1, 1]].im.abs() < 1e-10);
}

#[test]
fn test_dm_identity() {
    // No gates: rho = |0><0| = [[1, 0], [0, 0]]
    let circuit = Circuit::new(vec![2], vec![]).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]] - c(1.0, 0.0)).norm() < 1e-10);
    assert!((rho[[0, 1]] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((rho[[1, 0]] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((rho[[1, 1]] - c(0.0, 0.0)).norm() < 1e-10);
}
