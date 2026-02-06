#![cfg(feature = "torch")]

use yao_rs::circuit::{Circuit, control, put};
use yao_rs::einsum::circuit_to_einsum_with_boundary;
use yao_rs::gate::Gate;
use yao_rs::torch_contractor::{Device, contract};

#[test]
fn test_contract_identity() {
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn, Device::Cpu);

    // Scalar result: ⟨00|I|00⟩ = 1
    let re = f64::try_from(result.real()).unwrap();
    let im = f64::try_from(result.imag()).unwrap();
    assert!((re - 1.0).abs() < 1e-10);
    assert!(im.abs() < 1e-10);
}

#[test]
fn test_contract_h_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract(&tn, Device::Cpu);

    // ⟨0|H|0⟩ = 1/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    let re = f64::try_from(result.real()).unwrap();
    let im = f64::try_from(result.imag()).unwrap();
    assert!((re - expected).abs() < 1e-10);
    assert!(im.abs() < 1e-10);
}

#[test]
fn test_contract_bell_state() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn, Device::Cpu);

    // ⟨00|Bell⟩ = 1/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    let re = f64::try_from(result.real()).unwrap();
    let im = f64::try_from(result.imag()).unwrap();
    assert!((re - expected).abs() < 1e-10);
    assert!(im.abs() < 1e-10);
}
