use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use crate::ad::expect_grad;
use crate::circuit::{Circuit, control, put};
use crate::gate::Gate;
use crate::operator::{Op, OperatorPolynomial};
use crate::register::ArrayReg;

fn h_z(site: usize) -> OperatorPolynomial {
    OperatorPolynomial::single(site, Op::Z, Complex64::new(1.0, 0.0))
}

fn h_x(site: usize) -> OperatorPolynomial {
    OperatorPolynomial::single(site, Op::X, Complex64::new(1.0, 0.0))
}

#[test]
fn rx_on_zero_vs_z() {
    for &theta in &[0.0_f64, 0.3, 0.7, 1.57, 2.0, -0.5] {
        let c = Circuit::qubits(1, vec![put(vec![0], Gate::Rx(theta))]).unwrap();
        let psi0 = ArrayReg::zero_state(1);
        let (v, g) = expect_grad(&h_z(0), &c, &psi0);
        assert_abs_diff_eq!(v, theta.cos(), epsilon = 1e-10);
        assert_eq!(g.len(), 1);
        assert_abs_diff_eq!(g[0], -theta.sin(), epsilon = 1e-10);
    }
}

#[test]
fn ry_on_zero_vs_x() {
    for &theta in &[0.0_f64, 0.3, 0.7, 1.57, -0.2] {
        let c = Circuit::qubits(1, vec![put(vec![0], Gate::Ry(theta))]).unwrap();
        let psi0 = ArrayReg::zero_state(1);
        let (v, g) = expect_grad(&h_x(0), &c, &psi0);
        assert_abs_diff_eq!(v, theta.sin(), epsilon = 1e-10);
        assert_abs_diff_eq!(g[0], theta.cos(), epsilon = 1e-10);
    }
}

#[test]
fn rz_on_plus_vs_x() {
    for &theta in &[0.1_f64, 0.9, 1.3, -0.4] {
        let c = Circuit::qubits(
            1,
            vec![put(vec![0], Gate::H), put(vec![0], Gate::Rz(theta))],
        )
        .unwrap();
        let psi0 = ArrayReg::zero_state(1);
        let (v, g) = expect_grad(&h_x(0), &c, &psi0);
        assert_abs_diff_eq!(v, theta.cos(), epsilon = 1e-10);
        assert_abs_diff_eq!(g[0], -theta.sin(), epsilon = 1e-10);
    }
}

#[test]
fn phase_on_plus_vs_x() {
    for &theta in &[0.2_f64, 0.8, 1.4] {
        let c = Circuit::qubits(
            1,
            vec![put(vec![0], Gate::H), put(vec![0], Gate::Phase(theta))],
        )
        .unwrap();
        let psi0 = ArrayReg::zero_state(1);
        let (v, g) = expect_grad(&h_x(0), &c, &psi0);
        assert_abs_diff_eq!(v, theta.cos(), epsilon = 1e-10);
        assert_abs_diff_eq!(g[0], -theta.sin(), epsilon = 1e-10);
    }
}

#[test]
fn gradient_ordering_matches_circuit_parameters() {
    let c = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::Rx(0.1)),
            put(vec![1], Gate::Ry(0.2)),
            control(vec![0], vec![1], Gate::Rz(0.3)),
        ],
    )
    .unwrap();
    assert_eq!(c.parameters(), vec![0.1, 0.2, 0.3]);
    let h = OperatorPolynomial::single(1, Op::Z, Complex64::new(1.0, 0.0));
    let psi0 = ArrayReg::zero_state(2);
    let (_v, g) = expect_grad(&h, &c, &psi0);
    assert_eq!(g.len(), 3);
}

#[test]
#[should_panic(expected = "noise channel")]
fn rejects_channels() {
    use crate::circuit::channel;
    use crate::noise::NoiseChannel;

    let c = Circuit::qubits(
        1,
        vec![
            put(vec![0], Gate::Rx(0.3)),
            channel(vec![0], NoiseChannel::BitFlip { p: 0.01 }),
        ],
    )
    .unwrap();
    let psi0 = ArrayReg::zero_state(1);
    let _ = expect_grad(&h_z(0), &c, &psi0);
}

#[test]
fn finite_difference_cross_check() {
    let elements = vec![
        put(vec![0], Gate::H),
        put(vec![1], Gate::H),
        put(vec![0], Gate::Rx(0.31)),
        put(vec![1], Gate::Ry(-0.47)),
        control(vec![0], vec![1], Gate::Rz(0.22)),
        put(vec![0, 1], Gate::FSim(0.15, 0.37)),
        control(vec![1], vec![0], Gate::Phase(-0.63)),
    ];
    let circuit = Circuit::qubits(2, elements.clone()).unwrap();
    let h = &OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0))
        + &OperatorPolynomial::single(1, Op::X, Complex64::new(0.5, 0.0));
    let psi0 = ArrayReg::zero_state(2);
    let (_, grad_ad) = expect_grad(&h, &circuit, &psi0);
    assert_eq!(grad_ad.len(), circuit.num_params());

    let params = circuit.parameters();
    let h_step = 1e-5_f64;
    let mut grad_fd = vec![0.0; params.len()];
    for i in 0..params.len() {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[i] += h_step;
        minus[i] -= h_step;
        let mut c_plus = Circuit::qubits(2, elements.clone()).unwrap();
        c_plus.dispatch(&plus);
        let mut c_minus = Circuit::qubits(2, elements.clone()).unwrap();
        c_minus.dispatch(&minus);
        let (vp, _) = expect_grad(&h, &c_plus, &psi0);
        let (vm, _) = expect_grad(&h, &c_minus, &psi0);
        grad_fd[i] = (vp - vm) / (2.0 * h_step);
    }

    for i in 0..params.len() {
        assert_abs_diff_eq!(grad_ad[i], grad_fd[i], epsilon = 1e-6);
    }
}

#[test]
fn active_low_control_gradient_matches_independent_finite_difference() {
    use crate::{CircuitElement, PositionedGate, apply, expect_arrayreg};
    let circuit = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::Ry(0.61)),
            CircuitElement::Gate(PositionedGate::new(
                Gate::Rx(-0.37),
                vec![1],
                vec![0],
                vec![false],
            )),
            put(vec![0, 1], Gate::FSim(0.23, -0.17)),
            put(vec![1], Gate::Phase(0.44)),
        ],
    )
    .unwrap();
    let initial = ArrayReg::deterministic_state(2);
    let observable = &h_z(0) + &h_x(1);
    let (value, gradient) = expect_grad(&observable, &circuit, &initial);
    assert_abs_diff_eq!(
        value,
        expect_arrayreg(&apply(&circuit, &initial), &observable).re,
        epsilon = 1e-12
    );
    let params = circuit.parameters();
    for (i, &g) in gradient.iter().enumerate() {
        let mut shifted = circuit.clone();
        let mut p = params.clone();
        p[i] += 1e-6;
        shifted.dispatch(&p);
        let plus = expect_arrayreg(&apply(&shifted, &initial), &observable).re;
        p[i] -= 2e-6;
        shifted.dispatch(&p);
        let minus = expect_arrayreg(&apply(&shifted, &initial), &observable).re;
        assert_abs_diff_eq!(g, (plus - minus) / 2e-6, epsilon = 1e-8);
    }
}

#[test]
fn matches_yao_jl_reference_gradient() {
    // Generated by scripts/generate_ad_reference.jl using local Yao.jl
    // commit 31c7c1333b14b1e89123c511eff5742e7ac24edd.
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/data/ad_yao_reference.json")).unwrap();
    let circuit = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            put(vec![0], Gate::Rx(0.31)),
            put(vec![1], Gate::Ry(-0.47)),
            control(vec![0], vec![1], Gate::Rz(0.22)),
            crate::CircuitElement::Gate(crate::PositionedGate::new(
                Gate::Phase(-0.63),
                vec![0],
                vec![1],
                vec![false],
            )),
        ],
    )
    .unwrap();
    let observable = &h_z(0) + &OperatorPolynomial::single(1, Op::X, Complex64::new(0.5, 0.0));
    let (value, gradient) = expect_grad(&observable, &circuit, &ArrayReg::zero_state(2));
    assert_abs_diff_eq!(value, reference["value"].as_f64().unwrap(), epsilon = 1e-12);
    for ((parameter, derivative), (expected_parameter, expected_derivative)) in
        circuit.parameters().iter().zip(&gradient).zip(
            reference["parameters"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reference["gradient"].as_array().unwrap()),
        )
    {
        assert_abs_diff_eq!(
            *parameter,
            expected_parameter.as_f64().unwrap(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            *derivative,
            expected_derivative.as_f64().unwrap(),
            epsilon = 1e-12
        );
    }
    assert_eq!(
        gradient.len(),
        reference["gradient"].as_array().unwrap().len()
    );
}
