use super::*;
use crate::Gate;
use crate::{Op, OperatorPolynomial, control, expect_grad, put};

fn state(n: usize, shift: f64) -> ArrayReg {
    ArrayReg::from_vec(
        n,
        (0..1usize << n)
            .map(|i| C::new((i as f64 + shift).cos(), (0.3 * i as f64 - shift).sin()))
            .collect(),
    )
}
fn pairing(a: &ArrayReg, b: &ArrayReg) -> f64 {
    a.state
        .iter()
        .zip(&b.state)
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} != {b}");
}
fn circuit() -> DifferentiableCircuit {
    let mut low = control(vec![2], vec![0, 1], Gate::FSim(0.2, -0.4));
    if let CircuitElement::Gate(pg) = &mut low {
        pg.control_configs[0] = false;
    }
    DifferentiableCircuit::from_circuit(
        Circuit::qubits(
            3,
            vec![
                put(vec![0], Gate::H),
                put(vec![1], Gate::Rx(0.7)),
                control(vec![0], vec![2], Gate::Ry(-0.2)),
                low,
                put(vec![0], Gate::Phase(0.3)),
                put(vec![2], Gate::Rz(-0.1)),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn arbitrary_cotangent_matches_all_parameter_and_complex_input_differences() {
    let c = circuit();
    let p = c.template().parameters().to_vec();
    let x = state(3, 0.4);
    let bar = state(3, -0.7);
    let vjp = c.vjp(&p, &x, &bar).unwrap();
    for eps in [1e-4, 1e-5, 1e-6] {
        for i in 0..p.len() {
            let mut plus = p.clone();
            let mut minus = p.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (pairing(&bar, &c.forward(&plus, &x).unwrap())
                - pairing(&bar, &c.forward(&minus, &x).unwrap()))
                / (2. * eps);
            close(vjp.parameters[i], fd, 2e-8);
        }
        for i in 0..x.state.len() {
            for imaginary in [false, true] {
                let delta = if imaginary {
                    C::new(0., eps)
                } else {
                    C::new(eps, 0.)
                };
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus.state[i] += delta;
                minus.state[i] -= delta;
                let fd = (pairing(&bar, &c.forward(&p, &plus).unwrap())
                    - pairing(&bar, &c.forward(&p, &minus).unwrap()))
                    / (2. * eps);
                close(
                    if imaginary {
                        vjp.input.state[i].im
                    } else {
                        vjp.input.state[i].re
                    },
                    fd,
                    2e-8,
                );
            }
        }
    }
}

#[test]
fn jvp_vjp_duality_and_directional_finite_difference() {
    let c = circuit();
    let p = c.template().parameters();
    let x = state(3, 0.2);
    let dx = state(3, -0.2);
    let bar = state(3, 0.9);
    let dp = (0..p.len())
        .map(|i| (i as f64 + 0.1).sin())
        .collect::<Vec<_>>();
    let (out, dy) = c.jvp(p, &x, &dp, &dx).unwrap();
    let v = c.vjp(p, &x, &bar).unwrap();
    close(
        pairing(&bar, &dy),
        pairing(&v.input, &dx)
            + v.parameters
                .iter()
                .zip(&dp)
                .map(|(a, b)| a * b)
                .sum::<f64>(),
        1e-11,
    );
    for (a, b) in out.state.iter().zip(c.forward(p, &x).unwrap().state) {
        close((*a - b).norm(), 0., 1e-13);
    }
    for eps in [1e-4, 1e-5, 1e-6] {
        let shifted = |s: f64| {
            let ps = p
                .iter()
                .zip(&dp)
                .map(|(a, b)| a + s * eps * b)
                .collect::<Vec<_>>();
            let mut xs = x.clone();
            for (a, b) in xs.state.iter_mut().zip(&dx.state) {
                *a += s * eps * b;
            }
            c.forward(&ps, &xs).unwrap()
        };
        let plus = shifted(1.);
        let minus = shifted(-1.);
        for ((a, b), d) in plus.state.iter().zip(&minus.state).zip(&dy.state) {
            close(((a - b) / (2. * eps) - d).norm(), 0., 3e-8);
        }
    }
}

#[test]
fn shared_time_and_couplings_at_zero_and_nonzero() {
    use crate::hamiltonian::{Boundary, ProductFormula, ising};
    for t in [0., 0.6] {
        let bound = ising(3, -0.7, 0.4, Boundary::Open)
            .unwrap()
            .evolve(t, 3, ProductFormula::Suzuki2)
            .unwrap();
        let c = DifferentiableCircuit::new(bound).unwrap();
        let x = state(3, 0.4);
        let bar = state(3, 0.8);
        let p = c.template().parameters();
        let v = c.vjp(p, &x, &bar).unwrap();
        assert_eq!(v.parameters.len(), 3);
        for eps in [1e-4, 1e-5, 1e-6] {
            for i in 0..p.len() {
                let mut a = p.to_vec();
                let mut b = p.to_vec();
                a[i] += eps;
                b[i] -= eps;
                close(
                    v.parameters[i],
                    (pairing(&bar, &c.forward(&a, &x).unwrap())
                        - pairing(&bar, &c.forward(&b, &x).unwrap()))
                        / (2. * eps),
                    3e-7,
                );
            }
        }
    }
}

#[test]
fn hermitian_expectation_uses_twice_h_psi_seed() {
    let c = circuit();
    let x = state(3, 0.3);
    let p = c.template().parameters();
    let y = c.forward(p, &x).unwrap();
    let mut bar = crate::apply(
        &Circuit::qubits(3, vec![put(vec![1], Gate::Z)]).unwrap(),
        &y,
    );
    for z in &mut bar.state {
        *z *= 2.;
    }
    let v = c.vjp(p, &x, &bar).unwrap();
    let (_, g) = expect_grad(
        &OperatorPolynomial::single(1, Op::Z, 1.0.into()),
        c.template().circuit(),
        &x,
    );
    for (a, b) in v.parameters.iter().zip(g) {
        close(*a, b, 1e-12);
    }
}

#[test]
fn joint_loss_gradient_preserves_global_phase_and_shared_squared_parameters() {
    use crate::{OperatorString, hamiltonian::pauli_rotation};
    let bound = pauli_rotation(1, &OperatorString::new(vec![]), 0.3).unwrap();
    let c = DifferentiableCircuit::new(bound).unwrap();
    let x = ArrayReg::zero_state(1);
    let (value, g) = c
        .value_and_grad(&[0.3], &x, |y| {
            Ok((
                y.state[0].im,
                ArrayReg::from_vec(1, vec![C::new(0., 1.), C::new(0., 0.)]),
            ))
        })
        .unwrap();
    close(value, -0.15_f64.sin(), 1e-12);
    close(g.parameters[0], -0.5 * 0.15_f64.cos(), 1e-12);
    assert!(
        c.value_and_grad(&[0.3], &x, |_| Ok((f64::NAN, x.clone())))
            .is_err()
    );
    let bound = BoundCircuit::new(
        Circuit::qubits(
            1,
            vec![put(vec![0], Gate::Ry(0.)), put(vec![0], Gate::Ry(0.))],
        )
        .unwrap(),
        vec![0.4, 9.],
        vec![
            ParameterBinding::Product {
                left: 0,
                right: 0,
                scale: 0.7,
            },
            ParameterBinding::Scaled {
                index: 0,
                scale: -0.2,
            },
        ],
    )
    .unwrap();
    let c = DifferentiableCircuit::new(bound).unwrap();
    let g = c.vjp(&[0.4, 9.], &x, &x).unwrap();
    close(
        g.parameters[0],
        -0.5 * (0.5_f64 * (0.7 * 0.4 * 0.4 - 0.2 * 0.4)).sin() * (1.4 * 0.4 - 0.2),
        1e-12,
    );
    assert_eq!(g.parameters[1], 0.);
}

#[test]
fn fixed_only_empty_and_unitary_custom_circuits() {
    for gates in [
        vec![],
        vec![put(vec![0], Gate::H)],
        vec![put(
            vec![0],
            Gate::Custom {
                matrix: Gate::Y.matrix(),
                is_diagonal: false,
                label: "Y".into(),
            },
        )],
    ] {
        let c = DifferentiableCircuit::from_circuit(Circuit::qubits(1, gates).unwrap()).unwrap();
        let x = state(1, 0.1);
        let bar = state(1, -0.2);
        let v = c.vjp(&[], &x, &bar).unwrap();
        assert!(v.parameters.is_empty());
        let (_, dy) = c.jvp(&[], &x, &[], &bar).unwrap();
        close(pairing(&bar, &dy), pairing(&v.input, &bar), 1e-12);
    }
    let bound = BoundCircuit::new(
        Circuit::qubits(1, vec![put(vec![0], Gate::Ry(0.3))]).unwrap(),
        vec![],
        vec![ParameterBinding::Fixed(0.3)],
    )
    .unwrap();
    let c = DifferentiableCircuit::new(bound).unwrap();
    assert!(
        c.vjp(&[], &state(1, 0.1), &state(1, 0.3))
            .unwrap()
            .parameters
            .is_empty()
    );
}

#[test]
fn rejects_nonunitary_channels_and_bad_inputs() {
    let noisy = Circuit::qubits(
        1,
        vec![crate::circuit::channel(
            vec![0],
            crate::noise::NoiseChannel::BitFlip { p: 0.2 },
        )],
    )
    .unwrap();
    assert!(
        DifferentiableCircuit::from_circuit(noisy)
            .unwrap_err()
            .contains("noise channels")
    );
    let huge = Circuit::qubits(usize::BITS as usize - 4, vec![]).unwrap();
    assert!(DifferentiableCircuit::from_circuit(huge).is_err());
    for (matrix, diagonal) in [
        (Gate::H.matrix().mapv(|x| 2. * x), false),
        (Gate::H.matrix(), true),
    ] {
        let c = Circuit::qubits(
            1,
            vec![put(
                vec![0],
                Gate::Custom {
                    matrix,
                    is_diagonal: diagonal,
                    label: "bad".into(),
                },
            )],
        )
        .unwrap();
        assert!(DifferentiableCircuit::from_circuit(c).is_err());
    }
    let c = circuit();
    let p = c.template().parameters();
    let x = state(3, 0.2);
    assert!(c.forward(&[], &x).is_err());
    let mut bad = p.to_vec();
    bad[0] = f64::NAN;
    assert!(c.forward(&bad, &x).is_err());
    let mut bad = x.clone();
    bad.state.pop();
    assert!(c.vjp(p, &bad, &x).is_err());
    assert!(c.vjp(p, &x, &bad).is_err());
    bad = x.clone();
    bad.state[0].im = f64::INFINITY;
    assert!(c.forward(p, &bad).is_err());
    assert!(c.jvp(p, &x, &[], &x).is_err());
}
