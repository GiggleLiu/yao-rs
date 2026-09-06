use super::*;
use crate::{ArrayReg, expect_arrayreg, expect_grad};
use ndarray::{Array1, Array2};
use num_complex::Complex64 as C;

fn close(a: &[C], b: &[C], tolerance: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).norm() < tolerance, "{a} != {b}");
    }
}

// Independent small dense oracle: build each Pauli column directly from its
// action on computational basis bits, then sum the exponential power series.
fn dense(n: usize, poly: &OperatorPolynomial) -> Array2<C> {
    let dim = 1 << n;
    let mut matrix = Array2::zeros((dim, dim));
    for (coefficient, word) in poly.iter() {
        for column in 0..dim {
            let mut row = column;
            let mut value = *coefficient;
            for &(site, op) in word.ops() {
                let bit = 1 << (n - 1 - site);
                let one = column & bit != 0;
                match op {
                    Op::X => row ^= bit,
                    Op::Y => {
                        row ^= bit;
                        value *= C::new(0., if one { -1. } else { 1. });
                    }
                    Op::Z => {
                        if one {
                            value = -value
                        }
                    }
                    Op::I => {}
                    _ => panic!("non-Pauli oracle input"),
                }
            }
            matrix[[row, column]] += value;
        }
    }
    matrix
}
fn exact(n: usize, poly: &OperatorPolynomial, time: f64, input: &[C]) -> Vec<C> {
    let matrix = dense(n, poly).mapv(|v| C::new(0., -time) * v);
    let mut term = Array1::from_vec(input.to_vec());
    let mut sum = term.clone();
    for k in 1..=150 {
        term = matrix.dot(&term).mapv(|v| v / k as f64);
        sum += &term;
        if term.iter().map(|z| z.norm()).sum::<f64>() < 1e-16 {
            return sum.to_vec();
        }
    }
    panic!("test exponential series did not converge");
}
fn error(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

#[test]
fn analytic_pauli_rotations_and_identity_phase() {
    let input = ArrayReg::deterministic_state(3);
    for ops in [
        vec![],
        vec![(0, Op::X)],
        vec![(2, Op::Y)],
        vec![(0, Op::Z), (2, Op::Y)],
        vec![(0, Op::Y), (1, Op::X), (2, Op::Y)],
    ] {
        let word = OperatorString::new(ops);
        let poly = OperatorPolynomial::new(vec![C::new(1., 0.)], vec![word.clone()]);
        let ppsi = dense(3, &poly).dot(&Array1::from_vec(input.state.clone()));
        for theta in [-1.3, 0., 0.7] {
            let rotation = pauli_rotation(3, &word, theta).unwrap();
            let got = rotation.apply(&input).unwrap();
            let want = input
                .state
                .iter()
                .zip(&ppsi)
                .map(|(v, p)| (theta / 2.).cos() * v - C::new(0., (theta / 2.).sin()) * p)
                .collect::<Vec<_>>();
            close(&got.state, &want, 2e-14);
            if theta == 0. {
                assert_eq!(got.state, input.state);
            }
        }
    }
}

#[test]
fn identity_global_phase_is_visible_under_control() {
    let rotation = pauli_rotation(1, &OperatorString::identity(), 0.7).unwrap();
    for config in [true, false] {
        let elements = rotation
            .circuit()
            .elements
            .iter()
            .map(|element| {
                let CircuitElement::Gate(pg) = element else {
                    unreachable!()
                };
                let mut gate = control(vec![0], vec![1], pg.gate.clone());
                if let CircuitElement::Gate(pg) = &mut gate {
                    pg.control_configs[0] = config;
                }
                gate
            })
            .collect();
        let circuit = Circuit::qubits(2, elements).unwrap();
        let input = ArrayReg::deterministic_state(2);
        let got = crate::apply(&circuit, &input);
        let phase = C::from_polar(1., -0.35);
        let want = input
            .state
            .iter()
            .enumerate()
            .map(|(i, z)| if (i >= 2) == config { z * phase } else { *z })
            .collect::<Vec<_>>();
        close(&got.state, &want, 1e-14);
    }
}

#[test]
fn commuting_sums_negative_time_and_zero_hamiltonian() {
    let poly = OperatorPolynomial::new(
        vec![0.3.into(), (-0.7).into(), 0.2.into(), 0.4.into()],
        vec![
            OperatorString::new(vec![(0, Op::Z), (2, Op::Z)]),
            OperatorString::new(vec![(0, Op::X), (2, Op::X)]),
            OperatorString::new(vec![(1, Op::Y)]),
            OperatorString::identity(),
        ],
    );
    let h = PauliHamiltonian::new(3, &poly).unwrap();
    let input = ArrayReg::deterministic_state(3);
    for time in [-0.8, 0., 0.9] {
        for steps in [1, 3] {
            for formula in [ProductFormula::LieTrotter, ProductFormula::Suzuki2] {
                let got = h
                    .evolve(time, steps, formula)
                    .unwrap()
                    .apply(&input)
                    .unwrap();
                close(&got.state, &exact(3, &poly, time, &input.state), 2e-13);
                if time == 0. {
                    assert_eq!(got.state, input.state);
                }
            }
        }
    }
    let zero = PauliHamiltonian::new(3, &OperatorPolynomial::zero()).unwrap();
    let evolved = zero
        .evolve(0.7, usize::MAX, ProductFormula::Suzuki2)
        .unwrap();
    assert!(evolved.circuit().elements.is_empty());
    assert_eq!(evolved.apply(&input).unwrap().state, input.state);
    let zero = ising(3, 0., 0., Boundary::Open).unwrap();
    assert_eq!(
        zero.evolve(1., 3, ProductFormula::Suzuki2)
            .unwrap()
            .apply(&input)
            .unwrap()
            .state,
        input.state
    );
}

#[test]
fn product_formula_convergence_order_against_dense_exponential() {
    for h in [
        ising(3, -0.7, 0.4, Boundary::Open).unwrap(),
        heisenberg(3, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic).unwrap(),
    ] {
        let input = ArrayReg::deterministic_state(3);
        let reference = exact(3, &h.polynomial(), 0.8, &input.state);
        for (formula, min_ratio) in [
            (ProductFormula::LieTrotter, 1.8),
            (ProductFormula::Suzuki2, 3.5),
        ] {
            let errors = [2, 4, 8, 16].map(|steps| {
                error(
                    &h.evolve(0.8, steps, formula)
                        .unwrap()
                        .apply(&input)
                        .unwrap()
                        .state,
                    &reference,
                )
            });
            for pair in errors.windows(2) {
                assert!(pair[0] / pair[1] > min_ratio, "{formula:?}: {errors:?}");
            }
        }
    }
}

#[test]
fn shared_time_and_model_coupling_gradients_including_zero() {
    let input = ArrayReg::deterministic_state(3);
    let observable = &OperatorPolynomial::single(0, Op::Y, 1.0.into())
        + &OperatorPolynomial::single(2, Op::Z, 0.3.into());
    for h in [
        ising(3, -0.6, 0.4, Boundary::Open).unwrap(),
        heisenberg(3, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic).unwrap(),
    ] {
        for time in [0., 0.41] {
            let bound = h.evolve(time, 3, ProductFormula::Suzuki2).unwrap();
            assert_eq!(bound.parameters().len(), h.coefficients().len() + 1);
            let (_, angles) = expect_grad(&observable, bound.circuit(), &input);
            let gradient = bound.pullback(&angles).unwrap();
            for (i, &want) in gradient.iter().enumerate() {
                for eps in [1e-4, 1e-5, 1e-6] {
                    let mut perturbed = bound.clone();
                    let mut params = bound.parameters().to_vec();
                    params[i] += eps;
                    perturbed.dispatch(&params).unwrap();
                    let plus = expect_arrayreg(&perturbed.apply(&input).unwrap(), &observable).re;
                    params[i] -= 2. * eps;
                    perturbed.dispatch(&params).unwrap();
                    let minus = expect_arrayreg(&perturbed.apply(&input).unwrap(), &observable).re;
                    assert!(
                        ((plus - minus) / (2. * eps) - want).abs() < 2e-7,
                        "parameter {i}, time {time}, eps {eps}: FD {} != {want}",
                        (plus - minus) / (2. * eps)
                    );
                }
            }
        }
    }
}

#[test]
fn validation_model_conventions_and_large_words() {
    assert_eq!(
        ising(3, 0.7, -0.2, Boundary::Open)
            .unwrap()
            .polynomial()
            .coeffs(),
        &[
            0.7.into(),
            0.7.into(),
            (-0.2).into(),
            (-0.2).into(),
            (-0.2).into()
        ]
    );
    assert_eq!(
        heisenberg(3, [1., 2., 3.], 0.4, Boundary::Periodic)
            .unwrap()
            .polynomial()
            .len(),
        12
    );
    assert_eq!(
        ising(1, 1., 0.4, Boundary::Open)
            .unwrap()
            .polynomial()
            .len(),
        1
    );
    assert!(ising(2, 1., 1., Boundary::Periodic).is_err());
    assert!(ising(0, 1., 1., Boundary::Open).is_err());
    assert!(heisenberg(3, [1., f64::NAN, 1.], 0., Boundary::Open).is_err());
    for word in [
        OperatorString::new(vec![(3, Op::X)]),
        OperatorString::new(vec![(0, Op::X), (0, Op::Y)]),
        OperatorString::new(vec![(0, Op::P0)]),
        serde_json::from_str::<OperatorString>(r#"{"ops":[[2,"X"],[0,"Y"]]}"#).unwrap(),
    ] {
        assert!(pauli_rotation(3, &word, 0.2).is_err());
    }
    let with_identity: OperatorString =
        serde_json::from_str(r#"{"ops":[[0,"I"],[2,"Y"]]}"#).unwrap();
    let without_identity = OperatorString::new(vec![(2, Op::Y)]);
    let state = ArrayReg::deterministic_state(3);
    close(
        &pauli_rotation(3, &with_identity, 0.2)
            .unwrap()
            .apply(&state)
            .unwrap()
            .state,
        &pauli_rotation(3, &without_identity, 0.2)
            .unwrap()
            .apply(&state)
            .unwrap()
            .state,
        1e-14,
    );
    let single =
        PauliHamiltonian::new(1, &OperatorPolynomial::single(0, Op::X, 1.0.into())).unwrap();
    assert!(
        single
            .evolve(0.1, usize::MAX, ProductFormula::LieTrotter)
            .is_err()
    );
    let h = ising(3, 1., 1., Boundary::Open).unwrap();
    assert!(h.evolve(0.1, 0, ProductFormula::Suzuki2).is_err());
    assert!(h.evolve(f64::INFINITY, 1, ProductFormula::Suzuki2).is_err());
    assert!(h.evolve(0.1, usize::MAX, ProductFormula::Suzuki2).is_err());
    assert!(pauli_rotation(3, &OperatorString::identity(), f64::NAN).is_err());
    let poly = OperatorPolynomial::single(0, Op::X, C::new(1., 0.1));
    assert!(PauliHamiltonian::new(1, &poly).is_err());
    // Construction scales with word length, not the dense operator dimension.
    let word = OperatorString::new((0..64).map(|i| (i, Op::Y)).collect());
    let rotation = pauli_rotation(64, &word, 0.3).unwrap();
    assert!(rotation.circuit().elements.len() < 7 * 64);
    assert!(rotation.circuit().elements.iter().all(|e| match e {
        CircuitElement::Gate(g) => g.target_locs.len() <= 1,
        _ => false,
    }));
}

#[cfg(feature = "tenferro")]
#[test]
fn lowered_evolution_matches_tenferro_and_json_round_trip() {
    let cpu = crate::tenferro::CpuContractor::new(1).unwrap();
    let h = heisenberg(3, [0.3, -0.4, 0.7], 0.2, Boundary::Open).unwrap();
    let bound = h.evolve(-0.37, 2, ProductFormula::Suzuki2).unwrap();
    let tn = crate::circuit_to_einsum_with_boundary(bound.circuit(), &[]);
    let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
    let got = cpu.execute(&plan, &tn.tensors).unwrap();
    let reference = bound.apply(&ArrayReg::zero_state(3)).unwrap();
    close(
        &got.iter().copied().collect::<Vec<_>>(),
        &reference.state,
        1e-12,
    );
    let json = crate::circuit_to_json(bound.circuit());
    let restored = crate::circuit_from_json(&json).unwrap();
    close(
        &crate::apply(&restored, &ArrayReg::zero_state(3)).state,
        &reference.state,
        1e-12,
    );
}
