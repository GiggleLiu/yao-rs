use super::*;
use crate::hamiltonian::{Boundary, heisenberg, ising};
use crate::{ArrayReg, Op, OperatorPolynomial, OperatorString};

fn distance(a: &[C], b: &[C]) -> f64 {
    norm(&a.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>())
}

// A dense power series is independent of Lanczos and its projected eigensolver.
fn series(matrix: &[Vec<C>], input: &[C], time: f64) -> Vec<C> {
    let mut sum = input.to_vec();
    let mut term = input.to_vec();
    for k in 1..=200 {
        term = matrix
            .iter()
            .map(|row| {
                row.iter().zip(&term).map(|(a, x)| a * x).sum::<C>() * C::new(0., -time / k as f64)
            })
            .collect();
        for (out, value) in sum.iter_mut().zip(&term) {
            *out += value;
        }
        if norm(&term) < 1e-16 * norm(input) {
            return sum;
        }
    }
    panic!("dense reference did not converge");
}

fn action(matrix: &[Vec<C>], x: &[C], y: &mut [C]) -> Result<(), String> {
    for (out, row) in y.iter_mut().zip(matrix) {
        *out = row.iter().zip(x).map(|(a, x)| a * x).sum();
    }
    Ok(())
}

fn complex_matrix(n: usize) -> Vec<Vec<C>> {
    (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    if r == c {
                        C::new(0.13 * r as f64 - 0.4, 0.)
                    } else {
                        C::new(
                            ((r + c + 1) as f64).sin() * 0.3,
                            (r as f64 - c as f64).sin() * 0.2,
                        )
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn complex_callbacks_match_dense_exponential_and_preserve_norm() {
    for n in [1, 3, 7, 12] {
        let matrix = complex_matrix(n);
        let input: Vec<_> = (0..n)
            .map(|i| C::new(0.2 + i as f64 * 0.1, (i as f64 + 0.4).cos()))
            .collect();
        for time in [-1.2, 0.3, 1.7] {
            let result = exponential_action(&input, time, EvolutionOptions::default(), |x, y| {
                action(&matrix, x, y)
            })
            .unwrap();
            let reference = series(&matrix, &input, time);
            assert!(distance(&result.state, &reference) < 2e-10);
            assert!((norm(&result.state) - norm(&input)).abs() < 1e-12);
            assert_eq!(result.info.time_reached, time);
            assert!(result.info.estimated_error <= result.info.tolerance);
        }
    }
}

#[test]
fn adaptive_restarts_and_tolerance_refinement() {
    let matrix = complex_matrix(9);
    let input: Vec<_> = (0..9)
        .map(|i| C::new((i as f64 + 0.2).sin(), 0.3))
        .collect();
    let exact = series(&matrix, &input, 2.1);
    let mut errors = Vec::new();
    for rtol in [1e-3, 1e-5, 1e-7] {
        let options = EvolutionOptions {
            krylov_dim: 4,
            atol: 0.,
            rtol,
            ..Default::default()
        };
        let got = exponential_action(&input, 2.1, options, |x, y| action(&matrix, x, y)).unwrap();
        assert!(got.info.steps > 1);
        assert!(got.info.max_krylov_dim <= 4);
        let error = distance(&got.state, &exact);
        assert!(error <= got.info.estimated_error + 3e-13);
        assert!(got.info.estimated_error <= got.info.tolerance);
        errors.push(error);
    }
    assert!(errors.windows(2).all(|w| w[1] < w[0] / 10.), "{errors:?}");
}

#[test]
fn pauli_masks_match_independent_tensor_product_action() {
    let n = 4;
    let words = vec![
        OperatorString::identity(),
        OperatorString::new(vec![(0, Op::Y), (3, Op::X)]),
        OperatorString::new(vec![(1, Op::Z), (2, Op::Y)]),
        OperatorString::new(vec![(0, Op::Y), (1, Op::Y), (3, Op::Y)]),
    ];
    let poly = OperatorPolynomial::new(
        vec![0.7.into(), (-0.3).into(), 0.4.into(), 0.2.into()],
        words,
    );
    let h = crate::hamiltonian::PauliHamiltonian::new(n, &poly).unwrap();
    let d = 1 << n;
    let mut dense = vec![vec![C::new(0., 0.); d]; d];
    for (coeff, word) in poly.iter() {
        for (row, entries) in dense.iter_mut().enumerate() {
            for (col, entry) in entries.iter_mut().enumerate() {
                let mut value = *coeff;
                for site in 0..n {
                    let op = word
                        .ops()
                        .iter()
                        .find(|(s, _)| *s == site)
                        .map_or(Op::I, |(_, o)| *o);
                    let matrix = crate::op_matrix(&op);
                    value *= matrix[[(row >> (n - 1 - site)) & 1, (col >> (n - 1 - site)) & 1]];
                }
                *entry += value;
            }
        }
    }
    let input = ArrayReg::deterministic_state(n);
    for time in [-0.83, 0.61] {
        let got = h
            .evolve_krylov(&input, time, EvolutionOptions::default())
            .unwrap();
        assert!(distance(&got.state, &series(&dense, &input.state, time)) < 2e-11);
    }
}

#[test]
fn exact_zero_identity_and_pauli_invariant_subspaces() {
    let input = ArrayReg::deterministic_state(3);
    let zero = crate::hamiltonian::PauliHamiltonian::new(3, &OperatorPolynomial::zero()).unwrap();
    assert_eq!(
        zero.evolve_krylov(&input, 5., EvolutionOptions::default())
            .unwrap()
            .state,
        input.state
    );
    let options = EvolutionOptions::default();
    let got = exponential_action(&input.state, 0., options, |_, _| {
        panic!("zero time needs no callback")
    })
    .unwrap();
    assert_eq!(got.state, input.state);
    assert_eq!(got.info.matvecs, 0);
    let zeros = vec![C::new(0., 0.); 7];
    assert_eq!(
        exponential_action(&zeros, 7., options, |_, _| panic!(
            "zero vector needs no callback"
        ))
        .unwrap()
        .state,
        zeros
    );
    for op in [Op::I, Op::X, Op::Y, Op::Z] {
        let poly = OperatorPolynomial::single(2, op, 0.7.into());
        let h = crate::hamiltonian::PauliHamiltonian::new(3, &poly).unwrap();
        let got = h.evolve_krylov(&input, -1.3, options).unwrap();
        let rotation =
            crate::hamiltonian::pauli_rotation(3, &OperatorString::new(vec![(2, op)]), -1.82)
                .unwrap();
        assert!(distance(&got.state, &rotation.apply(&input).unwrap().state) < 1e-13);
        assert!(got.info.matvecs <= 2);
    }
}

#[test]
fn work_limit_returns_partial_time_and_can_resume() {
    let matrix = complex_matrix(9);
    let input = vec![C::new(0.3, 0.2); 9];
    let options = EvolutionOptions {
        krylov_dim: 3,
        max_matvecs: 3,
        rtol: 1e-4,
        atol: 0.,
    };
    let error =
        exponential_action(&input, -1.1, options, |x, y| action(&matrix, x, y)).unwrap_err();
    assert_eq!(error.kind, EvolutionFailure::WorkLimit);
    let partial = error.partial.unwrap();
    assert_eq!(partial.info.matvecs, 3);
    assert_eq!(partial.info.steps, 1);
    assert!(partial.info.time_reached < 0. && partial.info.time_reached > -1.1);
    assert!(
        distance(
            &partial.state,
            &series(&matrix, &input, partial.info.time_reached)
        ) < 1e-4 * norm(&input)
    );
    let resumed = exponential_action(
        &partial.state,
        -1.1 - partial.info.time_reached,
        EvolutionOptions::default(),
        |x, y| action(&matrix, x, y),
    )
    .unwrap();
    assert!(distance(&resumed.state, &series(&matrix, &input, -1.1)) < 1e-4 * norm(&input));
}

#[test]
fn validation_callback_failures_and_nonhermitian_rejection() {
    let x = vec![C::new(1., 0.), C::new(0., 0.)];
    for options in [
        EvolutionOptions {
            krylov_dim: 1,
            ..Default::default()
        },
        EvolutionOptions {
            max_matvecs: 0,
            ..Default::default()
        },
        EvolutionOptions {
            atol: -1.,
            ..Default::default()
        },
        EvolutionOptions {
            rtol: f64::NAN,
            ..Default::default()
        },
        EvolutionOptions {
            atol: 0.,
            rtol: 0.,
            ..Default::default()
        },
    ] {
        assert_eq!(
            exponential_action(&x, 1., options, |_, _| panic!())
                .unwrap_err()
                .kind,
            EvolutionFailure::InvalidInput
        );
    }
    let options = EvolutionOptions::default();
    assert!(exponential_action(&[], 1., options, |_, _| panic!()).is_err());
    assert!(exponential_action(&[C::new(f64::NAN, 0.)], 1., options, |_, _| panic!()).is_err());
    assert!(exponential_action(&x, f64::INFINITY, options, |_, _| panic!()).is_err());
    let err =
        exponential_action(&x, 1., options, |_, _| Err("operator unavailable".into())).unwrap_err();
    assert_eq!(err.kind, EvolutionFailure::Operator);
    assert!(err.to_string().contains("operator unavailable"));
    assert!(exponential_action(&x, 1., options, |_, _| Ok(())).is_err());
    let nonhermitian = vec![
        vec![C::new(0., 0.), C::new(1., 0.)],
        vec![C::new(2., 0.), C::new(0., 0.)],
    ];
    assert!(exponential_action(&x, 1., options, |x, y| action(&nonhermitian, x, y)).is_err());
    assert!(
        exponential_action(&x, 1., options, |x, y| {
            for (y, x) in y.iter_mut().zip(x) {
                *y = C::new(0., 1.) * x;
            }
            Ok(())
        })
        .is_err()
    );
    let h = ising(3, 0.4, 0.7, Boundary::Open).unwrap();
    assert!(
        h.evolve_krylov(&ArrayReg::zero_state(2), 1., options)
            .is_err()
    );
    let mut malformed = ArrayReg::zero_state(3);
    malformed.state.pop();
    assert!(h.evolve_krylov(&malformed, 1., options).is_err());
    let huge = crate::hamiltonian::PauliHamiltonian::with_shared_coefficients(
        usize::BITS as usize,
        vec![],
        vec![],
    )
    .unwrap();
    assert!(
        huge.evolve_krylov(&ArrayReg::zero_state(1), 1., options)
            .is_err()
    );
}

#[test]
fn large_matrix_free_model_and_reversible_evolution() {
    let h = heisenberg(12, [0.3, -0.4, 0.5], 0.2, Boundary::Open).unwrap();
    let input = ArrayReg::deterministic_state(12);
    let options = EvolutionOptions {
        rtol: 1e-9,
        ..Default::default()
    };
    let forward = h.evolve_krylov(&input, 0.7, options).unwrap();
    let backward = h
        .evolve_krylov(&ArrayReg::from_vec(12, forward.state), -0.7, options)
        .unwrap();
    assert!(distance(&input.state, &backward.state) < 2e-9);
    assert!(backward.info.max_krylov_dim <= options.krylov_dim);
}

#[test]
fn scaling_shift_and_roundoff_limits_are_explicit() {
    let matrix = complex_matrix(5);
    let base: Vec<_> = (0..5).map(|i| C::new(0.31 * i as f64, -0.27)).collect();
    let expected = series(&matrix, &base, 0.43);
    for scale in [1e-200, 1., 1e200] {
        let input: Vec<_> = base.iter().map(|x| x * scale).collect();
        let options = EvolutionOptions {
            atol: 0.,
            rtol: 1e-10,
            ..Default::default()
        };
        let got = exponential_action(&input, 0.43, options, |x, y| action(&matrix, x, y)).unwrap();
        let rescaled: Vec<_> = got.state.iter().map(|x| x / scale).collect();
        assert!(distance(&rescaled, &expected) < 1e-10);
        let unchanged = exponential_action(&input, 0.43, options, |_, y| {
            y.fill(C::new(0., 0.));
            Ok(())
        })
        .unwrap();
        assert_eq!(unchanged.state, input);
    }
    // Identity shifts change global phase, not the Krylov subspace.
    let shift = 100.;
    let shifted = exponential_action(&base, 0.43, EvolutionOptions::default(), |x, y| {
        action(&matrix, x, y)?;
        for (y, x) in y.iter_mut().zip(x) {
            *y += shift * x;
        }
        Ok(())
    })
    .unwrap();
    let want: Vec<_> = expected
        .iter()
        .map(|x| C::from_polar(1., -shift * 0.43) * x)
        .collect();
    assert!(distance(&shifted.state, &want) < 2e-12);
    let strict = EvolutionOptions {
        atol: 0.,
        rtol: 1e-30,
        ..Default::default()
    };
    let err = exponential_action(&base, 0.43, strict, |x, y| action(&matrix, x, y)).unwrap_err();
    assert_eq!(err.kind, EvolutionFailure::PrecisionLimit);
    assert!(err.partial.is_some());
    let overflow = exponential_action(
        &[C::new(1., 0.)],
        1e300,
        EvolutionOptions::default(),
        |x, y| {
            y[0] = x[0] * 1e300;
            Ok(())
        },
    )
    .unwrap_err();
    assert_eq!(overflow.kind, EvolutionFailure::Numerical);
}

#[test]
fn work_limit_mid_basis_does_not_advance_state() {
    let matrix = complex_matrix(7);
    let input = vec![C::new(0.7, 0.2); 7];
    let error = exponential_action(
        &input,
        1.,
        EvolutionOptions {
            max_matvecs: 1,
            ..Default::default()
        },
        |x, y| action(&matrix, x, y),
    )
    .unwrap_err();
    assert_eq!(error.kind, EvolutionFailure::WorkLimit);
    let partial = error.partial.unwrap();
    assert_eq!(partial.state, input);
    assert_eq!(partial.info.steps, 0);
    assert_eq!(partial.info.time_reached, 0.);
    assert_eq!(partial.info.matvecs, 1);
}

#[test]
fn large_basis_reductions_support_tight_reference_accuracy() {
    let h = heisenberg(16, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic).unwrap();
    let input = ArrayReg::zero_state(16);
    let result = h
        .evolve_krylov(
            &input,
            2.1,
            EvolutionOptions {
                atol: 0.,
                rtol: 1e-13,
                krylov_dim: 40,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.info.time_reached, 2.1);
    assert!(result.info.estimated_error <= 1e-13);
    assert!((norm(&result.state) - 1.).abs() < 2e-13);
}
