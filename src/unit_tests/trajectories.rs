use super::*;
use crate::{
    DensityMatrix, Gate, NoiseChannel, Op, OperatorString, Register, channel, control, put,
};
use ndarray::Array2;

fn options(trajectories: usize) -> TrajectoryOptions {
    TrajectoryOptions {
        trajectories,
        seed: 817,
        threads: 1,
    }
}
fn observable() -> OperatorPolynomial {
    OperatorPolynomial::new(
        vec![C::new(0.7, 0.3), C::new(-0.2, 0.8), C::new(0.1, -0.1)],
        vec![
            OperatorString::new(vec![(0, Op::Y), (2, Op::Z)]),
            OperatorString::new(vec![(1, Op::Pu)]),
            OperatorString::new(vec![]),
        ],
    )
}
fn error_bound(stats: &TrajectoryStatistics, expected: C) {
    let se = stats.standard_error.unwrap();
    assert!(
        (stats.mean.re - expected.re).abs() < 6. * se.re + 1e-11,
        "{stats:?} expected {expected}"
    );
    assert!(
        (stats.mean.im - expected.im).abs() < 6. * se.im + 1e-11,
        "{stats:?} expected {expected}"
    );
}

#[test]
fn built_in_ensembles_match_exact_density() {
    let channels = vec![
        NoiseChannel::BitFlip { p: 0.3 },
        NoiseChannel::PhaseFlip { p: 0.4 },
        NoiseChannel::Depolarizing { n: 1, p: 0.7 },
        NoiseChannel::PauliChannel {
            px: 0.1,
            py: 0.2,
            pz: 0.3,
        },
        NoiseChannel::Reset { p0: 0.2, p1: 0.3 },
        NoiseChannel::AmplitudeDamping {
            gamma: 0.45,
            excited_population: 0.2,
        },
        NoiseChannel::PhaseDamping { gamma: 0.7 },
        NoiseChannel::PhaseAmplitudeDamping {
            amplitude: 0.2,
            phase: 0.3,
            excited_population: 0.4,
        },
        NoiseChannel::ThermalRelaxation {
            t1: 100.,
            t2: 80.,
            time: 300.,
            excited_population: 0.2,
        },
        NoiseChannel::Coherent {
            matrix: Gate::Ry(0.8).matrix(),
        },
    ];
    let input = ArrayReg::deterministic_state(3);
    for noise in channels {
        let circuit = Circuit::qubits(
            3,
            vec![
                put(vec![2], Gate::Ry(0.8)),
                control(vec![2], vec![0], Gate::X),
                channel(vec![1], noise.clone()),
                channel(vec![0], noise),
                put(vec![0], Gate::Rx(0.5)),
            ],
        )
        .unwrap();
        let mut exact = DensityMatrix::from_reg(&input);
        exact.apply(&circuit);
        let simulator = TrajectoryCircuit::new(circuit).unwrap();
        let stats = simulator
            .expectation(&input, &observable(), options(4096))
            .unwrap();
        error_bound(&stats, crate::expect_dm(&exact, &observable()));
        let state = simulator.sample(&input, 63, 29).unwrap();
        assert!((state.norm() - 1.).abs() < 1e-12);
    }
}

#[test]
fn endpoints_and_state_dependent_branches() {
    let z = OperatorPolynomial::single(0, Op::Z, C::new(1., 0.));
    for (noise, input, expected) in [
        (NoiseChannel::BitFlip { p: 0. }, ArrayReg::zero_state(1), 1.),
        (
            NoiseChannel::BitFlip { p: 1. },
            ArrayReg::zero_state(1),
            -1.,
        ),
        (
            NoiseChannel::AmplitudeDamping {
                gamma: 1.,
                excited_population: 0.,
            },
            ArrayReg::uniform_state(1),
            1.,
        ),
        (
            NoiseChannel::Reset { p0: 0., p1: 1. },
            ArrayReg::zero_state(1),
            -1.,
        ),
    ] {
        let sim =
            TrajectoryCircuit::new(Circuit::qubits(1, vec![channel(vec![0], noise)]).unwrap())
                .unwrap();
        let stats = sim.expectation(&input, &z, options(16)).unwrap();
        assert!((stats.mean.re - expected).abs() < 1e-14);
        assert!(stats.standard_error.unwrap().norm() < 1e-14);
    }
    // Projectors have state-dependent probabilities, unlike a mixed-unitary channel.
    let noise = NoiseChannel::Custom {
        kraus_ops: vec![crate::op_matrix(&Op::P0), crate::op_matrix(&Op::P1)],
    };
    let sim =
        TrajectoryCircuit::new(Circuit::qubits(1, vec![channel(vec![0], noise)]).unwrap()).unwrap();
    let input = ArrayReg::from_vec(1, vec![C::new(0.6, 0.), C::new(0., 0.8)]);
    let stats = sim.expectation(&input, &z, options(8192)).unwrap();
    error_bound(&stats, C::new(-0.28, 0.));
    assert!((stats.sample_variance.unwrap().re - (1. - 0.28_f64.powi(2))).abs() < 0.04);
}

#[test]
fn nonadjacent_multiqubit_custom_channel_and_controls() {
    let k = Gate::FSim(0.7, 0.3).matrix();
    let noise = NoiseChannel::Custom {
        kraus_ops: vec![
            Array2::eye(4).mapv(|z: C| z * 0.3_f64.sqrt()),
            k.mapv(|z| z * 0.7_f64.sqrt()),
        ],
    };
    for noise in [noise, NoiseChannel::Depolarizing { n: 2, p: 0.5 }] {
        let mut ctrl = match control(vec![1], vec![2], Gate::Ry(0.31)) {
            CircuitElement::Gate(g) => g,
            _ => unreachable!(),
        };
        ctrl.control_configs[0] = false;
        let circuit = Circuit::qubits(
            3,
            vec![
                CircuitElement::Gate(ctrl),
                channel(vec![2, 0], noise),
                put(vec![0, 2], Gate::SWAP),
            ],
        )
        .unwrap();
        let input = ArrayReg::deterministic_state(3);
        let mut dm = DensityMatrix::from_reg(&input);
        dm.apply(&circuit);
        let sim = TrajectoryCircuit::new(circuit).unwrap();
        error_bound(
            &sim.expectation(&input, &observable(), options(4096))
                .unwrap(),
            crate::expect_dm(&dm, &observable()),
        );
    }
}

#[test]
fn seed_streams_reproduce_and_moments_match_explicit_samples() {
    let sim = TrajectoryCircuit::new(
        Circuit::qubits(
            3,
            vec![
                channel(vec![0], NoiseChannel::BitFlip { p: 0.4 }),
                channel(
                    vec![1],
                    NoiseChannel::AmplitudeDamping {
                        gamma: 0.3,
                        excited_population: 0.1,
                    },
                ),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let input = ArrayReg::deterministic_state(3);
    let op = observable();
    let stats = sim.expectation(&input, &op, options(257)).unwrap();
    assert_eq!(stats, sim.expectation(&input, &op, options(257)).unwrap());
    let samples: Vec<_> = (0..257)
        .map(|id| crate::expect_arrayreg(&sim.sample(&input, 817, id).unwrap(), &op))
        .collect();
    let mean: C = samples.iter().sum::<C>() / 257.;
    let variance = samples
        .iter()
        .map(|x| C::new((x.re - mean.re).powi(2), (x.im - mean.im).powi(2)))
        .sum::<C>()
        / 256.;
    let covariance = samples
        .iter()
        .map(|x| (x.re - mean.re) * (x.im - mean.im))
        .sum::<f64>()
        / 256.;
    assert!((stats.mean - mean).norm() < 1e-14);
    assert!((stats.sample_variance.unwrap() - variance).norm() < 1e-14);
    assert!((stats.covariance.unwrap() - covariance).abs() < 1e-14);
    assert_ne!(
        sim.expectation(
            &input,
            &op,
            TrajectoryOptions {
                seed: 818,
                ..options(257)
            }
        )
        .unwrap()
        .mean,
        stats.mean
    );
    #[cfg(feature = "parallel")]
    for threads in [2, 4] {
        let parallel = sim
            .expectation(
                &input,
                &op,
                TrajectoryOptions {
                    threads,
                    ..options(257)
                },
            )
            .unwrap();
        assert_eq!(parallel.mean, stats.mean);
        assert_eq!(parallel.sample_variance, stats.sample_variance);
        assert_eq!(parallel.covariance, stats.covariance);
    }
}

#[test]
fn identity_empty_and_one_sample_uncertainty() {
    for n in [0, 3] {
        let sim = TrajectoryCircuit::new(Circuit::qubits(n, vec![]).unwrap()).unwrap();
        let input = ArrayReg::zero_state(n);
        let op = OperatorPolynomial::new(vec![C::new(2., -0.3)], vec![OperatorString::new(vec![])]);
        let stats = sim.expectation(&input, &op, options(1)).unwrap();
        assert_eq!(stats.mean, C::new(2., -0.3));
        assert_eq!(stats.standard_error, None);
        assert_eq!(stats.sample_variance, None);
        assert_eq!(stats.covariance, None);
        assert_eq!(sim.sample(&input, 0, 0).unwrap().state, input.state);
        let zero = OperatorPolynomial::new(vec![], vec![]);
        assert_eq!(
            sim.expectation(&input, &zero, options(3)).unwrap().mean,
            C::new(0., 0.)
        );
    }
}

#[test]
fn invalid_domains_fail_before_execution() {
    let sim = TrajectoryCircuit::new(Circuit::qubits(1, vec![]).unwrap()).unwrap();
    let op = OperatorPolynomial::single(0, Op::Z, C::new(1., 0.));
    for input in [
        ArrayReg::zero_state(2),
        ArrayReg::from_vec(1, vec![C::new(2., 0.), C::new(0., 0.)]),
        ArrayReg::from_vec(1, vec![C::new(f64::NAN, 0.), C::new(0., 0.)]),
    ] {
        assert!(sim.sample(&input, 0, 0).is_err());
    }
    for config in [
        options(0),
        TrajectoryOptions {
            threads: 0,
            ..options(2)
        },
    ] {
        assert!(
            sim.expectation(&ArrayReg::zero_state(1), &op, config)
                .is_err()
        );
    }
    let mut circuit = Circuit::qubits(1, vec![]).unwrap();
    circuit.nbits = 2;
    assert!(TrajectoryCircuit::new(circuit).is_err());
    let invalids = vec![
        channel(vec![0], NoiseChannel::Custom { kraus_ops: vec![] }),
        channel(vec![0], NoiseChannel::BitFlip { p: 2. }),
        channel(vec![0, 0], NoiseChannel::Depolarizing { n: 2, p: 0.1 }),
        put(
            vec![0],
            Gate::Custom {
                matrix: Array2::eye(2).mapv(|z: C| z * 2.),
                is_diagonal: false,
                label: "nonunitary".into(),
            },
        ),
        put(vec![0], Gate::Rx(f64::NAN)),
        control(vec![0, 0], vec![1], Gate::X),
    ];
    for element in invalids {
        assert!(
            TrajectoryCircuit::new(Circuit {
                nbits: 2,
                dims: vec![2, 2],
                elements: vec![element]
            })
            .is_err()
        );
    }
    let badop = OperatorPolynomial::single(1, Op::Z, C::new(1., 0.));
    assert!(
        sim.expectation(&ArrayReg::zero_state(1), &badop, options(2))
            .is_err()
    );
    #[cfg(not(feature = "parallel"))]
    assert!(
        sim.expectation(
            &ArrayReg::zero_state(1),
            &op,
            TrajectoryOptions {
                threads: 2,
                ..options(2)
            }
        )
        .is_err()
    );
}

#[test]
fn register_wide_depolarizing_avoids_dense_kraus_storage() {
    // Dense Kraus matrices here would need 16^12 complex entries.
    let n = 12;
    let circuit = Circuit::qubits(
        n,
        vec![channel(
            (0..n).collect(),
            NoiseChannel::Depolarizing { n, p: 1. },
        )],
    )
    .unwrap();
    let simulator = TrajectoryCircuit::new(circuit).unwrap();
    let input = ArrayReg::zero_state(n);
    let op = OperatorPolynomial::single(n - 1, Op::Z, C::new(1., 0.));
    let stats = simulator.expectation(&input, &op, options(256)).unwrap();
    error_bound(&stats, C::new(0., 0.));
    assert!((simulator.sample(&input, 817, 0).unwrap().norm() - 1.).abs() < 1e-14);
}

#[test]
fn malformed_matrices_and_observables_return_errors() {
    for matrix in [
        Array2::zeros((2, 3)),
        Array2::zeros((0, 0)),
        Array2::from_elem((2, 2), C::new(f64::INFINITY, 0.)),
    ] {
        let c = Circuit {
            nbits: 1,
            dims: vec![2],
            elements: vec![put(
                vec![0],
                Gate::Custom {
                    matrix,
                    is_diagonal: false,
                    label: "bad".into(),
                },
            )],
        };
        assert!(TrajectoryCircuit::new(c).is_err());
    }
    let sim = TrajectoryCircuit::new(Circuit::qubits(1, vec![]).unwrap()).unwrap();
    let malformed: OperatorPolynomial =
        serde_json::from_str(r#"{"coeffs":[[1.0,0.0]],"opstrings":[]}"#).unwrap();
    assert!(
        sim.expectation(&ArrayReg::zero_state(1), &malformed, options(2))
            .is_err()
    );
    let duplicate = OperatorPolynomial::new(
        vec![C::new(1., 0.)],
        vec![OperatorString::new(vec![(0, Op::X), (0, Op::Y)])],
    );
    assert!(
        sim.expectation(&ArrayReg::zero_state(1), &duplicate, options(2))
            .is_err()
    );
}

#[test]
fn generic_three_target_kraus_preserves_complex_amplitudes_and_site_order() {
    let n = 4;
    let locs = vec![3, 0, 2];
    let matrix = Array2::from_shape_fn((8, 8), |(row, col)| {
        if row == (col + 1) % 8 {
            C::from_polar(1., 0.13 * col as f64)
        } else {
            C::new(0., 0.)
        }
    });
    let input = ArrayReg::deterministic_state(n);
    // Independent embedded-matrix oracle: first target is the local least
    // significant bit, while register site zero is the global most significant.
    let local = |basis: usize| {
        locs.iter()
            .enumerate()
            .map(|(j, &loc)| ((basis >> (n - 1 - loc)) & 1) << j)
            .sum::<usize>()
    };
    let spectator = |basis: usize| (basis >> (n - 1 - 1)) & 1;
    let expected: Vec<C> = (0..16)
        .map(|row| {
            (0..16)
                .filter(|&col| spectator(row) == spectator(col))
                .map(|col| matrix[[local(row), local(col)]] * input.state[col])
                .sum()
        })
        .collect();
    let noise = NoiseChannel::Custom {
        kraus_ops: vec![
            Array2::eye(8).mapv(|z: C| z * 0.3_f64.sqrt()),
            matrix.mapv(|z| z * 0.7_f64.sqrt()),
        ],
    };
    let simulator =
        TrajectoryCircuit::new(Circuit::qubits(n, vec![channel(locs, noise)]).unwrap()).unwrap();
    let mut seen = [false; 2];
    for id in 0..32 {
        let result = simulator.sample(&input, 19, id).unwrap();
        let close = |want: &[C]| {
            result
                .state
                .iter()
                .zip(want)
                .all(|(a, b)| (a - b).norm() < 1e-13)
        };
        if close(&input.state) {
            seen[0] = true;
        } else {
            assert!(close(&expected));
            seen[1] = true;
        }
    }
    assert_eq!(seen, [true, true]);
}
