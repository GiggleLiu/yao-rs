#[test]
fn num_params_covers_all_variants() {
    use crate::gate::Gate;
    use ndarray::Array2;
    use num_complex::Complex64;

    assert_eq!(Gate::X.num_params(), 0);
    assert_eq!(Gate::Y.num_params(), 0);
    assert_eq!(Gate::Z.num_params(), 0);
    assert_eq!(Gate::H.num_params(), 0);
    assert_eq!(Gate::S.num_params(), 0);
    assert_eq!(Gate::T.num_params(), 0);
    assert_eq!(Gate::SWAP.num_params(), 0);
    assert_eq!(Gate::SqrtX.num_params(), 0);
    assert_eq!(Gate::SqrtY.num_params(), 0);
    assert_eq!(Gate::SqrtW.num_params(), 0);
    assert_eq!(Gate::ISWAP.num_params(), 0);
    assert_eq!(Gate::Rx(0.3).num_params(), 1);
    assert_eq!(Gate::Ry(0.3).num_params(), 1);
    assert_eq!(Gate::Rz(0.3).num_params(), 1);
    assert_eq!(Gate::Phase(0.3).num_params(), 1);
    assert_eq!(Gate::FSim(0.2, 0.5).num_params(), 2);

    let m = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(
        Gate::Custom {
            matrix: m,
            is_diagonal: true,
            label: "U".to_string()
        }
        .num_params(),
        0
    );
}

#[test]
fn get_and_set_params_round_trip() {
    use crate::gate::Gate;

    let mut g = Gate::Rx(0.1);
    assert_eq!(g.get_params(), vec![0.1]);
    g.set_params(&[0.9]);
    assert_eq!(g.get_params(), vec![0.9]);
    assert!(matches!(g, Gate::Rx(x) if (x - 0.9).abs() < 1e-15));

    let mut g = Gate::FSim(0.2, 0.5);
    assert_eq!(g.get_params(), vec![0.2, 0.5]);
    g.set_params(&[1.1, 1.3]);
    assert_eq!(g.get_params(), vec![1.1, 1.3]);
    assert!(
        matches!(g, Gate::FSim(t, p) if (t - 1.1).abs() < 1e-15 && (p - 1.3).abs() < 1e-15)
    );

    let g = Gate::X;
    assert!(g.get_params().is_empty());
}

#[test]
#[should_panic(expected = "set_params length")]
fn set_params_rejects_wrong_length() {
    let mut g = crate::gate::Gate::Rx(0.0);
    g.set_params(&[0.1, 0.2]);
}

#[test]
fn generator_matrix_matches_finite_difference() {
    use crate::gate::Gate;
    use num_complex::Complex64;

    let h: f64 = 1e-6;
    let i_c = Complex64::new(0.0, 1.0);
    let c0 = Complex64::new(0.0, 0.0);

    let cases: Vec<(Box<dyn Fn(f64) -> Gate>, usize)> = vec![
        (Box::new(Gate::Rx), 0),
        (Box::new(Gate::Ry), 0),
        (Box::new(Gate::Rz), 0),
        (Box::new(Gate::Phase), 0),
    ];

    for (ctor, idx) in cases {
        let theta = 0.37_f64;
        let u = ctor(theta).matrix();
        let up = ctor(theta + h).matrix();
        let um = ctor(theta - h).matrix();
        // dU/dtheta via central difference
        let du = (&up - &um).mapv(|x| x / Complex64::new(2.0 * h, 0.0));
        // Expected G = dU * U^{-1}. For unitary U, U^{-1} = U^dagger.
        let u_dag = {
            let mut t = u.clone();
            for i in 0..u.nrows() {
                for j in 0..u.ncols() {
                    t[[j, i]] = u[[i, j]].conj();
                }
            }
            t
        };
        // matmul 2x2
        let mut expected = ndarray::Array2::<Complex64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = c0;
                for k in 0..2 {
                    acc += du[[i, k]] * u_dag[[k, j]];
                }
                expected[[i, j]] = acc;
            }
        }
        let got = ctor(theta).generator_matrix(idx);
        for i in 0..2 {
            for j in 0..2 {
                let diff = got[[i, j]] - expected[[i, j]];
                assert!(
                    diff.norm() < 1e-5,
                    "generator mismatch for idx={idx}: got={:?} expected={:?}",
                    got[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    // FSim theta
    let theta = 0.27_f64;
    let phi = 0.43_f64;
    let up = Gate::FSim(theta + h, phi).matrix();
    let um = Gate::FSim(theta - h, phi).matrix();
    let du = (&up - &um).mapv(|x| x / Complex64::new(2.0 * h, 0.0));
    let u = Gate::FSim(theta, phi).matrix();
    let mut u_dag = u.clone();
    for i in 0..4 {
        for j in 0..4 {
            u_dag[[j, i]] = u[[i, j]].conj();
        }
    }
    let mut expected = ndarray::Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                expected[[i, j]] += du[[i, k]] * u_dag[[k, j]];
            }
        }
    }
    let got = Gate::FSim(theta, phi).generator_matrix(0);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (got[[i, j]] - expected[[i, j]]).norm() < 1e-5,
                "FSim theta generator mismatch at [{i},{j}]"
            );
        }
    }

    // FSim phi
    let up = Gate::FSim(theta, phi + h).matrix();
    let um = Gate::FSim(theta, phi - h).matrix();
    let du = (&up - &um).mapv(|x| x / Complex64::new(2.0 * h, 0.0));
    let mut expected = ndarray::Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                expected[[i, j]] += du[[i, k]] * u_dag[[k, j]];
            }
        }
    }
    let got = Gate::FSim(theta, phi).generator_matrix(1);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (got[[i, j]] - expected[[i, j]]).norm() < 1e-5,
                "FSim phi generator mismatch at [{i},{j}]"
            );
        }
    }

    // suppress unused warnings
    let _ = i_c;
}

#[test]
#[should_panic]
fn generator_matrix_rejects_bad_index() {
    let _ = crate::gate::Gate::Rx(0.1).generator_matrix(1);
}

#[test]
#[should_panic]
fn generator_matrix_rejects_non_parametric() {
    let _ = crate::gate::Gate::H.generator_matrix(0);
}
