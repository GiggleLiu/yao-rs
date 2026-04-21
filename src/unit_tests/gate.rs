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
