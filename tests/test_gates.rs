use approx::assert_abs_diff_eq;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI};
use yao_rs::gate::Gate;

/// Helper to check that two complex numbers are approximately equal.
fn assert_complex_approx(a: Complex64, b: Complex64, _msg: &str) {
    assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-12);
    assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-12);
}

// ============================================================
// Matrix value tests for each gate
// ============================================================

#[test]
fn test_x_gate_matrix() {
    let m = Gate::X.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    assert_complex_approx(m[[0, 0]], zero, "X[0,0]");
    assert_complex_approx(m[[0, 1]], one, "X[0,1]");
    assert_complex_approx(m[[1, 0]], one, "X[1,0]");
    assert_complex_approx(m[[1, 1]], zero, "X[1,1]");
}

#[test]
fn test_y_gate_matrix() {
    let m = Gate::Y.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    assert_complex_approx(m[[0, 0]], zero, "Y[0,0]");
    assert_complex_approx(m[[0, 1]], neg_i, "Y[0,1]");
    assert_complex_approx(m[[1, 0]], i, "Y[1,0]");
    assert_complex_approx(m[[1, 1]], zero, "Y[1,1]");
}

#[test]
fn test_z_gate_matrix() {
    let m = Gate::Z.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    assert_complex_approx(m[[0, 0]], one, "Z[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Z[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Z[1,0]");
    assert_complex_approx(m[[1, 1]], neg_one, "Z[1,1]");
}

#[test]
fn test_h_gate_matrix() {
    let m = Gate::H.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
    assert_complex_approx(m[[0, 0]], s, "H[0,0]");
    assert_complex_approx(m[[0, 1]], s, "H[0,1]");
    assert_complex_approx(m[[1, 0]], s, "H[1,0]");
    assert_complex_approx(m[[1, 1]], neg_s, "H[1,1]");
}

#[test]
fn test_s_gate_matrix() {
    let m = Gate::S.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    assert_complex_approx(m[[0, 0]], one, "S[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "S[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "S[1,0]");
    assert_complex_approx(m[[1, 1]], i, "S[1,1]");
}

#[test]
fn test_t_gate_matrix() {
    let m = Gate::T.matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
    assert_complex_approx(m[[0, 0]], one, "T[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "T[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "T[1,0]");
    assert_complex_approx(m[[1, 1]], t_phase, "T[1,1]");
}

#[test]
fn test_swap_gate_matrix() {
    let m = Gate::SWAP.matrix(2);
    assert_eq!(m.dim(), (4, 4));
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // |00> -> |00>
    assert_complex_approx(m[[0, 0]], one, "SWAP[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "SWAP[0,1]");
    assert_complex_approx(m[[0, 2]], zero, "SWAP[0,2]");
    assert_complex_approx(m[[0, 3]], zero, "SWAP[0,3]");

    // |01> -> |10>
    assert_complex_approx(m[[1, 0]], zero, "SWAP[1,0]");
    assert_complex_approx(m[[1, 1]], zero, "SWAP[1,1]");
    assert_complex_approx(m[[1, 2]], one, "SWAP[1,2]");
    assert_complex_approx(m[[1, 3]], zero, "SWAP[1,3]");

    // |10> -> |01>
    assert_complex_approx(m[[2, 0]], zero, "SWAP[2,0]");
    assert_complex_approx(m[[2, 1]], one, "SWAP[2,1]");
    assert_complex_approx(m[[2, 2]], zero, "SWAP[2,2]");
    assert_complex_approx(m[[2, 3]], zero, "SWAP[2,3]");

    // |11> -> |11>
    assert_complex_approx(m[[3, 0]], zero, "SWAP[3,0]");
    assert_complex_approx(m[[3, 1]], zero, "SWAP[3,1]");
    assert_complex_approx(m[[3, 2]], zero, "SWAP[3,2]");
    assert_complex_approx(m[[3, 3]], one, "SWAP[3,3]");
}

#[test]
fn test_rx_gate_matrix() {
    let theta = PI / 3.0;
    let m = Gate::Rx(theta).matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let cos = Complex64::new((theta / 2.0).cos(), 0.0);
    let neg_i_sin = Complex64::new(0.0, -(theta / 2.0).sin());
    assert_complex_approx(m[[0, 0]], cos, "Rx[0,0]");
    assert_complex_approx(m[[0, 1]], neg_i_sin, "Rx[0,1]");
    assert_complex_approx(m[[1, 0]], neg_i_sin, "Rx[1,0]");
    assert_complex_approx(m[[1, 1]], cos, "Rx[1,1]");
}

#[test]
fn test_rx_gate_zero_angle() {
    let m = Gate::Rx(0.0).matrix(2);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_complex_approx(m[[0, 0]], one, "Rx(0)[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Rx(0)[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Rx(0)[1,0]");
    assert_complex_approx(m[[1, 1]], one, "Rx(0)[1,1]");
}

#[test]
fn test_ry_gate_matrix() {
    let theta = PI / 4.0;
    let m = Gate::Ry(theta).matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let cos = Complex64::new((theta / 2.0).cos(), 0.0);
    let sin = Complex64::new((theta / 2.0).sin(), 0.0);
    let neg_sin = Complex64::new(-(theta / 2.0).sin(), 0.0);
    assert_complex_approx(m[[0, 0]], cos, "Ry[0,0]");
    assert_complex_approx(m[[0, 1]], neg_sin, "Ry[0,1]");
    assert_complex_approx(m[[1, 0]], sin, "Ry[1,0]");
    assert_complex_approx(m[[1, 1]], cos, "Ry[1,1]");
}

#[test]
fn test_ry_gate_zero_angle() {
    let m = Gate::Ry(0.0).matrix(2);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_complex_approx(m[[0, 0]], one, "Ry(0)[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Ry(0)[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Ry(0)[1,0]");
    assert_complex_approx(m[[1, 1]], one, "Ry(0)[1,1]");
}

#[test]
fn test_rz_gate_matrix() {
    let theta = PI / 6.0;
    let m = Gate::Rz(theta).matrix(2);
    assert_eq!(m.dim(), (2, 2));
    let zero = Complex64::new(0.0, 0.0);
    let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
    assert_complex_approx(m[[0, 0]], phase_neg, "Rz[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Rz[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Rz[1,0]");
    assert_complex_approx(m[[1, 1]], phase_pos, "Rz[1,1]");
}

#[test]
fn test_rz_gate_zero_angle() {
    let m = Gate::Rz(0.0).matrix(2);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_complex_approx(m[[0, 0]], one, "Rz(0)[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Rz(0)[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Rz(0)[1,0]");
    assert_complex_approx(m[[1, 1]], one, "Rz(0)[1,1]");
}

// ============================================================
// Custom gate tests
// ============================================================

#[test]
fn test_custom_gate_matrix_passthrough() {
    let one = Complex64::new(1.0, 0.0);
    let custom_matrix = Array2::from_shape_vec(
        (2, 2),
        vec![one, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    )
    .unwrap();

    let gate = Gate::Custom {
        matrix: custom_matrix.clone(),
        is_diagonal: true,
    };

    let result = gate.matrix(2);
    assert_eq!(result, custom_matrix);

    // Custom gate should work with any d value (no panic)
    let result2 = gate.matrix(3);
    assert_eq!(result2, custom_matrix);
}

#[test]
fn test_custom_gate_larger_matrix() {
    let one = Complex64::new(1.0, 0.0);

    // 4x4 identity-like custom gate
    let mut custom_matrix = Array2::zeros((4, 4));
    custom_matrix[[0, 0]] = one;
    custom_matrix[[1, 1]] = one;
    custom_matrix[[2, 2]] = one;
    custom_matrix[[3, 3]] = one;

    let gate = Gate::Custom {
        matrix: custom_matrix.clone(),
        is_diagonal: false,
    };

    let result = gate.matrix(2);
    assert_eq!(result, custom_matrix);
}

// ============================================================
// Wrong dimension panic tests
// ============================================================

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_x_gate_wrong_dimension() {
    Gate::X.matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_y_gate_wrong_dimension() {
    Gate::Y.matrix(4);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_z_gate_wrong_dimension() {
    Gate::Z.matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_h_gate_wrong_dimension() {
    Gate::H.matrix(5);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_s_gate_wrong_dimension() {
    Gate::S.matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_t_gate_wrong_dimension() {
    Gate::T.matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_swap_gate_wrong_dimension() {
    Gate::SWAP.matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_rx_gate_wrong_dimension() {
    Gate::Rx(1.0).matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_ry_gate_wrong_dimension() {
    Gate::Ry(1.0).matrix(3);
}

#[test]
#[should_panic(expected = "Named gates only support d=2")]
fn test_rz_gate_wrong_dimension() {
    Gate::Rz(1.0).matrix(3);
}

// ============================================================
// num_sites tests
// ============================================================

#[test]
fn test_num_sites_single_qubit_gates() {
    assert_eq!(Gate::X.num_sites(2), 1);
    assert_eq!(Gate::Y.num_sites(2), 1);
    assert_eq!(Gate::Z.num_sites(2), 1);
    assert_eq!(Gate::H.num_sites(2), 1);
    assert_eq!(Gate::S.num_sites(2), 1);
    assert_eq!(Gate::T.num_sites(2), 1);
    assert_eq!(Gate::Rx(1.0).num_sites(2), 1);
    assert_eq!(Gate::Ry(1.0).num_sites(2), 1);
    assert_eq!(Gate::Rz(1.0).num_sites(2), 1);
}

#[test]
fn test_num_sites_swap() {
    assert_eq!(Gate::SWAP.num_sites(2), 2);
}

#[test]
fn test_num_sites_custom_2x2() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let m = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    assert_eq!(gate.num_sites(2), 1);
}

#[test]
fn test_num_sites_custom_4x4_d2() {
    let one = Complex64::new(1.0, 0.0);
    let mut m = Array2::zeros((4, 4));
    m[[0, 0]] = one;
    m[[1, 1]] = one;
    m[[2, 2]] = one;
    m[[3, 3]] = one;
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    assert_eq!(gate.num_sites(2), 2);
}

#[test]
fn test_num_sites_custom_8x8_d2() {
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((8, 8));
    for i in 0..8 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: false,
    };
    assert_eq!(gate.num_sites(2), 3);
}

#[test]
fn test_num_sites_custom_9x9_d3() {
    // 9 = 3^2, so 2 sites with d=3
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((9, 9));
    for i in 0..9 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    assert_eq!(gate.num_sites(3), 2);
}

// ============================================================
// is_diagonal tests
// ============================================================

#[test]
fn test_is_diagonal_true_gates() {
    assert!(Gate::Z.is_diagonal());
    assert!(Gate::S.is_diagonal());
    assert!(Gate::T.is_diagonal());
    assert!(Gate::Rz(1.0).is_diagonal());
    assert!(Gate::Rz(0.0).is_diagonal());
    assert!(Gate::Rz(PI).is_diagonal());
}

#[test]
fn test_is_diagonal_false_gates() {
    assert!(!Gate::X.is_diagonal());
    assert!(!Gate::Y.is_diagonal());
    assert!(!Gate::H.is_diagonal());
    assert!(!Gate::SWAP.is_diagonal());
    assert!(!Gate::Rx(1.0).is_diagonal());
    assert!(!Gate::Ry(1.0).is_diagonal());
}

#[test]
fn test_is_diagonal_custom_true() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let m = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    assert!(gate.is_diagonal());
}

#[test]
fn test_is_diagonal_custom_false() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let m = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: false,
    };
    assert!(!gate.is_diagonal());
}

// ============================================================
// Additional edge case tests
// ============================================================

#[test]
fn test_rx_pi_gives_minus_i_times_x() {
    // Rx(pi) = [[0, -i], [-i, 0]]
    let m = Gate::Rx(PI).matrix(2);
    let zero = Complex64::new(0.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    assert_complex_approx(m[[0, 0]], zero, "Rx(pi)[0,0]");
    assert_complex_approx(m[[0, 1]], neg_i, "Rx(pi)[0,1]");
    assert_complex_approx(m[[1, 0]], neg_i, "Rx(pi)[1,0]");
    assert_complex_approx(m[[1, 1]], zero, "Rx(pi)[1,1]");
}

#[test]
fn test_ry_pi_gives_rotation() {
    // Ry(pi) = [[0, -1], [1, 0]]
    let m = Gate::Ry(PI).matrix(2);
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    assert_complex_approx(m[[0, 0]], zero, "Ry(pi)[0,0]");
    assert_complex_approx(m[[0, 1]], neg_one, "Ry(pi)[0,1]");
    assert_complex_approx(m[[1, 0]], one, "Ry(pi)[1,0]");
    assert_complex_approx(m[[1, 1]], zero, "Ry(pi)[1,1]");
}

#[test]
fn test_rz_pi_gives_z_up_to_phase() {
    // Rz(pi) = [[e^{-i*pi/2}, 0], [0, e^{i*pi/2}]] = [[-i, 0], [0, i]]
    let m = Gate::Rz(PI).matrix(2);
    let zero = Complex64::new(0.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let i = Complex64::new(0.0, 1.0);
    assert_complex_approx(m[[0, 0]], neg_i, "Rz(pi)[0,0]");
    assert_complex_approx(m[[0, 1]], zero, "Rz(pi)[0,1]");
    assert_complex_approx(m[[1, 0]], zero, "Rz(pi)[1,0]");
    assert_complex_approx(m[[1, 1]], i, "Rz(pi)[1,1]");
}

#[test]
fn test_h_squared_is_identity() {
    // H^2 = I
    let h = Gate::H.matrix(2);
    let h2 = h.dot(&h);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_complex_approx(h2[[0, 0]], one, "H^2[0,0]");
    assert_complex_approx(h2[[0, 1]], zero, "H^2[0,1]");
    assert_complex_approx(h2[[1, 0]], zero, "H^2[1,0]");
    assert_complex_approx(h2[[1, 1]], one, "H^2[1,1]");
}

#[test]
fn test_x_squared_is_identity() {
    let x = Gate::X.matrix(2);
    let x2 = x.dot(&x);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_complex_approx(x2[[0, 0]], one, "X^2[0,0]");
    assert_complex_approx(x2[[0, 1]], zero, "X^2[0,1]");
    assert_complex_approx(x2[[1, 0]], zero, "X^2[1,0]");
    assert_complex_approx(x2[[1, 1]], one, "X^2[1,1]");
}

#[test]
fn test_s_squared_is_z() {
    // S^2 = Z
    let s = Gate::S.matrix(2);
    let s2 = s.dot(&s);
    let z = Gate::Z.matrix(2);
    for i in 0..2 {
        for j in 0..2 {
            assert_complex_approx(s2[[i, j]], z[[i, j]], &format!("S^2[{},{}]", i, j));
        }
    }
}

#[test]
fn test_t_squared_is_s() {
    // T^2 = S
    let t = Gate::T.matrix(2);
    let t2 = t.dot(&t);
    let s = Gate::S.matrix(2);
    for i in 0..2 {
        for j in 0..2 {
            assert_complex_approx(t2[[i, j]], s[[i, j]], &format!("T^2[{},{}]", i, j));
        }
    }
}
