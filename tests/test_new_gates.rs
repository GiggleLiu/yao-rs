use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use yao_rs::gate::Gate;

/// Helper: check two complex numbers are approximately equal.
fn assert_complex_eq(a: Complex64, b: Complex64, _msg: &str) {
    assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
    assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-10);
}

/// Helper: check that an NxN matrix is unitary (M^dag * M = I).
fn assert_unitary(m: &ndarray::Array2<Complex64>, n: usize) {
    // Compute M^dag * M
    let m_dag = m.t().mapv(|c| c.conj());
    let product = m_dag.dot(m);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert_abs_diff_eq!(product[[i, j]].re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(product[[i, j]].im, expected.im, epsilon = 1e-10);
        }
    }
}

// ============================================================
// Test 1: SqrtX matrix values
// ============================================================

#[test]
fn test_sqrtx_matrix_values() {
    let m = Gate::SqrtX.matrix(2);
    assert_eq!(m.dim(), (2, 2));

    // Expected: (1+i)/2 * [[1, -i], [-i, 1]]
    let f = Complex64::new(0.5, 0.5); // (1+i)/2
    let one = Complex64::new(1.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);

    assert_complex_eq(m[[0, 0]], f * one, "SqrtX[0,0]");
    assert_complex_eq(m[[0, 1]], f * neg_i, "SqrtX[0,1]");
    assert_complex_eq(m[[1, 0]], f * neg_i, "SqrtX[1,0]");
    assert_complex_eq(m[[1, 1]], f * one, "SqrtX[1,1]");
}

// ============================================================
// Test 2: SqrtX^2 = X
// ============================================================

#[test]
fn test_sqrtx_squared_is_x() {
    let sqrtx = Gate::SqrtX.matrix(2);
    let sqrtx2 = sqrtx.dot(&sqrtx);
    let x = Gate::X.matrix(2);

    for i in 0..2 {
        for j in 0..2 {
            assert_complex_eq(sqrtx2[[i, j]], x[[i, j]], &format!("SqrtX^2[{},{}]", i, j));
        }
    }
}

// ============================================================
// Test 3: SqrtY^2 = Y
// ============================================================

#[test]
fn test_sqrty_squared_is_y() {
    let sqrty = Gate::SqrtY.matrix(2);
    let sqrty2 = sqrty.dot(&sqrty);
    let y = Gate::Y.matrix(2);

    for i in 0..2 {
        for j in 0..2 {
            assert_complex_eq(sqrty2[[i, j]], y[[i, j]], &format!("SqrtY^2[{},{}]", i, j));
        }
    }
}

// ============================================================
// Test 4: SqrtW is unitary
// ============================================================

#[test]
fn test_sqrtw_is_unitary() {
    let m = Gate::SqrtW.matrix(2);
    assert_unitary(&m, 2);
}

// ============================================================
// Test 5: ISWAP matrix values
// ============================================================

#[test]
fn test_iswap_matrix_values() {
    let m = Gate::ISWAP.matrix(2);
    assert_eq!(m.dim(), (4, 4));

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);

    // [0,0] = 1
    assert_complex_eq(m[[0, 0]], one, "ISWAP[0,0]");
    // [3,3] = 1
    assert_complex_eq(m[[3, 3]], one, "ISWAP[3,3]");
    // [1,2] = i
    assert_complex_eq(m[[1, 2]], i, "ISWAP[1,2]");
    // [2,1] = i
    assert_complex_eq(m[[2, 1]], i, "ISWAP[2,1]");

    // All other entries should be zero
    assert_complex_eq(m[[0, 1]], zero, "ISWAP[0,1]");
    assert_complex_eq(m[[0, 2]], zero, "ISWAP[0,2]");
    assert_complex_eq(m[[0, 3]], zero, "ISWAP[0,3]");
    assert_complex_eq(m[[1, 0]], zero, "ISWAP[1,0]");
    assert_complex_eq(m[[1, 1]], zero, "ISWAP[1,1]");
    assert_complex_eq(m[[1, 3]], zero, "ISWAP[1,3]");
    assert_complex_eq(m[[2, 0]], zero, "ISWAP[2,0]");
    assert_complex_eq(m[[2, 2]], zero, "ISWAP[2,2]");
    assert_complex_eq(m[[2, 3]], zero, "ISWAP[2,3]");
    assert_complex_eq(m[[3, 0]], zero, "ISWAP[3,0]");
    assert_complex_eq(m[[3, 1]], zero, "ISWAP[3,1]");
    assert_complex_eq(m[[3, 2]], zero, "ISWAP[3,2]");
}

// ============================================================
// Test 6: ISWAP is unitary
// ============================================================

#[test]
fn test_iswap_is_unitary() {
    let m = Gate::ISWAP.matrix(2);
    assert_unitary(&m, 4);
}

// ============================================================
// Test 9: num_sites for new gates (Task 2)
// ============================================================

#[test]
fn test_num_sites_new_gates() {
    assert_eq!(Gate::SqrtX.num_sites(2), 1);
    assert_eq!(Gate::SqrtY.num_sites(2), 1);
    assert_eq!(Gate::SqrtW.num_sites(2), 1);
    assert_eq!(Gate::ISWAP.num_sites(2), 2);
}

// ============================================================
// Test 10: is_diagonal for new gates (all false, Task 2)
// ============================================================

#[test]
fn test_is_diagonal_new_gates() {
    assert!(!Gate::SqrtX.is_diagonal());
    assert!(!Gate::SqrtY.is_diagonal());
    assert!(!Gate::SqrtW.is_diagonal());
    assert!(!Gate::ISWAP.is_diagonal());
}
