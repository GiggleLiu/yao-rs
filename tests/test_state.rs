use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use yao_rs::state::State;

#[test]
fn test_zero_state_qubits() {
    let state = State::zero_state(&[2, 2]);
    assert_eq!(state.total_dim(), 4);
    assert_eq!(state.data[0], Complex64::new(1.0, 0.0));
    for i in 1..4 {
        assert_eq!(state.data[i], Complex64::new(0.0, 0.0));
    }
}

#[test]
fn test_zero_state_qutrits() {
    let state = State::zero_state(&[3, 3]);
    assert_eq!(state.total_dim(), 9);
    assert_eq!(state.data[0], Complex64::new(1.0, 0.0));
    for i in 1..9 {
        assert_eq!(state.data[i], Complex64::new(0.0, 0.0));
    }
}

#[test]
fn test_product_state_qubits_1_0() {
    // |1,0> on 2 qubits: index = 1*2 + 0 = 2
    let state = State::product_state(&[2, 2], &[1, 0]);
    assert_eq!(state.total_dim(), 4);
    assert_eq!(state.data[2], Complex64::new(1.0, 0.0));
    // All other elements should be zero
    for i in [0, 1, 3] {
        assert_eq!(state.data[i], Complex64::new(0.0, 0.0));
    }
}

#[test]
fn test_product_state_qutrits_2_1() {
    // |2,1> on 2 qutrits: index = 2*3 + 1 = 7
    let state = State::product_state(&[3, 3], &[2, 1]);
    assert_eq!(state.total_dim(), 9);
    assert_eq!(state.data[7], Complex64::new(1.0, 0.0));
    // All other elements should be zero
    for i in 0..9 {
        if i != 7 {
            assert_eq!(state.data[i], Complex64::new(0.0, 0.0));
        }
    }
}

#[test]
fn test_norm_zero_state() {
    let state = State::zero_state(&[2, 2]);
    assert_abs_diff_eq!(state.norm(), 1.0, epsilon = 1e-12);
}

#[test]
fn test_norm_product_state() {
    let state = State::product_state(&[2, 3], &[1, 2]);
    assert_abs_diff_eq!(state.norm(), 1.0, epsilon = 1e-12);
}

#[test]
fn test_dims_preserved() {
    let state = State::zero_state(&[2, 3, 4]);
    assert_eq!(state.dims, vec![2, 3, 4]);
    assert_eq!(state.total_dim(), 24);
}

#[test]
#[should_panic(expected = "dims and levels must have the same length")]
fn test_product_state_mismatched_lengths() {
    State::product_state(&[2, 2], &[0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn test_product_state_level_out_of_range() {
    State::product_state(&[2, 2], &[2, 0]);
}
