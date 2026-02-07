use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI};

use yao_rs::gate::Gate;
use yao_rs::instruct::{
    instruct_controlled, instruct_diagonal, instruct_single, mulrow, u1rows, udrows,
};
use yao_rs::state::State;

fn approx_eq(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() < 1e-10
}

#[test]
fn test_u1rows_hadamard_on_zero() {
    // H|0⟩ = (|0⟩ + |1⟩) / √2
    let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    u1rows(&mut state, 0, 1, &h_gate);

    assert!(approx_eq(state[0], s));
    assert!(approx_eq(state[1], s));
}

#[test]
fn test_u1rows_hadamard_on_one() {
    // H|1⟩ = (|0⟩ - |1⟩) / √2
    let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    u1rows(&mut state, 0, 1, &h_gate);

    assert!(approx_eq(state[0], s));
    assert!(approx_eq(state[1], -s));
}

#[test]
fn test_u1rows_x_gate() {
    // X|0⟩ = |1⟩
    let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    u1rows(&mut state, 0, 1, &x_gate);

    assert!(approx_eq(state[0], zero));
    assert!(approx_eq(state[1], one));
}

#[test]
fn test_u1rows_x_gate_on_one() {
    // X|1⟩ = |0⟩
    let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    u1rows(&mut state, 0, 1, &x_gate);

    assert!(approx_eq(state[0], one));
    assert!(approx_eq(state[1], zero));
}

#[test]
fn test_u1rows_non_contiguous_indices() {
    // Test with non-contiguous indices in a 4-element state vector
    // This simulates applying a single-qubit gate to qubit 0 in a 2-qubit system
    // where indices 0 and 2 correspond to the qubit-0 basis states
    let mut state = vec![
        Complex64::new(1.0, 0.0), // |00⟩
        Complex64::new(0.0, 0.0), // |01⟩
        Complex64::new(0.0, 0.0), // |10⟩
        Complex64::new(0.0, 0.0), // |11⟩
    ];

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    // Apply H to qubit 0: indices 0 (|00⟩) and 2 (|10⟩)
    u1rows(&mut state, 0, 2, &h_gate);

    // Result: (|00⟩ + |10⟩) / √2
    assert!(approx_eq(state[0], s));
    assert!(approx_eq(state[1], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state[2], s));
    assert!(approx_eq(state[3], Complex64::new(0.0, 0.0)));
}

#[test]
fn test_udrows_qutrit_cyclic_permutation() {
    // Test with a 3x3 cyclic permutation on a qutrit
    // |0⟩ -> |1⟩ -> |2⟩ -> |0⟩
    let mut state = vec![
        Complex64::new(1.0, 0.0), // |0⟩
        Complex64::new(0.0, 0.0), // |1⟩
        Complex64::new(0.0, 0.0), // |2⟩
    ];

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Cyclic permutation matrix: P|k⟩ = |k+1 mod 3⟩
    // P = [[0,0,1], [1,0,0], [0,1,0]]
    let perm = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    udrows(&mut state, &[0, 1, 2], &perm);

    // |0⟩ -> |1⟩
    assert!(approx_eq(state[0], zero));
    assert!(approx_eq(state[1], one));
    assert!(approx_eq(state[2], zero));
}

#[test]
fn test_udrows_qutrit_superposition() {
    // Start with |0⟩, apply a unitary that creates superposition
    let mut state = vec![
        Complex64::new(1.0, 0.0), // |0⟩
        Complex64::new(0.0, 0.0), // |1⟩
        Complex64::new(0.0, 0.0), // |2⟩
    ];

    // Create a unitary that maps |0⟩ to (|0⟩ + |1⟩ + |2⟩)/√3
    let s = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
    // First column is [s, s, s]^T
    // We need a full unitary; use a simple one
    let gate = Array2::from_shape_vec(
        (3, 3),
        vec![
            s,
            s,
            s,
            s,
            s * Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0),
            s * Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 3.0),
            s,
            s * Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 3.0),
            s * Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0),
        ],
    )
    .unwrap();

    udrows(&mut state, &[0, 1, 2], &gate);

    // Check that |0⟩ component has amplitude s
    assert!(approx_eq(state[0], s));
    assert!(approx_eq(state[1], s));
    assert!(approx_eq(state[2], s));
}

#[test]
fn test_udrows_4x4_gate() {
    // Test with a 4x4 gate (e.g., SWAP-like operation)
    let mut state = vec![
        Complex64::new(1.0, 0.0), // index 0
        Complex64::new(0.0, 0.0), // index 1
        Complex64::new(0.0, 0.0), // index 2
        Complex64::new(0.0, 0.0), // index 3
    ];

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Permutation that swaps 0<->3 and 1<->2
    let gate = Array2::from_shape_vec(
        (4, 4),
        vec![
            zero, zero, zero, one, zero, zero, one, zero, zero, one, zero, zero, one, zero, zero,
            zero,
        ],
    )
    .unwrap();

    udrows(&mut state, &[0, 1, 2, 3], &gate);

    assert!(approx_eq(state[0], zero));
    assert!(approx_eq(state[1], zero));
    assert!(approx_eq(state[2], zero));
    assert!(approx_eq(state[3], one));
}

#[test]
fn test_udrows_non_contiguous_indices() {
    // Test udrows with non-contiguous indices
    let mut state = vec![
        Complex64::new(1.0, 0.0), // index 0
        Complex64::new(0.5, 0.0), // index 1
        Complex64::new(0.0, 0.0), // index 2
        Complex64::new(0.5, 0.0), // index 3
    ];

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Apply X gate to indices 0 and 2 only
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    udrows(&mut state, &[0, 2], &x_gate);

    // state[0] and state[2] should be swapped
    assert!(approx_eq(state[0], zero));
    assert!(approx_eq(state[1], Complex64::new(0.5, 0.0))); // unchanged
    assert!(approx_eq(state[2], one));
    assert!(approx_eq(state[3], Complex64::new(0.5, 0.0))); // unchanged
}

#[test]
fn test_mulrow_phase() {
    // Apply phase e^(iπ/4) to an amplitude
    let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

    let phase = Complex64::from_polar(1.0, FRAC_PI_4);
    mulrow(&mut state, 1, phase);

    assert!(approx_eq(state[0], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state[1], phase));
}

#[test]
fn test_mulrow_z_gate_diagonal() {
    // Z gate on |1⟩ component: multiply by -1
    let mut state = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];

    let neg_one = Complex64::new(-1.0, 0.0);
    mulrow(&mut state, 1, neg_one);

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert!(approx_eq(state[0], s));
    assert!(approx_eq(state[1], -s));
}

#[test]
fn test_mulrow_s_gate() {
    // S gate diagonal element: multiply by i
    let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

    let i = Complex64::new(0.0, 1.0);
    mulrow(&mut state, 1, i);

    assert!(approx_eq(state[0], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state[1], i));
}

#[test]
fn test_mulrow_preserves_other_amplitudes() {
    // Ensure mulrow only affects the specified index
    let mut state = vec![
        Complex64::new(0.5, 0.5),
        Complex64::new(0.3, 0.4),
        Complex64::new(0.1, 0.2),
        Complex64::new(0.6, 0.7),
    ];

    let original = state.clone();
    let phase = Complex64::from_polar(1.0, FRAC_PI_4);
    mulrow(&mut state, 2, phase);

    assert!(approx_eq(state[0], original[0]));
    assert!(approx_eq(state[1], original[1]));
    assert!(approx_eq(state[2], original[2] * phase));
    assert!(approx_eq(state[3], original[3]));
}

#[test]
fn test_u1rows_preserves_normalization() {
    // Unitary operations should preserve norm
    let mut state = vec![Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)];

    let norm_before: f64 = state.iter().map(|c| c.norm_sqr()).sum();

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    u1rows(&mut state, 0, 1, &h_gate);

    let norm_after: f64 = state.iter().map(|c| c.norm_sqr()).sum();

    assert!((norm_before - norm_after).abs() < 1e-10);
}

// Tests for instruct_single and instruct_diagonal

#[test]
fn test_instruct_single_h_gate() {
    // Apply H gate to qubit 0 in a 2-qubit system |00⟩
    // H|0⟩ = (|0⟩ + |1⟩) / √2
    // Result: (|00⟩ + |10⟩) / √2
    let mut state = State::zero_state(&[2, 2]);

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    instruct_single(&mut state, &h_gate, 0);

    // |00⟩ = index 0, |01⟩ = index 1, |10⟩ = index 2, |11⟩ = index 3
    assert!(approx_eq(state.data[0], s)); // |00⟩
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0))); // |01⟩
    assert!(approx_eq(state.data[2], s)); // |10⟩
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0))); // |11⟩
}

#[test]
fn test_instruct_single_x_gate() {
    // Apply X gate to qubit 1 in a 2-qubit system |00⟩
    // X on qubit 1: |00⟩ -> |01⟩
    let mut state = State::zero_state(&[2, 2]);

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    instruct_single(&mut state, &x_gate, 1);

    // |00⟩ = index 0, |01⟩ = index 1, |10⟩ = index 2, |11⟩ = index 3
    assert!(approx_eq(state.data[0], zero)); // |00⟩
    assert!(approx_eq(state.data[1], one)); // |01⟩
    assert!(approx_eq(state.data[2], zero)); // |10⟩
    assert!(approx_eq(state.data[3], zero)); // |11⟩
}

#[test]
fn test_instruct_single_qutrit() {
    // Apply X-like (cyclic shift) gate on a qutrit (d=3)
    // |0⟩ -> |1⟩ -> |2⟩ -> |0⟩
    let mut state = State::zero_state(&[3]); // Single qutrit |0⟩

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Cyclic permutation: X|k⟩ = |k+1 mod 3⟩
    // Matrix representation: [[0,0,1], [1,0,0], [0,1,0]]
    let x_qutrit = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    instruct_single(&mut state, &x_qutrit, 0);

    // |0⟩ -> |1⟩
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));
    assert!(approx_eq(state.data[2], zero));
}

#[test]
fn test_instruct_diagonal_z_gate() {
    // Apply Z gate: Z = diag(1, -1)
    // Start with |+⟩ = (|0⟩ + |1⟩) / √2
    // Z|+⟩ = (|0⟩ - |1⟩) / √2 = |-⟩
    let mut state = State::zero_state(&[2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Manually set to |+⟩ state
    state.data[0] = s;
    state.data[1] = s;

    let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    instruct_diagonal(&mut state, &z_phases, 0);

    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], -s));
}

#[test]
fn test_instruct_diagonal_phase() {
    // Apply Phase(π/4) gate: P(θ) = diag(1, e^(iθ))
    // On |+⟩ state
    let mut state = State::zero_state(&[2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Manually set to |+⟩ state
    state.data[0] = s;
    state.data[1] = s;

    let phase = Complex64::from_polar(1.0, FRAC_PI_4);
    let p_phases = [Complex64::new(1.0, 0.0), phase];
    instruct_diagonal(&mut state, &p_phases, 0);

    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s * phase));
}

#[test]
fn test_instruct_single_preserves_normalization() {
    // Verify that applying a unitary preserves the norm
    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Start with some superposition
    state.data[0] = s;
    state.data[2] = s;

    let norm_before = state.norm();

    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    instruct_single(&mut state, &h_gate, 1);

    let norm_after = state.norm();

    assert!((norm_before - norm_after).abs() < 1e-10);
}

#[test]
fn test_instruct_diagonal_on_multi_qubit() {
    // Apply Z to qubit 0 in a 2-qubit system
    // Start with |+0⟩ = (|00⟩ + |10⟩) / √2
    // Z on qubit 0: |00⟩ -> |00⟩, |10⟩ -> -|10⟩
    // Result: (|00⟩ - |10⟩) / √2
    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    state.data[0] = s; // |00⟩
    state.data[2] = s; // |10⟩

    let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    instruct_diagonal(&mut state, &z_phases, 0);

    assert!(approx_eq(state.data[0], s)); // |00⟩ unchanged
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0))); // |01⟩ still zero
    assert!(approx_eq(state.data[2], -s)); // |10⟩ flipped sign
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0))); // |11⟩ still zero
}

// Tests for instruct_controlled

#[test]
fn test_instruct_controlled_cnot() {
    // CNOT: control=0, target=1
    // |00⟩ -> |00⟩, |01⟩ -> |01⟩, |10⟩ -> |11⟩, |11⟩ -> |10⟩

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    // Test |10⟩ -> |11⟩
    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

    assert!(approx_eq(state.data[0], zero)); // |00⟩
    assert!(approx_eq(state.data[1], zero)); // |01⟩
    assert!(approx_eq(state.data[2], zero)); // |10⟩
    assert!(approx_eq(state.data[3], one)); // |11⟩

    // Test |00⟩ -> |00⟩ (control not active)
    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

    assert!(approx_eq(state.data[0], one)); // |00⟩ unchanged
    assert!(approx_eq(state.data[1], zero)); // |01⟩
    assert!(approx_eq(state.data[2], zero)); // |10⟩
    assert!(approx_eq(state.data[3], zero)); // |11⟩

    // Test |11⟩ -> |10⟩
    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

    assert!(approx_eq(state.data[0], zero)); // |00⟩
    assert!(approx_eq(state.data[1], zero)); // |01⟩
    assert!(approx_eq(state.data[2], one)); // |10⟩
    assert!(approx_eq(state.data[3], zero)); // |11⟩
}

#[test]
fn test_instruct_controlled_toffoli() {
    // Toffoli (CCX): two controls (sites 0 and 1), target (site 2)
    // Only flips target when both controls are |1⟩

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    // Test |110⟩ -> |111⟩ (both controls active)
    // dims=[2,2,2]: |110⟩ = 1*4 + 1*2 + 0 = 6
    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

    assert!(approx_eq(state.data[6], zero)); // |110⟩ -> 0
    assert!(approx_eq(state.data[7], one)); // |111⟩ -> 1

    // Test |100⟩ -> |100⟩ (only one control active)
    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

    assert!(approx_eq(state.data[4], one)); // |100⟩ unchanged
    assert!(approx_eq(state.data[5], zero)); // |101⟩

    // Test |010⟩ -> |010⟩ (only one control active)
    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

    assert!(approx_eq(state.data[2], one)); // |010⟩ unchanged
    assert!(approx_eq(state.data[3], zero)); // |011⟩
}

#[test]
fn test_instruct_controlled_qutrit() {
    // Control on a qutrit with value = 2
    // dims=[3, 2]: qutrit (control) and qubit (target)

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    // Test |20⟩ -> |21⟩ (control value = 2 is active)
    // dims=[3,2]: |20⟩ = 2*2 + 0 = 4
    let mut state = State::product_state(&[3, 2], &[2, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

    assert!(approx_eq(state.data[4], zero)); // |20⟩
    assert!(approx_eq(state.data[5], one)); // |21⟩

    // Test |10⟩ -> |10⟩ (control value = 1, not active)
    let mut state = State::product_state(&[3, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

    assert!(approx_eq(state.data[2], one)); // |10⟩ unchanged
    assert!(approx_eq(state.data[3], zero)); // |11⟩

    // Test |00⟩ -> |00⟩ (control value = 0, not active)
    let mut state = State::product_state(&[3, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

    assert!(approx_eq(state.data[0], one)); // |00⟩ unchanged
    assert!(approx_eq(state.data[1], zero)); // |01⟩
}

#[test]
fn test_instruct_controlled_no_controls() {
    // Empty ctrl_locs should behave like instruct_single

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    // Test with instruct_controlled (no controls)
    let mut state1 = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state1, &x_gate, &[], &[], &[1]);

    // Test with instruct_single for comparison
    let mut state2 = State::product_state(&[2, 2], &[0, 0]);
    instruct_single(&mut state2, &x_gate, 1);

    // Both should give |01⟩
    for i in 0..4 {
        assert!(approx_eq(state1.data[i], state2.data[i]));
    }
}

#[test]
fn test_instruct_controlled_preserves_normalization() {
    // Controlled unitary should preserve norm
    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Start with superposition (|00⟩ + |10⟩) / √2
    state.data[0] = s;
    state.data[2] = s;

    let norm_before = state.norm();

    let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();

    // Apply controlled-H: only affects |10⟩ and |11⟩
    instruct_controlled(&mut state, &h_gate, &[0], &[1], &[1]);

    let norm_after = state.norm();

    assert!((norm_before - norm_after).abs() < 1e-10);
}

#[test]
fn test_instruct_controlled_superposition() {
    // Test CNOT on a superposition state
    // Start with (|00⟩ + |10⟩) / √2 = |+0⟩
    // CNOT: (|00⟩ + |11⟩) / √2 (Bell state)

    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    state.data[0] = s; // |00⟩
    state.data[2] = s; // |10⟩

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

    // Result: (|00⟩ + |11⟩) / √2
    assert!(approx_eq(state.data[0], s)); // |00⟩
    assert!(approx_eq(state.data[1], zero)); // |01⟩
    assert!(approx_eq(state.data[2], zero)); // |10⟩ -> |11⟩
    assert!(approx_eq(state.data[3], s)); // |11⟩
}

// Tests for Pauli gate instructions

#[test]
fn test_instruct_pauli_x() {
    // X|0⟩ = |1⟩ on 2-qubit system
    // Test on qubit 0
    let mut state = State::zero_state(&[2, 2]); // |00⟩
    let x_gate = Gate::X.matrix(2);
    instruct_single(&mut state, &x_gate, 0);
    // Result: |10⟩ = index 2
    assert!(approx_eq(state.data[0], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[2], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0)));

    // Test on qubit 1
    let mut state = State::zero_state(&[2, 2]); // |00⟩
    instruct_single(&mut state, &x_gate, 1);
    // Result: |01⟩ = index 1
    assert!(approx_eq(state.data[0], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state.data[2], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0)));

    // X|1⟩ = |0⟩
    let mut state = State::product_state(&[2, 2], &[1, 0]); // |10⟩
    instruct_single(&mut state, &x_gate, 0);
    // Result: |00⟩ = index 0
    assert!(approx_eq(state.data[0], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[2], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0)));
}

#[test]
fn test_instruct_pauli_y() {
    // Y|0⟩ = i|1⟩
    let mut state = State::zero_state(&[2]); // |0⟩
    let y_gate = Gate::Y.matrix(2);
    instruct_single(&mut state, &y_gate, 0);
    // Result: i|1⟩
    assert!(approx_eq(state.data[0], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 1.0)));

    // Y|1⟩ = -i|0⟩
    let mut state = State::product_state(&[2], &[1]); // |1⟩
    instruct_single(&mut state, &y_gate, 0);
    // Result: -i|0⟩
    assert!(approx_eq(state.data[0], Complex64::new(0.0, -1.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0)));
}

#[test]
fn test_instruct_pauli_z() {
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    // Use instruct_diagonal since Z is diagonal: Z = diag(1, -1)
    let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

    // Test on |0⟩ state
    let mut state = State::zero_state(&[2]); // |0⟩
    instruct_diagonal(&mut state, &z_phases, 0);
    // Result: |0⟩ (unchanged)
    assert!(approx_eq(state.data[0], Complex64::new(1.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0)));

    // Test on |1⟩ state
    let mut state = State::product_state(&[2], &[1]); // |1⟩
    instruct_diagonal(&mut state, &z_phases, 0);
    // Result: -|1⟩
    assert!(approx_eq(state.data[0], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(-1.0, 0.0)));

    // Test Z on qubit 0 in 2-qubit system with |10⟩
    let mut state = State::product_state(&[2, 2], &[1, 0]); // |10⟩
    instruct_diagonal(&mut state, &z_phases, 0);
    // qubit 0 is in |1⟩, so multiply by -1
    // Result: -|10⟩
    assert!(approx_eq(state.data[0], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0)));
    assert!(approx_eq(state.data[2], Complex64::new(-1.0, 0.0)));
    assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0)));
}

#[test]
fn test_instruct_pauli_on_superposition() {
    // Create |+⟩ = (|0⟩ + |1⟩)/√2
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;

    // Apply X: X|+⟩ = X(|0⟩ + |1⟩)/√2 = (|1⟩ + |0⟩)/√2 = |+⟩
    let x_gate = Gate::X.matrix(2);
    instruct_single(&mut state, &x_gate, 0);
    // Result: |+⟩ (same state, amplitudes swapped but still equal)
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));

    // Reset to |+⟩
    state.data[0] = s;
    state.data[1] = s;

    // Apply Z: Z|+⟩ = Z(|0⟩ + |1⟩)/√2 = (|0⟩ - |1⟩)/√2 = |-⟩
    let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    instruct_diagonal(&mut state, &z_phases, 0);
    // Result: |-⟩ = (|0⟩ - |1⟩)/√2
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], -s));
}

// Tests for parallel variants
#[cfg(feature = "parallel")]
mod parallel_tests {
    use ndarray::Array2;
    use num_complex::Complex64;
    use std::f64::consts::FRAC_1_SQRT_2;

    use yao_rs::instruct::{
        instruct_diagonal, instruct_diagonal_parallel, instruct_single, instruct_single_parallel,
    };
    use yao_rs::state::State;

    fn approx_eq(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-10
    }

    #[test]
    fn test_instruct_diagonal_parallel_matches_sequential() {
        let mut state_seq = State::zero_state(&[2, 2, 2, 2]);
        let mut state_par = state_seq.clone();
        let amp = Complex64::new(0.25, 0.0);
        for i in 0..16 {
            state_seq.data[i] = amp;
            state_par.data[i] = amp;
        }
        let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        instruct_diagonal(&mut state_seq, &z_phases, 1);
        instruct_diagonal_parallel(&mut state_par, &z_phases, 1);
        for i in 0..16 {
            assert!(approx_eq(state_seq.data[i], state_par.data[i]));
        }
    }

    #[test]
    fn test_instruct_diagonal_parallel_large_state() {
        let mut state_seq = State::zero_state(&[2, 2, 2, 2, 2]);
        let mut state_par = state_seq.clone();
        for i in 0..32 {
            let re = (i as f64 * 0.1).sin();
            let im = (i as f64 * 0.1).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        let phases = [Complex64::new(1.0, 0.0), phase];
        instruct_diagonal(&mut state_seq, &phases, 2);
        instruct_diagonal_parallel(&mut state_par, &phases, 2);
        for i in 0..32 {
            assert!(approx_eq(state_seq.data[i], state_par.data[i]));
        }
    }

    #[test]
    fn test_instruct_single_parallel_matches_sequential() {
        let mut state_seq = State::zero_state(&[2, 2, 2, 2]);
        let mut state_par = state_seq.clone();
        let amp = Complex64::new(0.25, 0.0);
        for i in 0..16 {
            state_seq.data[i] = amp;
            state_par.data[i] = amp;
        }
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();
        instruct_single(&mut state_seq, &h_gate, 1);
        instruct_single_parallel(&mut state_par, &h_gate, 1);
        for i in 0..16 {
            assert!(approx_eq(state_seq.data[i], state_par.data[i]));
        }
    }

    #[test]
    fn test_instruct_single_parallel_large_state() {
        let mut state_seq = State::zero_state(&[2, 2, 2, 2, 2]);
        let mut state_par = state_seq.clone();
        for i in 0..32 {
            let re = (i as f64 * 0.1).sin();
            let im = (i as f64 * 0.1).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let x_gate = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();
        instruct_single(&mut state_seq, &x_gate, 2);
        instruct_single_parallel(&mut state_par, &x_gate, 2);
        for i in 0..32 {
            assert!(approx_eq(state_seq.data[i], state_par.data[i]));
        }
    }

    #[test]
    fn test_instruct_single_parallel_multiple_gates() {
        let mut state_seq = State::zero_state(&[2, 2, 2, 2]);
        let mut state_par = state_seq.clone();
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap();
        for loc in 0..4 {
            instruct_single(&mut state_seq, &h_gate, loc);
            instruct_single_parallel(&mut state_par, &h_gate, loc);
        }
        for i in 0..16 {
            assert!(approx_eq(state_seq.data[i], state_par.data[i]));
        }
    }
}

// Comprehensive parallel verification tests
#[cfg(feature = "parallel")]
mod parallel_comprehensive_tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use yao_rs::gate::Gate;
    use yao_rs::instruct::{
        instruct_diagonal, instruct_diagonal_parallel, instruct_single, instruct_single_parallel,
    };
    use yao_rs::state::State;

    fn approx_eq(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-10
    }

    fn states_match(state_a: &State, state_b: &State) -> bool {
        state_a.data.len() == state_b.data.len()
            && state_a
                .data
                .iter()
                .zip(state_b.data.iter())
                .all(|(&a, &b)| approx_eq(a, b))
    }

    #[test]
    fn test_parallel_all_paulis() {
        let n_qubits = 14;
        let n_amps = 1 << n_qubits;

        let x_gate = Gate::X.matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.001).sin();
            let im = (i as f64 * 0.001).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &x_gate, 7);
        instruct_single_parallel(&mut state_par, &x_gate, 7);
        assert!(
            states_match(&state_seq, &state_par),
            "X gate parallel mismatch"
        );

        let y_gate = Gate::Y.matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.002).cos();
            let im = (i as f64 * 0.002).sin();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &y_gate, 3);
        instruct_single_parallel(&mut state_par, &y_gate, 3);
        assert!(
            states_match(&state_seq, &state_par),
            "Y gate parallel mismatch"
        );

        let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.003).sin();
            let im = (i as f64 * 0.003).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_diagonal(&mut state_seq, &z_phases, 10);
        instruct_diagonal_parallel(&mut state_par, &z_phases, 10);
        assert!(
            states_match(&state_seq, &state_par),
            "Z gate parallel mismatch"
        );
    }

    #[test]
    fn test_parallel_rotations() {
        let n_qubits = 14;
        let n_amps = 1 << n_qubits;
        let theta = std::f64::consts::FRAC_PI_3;

        let rx_gate = Gate::Rx(theta).matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.001).sin();
            let im = (i as f64 * 0.001).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &rx_gate, 5);
        instruct_single_parallel(&mut state_par, &rx_gate, 5);
        assert!(
            states_match(&state_seq, &state_par),
            "Rx gate parallel mismatch"
        );

        let ry_gate = Gate::Ry(theta).matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.002).cos();
            let im = (i as f64 * 0.002).sin();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &ry_gate, 9);
        instruct_single_parallel(&mut state_par, &ry_gate, 9);
        assert!(
            states_match(&state_seq, &state_par),
            "Ry gate parallel mismatch"
        );

        let rz_phases = [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ];
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.003).sin();
            let im = (i as f64 * 0.003).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_diagonal(&mut state_seq, &rz_phases, 12);
        instruct_diagonal_parallel(&mut state_par, &rz_phases, 12);
        assert!(
            states_match(&state_seq, &state_par),
            "Rz gate parallel mismatch"
        );
    }

    #[test]
    fn test_parallel_controlled() {
        let n_qubits = 14;
        let n_amps = 1 << n_qubits;

        let x_gate = Gate::X.matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.001).sin();
            let im = (i as f64 * 0.001).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &x_gate, 3);
        instruct_single_parallel(&mut state_par, &x_gate, 3);
        assert!(
            states_match(&state_seq, &state_par),
            "CNOT-like gate parallel mismatch"
        );

        let h_gate = Gate::H.matrix(2);
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.002).cos();
            let im = (i as f64 * 0.002).sin();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &h_gate, 8);
        instruct_single_parallel(&mut state_par, &h_gate, 8);
        assert!(
            states_match(&state_seq, &state_par),
            "Toffoli-like gate parallel mismatch"
        );
    }

    #[test]
    fn test_parallel_large_circuit() {
        let n_qubits = 16;
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        let mut state_par = state_seq.clone();
        let h_gate = Gate::H.matrix(2);
        let x_gate = Gate::X.matrix(2);
        for loc in 0..n_qubits {
            instruct_single(&mut state_seq, &h_gate, loc);
            instruct_single_parallel(&mut state_par, &h_gate, loc);
        }
        assert!(
            states_match(&state_seq, &state_par),
            "H gates mismatch in large circuit"
        );
        for loc in 0..(n_qubits - 1) {
            instruct_single(&mut state_seq, &x_gate, loc);
            instruct_single_parallel(&mut state_par, &x_gate, loc);
        }
        assert!(
            states_match(&state_seq, &state_par),
            "CNOT chain mismatch in large circuit"
        );
        let norm_seq: f64 = state_seq.data.iter().map(|a| a.norm_sqr()).sum();
        let norm_par: f64 = state_par.data.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm_seq - norm_par).abs() < 1e-10,
            "Normalization mismatch"
        );
    }

    #[test]
    fn test_parallel_qutrit() {
        let n_qutrits = 9;
        let n_amps: usize = 3_usize.pow(n_qutrits as u32);
        let mut state_seq = State::zero_state(&vec![3; n_qutrits]);
        let mut state_par = state_seq.clone();
        for i in 0..n_amps {
            let re = (i as f64 * 0.0005).sin();
            let im = (i as f64 * 0.0005).cos();
            state_seq.data[i] = Complex64::new(re, im);
            state_par.data[i] = Complex64::new(re, im);
        }
        let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let omega2 = omega * omega;
        let s = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
        let qutrit_gate = Array2::from_shape_vec(
            (3, 3),
            vec![s, s, s, s, s * omega, s * omega2, s, s * omega2, s * omega],
        )
        .unwrap();
        instruct_single(&mut state_seq, &qutrit_gate, 4);
        instruct_single_parallel(&mut state_par, &qutrit_gate, 4);
        assert!(
            states_match(&state_seq, &state_par),
            "Qutrit gate parallel mismatch"
        );

        let phase1 = Complex64::new(1.0, 0.0);
        let phase2 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_3);
        let phase3 = Complex64::from_polar(1.0, 2.0 * std::f64::consts::FRAC_PI_3);
        let qutrit_phases = [phase1, phase2, phase3];
        instruct_diagonal(&mut state_seq, &qutrit_phases, 7);
        instruct_diagonal_parallel(&mut state_par, &qutrit_phases, 7);
        assert!(
            states_match(&state_seq, &state_par),
            "Qutrit diagonal gate parallel mismatch"
        );
    }

    #[test]
    fn test_parallel_deterministic() {
        let n_qubits = 14;
        let n_amps = 1 << n_qubits;
        let h_gate = Gate::H.matrix(2);
        let theta = std::f64::consts::FRAC_PI_4;
        let rz_phases = [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ];
        let mut results: Vec<State> = Vec::new();
        for _ in 0..5 {
            let mut state = State::zero_state(&vec![2; n_qubits]);
            for i in 0..n_amps {
                let re = (i as f64 * 0.001).sin();
                let im = (i as f64 * 0.001).cos();
                state.data[i] = Complex64::new(re, im);
            }
            instruct_single_parallel(&mut state, &h_gate, 0);
            instruct_single_parallel(&mut state, &h_gate, 5);
            instruct_diagonal_parallel(&mut state, &rz_phases, 10);
            instruct_single_parallel(&mut state, &h_gate, 13);
            instruct_diagonal_parallel(&mut state, &rz_phases, 3);
            results.push(state);
        }
        for i in 1..5 {
            assert!(
                states_match(&results[0], &results[i]),
                "Run {} differs from run 0",
                i
            );
        }
        let mut state_seq = State::zero_state(&vec![2; n_qubits]);
        for i in 0..n_amps {
            let re = (i as f64 * 0.001).sin();
            let im = (i as f64 * 0.001).cos();
            state_seq.data[i] = Complex64::new(re, im);
        }
        instruct_single(&mut state_seq, &h_gate, 0);
        instruct_single(&mut state_seq, &h_gate, 5);
        instruct_diagonal(&mut state_seq, &rz_phases, 10);
        instruct_single(&mut state_seq, &h_gate, 13);
        instruct_diagonal(&mut state_seq, &rz_phases, 3);
        assert!(
            states_match(&state_seq, &results[0]),
            "Parallel result differs from sequential"
        );
    }
}

#[test]
fn test_instruct_h_creates_superposition() {
    let mut state = State::zero_state(&[2]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &h_gate, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], -s));
}

#[test]
fn test_instruct_h_on_each_qubit() {
    let h_gate = Gate::H.matrix(2);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    let mut state = State::zero_state(&[2, 2, 2]);
    instruct_single(&mut state, &h_gate, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[4], s));
    for i in [1, 2, 3, 5, 6, 7] {
        assert!(approx_eq(state.data[i], zero));
    }

    let mut state = State::zero_state(&[2, 2, 2]);
    instruct_single(&mut state, &h_gate, 1);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[2], s));
    for i in [1, 3, 4, 5, 6, 7] {
        assert!(approx_eq(state.data[i], zero));
    }

    let mut state = State::zero_state(&[2, 2, 2]);
    instruct_single(&mut state, &h_gate, 2);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));
    for i in [2, 3, 4, 5, 6, 7] {
        assert!(approx_eq(state.data[i], zero));
    }
}

#[test]
fn test_instruct_s_gate() {
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let s_phases = [one, i];

    let mut state = State::zero_state(&[2]);
    instruct_diagonal(&mut state, &s_phases, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &s_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], i));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_diagonal(&mut state, &s_phases, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s * i));
}

#[test]
fn test_instruct_t_gate() {
    let one = Complex64::new(1.0, 0.0);
    let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
    let zero = Complex64::new(0.0, 0.0);
    let t_phases = [one, t_phase];

    let mut state = State::zero_state(&[2]);
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], t_phase));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s * t_phase));

    let i = Complex64::new(0.0, 1.0);
    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &t_phases, 0);
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], i));
}

#[test]
fn test_instruct_identity() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let identity = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();

    let mut state = State::zero_state(&[2]);
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 0);
    assert!(approx_eq(state.data[0], state_before.data[0]));
    assert!(approx_eq(state.data[1], state_before.data[1]));

    let mut state = State::product_state(&[2], &[1]);
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 0);
    assert!(approx_eq(state.data[0], state_before.data[0]));
    assert!(approx_eq(state.data[1], state_before.data[1]));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 0);
    assert!(approx_eq(state.data[0], state_before.data[0]));
    assert!(approx_eq(state.data[1], state_before.data[1]));

    let amp0 = Complex64::new(0.6, 0.2);
    let amp1 = Complex64::new(0.3, 0.7);
    let mut state = State::zero_state(&[2]);
    state.data[0] = amp0;
    state.data[1] = amp1;
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 0);
    assert!(approx_eq(state.data[0], state_before.data[0]));
    assert!(approx_eq(state.data[1], state_before.data[1]));

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = Complex64::new(0.5, 0.0);
    state.data[1] = Complex64::new(0.5, 0.0);
    state.data[2] = Complex64::new(0.5, 0.0);
    state.data[3] = Complex64::new(0.5, 0.0);
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 0);
    for i in 0..4 {
        assert!(approx_eq(state.data[i], state_before.data[i]));
    }

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = Complex64::new(0.5, 0.0);
    state.data[1] = Complex64::new(0.5, 0.0);
    state.data[2] = Complex64::new(0.5, 0.0);
    state.data[3] = Complex64::new(0.5, 0.0);
    let state_before = state.clone();
    instruct_single(&mut state, &identity, 1);
    for i in 0..4 {
        assert!(approx_eq(state.data[i], state_before.data[i]));
    }
}

#[test]
fn test_instruct_rx_various_angles() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);

    let rx_0 = Gate::Rx(0.0).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &rx_0, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &rx_0, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let rx_pi = Gate::Rx(PI).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &rx_pi, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], neg_i));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &rx_pi, 0);
    assert!(approx_eq(state.data[0], neg_i));
    assert!(approx_eq(state.data[1], zero));

    let rx_pi_2 = Gate::Rx(PI / 2.0).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &rx_pi_2, 0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let neg_i_s = Complex64::new(0.0, -FRAC_1_SQRT_2);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], neg_i_s));

    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_ry_various_angles() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);

    let ry_0 = Gate::Ry(0.0).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &ry_0, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &ry_0, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let ry_pi = Gate::Ry(PI).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &ry_pi, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &ry_pi, 0);
    assert!(approx_eq(state.data[0], neg_one));
    assert!(approx_eq(state.data[1], zero));

    let ry_pi_2 = Gate::Ry(PI / 2.0).matrix(2);
    let mut state = State::zero_state(&[2]);
    instruct_single(&mut state, &ry_pi_2, 0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));

    let mut state = State::product_state(&[2], &[1]);
    instruct_single(&mut state, &ry_pi_2, 0);
    let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
    assert!(approx_eq(state.data[0], neg_s));
    assert!(approx_eq(state.data[1], s));

    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_rz_various_angles() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    let rz_0 = Gate::Rz(0.0).matrix(2);
    let rz_phases_0 = [rz_0[[0, 0]], rz_0[[1, 1]]];
    let mut state = State::zero_state(&[2]);
    instruct_diagonal(&mut state, &rz_phases_0, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &rz_phases_0, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let rz_pi = Gate::Rz(PI).matrix(2);
    let rz_phases_pi = [rz_pi[[0, 0]], rz_pi[[1, 1]]];
    let neg_i = Complex64::new(0.0, -1.0);
    let i = Complex64::new(0.0, 1.0);
    assert!(approx_eq(rz_phases_pi[0], neg_i));
    assert!(approx_eq(rz_phases_pi[1], i));

    let mut state = State::zero_state(&[2]);
    instruct_diagonal(&mut state, &rz_phases_pi, 0);
    assert!(approx_eq(state.data[0], neg_i));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &rz_phases_pi, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], i));

    let rz_pi_2 = Gate::Rz(PI / 2.0).matrix(2);
    let rz_phases_pi_2 = [rz_pi_2[[0, 0]], rz_pi_2[[1, 1]]];
    let exp_neg_pi_4 = Complex64::from_polar(1.0, -FRAC_PI_4);
    let exp_pi_4 = Complex64::from_polar(1.0, FRAC_PI_4);
    assert!(approx_eq(rz_phases_pi_2[0], exp_neg_pi_4));
    assert!(approx_eq(rz_phases_pi_2[1], exp_pi_4));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_diagonal(&mut state, &rz_phases_pi_2, 0);
    assert!(approx_eq(state.data[0], s * exp_neg_pi_4));
    assert!(approx_eq(state.data[1], s * exp_pi_4));

    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_phase_various_angles() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_one = Complex64::new(-1.0, 0.0);

    let phase_pi_4 = Gate::Phase(FRAC_PI_4).matrix(2);
    let t_phases = [phase_pi_4[[0, 0]], phase_pi_4[[1, 1]]];
    let exp_pi_4 = Complex64::from_polar(1.0, FRAC_PI_4);
    assert!(approx_eq(t_phases[0], one));
    assert!(approx_eq(t_phases[1], exp_pi_4));

    let mut state = State::zero_state(&[2]);
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &t_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], exp_pi_4));

    let phase_pi_2 = Gate::Phase(PI / 2.0).matrix(2);
    let s_phases = [phase_pi_2[[0, 0]], phase_pi_2[[1, 1]]];
    assert!(approx_eq(s_phases[0], one));
    assert!(approx_eq(s_phases[1], i));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &s_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], i));

    let phase_pi = Gate::Phase(PI).matrix(2);
    let z_phases = [phase_pi[[0, 0]], phase_pi[[1, 1]]];
    assert!(approx_eq(z_phases[0], one));
    assert!(approx_eq(z_phases[1], neg_one));

    let mut state = State::product_state(&[2], &[1]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], neg_one));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_diagonal(&mut state, &s_phases, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s * i));

    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_rotation_on_superposition() {
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let i = Complex64::new(0.0, 1.0);

    let rx_pi = Gate::Rx(PI).matrix(2);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_single(&mut state, &rx_pi, 0);
    assert!(approx_eq(state.data[0], s * neg_i));
    assert!(approx_eq(state.data[1], s * neg_i));

    let ry_pi = Gate::Ry(PI).matrix(2);
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_single(&mut state, &ry_pi, 0);
    let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
    assert!(approx_eq(state.data[0], neg_s));
    assert!(approx_eq(state.data[1], s));

    let rz_pi = Gate::Rz(PI).matrix(2);
    let rz_phases_pi = [rz_pi[[0, 0]], rz_pi[[1, 1]]];
    let mut state = State::zero_state(&[2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_diagonal(&mut state, &rz_phases_pi, 0);
    assert!(approx_eq(state.data[0], s * neg_i));
    assert!(approx_eq(state.data[1], s * i));

    for angle in [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
        let rx = Gate::Rx(angle).matrix(2);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_single(&mut state, &rx, 0);
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);

        let ry = Gate::Ry(angle).matrix(2);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_single(&mut state, &ry, 0);
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);

        let rz = Gate::Rz(angle).matrix(2);
        let rz_phases = [rz[[0, 0]], rz[[1, 1]]];
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_diagonal(&mut state, &rz_phases, 0);
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}

// Tests for SWAP gate
#[test]
fn test_instruct_swap_basic() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], one));
    assert!(approx_eq(state.data[3], zero));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], zero));
}

#[test]
fn test_instruct_swap_symmetric() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], zero));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], one));
}

#[test]
fn test_instruct_swap_on_superposition() {
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = s;
    state.data[3] = s;
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], s));

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = zero;
    state.data[1] = s;
    state.data[2] = s;
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], s));
    assert!(approx_eq(state.data[2], s));
    assert!(approx_eq(state.data[3], zero));

    let amp1 = Complex64::new(1.0 / 5.0_f64.sqrt(), 0.0);
    let amp2 = Complex64::new(2.0 / 5.0_f64.sqrt(), 0.0);
    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = zero;
    state.data[1] = amp1;
    state.data[2] = amp2;
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], amp2));
    assert!(approx_eq(state.data[2], amp1));
    assert!(approx_eq(state.data[3], zero));
}

#[test]
fn test_instruct_swap_non_adjacent() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    for i in 0..8 {
        if i == 4 {
            assert!(approx_eq(state.data[i], one));
        } else {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    for i in 0..8 {
        if i == 1 {
            assert!(approx_eq(state.data[i], one));
        } else {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    for i in 0..8 {
        if i == 5 {
            assert!(approx_eq(state.data[i], one));
        } else {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    for i in 0..8 {
        if i == 6 {
            assert!(approx_eq(state.data[i], one));
        } else {
            assert!(approx_eq(state.data[i], zero));
        }
    }
}

#[test]
fn test_instruct_swap_preserves_norm() {
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = Complex64::new(0.3, 0.1);
    state.data[1] = Complex64::new(0.4, 0.2);
    state.data[2] = Complex64::new(0.5, 0.3);
    state.data[3] = Complex64::new(0.6, 0.4);
    let norm_before = state.norm();
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    let norm_after = state.norm();
    assert!((norm_before - norm_after).abs() < 1e-10);

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = Complex64::new(0.3, 0.1);
    state.data[1] = Complex64::new(0.4, 0.2);
    state.data[2] = Complex64::new(0.5, 0.3);
    state.data[3] = Complex64::new(0.6, 0.4);
    let original = state.clone();
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    for i in 0..4 {
        assert!(approx_eq(state.data[i], original.data[i]));
    }

    let mut state = State::zero_state(&[2, 2, 2]);
    for i in 0..8 {
        state.data[i] = Complex64::new((i as f64) * 0.1, (i as f64) * 0.05);
    }
    let norm_before = state.norm();
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    let norm_after = state.norm();
    assert!((norm_before - norm_after).abs() < 1e-10);
}

#[test]
fn test_instruct_cnot_all_basis_states() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], zero));

    let mut state = State::product_state(&[2, 2], &[0, 1]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], zero));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], one));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], one));
    assert!(approx_eq(state.data[3], zero));
}

#[test]
fn test_instruct_cnot_reversed() {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_gate, &[1], &[1], &[0]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2], &[0, 1]);
    instruct_controlled(&mut state, &x_gate, &[1], &[1], &[0]);
    assert!(approx_eq(state.data[3], one));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_gate, &[1], &[1], &[0]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &x_gate, &[1], &[1], &[0]);
    assert!(approx_eq(state.data[1], one));
}

#[test]
fn test_instruct_cz_gate() {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    let z_gate = Gate::Z.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &z_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2], &[0, 1]);
    instruct_controlled(&mut state, &z_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[1], one));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &z_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &z_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], neg_one));

    let mut state = State::zero_state(&[2, 2]);
    let half = Complex64::new(0.5, 0.0);
    state.data[0] = half;
    state.data[1] = half;
    state.data[2] = half;
    state.data[3] = half;
    instruct_controlled(&mut state, &z_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], half));
    assert!(approx_eq(state.data[1], half));
    assert!(approx_eq(state.data[2], half));
    assert!(approx_eq(state.data[3], -half));
}

#[test]
fn test_instruct_cy_gate() {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let y_gate = Gate::Y.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &y_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &y_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], i));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &y_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], neg_i));
}

#[test]
fn test_instruct_controlled_h() {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let h_gate = Gate::H.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &h_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &h_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], s));
    assert!(approx_eq(state.data[3], s));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &h_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], s));
    assert!(approx_eq(state.data[3], -s));
}

#[test]
fn test_instruct_controlled_phase() {
    let _zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    let theta = FRAC_PI_4;
    let phase = Complex64::from_polar(1.0, theta);
    let phase_gate = Gate::Phase(theta).matrix(2);

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &phase_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &phase_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], phase));

    let theta = std::f64::consts::FRAC_PI_2;
    let phase = Complex64::from_polar(1.0, theta);
    let phase_gate = Gate::Phase(theta).matrix(2);
    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &phase_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], phase));

    let theta = std::f64::consts::PI;
    let phase = Complex64::from_polar(1.0, theta);
    let phase_gate = Gate::Phase(theta).matrix(2);
    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &phase_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], phase));

    let theta = 2.0 * std::f64::consts::PI;
    let phase_gate = Gate::Phase(theta).matrix(2);
    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &phase_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], one));
}

#[test]
fn test_instruct_controlled_on_superposition() {
    let zero = Complex64::new(0.0, 0.0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = s;
    state.data[2] = s;
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], s));
    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = s;
    state.data[2] = -s;
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], -s));

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = s;
    state.data[1] = s;
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], zero));
}

#[test]
fn test_instruct_toffoli_all_basis() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[0], one));
    for i in 1..8 {
        assert!(approx_eq(state.data[i], zero));
    }

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[1], one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[3], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[4], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[5], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[6], zero));
    assert!(approx_eq(state.data[7], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[6], one));
    assert!(approx_eq(state.data[7], zero));
}

#[test]
fn test_instruct_ccz_gate() {
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    let z_gate = Gate::Z.matrix(2);

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 0]);
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[6], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 1]);
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[7], neg_one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 1]);
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[3], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[5], one));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2, 2, 2]);
    state.data[6] = s;
    state.data[7] = s;
    instruct_controlled(&mut state, &z_gate, &[0, 1], &[1, 1], &[2]);
    assert!(approx_eq(state.data[6], s));
    assert!(approx_eq(state.data[7], -s));
}

#[test]
fn test_instruct_three_controls() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2, 2, 2], &[1, 1, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1, 2], &[1, 1, 1], &[3]);
    assert!(approx_eq(state.data[14], zero));
    assert!(approx_eq(state.data[15], one));

    let mut state = State::product_state(&[2, 2, 2, 2], &[1, 1, 1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1, 2], &[1, 1, 1], &[3]);
    assert!(approx_eq(state.data[14], one));
    assert!(approx_eq(state.data[15], zero));

    let mut state = State::product_state(&[2, 2, 2, 2], &[1, 1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1, 2], &[1, 1, 1], &[3]);
    assert!(approx_eq(state.data[12], one));

    let mut state = State::product_state(&[2, 2, 2, 2], &[0, 1, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1, 2], &[1, 1, 1], &[3]);
    assert!(approx_eq(state.data[6], one));

    let mut state = State::product_state(&[2, 2, 2, 2], &[0, 0, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1, 2], &[1, 1, 1], &[3]);
    assert!(approx_eq(state.data[0], one));
}

#[test]
fn test_instruct_controlled_swap() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let swap_gate = Array2::from_shape_vec(
        (4, 4),
        vec![
            one, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, zero, zero, zero,
            one,
        ],
    )
    .unwrap();

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[4], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[6], zero));
    assert!(approx_eq(state.data[5], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[5], zero));
    assert!(approx_eq(state.data[6], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 1]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[7], one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 0]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 1]);
    instruct_controlled(&mut state, &swap_gate, &[0], &[1], &[1, 2]);
    assert!(approx_eq(state.data[1], one));
}

#[test]
fn test_instruct_mixed_control_values() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[0], &[1]);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let mut state = State::product_state(&[2, 2], &[0, 1]);
    instruct_controlled(&mut state, &x_gate, &[0], &[0], &[1]);
    assert!(approx_eq(state.data[0], one));
    assert!(approx_eq(state.data[1], zero));

    let mut state = State::product_state(&[2, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[0], &[1]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[2, 2], &[1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0], &[0], &[1]);
    assert!(approx_eq(state.data[3], one));
}

#[test]
fn test_instruct_multi_control_mixed_values() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_gate = Gate::X.matrix(2);

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], one));

    let mut state = State::product_state(&[2, 2, 2], &[0, 1, 1]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[2], one));
    assert!(approx_eq(state.data[3], zero));

    let mut state = State::product_state(&[2, 2, 2], &[0, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[6], one));

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[4], one));

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = State::zero_state(&[2, 2, 2]);
    state.data[2] = s;
    state.data[6] = s;
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[0, 1], &[2]);
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], s));
    assert!(approx_eq(state.data[6], s));
    assert!(approx_eq(state.data[7], zero));
}

// ========== Qudit (d>2) Tests ==========

#[test]
fn test_instruct_qutrit_x_gate() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_qutrit = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let mut state = State::zero_state(&[3]);
    instruct_single(&mut state, &x_qutrit, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));
    assert!(approx_eq(state.data[2], zero));

    let mut state = State::product_state(&[3], &[1]);
    instruct_single(&mut state, &x_qutrit, 0);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[3], &[2]);
    instruct_single(&mut state, &x_qutrit, 0);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::zero_state(&[3]);
    instruct_single(&mut state, &x_qutrit, 0);
    instruct_single(&mut state, &x_qutrit, 0);
    instruct_single(&mut state, &x_qutrit, 0);
    assert!(approx_eq(state.data[0], one));
}

#[test]
fn test_instruct_qutrit_z_gate() {
    let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let omega_sq = omega * omega;
    let one = Complex64::new(1.0, 0.0);
    let z_phases = [one, omega, omega_sq];

    let mut state = State::zero_state(&[3]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[3], &[1]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[1], omega));

    let mut state = State::product_state(&[3], &[2]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[2], omega_sq));

    let mut state = State::product_state(&[3], &[1]);
    instruct_diagonal(&mut state, &z_phases, 0);
    instruct_diagonal(&mut state, &z_phases, 0);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[1], one));
}

#[test]
fn test_instruct_qutrit_hadamard() {
    let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let scale = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
    let one = Complex64::new(1.0, 0.0);
    let h_qutrit = Array2::from_shape_vec(
        (3, 3),
        vec![
            scale * one,
            scale * one,
            scale * one,
            scale * one,
            scale * omega,
            scale * omega * omega,
            scale * one,
            scale * omega * omega,
            scale * omega,
        ],
    )
    .unwrap();

    let mut state = State::zero_state(&[3]);
    instruct_single(&mut state, &h_qutrit, 0);
    assert!(approx_eq(state.data[0], scale));
    assert!(approx_eq(state.data[1], scale));
    assert!(approx_eq(state.data[2], scale));
    let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_mixed_qubit_qutrit() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_qubit = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();
    let x_qutrit = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let mut state = State::zero_state(&[2, 3, 2]);
    instruct_single(&mut state, &x_qubit, 0);
    assert!(approx_eq(state.data[6], one));
    for i in 0..12 {
        if i != 6 {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::zero_state(&[2, 3, 2]);
    instruct_single(&mut state, &x_qutrit, 1);
    assert!(approx_eq(state.data[2], one));
    for i in 0..12 {
        if i != 2 {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::zero_state(&[2, 3, 2]);
    instruct_single(&mut state, &x_qubit, 2);
    assert!(approx_eq(state.data[1], one));
    for i in 0..12 {
        if i != 1 {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    let mut state = State::zero_state(&[2, 3, 2]);
    instruct_single(&mut state, &x_qubit, 0);
    instruct_single(&mut state, &x_qutrit, 1);
    instruct_single(&mut state, &x_qubit, 2);
    assert!(approx_eq(state.data[9], one));
    for i in 0..12 {
        if i != 9 {
            assert!(approx_eq(state.data[i], zero));
        }
    }
}

#[test]
fn test_instruct_controlled_qutrit_by_qubit() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_qutrit = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let mut state = State::product_state(&[2, 3], &[1, 0]);
    instruct_controlled(&mut state, &x_qutrit, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[3], zero));
    assert!(approx_eq(state.data[4], one));

    let mut state = State::product_state(&[2, 3], &[1, 1]);
    instruct_controlled(&mut state, &x_qutrit, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[4], zero));
    assert!(approx_eq(state.data[5], one));

    let mut state = State::product_state(&[2, 3], &[0, 0]);
    instruct_controlled(&mut state, &x_qutrit, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[2, 3], &[0, 1]);
    instruct_controlled(&mut state, &x_qutrit, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[1], one));
}

#[test]
fn test_instruct_qutrit_controls_qubit() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_qubit = Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap();

    let mut state = State::product_state(&[3, 2], &[2, 0]);
    instruct_controlled(&mut state, &x_qubit, &[0], &[2], &[1]);
    assert!(approx_eq(state.data[4], zero));
    assert!(approx_eq(state.data[5], one));

    let mut state = State::product_state(&[3, 2], &[2, 1]);
    instruct_controlled(&mut state, &x_qubit, &[0], &[2], &[1]);
    assert!(approx_eq(state.data[4], one));
    assert!(approx_eq(state.data[5], zero));

    let mut state = State::product_state(&[3, 2], &[0, 0]);
    instruct_controlled(&mut state, &x_qubit, &[0], &[2], &[1]);
    assert!(approx_eq(state.data[0], one));

    let mut state = State::product_state(&[3, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_qubit, &[0], &[2], &[1]);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[3, 2], &[1, 0]);
    instruct_controlled(&mut state, &x_qubit, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], one));
}

#[test]
fn test_instruct_ququart() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let x_ququart = Array2::from_shape_vec(
        (4, 4),
        vec![
            zero, zero, zero, one, one, zero, zero, zero, zero, one, zero, zero, zero, zero, one,
            zero,
        ],
    )
    .unwrap();

    let mut state = State::zero_state(&[4]);
    instruct_single(&mut state, &x_ququart, 0);
    assert!(approx_eq(state.data[1], one));

    let mut state = State::product_state(&[4], &[1]);
    instruct_single(&mut state, &x_ququart, 0);
    assert!(approx_eq(state.data[2], one));

    let mut state = State::product_state(&[4], &[3]);
    instruct_single(&mut state, &x_ququart, 0);
    assert!(approx_eq(state.data[0], one));

    let z_phases = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];

    let mut state = State::product_state(&[4], &[2]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[2], Complex64::new(-1.0, 0.0)));

    let mut state = State::product_state(&[4], &[3]);
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[3], Complex64::new(0.0, -1.0)));

    let mut state = State::product_state(&[4], &[1]);
    for _ in 0..4 {
        instruct_diagonal(&mut state, &z_phases, 0);
    }
    assert!(approx_eq(state.data[1], one));

    let mut state = State::zero_state(&[4, 2]);
    instruct_single(&mut state, &x_ququart, 0);
    assert!(approx_eq(state.data[2], one));
    for i in 0..8 {
        if i != 2 {
            assert!(approx_eq(state.data[i], zero));
        }
    }
}

// ========== Diagonal Optimization Verification Tests ==========

#[test]
fn test_diagonal_z_matches_general() {
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let z_matrix = Array2::from_shape_vec((2, 2), vec![one, zero, zero, neg_one]).unwrap();
    let z_phases = [one, neg_one];

    for initial_vals in [[0], [1]] {
        let mut state_general = State::product_state(&[2], &initial_vals);
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &z_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &z_phases, 0);
        for i in 0..2 {
            assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
        }
    }

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state_general = State::zero_state(&[2]);
    state_general.data[0] = s;
    state_general.data[1] = s;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &z_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &z_phases, 0);
    for i in 0..2 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }

    let mut state_general = State::zero_state(&[2, 2, 2]);
    let mut state_diagonal = State::zero_state(&[2, 2, 2]);
    for i in 0..8 {
        let amp = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.1).cos());
        state_general.data[i] = amp;
        state_diagonal.data[i] = amp;
    }
    instruct_single(&mut state_general, &z_matrix, 1);
    instruct_diagonal(&mut state_diagonal, &z_phases, 1);
    for i in 0..8 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }
}

#[test]
fn test_diagonal_s_matches_general() {
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let s_matrix = Array2::from_shape_vec((2, 2), vec![one, zero, zero, i]).unwrap();
    let s_phases = [one, i];

    for initial_vals in [[0], [1]] {
        let mut state_general = State::product_state(&[2], &initial_vals);
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &s_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &s_phases, 0);
        for idx in 0..2 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }
    }

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state_general = State::zero_state(&[2]);
    state_general.data[0] = s;
    state_general.data[1] = s;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &s_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &s_phases, 0);
    for idx in 0..2 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }

    for loc in 0..2 {
        let mut state_general = State::zero_state(&[2, 2]);
        let mut state_diagonal = State::zero_state(&[2, 2]);
        let half = Complex64::new(0.5, 0.0);
        for idx in 0..4 {
            state_general.data[idx] = half;
            state_diagonal.data[idx] = half;
        }
        instruct_single(&mut state_general, &s_matrix, loc);
        instruct_diagonal(&mut state_diagonal, &s_phases, loc);
        for idx in 0..4 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }
    }
}

#[test]
fn test_diagonal_t_matches_general() {
    let one = Complex64::new(1.0, 0.0);
    let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
    let zero = Complex64::new(0.0, 0.0);
    let t_matrix = Array2::from_shape_vec((2, 2), vec![one, zero, zero, t_phase]).unwrap();
    let t_phases = [one, t_phase];

    for initial_vals in [[0], [1]] {
        let mut state_general = State::product_state(&[2], &initial_vals);
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &t_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &t_phases, 0);
        for idx in 0..2 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }
    }

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state_general = State::zero_state(&[2]);
    state_general.data[0] = s;
    state_general.data[1] = s;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &t_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &t_phases, 0);
    for idx in 0..2 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }

    let mut state_general = State::product_state(&[2], &[1]);
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &t_matrix, 0);
    instruct_single(&mut state_general, &t_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &t_phases, 0);
    instruct_diagonal(&mut state_diagonal, &t_phases, 0);
    for idx in 0..2 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }
}

#[test]
fn test_diagonal_phase_matches_general() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    let test_angles = [
        0.0,
        FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        PI,
        3.0 * std::f64::consts::FRAC_PI_2,
        2.0 * PI,
        0.123,
        -0.456,
    ];

    for theta in test_angles {
        let phase = Complex64::from_polar(1.0, theta);
        let phase_matrix = Array2::from_shape_vec((2, 2), vec![one, zero, zero, phase]).unwrap();
        let phase_phases = [one, phase];

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state_general = State::zero_state(&[2]);
        state_general.data[0] = s;
        state_general.data[1] = s;
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &phase_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &phase_phases, 0);
        for idx in 0..2 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }

        let mut state_general = State::zero_state(&[2, 2, 2]);
        let mut state_diagonal = State::zero_state(&[2, 2, 2]);
        for i in 0..8 {
            let amp = Complex64::new((i as f64 * 0.2).cos(), (i as f64 * 0.3).sin());
            state_general.data[i] = amp;
            state_diagonal.data[i] = amp;
        }
        instruct_single(&mut state_general, &phase_matrix, 1);
        instruct_diagonal(&mut state_diagonal, &phase_phases, 1);
        for i in 0..8 {
            assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
        }
    }
}

#[test]
fn test_diagonal_rz_matches_general() {
    let zero = Complex64::new(0.0, 0.0);
    let test_angles = [
        0.0,
        FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        PI,
        3.0 * std::f64::consts::FRAC_PI_2,
        2.0 * PI,
        0.789,
        -1.234,
    ];

    for theta in test_angles {
        let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
        let rz_matrix =
            Array2::from_shape_vec((2, 2), vec![phase_neg, zero, zero, phase_pos]).unwrap();
        let rz_phases = [phase_neg, phase_pos];

        for initial_vals in [[0], [1]] {
            let mut state_general = State::product_state(&[2], &initial_vals);
            let mut state_diagonal = state_general.clone();
            instruct_single(&mut state_general, &rz_matrix, 0);
            instruct_diagonal(&mut state_diagonal, &rz_phases, 0);
            for idx in 0..2 {
                assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
            }
        }

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state_general = State::zero_state(&[2]);
        state_general.data[0] = s;
        state_general.data[1] = s;
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &rz_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &rz_phases, 0);
        for idx in 0..2 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }
        let norm: f64 = state_diagonal.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_diagonal_custom_gate() {
    let zero = Complex64::new(0.0, 0.0);
    let phase0 = Complex64::from_polar(1.0, 0.5);
    let phase1 = Complex64::from_polar(1.0, 1.2);
    let custom_matrix = Array2::from_shape_vec((2, 2), vec![phase0, zero, zero, phase1]).unwrap();
    let custom_phases = [phase0, phase1];

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state_general = State::zero_state(&[2]);
    state_general.data[0] = s;
    state_general.data[1] = s;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &custom_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &custom_phases, 0);
    for idx in 0..2 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }

    let amp0 = Complex64::new(0.6, 0.2);
    let amp1 = Complex64::new(0.3, 0.7);
    let mut state_general = State::zero_state(&[2]);
    state_general.data[0] = amp0;
    state_general.data[1] = amp1;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &custom_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &custom_phases, 0);
    for idx in 0..2 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }

    let phase0 = Complex64::from_polar(1.0, std::f64::consts::PI / 3.0);
    let phase1 = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let custom_matrix = Array2::from_shape_vec((2, 2), vec![phase0, zero, zero, phase1]).unwrap();
    let custom_phases = [phase0, phase1];

    let mut state_general = State::zero_state(&[2, 2, 2, 2]);
    let mut state_diagonal = State::zero_state(&[2, 2, 2, 2]);
    for i in 0..16 {
        let amp = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.15).cos());
        state_general.data[i] = amp;
        state_diagonal.data[i] = amp;
    }
    instruct_single(&mut state_general, &custom_matrix, 2);
    instruct_diagonal(&mut state_diagonal, &custom_phases, 2);
    for i in 0..16 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }
}

#[test]
fn test_diagonal_qutrit() {
    let zero = Complex64::new(0.0, 0.0);
    let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let omega_sq = omega * omega;
    let one = Complex64::new(1.0, 0.0);
    let z3_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![one, zero, zero, zero, omega, zero, zero, zero, omega_sq],
    )
    .unwrap();
    let z3_phases = [one, omega, omega_sq];

    for initial_val in 0..3 {
        let mut state_general = State::product_state(&[3], &[initial_val]);
        let mut state_diagonal = state_general.clone();
        instruct_single(&mut state_general, &z3_matrix, 0);
        instruct_diagonal(&mut state_diagonal, &z3_phases, 0);
        for idx in 0..3 {
            assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
        }
    }

    let scale = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
    let mut state_general = State::zero_state(&[3]);
    state_general.data[0] = scale;
    state_general.data[1] = scale;
    state_general.data[2] = scale;
    let mut state_diagonal = state_general.clone();
    instruct_single(&mut state_general, &z3_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &z3_phases, 0);
    for idx in 0..3 {
        assert!(approx_eq(state_general.data[idx], state_diagonal.data[idx]));
    }

    let mut state_general = State::zero_state(&[2, 3]);
    let mut state_diagonal = State::zero_state(&[2, 3]);
    for i in 0..6 {
        let amp = Complex64::new((i as f64 * 0.2).cos(), (i as f64 * 0.3).sin());
        state_general.data[i] = amp;
        state_diagonal.data[i] = amp;
    }
    instruct_single(&mut state_general, &z3_matrix, 1);
    instruct_diagonal(&mut state_diagonal, &z3_phases, 1);
    for i in 0..6 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }

    let z_matrix =
        Array2::from_shape_vec((2, 2), vec![one, zero, zero, Complex64::new(-1.0, 0.0)]).unwrap();
    let z_phases = [one, Complex64::new(-1.0, 0.0)];

    let mut state_general = State::zero_state(&[2, 3]);
    let mut state_diagonal = State::zero_state(&[2, 3]);
    for i in 0..6 {
        let amp = Complex64::new((i as f64 * 0.25).sin(), (i as f64 * 0.35).cos());
        state_general.data[i] = amp;
        state_diagonal.data[i] = amp;
    }
    instruct_single(&mut state_general, &z_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &z_phases, 0);
    for i in 0..6 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }

    let d4_phases = [
        Complex64::from_polar(1.0, 0.1),
        Complex64::from_polar(1.0, 0.5),
        Complex64::from_polar(1.0, 1.0),
        Complex64::from_polar(1.0, 1.5),
    ];
    let d4_matrix = Array2::from_shape_vec(
        (4, 4),
        vec![
            d4_phases[0],
            zero,
            zero,
            zero,
            zero,
            d4_phases[1],
            zero,
            zero,
            zero,
            zero,
            d4_phases[2],
            zero,
            zero,
            zero,
            zero,
            d4_phases[3],
        ],
    )
    .unwrap();

    let mut state_general = State::zero_state(&[4]);
    let mut state_diagonal = State::zero_state(&[4]);
    let amp = Complex64::new(0.5, 0.0);
    for i in 0..4 {
        state_general.data[i] = amp;
        state_diagonal.data[i] = amp;
    }
    instruct_single(&mut state_general, &d4_matrix, 0);
    instruct_diagonal(&mut state_diagonal, &d4_phases, 0);
    for i in 0..4 {
        assert!(approx_eq(state_general.data[i], state_diagonal.data[i]));
    }
}

// ==================== Edge Case and Error Handling Tests ====================

#[test]
fn test_instruct_single_qubit_system() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    let mut state = State::zero_state(&[2]);
    let x_gate = Gate::X.matrix(2);
    instruct_single(&mut state, &x_gate, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));

    let mut state = State::zero_state(&[2]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));

    let mut state = State::zero_state(&[2]);
    let y_gate = Gate::Y.matrix(2);
    instruct_single(&mut state, &y_gate, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 1.0)));

    let mut state = State::product_state(&[2], &[1]);
    let z_phases = [one, Complex64::new(-1.0, 0.0)];
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], Complex64::new(-1.0, 0.0)));

    let mut state = State::product_state(&[2], &[1]);
    let s_gate = Gate::S.matrix(2);
    instruct_single(&mut state, &s_gate, 0);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], Complex64::new(0.0, 1.0)));

    let mut state = State::product_state(&[2], &[1]);
    let t_gate = Gate::T.matrix(2);
    instruct_single(&mut state, &t_gate, 0);
    let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], t_phase));
}

#[test]
fn test_instruct_large_system() {
    let dims: Vec<usize> = vec![2; 10];
    let mut state = State::zero_state(&dims);
    assert_eq!(state.data.len(), 1024);
    assert_eq!(state.total_dim(), 1024);

    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[512], s));
    for i in 1..512 {
        assert!(
            approx_eq(state.data[i], zero),
            "Expected zero at index {}",
            i
        );
    }
    for i in 513..1024 {
        assert!(
            approx_eq(state.data[i], zero),
            "Expected zero at index {}",
            i
        );
    }
    assert!((state.norm() - 1.0).abs() < 1e-10);
}

#[test]
fn test_instruct_gate_on_last_qubit() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    let mut state = State::zero_state(&[2, 2, 2]);
    let x_gate = Gate::X.matrix(2);
    instruct_single(&mut state, &x_gate, 2);
    assert!(approx_eq(state.data[0], zero));
    assert!(approx_eq(state.data[1], one));
    for i in 2..8 {
        assert!(approx_eq(state.data[i], zero));
    }

    let mut state = State::zero_state(&[2, 2, 2, 2]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 3);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], s));
    for i in 2..16 {
        assert!(approx_eq(state.data[i], zero));
    }

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[2]);
    assert!(approx_eq(state.data[4], zero));
    assert!(approx_eq(state.data[5], one));
}

#[test]
fn test_instruct_non_contiguous_targets() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let swap_gate = Gate::SWAP.matrix(2);

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 2]);
    assert!(approx_eq(state.data[1], one));
    assert!(approx_eq(state.data[4], zero));

    let x_gate = Gate::X.matrix(2);
    let mut state = State::product_state(&[2, 2, 2, 2], &[1, 0, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[3]);
    assert!(approx_eq(state.data[8], zero));
    assert!(approx_eq(state.data[9], one));

    let mut state = State::product_state(&[2, 2, 2, 2], &[0, 1, 0, 0]);
    instruct_controlled(&mut state, &x_gate, &[1], &[1], &[3]);
    assert!(approx_eq(state.data[4], zero));
    assert!(approx_eq(state.data[5], one));
}

#[test]
fn test_instruct_preserves_normalization_comprehensive() {
    let tolerance = 1e-10;

    fn create_test_state(dims: &[usize]) -> State {
        let mut state = State::zero_state(dims);
        let n = state.data.len();
        for i in 0..n {
            let re = ((i as f64 * 0.7).sin() + 1.0) / 2.0;
            let im = ((i as f64 * 0.3).cos()) / 2.0;
            state.data[i] = Complex64::new(re, im);
        }
        let norm = state.norm();
        for amp in state.data.iter_mut() {
            *amp /= norm;
        }
        state
    }

    let mut state = create_test_state(&[2, 2]);
    let x_gate = Gate::X.matrix(2);
    instruct_single(&mut state, &x_gate, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let y_gate = Gate::Y.matrix(2);
    instruct_single(&mut state, &y_gate, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    instruct_diagonal(&mut state, &z_phases, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let s_gate = Gate::S.matrix(2);
    instruct_single(&mut state, &s_gate, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let t_gate = Gate::T.matrix(2);
    instruct_single(&mut state, &t_gate, 0);
    assert!((state.norm() - 1.0).abs() < tolerance);

    for theta in [
        0.0,
        0.5,
        1.0,
        std::f64::consts::PI,
        2.0 * std::f64::consts::PI,
    ] {
        let mut state = create_test_state(&[2, 2]);
        let rx_gate = Gate::Rx(theta).matrix(2);
        instruct_single(&mut state, &rx_gate, 0);
        assert!((state.norm() - 1.0).abs() < tolerance);
    }

    for theta in [0.0, 0.5, 1.0, std::f64::consts::PI] {
        let mut state = create_test_state(&[2, 2]);
        let ry_gate = Gate::Ry(theta).matrix(2);
        instruct_single(&mut state, &ry_gate, 0);
        assert!((state.norm() - 1.0).abs() < tolerance);
    }

    for theta in [0.0, 0.5, 1.0, std::f64::consts::PI] {
        let mut state = create_test_state(&[2, 2]);
        let rz_gate = Gate::Rz(theta).matrix(2);
        instruct_single(&mut state, &rz_gate, 0);
        assert!((state.norm() - 1.0).abs() < tolerance);
    }

    for theta in [
        0.0,
        FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
    ] {
        let mut state = create_test_state(&[2, 2]);
        let phase_gate = Gate::Phase(theta).matrix(2);
        instruct_single(&mut state, &phase_gate, 0);
        assert!((state.norm() - 1.0).abs() < tolerance);
    }

    let mut state = create_test_state(&[2, 2]);
    let x_gate = Gate::X.matrix(2);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2]);
    let swap_gate = Gate::SWAP.matrix(2);
    instruct_controlled(&mut state, &swap_gate, &[], &[], &[0, 1]);
    assert!((state.norm() - 1.0).abs() < tolerance);

    let mut state = create_test_state(&[2, 2, 2]);
    instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);
    assert!((state.norm() - 1.0).abs() < tolerance);
}

#[test]
fn test_instruct_multiple_gates_sequence() {
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let mut state = State::zero_state(&[2, 2]);

    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[2], s));

    let x_gate = Gate::X.matrix(2);
    instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
    assert!(approx_eq(state.data[0], s));
    assert!(approx_eq(state.data[1], zero));
    assert!(approx_eq(state.data[2], zero));
    assert!(approx_eq(state.data[3], s));

    let theta = FRAC_PI_4;
    let rz_phases = [
        Complex64::from_polar(1.0, -theta / 2.0),
        Complex64::from_polar(1.0, theta / 2.0),
    ];
    instruct_diagonal(&mut state, &rz_phases, 1);
    let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
    assert!(approx_eq(state.data[0], s * phase_neg));
    assert!(approx_eq(state.data[3], s * phase_pos));
    assert!((state.norm() - 1.0).abs() < 1e-10);

    let mut state = State::zero_state(&[2, 2, 2]);
    for i in 0..3 {
        instruct_single(&mut state, &h_gate, i);
    }
    let amp = Complex64::new(1.0 / (8.0_f64).sqrt(), 0.0);
    for i in 0..8 {
        assert!(approx_eq(state.data[i], amp));
    }
    assert!((state.norm() - 1.0).abs() < 1e-10);
}

#[test]
#[should_panic]
fn test_instruct_invalid_location() {
    let mut state = State::zero_state(&[2, 2]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 5);
}

#[test]
#[should_panic]
fn test_instruct_gate_dimension_mismatch() {
    let mut state = State::zero_state(&[3]);
    let h_gate = Gate::H.matrix(2);
    instruct_single(&mut state, &h_gate, 0);
}
