//! Primitive amplitude operations for state vector simulation.
//!
//! These functions directly manipulate amplitudes in the state vector,
//! providing O(1) operations for applying gates to specific indices.

use ndarray::Array2;
use num_complex::Complex64;

/// Apply a 2x2 unitary gate to a pair of amplitudes at indices i and j.
///
/// The gate matrix is [[a, b], [c, d]] and transforms:
/// - new_i = a * state[i] + b * state[j]
/// - new_j = c * state[i] + d * state[j]
///
/// # Arguments
/// * `state` - Mutable slice of complex amplitudes
/// * `i` - Index of the first amplitude (corresponds to |0⟩ in the gate's basis)
/// * `j` - Index of the second amplitude (corresponds to |1⟩ in the gate's basis)
/// * `gate` - 2x2 unitary matrix
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use yao_rs::instruct::u1rows;
///
/// // Apply X gate to |0⟩ state
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// u1rows(&mut state, 0, 1, &x_gate);
/// // Now state is |1⟩
/// assert!((state[0].norm() - 0.0).abs() < 1e-10);
/// assert!((state[1].norm() - 1.0).abs() < 1e-10);
/// ```
pub fn u1rows(state: &mut [Complex64], i: usize, j: usize, gate: &Array2<Complex64>) {
    debug_assert_eq!(gate.nrows(), 2);
    debug_assert_eq!(gate.ncols(), 2);

    let old_i = state[i];
    let old_j = state[j];

    state[i] = gate[[0, 0]] * old_i + gate[[0, 1]] * old_j;
    state[j] = gate[[1, 0]] * old_i + gate[[1, 1]] * old_j;
}

/// Apply a d x d unitary gate to d amplitudes at given indices.
///
/// The gate transforms the amplitudes as: new_amps = gate * old_amps
///
/// # Arguments
/// * `state` - Mutable slice of complex amplitudes
/// * `indices` - Slice of d indices corresponding to the d basis states
/// * `gate` - d x d unitary matrix
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use yao_rs::instruct::udrows;
///
/// // Apply a 3x3 permutation matrix to a qutrit
/// let mut state = vec![
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0),
/// ];
/// // Cyclic permutation: |0⟩ -> |1⟩ -> |2⟩ -> |0⟩
/// let perm = Array2::from_shape_vec((3, 3), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// udrows(&mut state, &[0, 1, 2], &perm);
/// // Now state is |1⟩
/// assert!((state[0].norm() - 0.0).abs() < 1e-10);
/// assert!((state[1].norm() - 1.0).abs() < 1e-10);
/// assert!((state[2].norm() - 0.0).abs() < 1e-10);
/// ```
pub fn udrows(state: &mut [Complex64], indices: &[usize], gate: &Array2<Complex64>) {
    let d = indices.len();
    debug_assert_eq!(gate.nrows(), d);
    debug_assert_eq!(gate.ncols(), d);

    // Collect old amplitudes
    let old_amps: Vec<Complex64> = indices.iter().map(|&idx| state[idx]).collect();

    // Compute new amplitudes: new[i] = sum_j gate[i,j] * old[j]
    for (i, &out_idx) in indices.iter().enumerate() {
        let mut new_amp = Complex64::new(0.0, 0.0);
        for (j, &old_amp) in old_amps.iter().enumerate() {
            new_amp += gate[[i, j]] * old_amp;
        }
        state[out_idx] = new_amp;
    }
}

/// Multiply an amplitude at index i by a scalar factor.
///
/// Used for diagonal gates like Z, S, T, and Phase gates.
///
/// # Arguments
/// * `state` - Mutable slice of complex amplitudes
/// * `i` - Index of the amplitude to multiply
/// * `factor` - Complex scalar to multiply by
///
/// # Example
/// ```
/// use num_complex::Complex64;
/// use yao_rs::instruct::mulrow;
///
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
/// // Apply phase e^(iπ/4) to second amplitude
/// let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
/// mulrow(&mut state, 1, phase);
/// assert!((state[1] - phase).norm() < 1e-10);
/// ```
pub fn mulrow(state: &mut [Complex64], i: usize, factor: Complex64) {
    state[i] *= factor;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

    fn approx_eq(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-10
    }

    #[test]
    fn test_u1rows_hadamard_on_zero() {
        // H|0⟩ = (|0⟩ + |1⟩) / √2
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

        u1rows(&mut state, 0, 1, &h_gate);

        assert!(approx_eq(state[0], s));
        assert!(approx_eq(state[1], s));
    }

    #[test]
    fn test_u1rows_hadamard_on_one() {
        // H|1⟩ = (|0⟩ - |1⟩) / √2
        let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

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
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

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
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

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
            Complex64::new(1.0, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),  // |01⟩
            Complex64::new(0.0, 0.0),  // |10⟩
            Complex64::new(0.0, 0.0),  // |11⟩
        ];

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

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
            Complex64::new(1.0, 0.0),  // |0⟩
            Complex64::new(0.0, 0.0),  // |1⟩
            Complex64::new(0.0, 0.0),  // |2⟩
        ];

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Cyclic permutation matrix: P|k⟩ = |k+1 mod 3⟩
        // P = [[0,0,1], [1,0,0], [0,1,0]]
        let perm = Array2::from_shape_vec(
            (3, 3),
            vec![
                zero, zero, one,
                one, zero, zero,
                zero, one, zero,
            ],
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
            Complex64::new(1.0, 0.0),  // |0⟩
            Complex64::new(0.0, 0.0),  // |1⟩
            Complex64::new(0.0, 0.0),  // |2⟩
        ];

        // Create a unitary that maps |0⟩ to (|0⟩ + |1⟩ + |2⟩)/√3
        let s = Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0);
        // First column is [s, s, s]^T
        // We need a full unitary; use a simple one
        let gate = Array2::from_shape_vec(
            (3, 3),
            vec![
                s, s, s,
                s, s * Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0),
                s * Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 3.0),
                s, s * Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 3.0),
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
            Complex64::new(1.0, 0.0),  // index 0
            Complex64::new(0.0, 0.0),  // index 1
            Complex64::new(0.0, 0.0),  // index 2
            Complex64::new(0.0, 0.0),  // index 3
        ];

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Permutation that swaps 0<->3 and 1<->2
        let gate = Array2::from_shape_vec(
            (4, 4),
            vec![
                zero, zero, zero, one,
                zero, zero, one, zero,
                zero, one, zero, zero,
                one, zero, zero, zero,
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
            Complex64::new(1.0, 0.0),  // index 0
            Complex64::new(0.5, 0.0),  // index 1
            Complex64::new(0.0, 0.0),  // index 2
            Complex64::new(0.5, 0.0),  // index 3
        ];

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Apply X gate to indices 0 and 2 only
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

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
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

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
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

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
        let mut state = vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
        ];

        let norm_before: f64 = state.iter().map(|c| c.norm_sqr()).sum();

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

        u1rows(&mut state, 0, 1, &h_gate);

        let norm_after: f64 = state.iter().map(|c| c.norm_sqr()).sum();

        assert!((norm_before - norm_after).abs() < 1e-10);
    }
}
