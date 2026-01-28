//! Primitive amplitude operations for state vector simulation.
//!
//! These functions directly manipulate amplitudes in the state vector,
//! providing O(1) operations for applying gates to specific indices.

use ndarray::Array2;
use num_complex::Complex64;

use crate::index::{insert_index, iter_basis, linear_to_indices, mixed_radix_index};
use crate::state::State;

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

/// Apply a general d×d gate at a single site location.
///
/// For each configuration of the other sites (not `loc`), this function
/// gathers the d amplitude indices corresponding to varying the site at `loc`
/// from 0 to d-1, then applies the gate using `udrows`.
///
/// # Arguments
/// * `state` - The quantum state to modify
/// * `gate` - d×d unitary matrix where d = state.dims[loc]
/// * `loc` - The site index where the gate is applied
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use yao_rs::state::State;
/// use yao_rs::instruct::instruct_single;
///
/// // Apply X gate to qubit 0 in a 2-qubit system
/// let mut state = State::zero_state(&[2, 2]); // |00⟩
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// instruct_single(&mut state, &x_gate, 0);
/// // Now state is |10⟩
/// assert!((state.data[2].norm() - 1.0).abs() < 1e-10);
/// ```
pub fn instruct_single(state: &mut State, gate: &Array2<Complex64>, loc: usize) {
    let d = state.dims[loc];
    debug_assert_eq!(gate.nrows(), d);
    debug_assert_eq!(gate.ncols(), d);

    // Build dims for the "other" sites (all sites except loc)
    let other_dims: Vec<usize> = state
        .dims
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != loc)
        .map(|(_, &dim)| dim)
        .collect();

    // If there are no other sites (single-site system), just apply the gate directly
    if other_dims.is_empty() {
        let indices: Vec<usize> = (0..d).collect();
        udrows(state.data.as_slice_mut().unwrap(), &indices, gate);
        return;
    }

    // Iterate over all configurations of the other sites
    for (_, other_basis) in iter_basis(&other_dims) {
        // For each value k at site loc, compute the flat index
        let indices: Vec<usize> = (0..d)
            .map(|k| {
                let full_indices = insert_index(&other_basis, loc, k);
                mixed_radix_index(&full_indices, &state.dims)
            })
            .collect();

        // Apply the gate to these d amplitudes
        udrows(state.data.as_slice_mut().unwrap(), &indices, gate);
    }
}

/// Apply a controlled gate to target sites when control sites have specified values.
///
/// This function applies a gate to the target sites only when all control sites
/// have their specified configuration values. This is used for controlled gates
/// like CNOT, Toffoli, and controlled-U operations.
///
/// # Arguments
/// * `state` - The quantum state to modify
/// * `gate` - The gate matrix to apply. Size must be d^n x d^n where d is the
///   target site dimension and n is the number of target sites
/// * `ctrl_locs` - Indices of the control sites
/// * `ctrl_configs` - Configuration values for each control site (gate applies when matched)
/// * `tgt_locs` - Indices of the target sites where the gate is applied
///
/// # Panics
/// * If `ctrl_locs` and `ctrl_configs` have different lengths
/// * If the gate dimension doesn't match the product of target site dimensions
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use yao_rs::state::State;
/// use yao_rs::instruct::instruct_controlled;
///
/// // CNOT gate: X on target when control is |1⟩
/// let mut state = State::product_state(&[2, 2], &[1, 0]); // |10⟩
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
/// // Now state is |11⟩
/// assert!((state.data[3].norm() - 1.0).abs() < 1e-10);
/// ```
pub fn instruct_controlled(
    state: &mut State,
    gate: &Array2<Complex64>,
    ctrl_locs: &[usize],
    ctrl_configs: &[usize],
    tgt_locs: &[usize],
) {
    debug_assert_eq!(
        ctrl_locs.len(),
        ctrl_configs.len(),
        "ctrl_locs and ctrl_configs must have the same length"
    );

    // Handle edge case: no controls is equivalent to instruct_single (for single target)
    if ctrl_locs.is_empty() && tgt_locs.len() == 1 {
        instruct_single(state, gate, tgt_locs[0]);
        return;
    }
    // For multi-target gates without controls, we fall through to the general algorithm

    // Calculate the dimension of the target subspace
    let tgt_dim: usize = tgt_locs.iter().map(|&loc| state.dims[loc]).product();
    debug_assert_eq!(gate.nrows(), tgt_dim, "gate rows must match target dimension");
    debug_assert_eq!(gate.ncols(), tgt_dim, "gate cols must match target dimension");

    // For single-target gates, we need to iterate over states where:
    // 1. Controls have their specified values
    // 2. All other sites (except target) vary freely
    //
    // For each such configuration, gather the indices where target varies from 0 to d-1

    if tgt_locs.len() == 1 {
        let tgt_loc = tgt_locs[0];
        let d = state.dims[tgt_loc];

        // Build dims for "other" sites (excluding target but including controls)
        // Controls are fixed, others vary freely
        let other_locs: Vec<usize> = (0..state.dims.len())
            .filter(|&i| i != tgt_loc)
            .collect();

        let other_dims: Vec<usize> = other_locs.iter().map(|&i| state.dims[i]).collect();

        // Map ctrl_locs to positions in other_locs
        let ctrl_positions_in_other: Vec<usize> = ctrl_locs
            .iter()
            .map(|&cl| other_locs.iter().position(|&ol| ol == cl).unwrap())
            .collect();

        // Iterate over all configurations of other sites where controls match
        for (_, other_basis) in iter_basis(&other_dims) {
            // Check if controls match
            let controls_match = ctrl_positions_in_other
                .iter()
                .zip(ctrl_configs.iter())
                .all(|(&pos, &config)| other_basis[pos] == config);

            if !controls_match {
                continue;
            }

            // For each value k at the target site, compute the flat index
            let indices: Vec<usize> = (0..d)
                .map(|k| {
                    let full_indices = insert_index(&other_basis, tgt_loc, k);
                    mixed_radix_index(&full_indices, &state.dims)
                })
                .collect();

            // Apply the gate to these d amplitudes
            udrows(state.data.as_slice_mut().unwrap(), &indices, gate);
        }
    } else {
        // Multi-target case: more complex
        // Build dimensions for "other" sites (excluding all targets)
        let other_locs: Vec<usize> = (0..state.dims.len())
            .filter(|&i| !tgt_locs.contains(&i))
            .collect();

        let other_dims: Vec<usize> = other_locs.iter().map(|&i| state.dims[i]).collect();

        // Map ctrl_locs to positions in other_locs
        let ctrl_positions_in_other: Vec<usize> = ctrl_locs
            .iter()
            .map(|&cl| other_locs.iter().position(|&ol| ol == cl).unwrap())
            .collect();

        // Iterate over all configurations of other sites where controls match
        for (_, other_basis) in iter_basis(&other_dims) {
            // Check if controls match
            let controls_match = ctrl_positions_in_other
                .iter()
                .zip(ctrl_configs.iter())
                .all(|(&pos, &config)| other_basis[pos] == config);

            if !controls_match {
                continue;
            }

            // Build indices for all target configurations
            let tgt_dims: Vec<usize> = tgt_locs.iter().map(|&loc| state.dims[loc]).collect();
            let indices: Vec<usize> = iter_basis(&tgt_dims)
                .map(|(_, tgt_basis)| {
                    // Reconstruct full indices: merge other_basis and tgt_basis
                    let mut full_indices = vec![0; state.dims.len()];
                    for (i, &loc) in other_locs.iter().enumerate() {
                        full_indices[loc] = other_basis[i];
                    }
                    for (i, &loc) in tgt_locs.iter().enumerate() {
                        full_indices[loc] = tgt_basis[i];
                    }
                    mixed_radix_index(&full_indices, &state.dims)
                })
                .collect();

            // Apply the gate to these indices
            udrows(state.data.as_slice_mut().unwrap(), &indices, gate);
        }
    }
}

/// Apply a diagonal gate at a single site location.
///
/// This is an optimized version for diagonal gates (Z, S, T, Phase, Rz, etc.)
/// where only the diagonal elements matter. Each amplitude is multiplied by
/// the appropriate phase based on the value at the target site.
///
/// # Arguments
/// * `state` - The quantum state to modify
/// * `phases` - Slice of d complex phases where d = state.dims[loc].
///   phases[k] is applied when the site at `loc` has value k.
/// * `loc` - The site index where the gate is applied
///
/// # Example
/// ```
/// use num_complex::Complex64;
/// use yao_rs::state::State;
/// use yao_rs::instruct::instruct_diagonal;
///
/// // Apply Z gate to qubit 0: Z = diag(1, -1)
/// let mut state = State::zero_state(&[2, 2]);
/// // First apply something to get superposition, then Z
/// let phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
/// instruct_diagonal(&mut state, &phases, 0);
/// ```
pub fn instruct_diagonal(state: &mut State, phases: &[Complex64], loc: usize) {
    let d = state.dims[loc];
    debug_assert_eq!(phases.len(), d);

    let total_dim = state.total_dim();

    // For each basis state, get the value at loc and multiply by the corresponding phase
    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);
        let val_at_loc = indices[loc];
        state.data[flat_idx] *= phases[val_at_loc];
    }
}

/// Parallel version of `instruct_diagonal` using Rayon.
///
/// Each amplitude can be processed independently, making this embarrassingly parallel.
/// The phase applied to each amplitude depends only on the value at the target site.
#[cfg(feature = "parallel")]
pub fn instruct_diagonal_parallel(state: &mut State, phases: &[Complex64], loc: usize) {
    use rayon::prelude::*;

    let d = state.dims[loc];
    debug_assert_eq!(phases.len(), d);

    let dims = state.dims.clone();

    state
        .data
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_idx, amp)| {
            let indices = linear_to_indices(flat_idx, &dims);
            let val_at_loc = indices[loc];
            *amp *= phases[val_at_loc];
        });
}

/// Parallel version of `instruct_single` using Rayon.
///
/// Partitions basis states into independent groups that can be processed in parallel.
/// Each group corresponds to a fixed configuration of "other" sites, and the gate
/// is applied to the d amplitudes that vary at the target site.
#[cfg(feature = "parallel")]
pub fn instruct_single_parallel(state: &mut State, gate: &Array2<Complex64>, loc: usize) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicPtr, Ordering};

    let d = state.dims[loc];
    debug_assert_eq!(gate.nrows(), d);
    debug_assert_eq!(gate.ncols(), d);

    // Build dims for the "other" sites (all sites except loc)
    let other_dims: Vec<usize> = state
        .dims
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != loc)
        .map(|(_, &dim)| dim)
        .collect();

    // If there are no other sites (single-site system), just apply the gate directly
    if other_dims.is_empty() {
        let indices: Vec<usize> = (0..d).collect();
        udrows(state.data.as_slice_mut().unwrap(), &indices, gate);
        return;
    }

    // Collect all the "other" basis configurations
    let other_bases: Vec<Vec<usize>> = iter_basis(&other_dims).map(|(_, basis)| basis).collect();

    let dims = state.dims.clone();

    // Process groups in parallel using atomic pointer wrapper for Send + Sync
    let state_ptr = AtomicPtr::new(state.data.as_mut_ptr());
    let state_len = state.data.len();

    // Safety: We partition the state space into disjoint groups.
    // Each group consists of d indices that differ only at `loc`.
    // Different `other_basis` configurations yield completely disjoint index sets.
    other_bases.par_iter().for_each(|other_basis| {
        let ptr = state_ptr.load(Ordering::Relaxed);

        // For each value k at site loc, compute the flat index
        let indices: Vec<usize> = (0..d)
            .map(|k| {
                let full_indices = insert_index(other_basis, loc, k);
                mixed_radix_index(&full_indices, &dims)
            })
            .collect();

        // Collect old amplitudes (safe because indices are disjoint across iterations)
        let old_amps: Vec<Complex64> = indices
            .iter()
            .map(|&idx| {
                debug_assert!(idx < state_len);
                // Safety: idx is within bounds and each idx is accessed by only one thread
                unsafe { *ptr.add(idx) }
            })
            .collect();

        // Compute new amplitudes and write back
        for (i, &out_idx) in indices.iter().enumerate() {
            let mut new_amp = Complex64::new(0.0, 0.0);
            for (j, &old_amp) in old_amps.iter().enumerate() {
                new_amp += gate[[i, j]] * old_amp;
            }
            // Safety: out_idx is within bounds and each out_idx is written by only one thread
            unsafe {
                *ptr.add(out_idx) = new_amp;
            }
        }
    });
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

    // Tests for instruct_single and instruct_diagonal
    use crate::state::State;

    #[test]
    fn test_instruct_single_h_gate() {
        // Apply H gate to qubit 0 in a 2-qubit system |00⟩
        // H|0⟩ = (|0⟩ + |1⟩) / √2
        // Result: (|00⟩ + |10⟩) / √2
        let mut state = State::zero_state(&[2, 2]);

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

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
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

        instruct_single(&mut state, &x_gate, 1);

        // |00⟩ = index 0, |01⟩ = index 1, |10⟩ = index 2, |11⟩ = index 3
        assert!(approx_eq(state.data[0], zero)); // |00⟩
        assert!(approx_eq(state.data[1], one));  // |01⟩
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
            vec![
                zero, zero, one,
                one, zero, zero,
                zero, one, zero,
            ],
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

        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

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

        state.data[0] = s;  // |00⟩
        state.data[2] = s;  // |10⟩

        let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        instruct_diagonal(&mut state, &z_phases, 0);

        assert!(approx_eq(state.data[0], s));           // |00⟩ unchanged
        assert!(approx_eq(state.data[1], Complex64::new(0.0, 0.0))); // |01⟩ still zero
        assert!(approx_eq(state.data[2], -s));          // |10⟩ flipped sign
        assert!(approx_eq(state.data[3], Complex64::new(0.0, 0.0))); // |11⟩ still zero
    }

    // Tests for instruct_controlled

    #[test]
    fn test_instruct_controlled_cnot() {
        // CNOT: control=0, target=1
        // |00⟩ -> |00⟩, |01⟩ -> |01⟩, |10⟩ -> |11⟩, |11⟩ -> |10⟩

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

        // Test |10⟩ -> |11⟩
        let mut state = State::product_state(&[2, 2], &[1, 0]);
        instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

        assert!(approx_eq(state.data[0], zero)); // |00⟩
        assert!(approx_eq(state.data[1], zero)); // |01⟩
        assert!(approx_eq(state.data[2], zero)); // |10⟩
        assert!(approx_eq(state.data[3], one));  // |11⟩

        // Test |00⟩ -> |00⟩ (control not active)
        let mut state = State::product_state(&[2, 2], &[0, 0]);
        instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

        assert!(approx_eq(state.data[0], one));  // |00⟩ unchanged
        assert!(approx_eq(state.data[1], zero)); // |01⟩
        assert!(approx_eq(state.data[2], zero)); // |10⟩
        assert!(approx_eq(state.data[3], zero)); // |11⟩

        // Test |11⟩ -> |10⟩
        let mut state = State::product_state(&[2, 2], &[1, 1]);
        instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

        assert!(approx_eq(state.data[0], zero)); // |00⟩
        assert!(approx_eq(state.data[1], zero)); // |01⟩
        assert!(approx_eq(state.data[2], one));  // |10⟩
        assert!(approx_eq(state.data[3], zero)); // |11⟩
    }

    #[test]
    fn test_instruct_controlled_toffoli() {
        // Toffoli (CCX): two controls (sites 0 and 1), target (site 2)
        // Only flips target when both controls are |1⟩

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

        // Test |110⟩ -> |111⟩ (both controls active)
        // dims=[2,2,2]: |110⟩ = 1*4 + 1*2 + 0 = 6
        let mut state = State::product_state(&[2, 2, 2], &[1, 1, 0]);
        instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

        assert!(approx_eq(state.data[6], zero)); // |110⟩ -> 0
        assert!(approx_eq(state.data[7], one));  // |111⟩ -> 1

        // Test |100⟩ -> |100⟩ (only one control active)
        let mut state = State::product_state(&[2, 2, 2], &[1, 0, 0]);
        instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

        assert!(approx_eq(state.data[4], one));  // |100⟩ unchanged
        assert!(approx_eq(state.data[5], zero)); // |101⟩

        // Test |010⟩ -> |010⟩ (only one control active)
        let mut state = State::product_state(&[2, 2, 2], &[0, 1, 0]);
        instruct_controlled(&mut state, &x_gate, &[0, 1], &[1, 1], &[2]);

        assert!(approx_eq(state.data[2], one));  // |010⟩ unchanged
        assert!(approx_eq(state.data[3], zero)); // |011⟩
    }

    #[test]
    fn test_instruct_controlled_qutrit() {
        // Control on a qutrit with value = 2
        // dims=[3, 2]: qutrit (control) and qubit (target)

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

        // Test |20⟩ -> |21⟩ (control value = 2 is active)
        // dims=[3,2]: |20⟩ = 2*2 + 0 = 4
        let mut state = State::product_state(&[3, 2], &[2, 0]);
        instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

        assert!(approx_eq(state.data[4], zero)); // |20⟩
        assert!(approx_eq(state.data[5], one));  // |21⟩

        // Test |10⟩ -> |10⟩ (control value = 1, not active)
        let mut state = State::product_state(&[3, 2], &[1, 0]);
        instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

        assert!(approx_eq(state.data[2], one));  // |10⟩ unchanged
        assert!(approx_eq(state.data[3], zero)); // |11⟩

        // Test |00⟩ -> |00⟩ (control value = 0, not active)
        let mut state = State::product_state(&[3, 2], &[0, 0]);
        instruct_controlled(&mut state, &x_gate, &[0], &[2], &[1]);

        assert!(approx_eq(state.data[0], one));  // |00⟩ unchanged
        assert!(approx_eq(state.data[1], zero)); // |01⟩
    }

    #[test]
    fn test_instruct_controlled_no_controls() {
        // Empty ctrl_locs should behave like instruct_single

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

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

        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![s, s, s, -s],
        )
        .unwrap();

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
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![zero, one, one, zero],
        )
        .unwrap();

        instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);

        // Result: (|00⟩ + |11⟩) / √2
        assert!(approx_eq(state.data[0], s));    // |00⟩
        assert!(approx_eq(state.data[1], zero)); // |01⟩
        assert!(approx_eq(state.data[2], zero)); // |10⟩ -> |11⟩
        assert!(approx_eq(state.data[3], s));    // |11⟩
    }

    // Tests for Pauli gate instructions
    use crate::gate::Gate;

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
        use super::*;

        fn approx_eq(a: Complex64, b: Complex64) -> bool {
            (a - b).norm() < 1e-10
        }

        #[test]
        fn test_instruct_diagonal_parallel_matches_sequential() {
            // Test that parallel diagonal gives same results as sequential
            let mut state_seq = State::zero_state(&[2, 2, 2, 2]); // 16 amplitudes
            let mut state_par = state_seq.clone();

            // Create superposition state
            let amp = Complex64::new(0.25, 0.0);
            for i in 0..16 {
                state_seq.data[i] = amp;
                state_par.data[i] = amp;
            }

            let z_phases = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

            instruct_diagonal(&mut state_seq, &z_phases, 1);
            super::super::instruct_diagonal_parallel(&mut state_par, &z_phases, 1);

            for i in 0..16 {
                assert!(approx_eq(state_seq.data[i], state_par.data[i]));
            }
        }

        #[test]
        fn test_instruct_diagonal_parallel_large_state() {
            // Test with a larger state (5 qubits = 32 amplitudes)
            let mut state_seq = State::zero_state(&[2, 2, 2, 2, 2]);
            let mut state_par = state_seq.clone();

            // Create random-ish amplitudes
            for i in 0..32 {
                let re = (i as f64 * 0.1).sin();
                let im = (i as f64 * 0.1).cos();
                state_seq.data[i] = Complex64::new(re, im);
                state_par.data[i] = Complex64::new(re, im);
            }

            // Apply phase gate
            let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
            let phases = [Complex64::new(1.0, 0.0), phase];

            instruct_diagonal(&mut state_seq, &phases, 2);
            super::super::instruct_diagonal_parallel(&mut state_par, &phases, 2);

            for i in 0..32 {
                assert!(approx_eq(state_seq.data[i], state_par.data[i]));
            }
        }

        #[test]
        fn test_instruct_single_parallel_matches_sequential() {
            // Test that parallel single gate gives same results as sequential
            let mut state_seq = State::zero_state(&[2, 2, 2, 2]); // 16 amplitudes
            let mut state_par = state_seq.clone();

            // Create superposition state
            let amp = Complex64::new(0.25, 0.0);
            for i in 0..16 {
                state_seq.data[i] = amp;
                state_par.data[i] = amp;
            }

            let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
            let h_gate = Array2::from_shape_vec(
                (2, 2),
                vec![s, s, s, -s],
            )
            .unwrap();

            instruct_single(&mut state_seq, &h_gate, 1);
            super::super::instruct_single_parallel(&mut state_par, &h_gate, 1);

            for i in 0..16 {
                assert!(approx_eq(state_seq.data[i], state_par.data[i]));
            }
        }

        #[test]
        fn test_instruct_single_parallel_large_state() {
            // Test with a larger state (5 qubits = 32 amplitudes)
            let mut state_seq = State::zero_state(&[2, 2, 2, 2, 2]);
            let mut state_par = state_seq.clone();

            // Create random-ish amplitudes
            for i in 0..32 {
                let re = (i as f64 * 0.1).sin();
                let im = (i as f64 * 0.1).cos();
                state_seq.data[i] = Complex64::new(re, im);
                state_par.data[i] = Complex64::new(re, im);
            }

            // Apply X gate
            let zero = Complex64::new(0.0, 0.0);
            let one = Complex64::new(1.0, 0.0);
            let x_gate = Array2::from_shape_vec(
                (2, 2),
                vec![zero, one, one, zero],
            )
            .unwrap();

            instruct_single(&mut state_seq, &x_gate, 2);
            super::super::instruct_single_parallel(&mut state_par, &x_gate, 2);

            for i in 0..32 {
                assert!(approx_eq(state_seq.data[i], state_par.data[i]));
            }
        }

        #[test]
        fn test_instruct_single_parallel_multiple_gates() {
            // Apply multiple gates and verify results match
            let mut state_seq = State::zero_state(&[2, 2, 2, 2]);
            let mut state_par = state_seq.clone();

            let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
            let h_gate = Array2::from_shape_vec(
                (2, 2),
                vec![s, s, s, -s],
            )
            .unwrap();

            // Apply H to all qubits
            for loc in 0..4 {
                instruct_single(&mut state_seq, &h_gate, loc);
                super::super::instruct_single_parallel(&mut state_par, &h_gate, loc);
            }

            for i in 0..16 {
                assert!(approx_eq(state_seq.data[i], state_par.data[i]));
            }
        }
    }

    #[test]
    fn test_instruct_h_creates_superposition() {
        // H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
        let mut state = State::zero_state(&[2]); // |0⟩
        let h_gate = Gate::H.matrix(2);
        instruct_single(&mut state, &h_gate, 0);

        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], s));

        // H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2
        let mut state = State::product_state(&[2], &[1]); // |1⟩
        instruct_single(&mut state, &h_gate, 0);

        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], -s));
    }

    #[test]
    fn test_instruct_h_on_each_qubit() {
        // 3-qubit system, apply H to qubit 0, 1, 2 separately
        // Verify each creates proper superposition
        let h_gate = Gate::H.matrix(2);
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        // Test H on qubit 0: |000⟩ -> (|000⟩ + |100⟩)/√2
        let mut state = State::zero_state(&[2, 2, 2]); // |000⟩
        instruct_single(&mut state, &h_gate, 0);
        // |000⟩ = index 0, |100⟩ = index 4
        assert!(approx_eq(state.data[0], s)); // |000⟩
        assert!(approx_eq(state.data[4], s)); // |100⟩
        for i in [1, 2, 3, 5, 6, 7] {
            assert!(approx_eq(state.data[i], zero));
        }

        // Test H on qubit 1: |000⟩ -> (|000⟩ + |010⟩)/√2
        let mut state = State::zero_state(&[2, 2, 2]); // |000⟩
        instruct_single(&mut state, &h_gate, 1);
        // |000⟩ = index 0, |010⟩ = index 2
        assert!(approx_eq(state.data[0], s)); // |000⟩
        assert!(approx_eq(state.data[2], s)); // |010⟩
        for i in [1, 3, 4, 5, 6, 7] {
            assert!(approx_eq(state.data[i], zero));
        }

        // Test H on qubit 2: |000⟩ -> (|000⟩ + |001⟩)/√2
        let mut state = State::zero_state(&[2, 2, 2]); // |000⟩
        instruct_single(&mut state, &h_gate, 2);
        // |000⟩ = index 0, |001⟩ = index 1
        assert!(approx_eq(state.data[0], s)); // |000⟩
        assert!(approx_eq(state.data[1], s)); // |001⟩
        for i in [2, 3, 4, 5, 6, 7] {
            assert!(approx_eq(state.data[i], zero));
        }
    }

    #[test]
    fn test_instruct_s_gate() {
        // S = diag(1, i)
        // S|0⟩ = |0⟩, S|1⟩ = i|1⟩
        // Use instruct_diagonal
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let zero = Complex64::new(0.0, 0.0);
        let s_phases = [one, i];

        // S|0⟩ = |0⟩
        let mut state = State::zero_state(&[2]); // |0⟩
        instruct_diagonal(&mut state, &s_phases, 0);
        assert!(approx_eq(state.data[0], one));
        assert!(approx_eq(state.data[1], zero));

        // S|1⟩ = i|1⟩
        let mut state = State::product_state(&[2], &[1]); // |1⟩
        instruct_diagonal(&mut state, &s_phases, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], i));

        // S on |+⟩ = (|0⟩ + |1⟩)/√2 -> (|0⟩ + i|1⟩)/√2
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
        // T = diag(1, e^(iπ/4))
        // T|0⟩ = |0⟩, T|1⟩ = e^(iπ/4)|1⟩
        // Use instruct_diagonal
        let one = Complex64::new(1.0, 0.0);
        let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
        let zero = Complex64::new(0.0, 0.0);
        let t_phases = [one, t_phase];

        // T|0⟩ = |0⟩
        let mut state = State::zero_state(&[2]); // |0⟩
        instruct_diagonal(&mut state, &t_phases, 0);
        assert!(approx_eq(state.data[0], one));
        assert!(approx_eq(state.data[1], zero));

        // T|1⟩ = e^(iπ/4)|1⟩
        let mut state = State::product_state(&[2], &[1]); // |1⟩
        instruct_diagonal(&mut state, &t_phases, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], t_phase));

        // T on |+⟩ = (|0⟩ + |1⟩)/√2 -> (|0⟩ + e^(iπ/4)|1⟩)/√2
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_diagonal(&mut state, &t_phases, 0);
        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], s * t_phase));

        // Verify T^2 = S: T|1⟩ then T|1⟩ should give S|1⟩ = i|1⟩
        let i = Complex64::new(0.0, 1.0);
        let mut state = State::product_state(&[2], &[1]); // |1⟩
        instruct_diagonal(&mut state, &t_phases, 0);
        instruct_diagonal(&mut state, &t_phases, 0);
        // e^(iπ/4) * e^(iπ/4) = e^(iπ/2) = i
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], i));
    }

    #[test]
    fn test_instruct_identity() {
        // I|ψ⟩ = |ψ⟩ for any state
        // Test on various states
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        // 2x2 identity matrix
        let identity = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();

        // Test I|0⟩ = |0⟩
        let mut state = State::zero_state(&[2]); // |0⟩
        let state_before = state.clone();
        instruct_single(&mut state, &identity, 0);
        assert!(approx_eq(state.data[0], state_before.data[0]));
        assert!(approx_eq(state.data[1], state_before.data[1]));

        // Test I|1⟩ = |1⟩
        let mut state = State::product_state(&[2], &[1]); // |1⟩
        let state_before = state.clone();
        instruct_single(&mut state, &identity, 0);
        assert!(approx_eq(state.data[0], state_before.data[0]));
        assert!(approx_eq(state.data[1], state_before.data[1]));

        // Test I|+⟩ = |+⟩ (superposition state)
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        let state_before = state.clone();
        instruct_single(&mut state, &identity, 0);
        assert!(approx_eq(state.data[0], state_before.data[0]));
        assert!(approx_eq(state.data[1], state_before.data[1]));

        // Test I on arbitrary state
        let amp0 = Complex64::new(0.6, 0.2);
        let amp1 = Complex64::new(0.3, 0.7);
        let mut state = State::zero_state(&[2]);
        state.data[0] = amp0;
        state.data[1] = amp1;
        let state_before = state.clone();
        instruct_single(&mut state, &identity, 0);
        assert!(approx_eq(state.data[0], state_before.data[0]));
        assert!(approx_eq(state.data[1], state_before.data[1]));

        // Test I on 2-qubit system (apply to qubit 0)
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

        // Test I on 2-qubit system (apply to qubit 1)
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

    // Tests for parametric rotation gates
    use std::f64::consts::PI;

    #[test]
    fn test_instruct_rx_various_angles() {
        // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let neg_i = Complex64::new(0.0, -1.0);

        // Rx(0) = I
        let rx_0 = Gate::Rx(0.0).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &rx_0, 0);
        assert!(approx_eq(state.data[0], one));
        assert!(approx_eq(state.data[1], zero));

        let mut state = State::product_state(&[2], &[1]);
        instruct_single(&mut state, &rx_0, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], one));

        // Rx(π) ≈ -iX (up to global phase)
        // Rx(π)|0⟩ = -i|1⟩
        let rx_pi = Gate::Rx(PI).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &rx_pi, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], neg_i));

        // Rx(π)|1⟩ = -i|0⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_single(&mut state, &rx_pi, 0);
        assert!(approx_eq(state.data[0], neg_i));
        assert!(approx_eq(state.data[1], zero));

        // Rx(π/2) creates superposition
        // Rx(π/2)|0⟩ = (|0⟩ - i|1⟩)/√2
        let rx_pi_2 = Gate::Rx(PI / 2.0).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &rx_pi_2, 0);
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let neg_i_s = Complex64::new(0.0, -FRAC_1_SQRT_2);
        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], neg_i_s));

        // Verify normalization
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_instruct_ry_various_angles() {
        // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);

        // Ry(0) = I
        let ry_0 = Gate::Ry(0.0).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &ry_0, 0);
        assert!(approx_eq(state.data[0], one));
        assert!(approx_eq(state.data[1], zero));

        let mut state = State::product_state(&[2], &[1]);
        instruct_single(&mut state, &ry_0, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], one));

        // Ry(π) maps |0⟩ -> |1⟩, |1⟩ -> -|0⟩
        // cos(π/2) = 0, sin(π/2) = 1
        // Ry(π)|0⟩ = |1⟩
        let ry_pi = Gate::Ry(PI).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &ry_pi, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], one));

        // Ry(π)|1⟩ = -|0⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_single(&mut state, &ry_pi, 0);
        assert!(approx_eq(state.data[0], neg_one));
        assert!(approx_eq(state.data[1], zero));

        // Ry(π/2) creates superposition
        // Ry(π/2)|0⟩ = (|0⟩ + |1⟩)/√2
        let ry_pi_2 = Gate::Ry(PI / 2.0).matrix(2);
        let mut state = State::zero_state(&[2]);
        instruct_single(&mut state, &ry_pi_2, 0);
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], s));

        // Ry(π/2)|1⟩ = (-|0⟩ + |1⟩)/√2
        let mut state = State::product_state(&[2], &[1]);
        instruct_single(&mut state, &ry_pi_2, 0);
        let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
        assert!(approx_eq(state.data[0], neg_s));
        assert!(approx_eq(state.data[1], s));

        // Verify normalization
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_instruct_rz_various_angles() {
        // Rz(θ) = diag(e^(-iθ/2), e^(iθ/2))
        // Use instruct_diagonal since Rz is diagonal
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        // Rz(0) = I
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

        // Rz(π) = diag(e^(-iπ/2), e^(iπ/2)) = diag(-i, i)
        // Rz(π)|0⟩ = -i|0⟩
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

        // Rz(π)|1⟩ = i|1⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_diagonal(&mut state, &rz_phases_pi, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], i));

        // Rz(π/2) = diag(e^(-iπ/4), e^(iπ/4))
        let rz_pi_2 = Gate::Rz(PI / 2.0).matrix(2);
        let rz_phases_pi_2 = [rz_pi_2[[0, 0]], rz_pi_2[[1, 1]]];
        let exp_neg_pi_4 = Complex64::from_polar(1.0, -FRAC_PI_4);
        let exp_pi_4 = Complex64::from_polar(1.0, FRAC_PI_4);
        assert!(approx_eq(rz_phases_pi_2[0], exp_neg_pi_4));
        assert!(approx_eq(rz_phases_pi_2[1], exp_pi_4));

        // Test on |+⟩ state
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_diagonal(&mut state, &rz_phases_pi_2, 0);
        assert!(approx_eq(state.data[0], s * exp_neg_pi_4));
        assert!(approx_eq(state.data[1], s * exp_pi_4));

        // Verify normalization
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_instruct_phase_various_angles() {
        // Phase(θ) = diag(1, e^(iθ))
        // Use instruct_diagonal
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_one = Complex64::new(-1.0, 0.0);

        // Phase(π/4) = T gate: diag(1, e^(iπ/4))
        let phase_pi_4 = Gate::Phase(FRAC_PI_4).matrix(2);
        let t_phases = [phase_pi_4[[0, 0]], phase_pi_4[[1, 1]]];
        let exp_pi_4 = Complex64::from_polar(1.0, FRAC_PI_4);
        assert!(approx_eq(t_phases[0], one));
        assert!(approx_eq(t_phases[1], exp_pi_4));

        // Phase(π/4)|0⟩ = |0⟩
        let mut state = State::zero_state(&[2]);
        instruct_diagonal(&mut state, &t_phases, 0);
        assert!(approx_eq(state.data[0], one));
        assert!(approx_eq(state.data[1], zero));

        // Phase(π/4)|1⟩ = e^(iπ/4)|1⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_diagonal(&mut state, &t_phases, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], exp_pi_4));

        // Phase(π/2) = S gate: diag(1, i)
        let phase_pi_2 = Gate::Phase(PI / 2.0).matrix(2);
        let s_phases = [phase_pi_2[[0, 0]], phase_pi_2[[1, 1]]];
        assert!(approx_eq(s_phases[0], one));
        assert!(approx_eq(s_phases[1], i));

        // Phase(π/2)|1⟩ = i|1⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_diagonal(&mut state, &s_phases, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], i));

        // Phase(π) = Z gate: diag(1, -1)
        let phase_pi = Gate::Phase(PI).matrix(2);
        let z_phases = [phase_pi[[0, 0]], phase_pi[[1, 1]]];
        assert!(approx_eq(z_phases[0], one));
        assert!(approx_eq(z_phases[1], neg_one));

        // Phase(π)|1⟩ = -|1⟩
        let mut state = State::product_state(&[2], &[1]);
        instruct_diagonal(&mut state, &z_phases, 0);
        assert!(approx_eq(state.data[0], zero));
        assert!(approx_eq(state.data[1], neg_one));

        // Test on |+⟩ state with S gate
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_diagonal(&mut state, &s_phases, 0);
        assert!(approx_eq(state.data[0], s));
        assert!(approx_eq(state.data[1], s * i));

        // Verify normalization
        let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_instruct_rotation_on_superposition() {
        // Create |+⟩ = (|0⟩ + |1⟩)/√2, apply rotations, verify results
        let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let neg_i = Complex64::new(0.0, -1.0);
        let i = Complex64::new(0.0, 1.0);

        // Apply Rx(π) to |+⟩
        // Rx(π)|+⟩ = -iX|+⟩ = -i|+⟩ (since X|+⟩ = |+⟩)
        let rx_pi = Gate::Rx(PI).matrix(2);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_single(&mut state, &rx_pi, 0);
        // Rx(π) = [[0, -i], [-i, 0]]
        // Rx(π)|+⟩ = -i(|0⟩ + |1⟩)/√2 = -i|+⟩
        assert!(approx_eq(state.data[0], s * neg_i));
        assert!(approx_eq(state.data[1], s * neg_i));

        // Apply Ry(π) to |+⟩
        // Ry(π)|+⟩ = Y|+⟩ (without global phase) = (|1⟩ - |0⟩)/√2 = |-⟩ (but flipped)
        let ry_pi = Gate::Ry(PI).matrix(2);
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_single(&mut state, &ry_pi, 0);
        // Ry(π) = [[0, -1], [1, 0]]
        // Ry(π)(|0⟩+|1⟩)/√2 = (-|1⟩+|0⟩)/√2
        let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
        assert!(approx_eq(state.data[0], neg_s));
        assert!(approx_eq(state.data[1], s));

        // Apply Rz(π) to |+⟩
        // Rz(π)|+⟩ = (e^(-iπ/2)|0⟩ + e^(iπ/2)|1⟩)/√2 = (-i|0⟩ + i|1⟩)/√2
        let rz_pi = Gate::Rz(PI).matrix(2);
        let rz_phases_pi = [rz_pi[[0, 0]], rz_pi[[1, 1]]];
        let mut state = State::zero_state(&[2]);
        state.data[0] = s;
        state.data[1] = s;
        instruct_diagonal(&mut state, &rz_phases_pi, 0);
        assert!(approx_eq(state.data[0], s * neg_i));
        assert!(approx_eq(state.data[1], s * i));

        // Verify all states are normalized
        for angle in [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            // Test Rx
            let rx = Gate::Rx(angle).matrix(2);
            let mut state = State::zero_state(&[2]);
            state.data[0] = s;
            state.data[1] = s;
            instruct_single(&mut state, &rx, 0);
            let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
            assert!((norm - 1.0).abs() < 1e-10);

            // Test Ry
            let ry = Gate::Ry(angle).matrix(2);
            let mut state = State::zero_state(&[2]);
            state.data[0] = s;
            state.data[1] = s;
            instruct_single(&mut state, &ry, 0);
            let norm: f64 = state.data.iter().map(|c| c.norm_sqr()).sum();
            assert!((norm - 1.0).abs() < 1e-10);

            // Test Rz
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
}
