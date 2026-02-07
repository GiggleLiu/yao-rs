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
/// * `i` - Index of the first amplitude (corresponds to |0> in the gate's basis)
/// * `j` - Index of the second amplitude (corresponds to |1> in the gate's basis)
/// * `gate` - 2x2 unitary matrix
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use yao_rs::instruct::u1rows;
///
/// // Apply X gate to |0> state
/// let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// u1rows(&mut state, 0, 1, &x_gate);
/// // Now state is |1>
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
/// // Cyclic permutation: |0> -> |1> -> |2> -> |0>
/// let perm = Array2::from_shape_vec((3, 3), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// udrows(&mut state, &[0, 1, 2], &perm);
/// // Now state is |1>
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
/// // Apply phase e^(ipi/4) to second amplitude
/// let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
/// mulrow(&mut state, 1, phase);
/// assert!((state[1] - phase).norm() < 1e-10);
/// ```
pub fn mulrow(state: &mut [Complex64], i: usize, factor: Complex64) {
    state[i] *= factor;
}

/// Apply a general d x d gate at a single site location.
///
/// For each configuration of the other sites (not `loc`), this function
/// gathers the d amplitude indices corresponding to varying the site at `loc`
/// from 0 to d-1, then applies the gate using `udrows`.
///
/// # Arguments
/// * `state` - The quantum state to modify
/// * `gate` - d x d unitary matrix where d = state.dims[loc]
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
/// let mut state = State::zero_state(&[2, 2]); // |00>
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// instruct_single(&mut state, &x_gate, 0);
/// // Now state is |10>
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
/// // CNOT gate: X on target when control is |1>
/// let mut state = State::product_state(&[2, 2], &[1, 0]); // |10>
/// let x_gate = Array2::from_shape_vec((2, 2), vec![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
/// ]).unwrap();
/// instruct_controlled(&mut state, &x_gate, &[0], &[1], &[1]);
/// // Now state is |11>
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
    debug_assert_eq!(
        gate.nrows(),
        tgt_dim,
        "gate rows must match target dimension"
    );
    debug_assert_eq!(
        gate.ncols(),
        tgt_dim,
        "gate cols must match target dimension"
    );

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
        let other_locs: Vec<usize> = (0..state.dims.len()).filter(|&i| i != tgt_loc).collect();

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
