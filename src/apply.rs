use ndarray::Array2;
use num_complex::Complex64;

#[cfg(test)]
use ndarray::Array1;

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::instruct::{instruct_controlled, instruct_diagonal, instruct_single};
#[cfg(feature = "parallel")]
use crate::instruct::{instruct_diagonal_parallel, instruct_single_parallel};

/// Threshold for switching to parallel execution (2^14 = 16384 amplitudes).
/// Below this threshold, the overhead of Rayon is not worth it.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 16384;
use crate::state::State;

/// Decompose a flat index into a multi-index given dimensions (row-major order).
#[cfg(test)]
fn linear_to_multi(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % dims[i];
        index /= dims[i];
    }
    multi
}

/// Compose a multi-index into a flat index given dimensions (row-major order).
#[cfg(test)]
fn multi_to_linear(indices: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = dims[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

/// Simple matrix-vector multiplication for complex matrices and vectors.
#[cfg(test)]
fn matrix_vector_mul(mat: &Array2<Complex64>, vec: &Array1<Complex64>) -> Array1<Complex64> {
    let n = mat.nrows();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..mat.ncols() {
            sum += mat[[i, j]] * vec[j];
        }
        result[i] = sum;
    }
    result
}

/// Build the controlled local matrix on all involved sites (controls + targets).
///
/// all_dims: dimensions for all involved sites (control_locs ++ target_locs order)
/// gate_matrix: the gate's matrix on target sites only
/// control_configs: which control configuration triggers the gate
/// num_controls: number of control sites
#[cfg(test)]
fn build_controlled_matrix(
    all_dims: &[usize],
    gate_matrix: &Array2<Complex64>,
    control_configs: &[bool],
    num_controls: usize,
) -> Array2<Complex64> {
    if num_controls == 0 {
        return gate_matrix.clone();
    }

    let involved_dim: usize = all_dims.iter().product();
    let control_dims = &all_dims[..num_controls];
    let target_dims = &all_dims[num_controls..];
    let target_dim: usize = target_dims.iter().product();

    let mut mat = Array2::zeros((involved_dim, involved_dim));

    // Compute the trigger index in the control subspace
    let trigger_index: usize = control_configs
        .iter()
        .enumerate()
        .map(|(i, &cfg)| {
            let val = if cfg { 1usize } else { 0usize };
            let stride: usize = control_dims[i + 1..].iter().product();
            val * stride
        })
        .sum();

    let control_dim: usize = control_dims.iter().product();

    for ctrl_idx in 0..control_dim {
        for t_row in 0..target_dim {
            let row = ctrl_idx * target_dim + t_row;
            if ctrl_idx == trigger_index {
                // Apply the gate matrix on the target portion
                for t_col in 0..target_dim {
                    let col = ctrl_idx * target_dim + t_col;
                    mat[[row, col]] = gate_matrix[[t_row, t_col]];
                }
            } else {
                // Identity on the target portion
                let col = ctrl_idx * target_dim + t_row;
                mat[[row, col]] = Complex64::new(1.0, 0.0);
            }
        }
    }

    mat
}

/// Check if a gate is diagonal.
fn is_diagonal(gate: &Gate) -> bool {
    gate.is_diagonal()
}

/// Extract the diagonal phases from a diagonal gate matrix.
///
/// For a diagonal matrix, returns the diagonal elements as a vector.
fn extract_diagonal_phases(matrix: &Array2<Complex64>) -> Vec<Complex64> {
    let d = matrix.nrows();
    (0..d).map(|i| matrix[[i, i]]).collect()
}

/// Apply a circuit to a quantum state in-place using efficient instruct functions.
///
/// This function modifies the state directly without allocating new matrices.
/// For each gate in the circuit, it dispatches to the appropriate instruct
/// function based on whether the gate is diagonal and whether it has controls.
///
/// When the `parallel` feature is enabled and the state has >= 16384 amplitudes,
/// parallel variants of `instruct_diagonal` and `instruct_single` are used
/// for improved performance on large states.
pub fn apply_inplace(circuit: &Circuit, state: &mut State) {
    let dims = &circuit.dims;

    #[cfg(feature = "parallel")]
    let use_parallel = state.data.len() >= PARALLEL_THRESHOLD;

    for pg in &circuit.gates {
        // Get the gate's local matrix on target sites
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);

        if pg.control_locs.is_empty() {
            // No controls
            if is_diagonal(&pg.gate) {
                let phases = extract_diagonal_phases(&gate_matrix);
                for &loc in &pg.target_locs {
                    #[cfg(feature = "parallel")]
                    {
                        if use_parallel {
                            instruct_diagonal_parallel(state, &phases, loc);
                        } else {
                            instruct_diagonal(state, &phases, loc);
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        instruct_diagonal(state, &phases, loc);
                    }
                }
            } else {
                // For multi-qubit gates (like SWAP), we need to handle them differently
                if pg.target_locs.len() == 1 {
                    for &loc in &pg.target_locs {
                        #[cfg(feature = "parallel")]
                        {
                            if use_parallel {
                                instruct_single_parallel(state, &gate_matrix, loc);
                            } else {
                                instruct_single(state, &gate_matrix, loc);
                            }
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            instruct_single(state, &gate_matrix, loc);
                        }
                    }
                } else {
                    // Multi-target gate without controls: use instruct_controlled with empty controls
                    instruct_controlled(
                        state,
                        &gate_matrix,
                        &[],
                        &[],
                        &pg.target_locs,
                    );
                }
            }
        } else {
            // Controlled gate
            // Convert control_configs from Vec<bool> to Vec<usize>
            let ctrl_configs: Vec<usize> = pg
                .control_configs
                .iter()
                .map(|&b| if b { 1 } else { 0 })
                .collect();
            instruct_controlled(
                state,
                &gate_matrix,
                &pg.control_locs,
                &ctrl_configs,
                &pg.target_locs,
            );
        }
    }
}

/// Apply a circuit to a quantum state, returning a new state.
///
/// This is a convenience wrapper that clones the input state and applies
/// the circuit in-place.
pub fn apply(circuit: &Circuit, state: &State) -> State {
    let mut result = state.clone();
    apply_inplace(circuit, &mut result);
    result
}

/// Apply a circuit to a quantum state by building and multiplying full-space matrices.
///
/// This is the original implementation kept for comparison testing.
/// It builds full matrices in the Hilbert space which is O(4^n) memory.
///
/// For each gate in the circuit:
/// 1. Build the controlled local matrix on all involved sites
/// 2. Embed it into the full Hilbert space
/// 3. Multiply by the state vector
#[cfg(test)]
pub fn apply_old(circuit: &Circuit, state: &State) -> State {
    let dims = &circuit.dims;
    let total_dim = circuit.total_dim();
    let mut current_data = state.data.clone();

    for pg in &circuit.gates {
        // Get the gate's local matrix on target sites
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);

        // Build the controlled local matrix on all involved sites
        let all_locs = pg.all_locs(); // control_locs ++ target_locs
        let all_dims: Vec<usize> = all_locs.iter().map(|&loc| dims[loc]).collect();
        let num_controls = pg.control_locs.len();

        let local_matrix = build_controlled_matrix(
            &all_dims,
            &gate_matrix,
            &pg.control_configs,
            num_controls,
        );

        // Embed into full Hilbert space and multiply
        let mut full_matrix = Array2::zeros((total_dim, total_dim));

        for row in 0..total_dim {
            let row_multi = linear_to_multi(row, dims);
            for col in 0..total_dim {
                let col_multi = linear_to_multi(col, dims);

                // Check that non-involved sites are the same between row and col
                let mut non_involved_match = true;
                for site in 0..dims.len() {
                    if !all_locs.contains(&site)
                        && row_multi[site] != col_multi[site]
                    {
                        non_involved_match = false;
                        break;
                    }
                }

                if !non_involved_match {
                    // entry is 0 (already initialized)
                    continue;
                }

                // Extract involved-site indices for row and col
                let row_involved: Vec<usize> =
                    all_locs.iter().map(|&loc| row_multi[loc]).collect();
                let col_involved: Vec<usize> =
                    all_locs.iter().map(|&loc| col_multi[loc]).collect();

                let local_row = multi_to_linear(&row_involved, &all_dims);
                let local_col = multi_to_linear(&col_involved, &all_dims);

                full_matrix[[row, col]] = local_matrix[[local_row, local_col]];
            }
        }

        current_data = matrix_vector_mul(&full_matrix, &current_data);
    }

    State {
        dims: dims.clone(),
        data: current_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_multi() {
        assert_eq!(linear_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3]), vec![0, 1]);
        assert_eq!(linear_to_multi(3, &[2, 3]), vec![1, 0]);
        assert_eq!(linear_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_multi_to_linear() {
        assert_eq!(multi_to_linear(&[0, 0], &[2, 3]), 0);
        assert_eq!(multi_to_linear(&[0, 1], &[2, 3]), 1);
        assert_eq!(multi_to_linear(&[1, 0], &[2, 3]), 3);
        assert_eq!(multi_to_linear(&[1, 2], &[2, 3]), 5);
    }

    #[test]
    fn test_roundtrip() {
        let dims = [2, 3, 2];
        let total: usize = dims.iter().product();
        for i in 0..total {
            let multi = linear_to_multi(i, &dims);
            assert_eq!(multi_to_linear(&multi, &dims), i);
        }
    }

    // Tests comparing new apply with old full_matrix approach
    use crate::circuit::{control, put};
    use crate::gate::Gate;
    use std::f64::consts::PI;

    fn states_approx_equal(a: &State, b: &State, tol: f64) -> bool {
        if a.dims != b.dims {
            return false;
        }
        for i in 0..a.data.len() {
            if (a.data[i] - b.data[i]).norm() > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_apply_vs_apply_old_single_h() {
        // Single H gate on 2-qubit system
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H)],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_diagonal_gates() {
        // Z, S, T gates (all diagonal)
        let circuit = Circuit::new(
            vec![2, 2, 2],
            vec![
                put(vec![0], Gate::Z),
                put(vec![1], Gate::S),
                put(vec![2], Gate::T),
            ],
        )
        .unwrap();

        // Start with superposition state
        let mut state = State::zero_state(&[2, 2, 2]);
        // Create |+++⟩ state manually by setting all amplitudes equal
        let amp = Complex64::new(1.0 / (8.0_f64).sqrt(), 0.0);
        for i in 0..8 {
            state.data[i] = amp;
        }

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_phase_gate() {
        // Phase gate with arbitrary angle
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::Phase(PI / 3.0))],
        )
        .unwrap();

        // Start with superposition
        let mut state = State::zero_state(&[2, 2]);
        let s = Complex64::new(0.5, 0.0);
        for i in 0..4 {
            state.data[i] = s;
        }

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_rz_gate() {
        // Rz gate (diagonal)
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![1], Gate::Rz(PI / 4.0))],
        )
        .unwrap();

        let mut state = State::zero_state(&[2, 2]);
        state.data[0] = Complex64::new(0.5, 0.0);
        state.data[1] = Complex64::new(0.5, 0.0);
        state.data[2] = Complex64::new(0.5, 0.0);
        state.data[3] = Complex64::new(0.5, 0.0);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_bell_circuit() {
        // Bell state circuit: H on qubit 0, then CNOT
        let circuit = Circuit::new(
            vec![2, 2],
            vec![
                put(vec![0], Gate::H),
                control(vec![0], vec![1], Gate::X),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_cnot() {
        // CNOT gate
        let circuit = Circuit::new(
            vec![2, 2],
            vec![control(vec![0], vec![1], Gate::X)],
        )
        .unwrap();

        // Test with |10⟩
        let state = State::product_state(&[2, 2], &[1, 0]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_toffoli() {
        // Toffoli (CCX) gate
        let circuit = Circuit::new(
            vec![2, 2, 2],
            vec![control(vec![0, 1], vec![2], Gate::X)],
        )
        .unwrap();

        // Test with |110⟩
        let state = State::product_state(&[2, 2, 2], &[1, 1, 0]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_controlled_phase() {
        // Controlled-Z gate
        let circuit = Circuit::new(
            vec![2, 2],
            vec![control(vec![0], vec![1], Gate::Z)],
        )
        .unwrap();

        // Start with superposition
        let mut state = State::zero_state(&[2, 2]);
        let s = Complex64::new(0.5, 0.0);
        for i in 0..4 {
            state.data[i] = s;
        }

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_rx_ry() {
        // Rx and Ry gates (not diagonal)
        let circuit = Circuit::new(
            vec![2, 2],
            vec![
                put(vec![0], Gate::Rx(PI / 5.0)),
                put(vec![1], Gate::Ry(PI / 7.0)),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_complex_circuit() {
        // More complex circuit with multiple gates
        let circuit = Circuit::new(
            vec![2, 2, 2],
            vec![
                put(vec![0], Gate::H),
                put(vec![1], Gate::H),
                put(vec![2], Gate::H),
                control(vec![0], vec![1], Gate::X),
                put(vec![0], Gate::T),
                control(vec![1], vec![2], Gate::X),
                put(vec![1], Gate::S),
                control(vec![0, 1], vec![2], Gate::X),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_4_qubit() {
        // 4 qubit circuit
        let circuit = Circuit::new(
            vec![2, 2, 2, 2],
            vec![
                put(vec![0], Gate::H),
                put(vec![1], Gate::H),
                control(vec![0], vec![2], Gate::X),
                control(vec![1], vec![3], Gate::X),
                put(vec![2], Gate::Z),
                put(vec![3], Gate::T),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2, 2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_random_gates() {
        // Circuit with X, Y, Z, H gates in various combinations
        let circuit = Circuit::new(
            vec![2, 2, 2],
            vec![
                put(vec![0], Gate::X),
                put(vec![1], Gate::Y),
                put(vec![2], Gate::Z),
                put(vec![0], Gate::H),
                control(vec![0], vec![1], Gate::X),
                put(vec![2], Gate::H),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_swap_gate() {
        // SWAP gate (multi-target, no controls)
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();

        // Test with |10⟩ -> |01⟩
        let state = State::product_state(&[2, 2], &[1, 0]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_vs_apply_old_qft_like_circuit() {
        // QFT-like circuit (simplified)
        // QFT on 3 qubits has H gates and controlled phase rotations
        let circuit = Circuit::new(
            vec![2, 2, 2],
            vec![
                // First qubit
                put(vec![0], Gate::H),
                control(vec![1], vec![0], Gate::Phase(PI / 2.0)),
                control(vec![2], vec![0], Gate::Phase(PI / 4.0)),
                // Second qubit
                put(vec![1], Gate::H),
                control(vec![2], vec![1], Gate::Phase(PI / 2.0)),
                // Third qubit
                put(vec![2], Gate::H),
            ],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2, 2]);

        let result_new = apply(&circuit, &state);
        let result_old = apply_old(&circuit, &state);

        assert!(states_approx_equal(&result_new, &result_old, 1e-10));
    }

    #[test]
    fn test_apply_inplace() {
        // Test that apply_inplace modifies state correctly
        let circuit = Circuit::new(
            vec![2, 2],
            vec![
                put(vec![0], Gate::H),
                control(vec![0], vec![1], Gate::X),
            ],
        )
        .unwrap();

        let mut state = State::zero_state(&[2, 2]);
        apply_inplace(&circuit, &mut state);

        // Should be Bell state (|00⟩ + |11⟩) / √2
        let s = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        assert!((state.data[0] - s).norm() < 1e-10);
        assert!(state.data[1].norm() < 1e-10);
        assert!(state.data[2].norm() < 1e-10);
        assert!((state.data[3] - s).norm() < 1e-10);
    }

    #[test]
    fn test_apply_preserves_input() {
        // Test that apply does not modify the input state
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::X)],
        )
        .unwrap();

        let state = State::zero_state(&[2, 2]);
        let state_clone = state.clone();
        let _result = apply(&circuit, &state);

        // Original state should be unchanged
        assert!(states_approx_equal(&state, &state_clone, 1e-10));
    }

    #[test]
    fn test_is_diagonal() {
        assert!(is_diagonal(&Gate::Z));
        assert!(is_diagonal(&Gate::S));
        assert!(is_diagonal(&Gate::T));
        assert!(is_diagonal(&Gate::Phase(1.0)));
        assert!(is_diagonal(&Gate::Rz(1.0)));
        assert!(!is_diagonal(&Gate::X));
        assert!(!is_diagonal(&Gate::Y));
        assert!(!is_diagonal(&Gate::H));
        assert!(!is_diagonal(&Gate::Rx(1.0)));
        assert!(!is_diagonal(&Gate::Ry(1.0)));
        assert!(!is_diagonal(&Gate::SWAP));
    }

    #[test]
    fn test_extract_diagonal_phases() {
        // Z gate: diag(1, -1)
        let z_matrix = Gate::Z.matrix(2);
        let phases = extract_diagonal_phases(&z_matrix);
        assert_eq!(phases.len(), 2);
        assert!((phases[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((phases[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);

        // S gate: diag(1, i)
        let s_matrix = Gate::S.matrix(2);
        let phases = extract_diagonal_phases(&s_matrix);
        assert_eq!(phases.len(), 2);
        assert!((phases[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((phases[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    }
}
