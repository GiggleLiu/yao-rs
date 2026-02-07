use ndarray::Array2;
use num_complex::Complex64;

use crate::circuit::{Circuit, CircuitElement};
use crate::gate::Gate;
use crate::instruct::{instruct_controlled, instruct_diagonal, instruct_single};
#[cfg(feature = "parallel")]
use crate::instruct::{instruct_diagonal_parallel, instruct_single_parallel};

/// Threshold for switching to parallel execution (2^14 = 16384 amplitudes).
/// Below this threshold, the overhead of Rayon is not worth it.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 16384;
use crate::state::State;

/// Check if a gate is diagonal.
pub(crate) fn is_diagonal(gate: &Gate) -> bool {
    gate.is_diagonal()
}

/// Extract the diagonal phases from a diagonal gate matrix.
///
/// For a diagonal matrix, returns the diagonal elements as a vector.
pub(crate) fn extract_diagonal_phases(matrix: &Array2<Complex64>) -> Vec<Complex64> {
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

    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) => continue, // Skip annotations
        };

        // Get the gate's local matrix on target sites
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);

        if pg.control_locs.is_empty() {
            // No controls
            // Diagonal optimization only applies to single-target gates
            // Multi-target diagonal gates (e.g., 2-qubit diagonal with d^2 phases)
            // cannot use this path as it would incorrectly apply phases per-site
            if is_diagonal(&pg.gate) && pg.target_locs.len() == 1 {
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
            } else if pg.target_locs.len() == 1 {
                // Single-target non-diagonal gate
                let loc = pg.target_locs[0];
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
            } else {
                // Multi-target gate without controls (including multi-target diagonal gates):
                // use instruct_controlled with empty controls
                instruct_controlled(state, &gate_matrix, &[], &[], &pg.target_locs);
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
