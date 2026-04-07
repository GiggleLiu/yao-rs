use ndarray::Array2;
use num_complex::Complex64;

use crate::circuit::{Circuit, CircuitElement, PositionedGate};
use crate::gate::Gate;
use crate::instruct::{instruct_controlled, instruct_diagonal, instruct_single};
#[cfg(feature = "parallel")]
use crate::instruct::{instruct_diagonal_parallel, instruct_single_parallel};
use crate::register::ArrayReg;

/// Threshold for switching to parallel execution (2^14 = 16384 amplitudes).
/// Below this threshold, the overhead of Rayon is not worth it.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 16384;
use crate::state::State;

#[doc(hidden)]
pub trait ApplyTarget {
    fn apply_circuit_inplace(circuit: &Circuit, target: &mut Self);
}

impl ApplyTarget for State {
    fn apply_circuit_inplace(circuit: &Circuit, target: &mut Self) {
        apply_state_inplace(circuit, target);
    }
}

impl ApplyTarget for ArrayReg {
    fn apply_circuit_inplace(circuit: &Circuit, target: &mut Self) {
        apply_arrayreg_inplace(circuit, target);
    }
}

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

/// Dispatch a single qubit gate through the bit-manipulation fast path.
fn apply_qubit_gate(pg: &PositionedGate, state: &mut State, nbits: usize) {
    let gate_matrix = pg.gate.matrix();
    let state_slice = state.data.as_slice_mut().unwrap();

    let ctrl_locs = &pg.control_locs;
    let ctrl_bits: Vec<usize> = pg.control_configs.iter().map(|&b| usize::from(b)).collect();
    let has_controls = !ctrl_locs.is_empty();

    if pg.target_locs.len() == 1 {
        let loc = pg.target_locs[0];
        if is_diagonal(&pg.gate) {
            let d0 = gate_matrix[[0, 0]];
            let d1 = gate_matrix[[1, 1]];
            if has_controls {
                crate::instruct_qubit::instruct_1q_diag_controlled(
                    state_slice,
                    nbits,
                    loc,
                    d0,
                    d1,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q_diag(state_slice, loc, d0, d1);
            }
        } else {
            let a = gate_matrix[[0, 0]];
            let b = gate_matrix[[0, 1]];
            let c = gate_matrix[[1, 0]];
            let d = gate_matrix[[1, 1]];
            if has_controls {
                crate::instruct_qubit::instruct_1q_controlled(
                    state_slice,
                    nbits,
                    loc,
                    a,
                    b,
                    c,
                    d,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q(state_slice, loc, a, b, c, d);
            }
        }
    } else if pg.target_locs.len() == 2 {
        let locs = &pg.target_locs;
        if is_diagonal(&pg.gate) {
            let diag = [
                gate_matrix[[0, 0]],
                gate_matrix[[1, 1]],
                gate_matrix[[2, 2]],
                gate_matrix[[3, 3]],
            ];
            if has_controls {
                crate::instruct_qubit::instruct_2q_diag_controlled(
                    state_slice,
                    nbits,
                    locs,
                    &diag,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q_diag(state_slice, nbits, locs, &diag);
            }
        } else {
            let mut gate_flat = Vec::with_capacity(16);
            for i in 0..4 {
                for j in 0..4 {
                    gate_flat.push(gate_matrix[[i, j]]);
                }
            }
            if has_controls {
                crate::instruct_qubit::instruct_2q_controlled(
                    state_slice,
                    nbits,
                    locs,
                    &gate_flat,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q(state_slice, nbits, locs, &gate_flat);
            }
        }
    } else {
        // >2 target qubits: fall through to generic path
        let ctrl_configs: Vec<usize> = pg.control_configs.iter().map(|&b| usize::from(b)).collect();
        instruct_controlled(
            state,
            &gate_matrix,
            &pg.control_locs,
            &ctrl_configs,
            &pg.target_locs,
        );
    }
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
fn apply_state_inplace(circuit: &Circuit, state: &mut State) {
    let dims = &circuit.dims;
    let all_qubit = dims.iter().all(|&d| d == 2);
    let nbits = dims.len();

    #[cfg(feature = "parallel")]
    let use_parallel = state.data.len() >= PARALLEL_THRESHOLD;

    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => continue,
        };

        // Qubit fast path: use bit-manipulation-based instruct functions
        if all_qubit {
            apply_qubit_gate(pg, state, nbits);
            continue;
        }

        // Generic path (qudits or mixed dimensions)
        let gate_matrix = pg.gate.matrix();

        if pg.control_locs.is_empty() {
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
                instruct_controlled(state, &gate_matrix, &[], &[], &pg.target_locs);
            }
        } else {
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

fn dispatch_arrayreg_gate(nbits: usize, state: &mut [Complex64], pg: &PositionedGate) {
    let has_controls = !pg.control_locs.is_empty();
    let ctrl_locs = &pg.control_locs;
    let ctrl_bits: Vec<usize> = pg
        .control_configs
        .iter()
        .map(|&value| usize::from(value))
        .collect();
    let gate = &pg.gate;

    match gate {
        Gate::X if !has_controls => {
            for &loc in &pg.target_locs {
                crate::instruct_qubit::instruct_x(state, nbits, loc);
            }
        }
        Gate::X => {
            for &loc in &pg.target_locs {
                crate::instruct_qubit::instruct_x_controlled(
                    state, nbits, loc, ctrl_locs, &ctrl_bits,
                );
            }
        }
        Gate::SWAP if !has_controls => {
            crate::instruct_qubit::instruct_swap(state, nbits, &pg.target_locs);
        }
        Gate::SWAP => {
            let gate_flat: Vec<Complex64> = gate.matrix().iter().copied().collect();
            crate::instruct_qubit::instruct_2q_controlled(
                state,
                nbits,
                &pg.target_locs,
                &gate_flat,
                ctrl_locs,
                &ctrl_bits,
            );
        }
        gate if gate.is_diagonal() && pg.target_locs.len() == 1 => {
            let matrix = gate.matrix();
            let d0 = matrix[[0, 0]];
            let d1 = matrix[[1, 1]];
            let loc = pg.target_locs[0];
            if has_controls {
                crate::instruct_qubit::instruct_1q_diag_controlled(
                    state, nbits, loc, d0, d1, ctrl_locs, &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q_diag(state, loc, d0, d1);
            }
        }
        gate if gate.is_diagonal() && pg.target_locs.len() == 2 => {
            let matrix = gate.matrix();
            let diag = [
                matrix[[0, 0]],
                matrix[[1, 1]],
                matrix[[2, 2]],
                matrix[[3, 3]],
            ];
            if has_controls {
                crate::instruct_qubit::instruct_2q_diag_controlled(
                    state,
                    nbits,
                    &pg.target_locs,
                    &diag,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q_diag(state, nbits, &pg.target_locs, &diag);
            }
        }
        _ if pg.target_locs.len() == 1 => {
            let matrix = gate.matrix();
            let loc = pg.target_locs[0];
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];
            if has_controls {
                crate::instruct_qubit::instruct_1q_controlled(
                    state, nbits, loc, a, b, c, d, ctrl_locs, &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q(state, loc, a, b, c, d);
            }
        }
        _ if pg.target_locs.len() == 2 => {
            let gate_flat: Vec<Complex64> = gate.matrix().iter().copied().collect();
            if has_controls {
                crate::instruct_qubit::instruct_2q_controlled(
                    state,
                    nbits,
                    &pg.target_locs,
                    &gate_flat,
                    ctrl_locs,
                    &ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q(state, nbits, &pg.target_locs, &gate_flat);
            }
        }
        _ => {
            let gate_flat: Vec<Complex64> = gate.matrix().iter().copied().collect();
            crate::instruct_qubit::instruct_nq(
                state,
                nbits,
                &pg.target_locs,
                &gate_flat,
                ctrl_locs,
                &ctrl_bits,
            );
        }
    }
}

fn apply_arrayreg_inplace(circuit: &Circuit, reg: &mut ArrayReg) {
    assert!(
        circuit.dims.iter().all(|&dim| dim == 2),
        "ArrayReg only supports qubit-only circuits"
    );
    assert_eq!(
        circuit.nbits,
        reg.nqubits(),
        "Register and circuit qubit count mismatch"
    );

    for element in &circuit.elements {
        match element {
            CircuitElement::Gate(pg) => {
                dispatch_arrayreg_gate(reg.nqubits(), reg.state_vec_mut(), pg);
            }
            CircuitElement::Channel(_) => {
                unimplemented!("Noise channels on ArrayReg are not implemented yet");
            }
            CircuitElement::Annotation(_) => {}
        }
    }
}

pub fn apply_inplace<T: ApplyTarget>(circuit: &Circuit, target: &mut T) {
    T::apply_circuit_inplace(circuit, target);
}

/// Apply a circuit to a quantum state or register, returning a new value.
pub fn apply<T: ApplyTarget + Clone>(circuit: &Circuit, target: &T) -> T {
    let mut result = target.clone();
    apply_inplace(circuit, &mut result);
    result
}
