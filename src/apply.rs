use num_complex::Complex64;

use crate::circuit::{Circuit, CircuitElement, PositionedGate};
use crate::gate::Gate;
use crate::register::ArrayReg;

pub(crate) fn dispatch_arrayreg_gate(nbits: usize, state: &mut [Complex64], pg: &PositionedGate) {
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

/// Apply a circuit to an ArrayReg in-place.
pub fn apply_inplace(circuit: &Circuit, reg: &mut ArrayReg) {
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
                // Noise channels are not applied during pure-state simulation;
                // they are only used by density-matrix or tensor-network paths.
                continue;
            }
            CircuitElement::Annotation(_) => {}
        }
    }
}

/// Apply a circuit to an ArrayReg, returning a new register.
pub fn apply(circuit: &Circuit, reg: &ArrayReg) -> ArrayReg {
    let mut result = reg.clone();
    apply_inplace(circuit, &mut result);
    result
}
