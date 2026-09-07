use num_complex::Complex64;

use crate::circuit::{Circuit, CircuitElement, PositionedGate};
use crate::gate::Gate;
use crate::register::ArrayReg;

pub(crate) fn dispatch_arrayreg_gate(nbits: usize, state: &mut [Complex64], pg: &PositionedGate) {
    let has_controls = !pg.control_locs.is_empty();
    let ctrl_locs = &pg.control_locs;
    let many_controls;
    let ctrl_bits: &[usize] = match pg.control_configs.as_slice() {
        [] => &[],
        [false] => &[0],
        [true] => &[1],
        configs => {
            many_controls = configs.iter().map(|&v| usize::from(v)).collect::<Vec<_>>();
            &many_controls
        }
    };
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
                    state, nbits, loc, ctrl_locs, ctrl_bits,
                );
            }
        }
        Gate::SWAP if !has_controls => {
            crate::instruct_qubit::instruct_swap(state, nbits, &pg.target_locs);
        }
        Gate::SWAP => {
            let gate_flat = gate.matrix_row_major();
            crate::instruct_qubit::instruct_2q_controlled(
                state,
                nbits,
                &pg.target_locs,
                &gate_flat,
                ctrl_locs,
                ctrl_bits,
            );
        }
        gate if gate.is_diagonal() && pg.target_locs.len() == 1 => {
            let [d0, _, _, d1] = gate.single_qubit_coefficients().unwrap();
            let loc = pg.target_locs[0];
            if has_controls {
                crate::instruct_qubit::instruct_1q_diag_controlled(
                    state, nbits, loc, d0, d1, ctrl_locs, ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q_diag(state, loc, d0, d1);
            }
        }
        gate if gate.is_diagonal() && pg.target_locs.len() == 2 => {
            let matrix = gate.matrix_row_major();
            let diag = [matrix[0], matrix[5], matrix[10], matrix[15]];
            if has_controls {
                crate::instruct_qubit::instruct_2q_diag_controlled(
                    state,
                    nbits,
                    &pg.target_locs,
                    &diag,
                    ctrl_locs,
                    ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q_diag(state, nbits, &pg.target_locs, &diag);
            }
        }
        _ if pg.target_locs.len() == 1 => {
            let [a, b, c, d] = gate.single_qubit_coefficients().unwrap();
            let loc = pg.target_locs[0];
            if has_controls {
                crate::instruct_qubit::instruct_1q_controlled(
                    state, nbits, loc, a, b, c, d, ctrl_locs, ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_1q(state, loc, a, b, c, d);
            }
        }
        _ if pg.target_locs.len() == 2 => {
            let gate_flat = gate.matrix_row_major();
            if has_controls {
                crate::instruct_qubit::instruct_2q_controlled(
                    state,
                    nbits,
                    &pg.target_locs,
                    &gate_flat,
                    ctrl_locs,
                    ctrl_bits,
                );
            } else {
                crate::instruct_qubit::instruct_2q(state, nbits, &pg.target_locs, &gate_flat);
            }
        }
        _ => {
            let gate_flat = gate.matrix_row_major();
            crate::instruct_qubit::instruct_nq(
                state,
                nbits,
                &pg.target_locs,
                &gate_flat,
                ctrl_locs,
                ctrl_bits,
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

/// Shared validation for stochastic and reversible unitary execution.
pub(crate) fn validate_unitary_gate(n: usize, pg: &PositionedGate) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    if pg
        .target_locs
        .iter()
        .chain(&pg.control_locs)
        .any(|&i| i >= n || !seen.insert(i))
    {
        return Err("Unitary gate locations must be distinct and in range".into());
    }
    Circuit::qubits(n, vec![CircuitElement::Gate(pg.clone())]).map_err(|e| e.to_string())?;
    let matrix = pg.gate.matrix();
    if pg.gate.is_diagonal()
        && matrix
            .indexed_iter()
            .any(|((i, j), &z)| i != j && z != Complex64::new(0., 0.))
    {
        return Err("Custom diagonal flag disagrees with matrix".into());
    }
    crate::noise::validate_kraus(std::slice::from_ref(&matrix))
        .map_err(|e| format!("Unitary gate validation: {e}"))
}
