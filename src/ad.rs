//! Adjoint-mode automatic differentiation for Pauli-polynomial expectation values.
//!
//! Given a circuit U(theta) producing `|psi(theta)> = U(theta) |psi_0>`, this module
//! computes both `value = <psi|H|psi>` and its gradient with respect to every
//! trainable parameter of U, using one forward pass plus one backward sweep.
//!
//! See `docs/superpowers/specs/2026-04-21-parameter-dispatch-and-ad-design.md`.

use ndarray::Array2;
use num_complex::Complex64;

use crate::apply::{apply_inplace, dispatch_arrayreg_gate};
use crate::circuit::{Circuit, CircuitElement, PositionedGate};
use crate::expect::expect_arrayreg;
use crate::gate::Gate;
use crate::operator::OperatorPolynomial;
use crate::register::ArrayReg;

/// Apply a generator matrix `g` at `pg.target_locs` with the same control
/// configuration as `pg`, to the qubit state vector `state` of size `2^nbits`.
///
/// This is a thin shim around `dispatch_arrayreg_gate` using a temporary
/// `Gate::Custom` that carries the generator matrix.
fn apply_generator(state: &mut [Complex64], nbits: usize, pg: &PositionedGate, g: Array2<Complex64>) {
    let temp = PositionedGate {
        gate: Gate::Custom {
            matrix: g,
            is_diagonal: false,
            label: "G".to_string(),
        },
        target_locs: pg.target_locs.clone(),
        control_locs: pg.control_locs.clone(),
        control_configs: pg.control_configs.clone(),
    };
    dispatch_arrayreg_gate(nbits, state, &temp);
}

/// Compute `<psi|H|psi>` and its gradient with respect to every trainable
/// parameter in `circuit`, where `psi = circuit |psi0>`.
///
/// The returned gradient vector has length `circuit.num_params()` and follows
/// the same ordering as `circuit.parameters()`.
///
/// Panics if the circuit contains any `CircuitElement::Channel`, or if `psi0`'s
/// qubit count does not match the circuit.
pub fn expect_grad(
    observable: &OperatorPolynomial,
    circuit: &Circuit,
    psi0: &ArrayReg,
) -> (f64, Vec<f64>) {
    for el in &circuit.elements {
        if matches!(el, CircuitElement::Channel(_)) {
            panic!("expect_grad: circuit contains a noise channel; AD is unsupported for channels");
        }
    }

    let mut psi = psi0.clone();
    apply_inplace(circuit, &mut psi);

    let value = expect_arrayreg(&psi, observable).re;

    let grad = vec![0.0; circuit.num_params()];
    let _ = apply_generator;
    (value, grad)
}

#[cfg(test)]
#[path = "unit_tests/ad.rs"]
mod tests;
