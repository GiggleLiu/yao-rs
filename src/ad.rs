//! Adjoint-mode automatic differentiation for Pauli-polynomial expectation values.
//!
//! Given a circuit U(theta) producing `|psi(theta)> = U(theta) |psi_0>`, this module
//! computes both `value = <psi|H|psi>` and its gradient with respect to every
//! trainable parameter of U, using one forward pass plus one backward sweep.
//!
//! Parameters follow [`Circuit::parameters`] order; update them with
//! [`Circuit::dispatch`]. Noise-channel differentiation is unsupported.

use ndarray::Array2;
use num_complex::Complex64;

use crate::apply::{apply_inplace, dispatch_arrayreg_gate};
use crate::circuit::{Circuit, CircuitElement, PositionedGate};
use crate::expect::expect_arrayreg;
use crate::gate::Gate;
use crate::operator::OperatorPolynomial;
use crate::register::ArrayReg;

fn controls_match(basis: usize, nbits: usize, pg: &PositionedGate) -> bool {
    pg.control_locs
        .iter()
        .zip(&pg.control_configs)
        .all(|(&loc, &config)| {
            let bit = nbits - 1 - loc;
            ((basis >> bit) & 1) == usize::from(config)
        })
}

/// Apply a generator matrix `g` at `pg.target_locs` with the same control
/// configuration as `pg`, to the qubit state vector `state` of size `2^nbits`.
///
/// Unlike a controlled unitary, a controlled generator maps inactive control
/// branches to zero because those amplitudes have no parameter derivative.
pub(crate) fn apply_generator(
    state: &mut [Complex64],
    nbits: usize,
    pg: &PositionedGate,
    g: Array2<Complex64>,
) {
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

    if pg.control_locs.is_empty() {
        dispatch_arrayreg_gate(nbits, state, &temp);
        return;
    }

    let mut transformed = state.to_vec();
    dispatch_arrayreg_gate(nbits, &mut transformed, &temp);
    for (basis, amp) in state.iter_mut().enumerate() {
        *amp = if controls_match(basis, nbits, pg) {
            transformed[basis]
        } else {
            Complex64::new(0.0, 0.0)
        };
    }
}

/// Compute `<psi|H|psi>` and its gradient with respect to every trainable
/// parameter in `circuit`, where `psi = circuit |psi0>`.
///
/// The returned gradient vector has length `circuit.num_params()` and follows
/// the same ordering as `circuit.parameters()`.
///
/// The observable must be Hermitian and the circuit gates must be unitary.
/// Supports Rx, Ry, Rz, Phase, and both FSim parameters, including controls.
///
/// ```
/// use yao_rs::{ArrayReg, Circuit, Gate, Op, OperatorPolynomial, expect_grad, put};
/// let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::Rx(0.3))]).unwrap();
/// let z = OperatorPolynomial::single(0, Op::Z, 1.0.into());
/// let (value, gradient) = expect_grad(&z, &circuit, &ArrayReg::zero_state(1));
/// assert!((value - 0.3_f64.cos()).abs() < 1e-12);
/// assert!((gradient[0] + 0.3_f64.sin()).abs() < 1e-12);
/// ```
///
/// # Panics
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

    let n_params = circuit.num_params();
    if n_params == 0 {
        return (value, Vec::new());
    }

    let nbits = psi.nqubits();
    let dim = 1usize << nbits;

    let mut lambda_state = vec![Complex64::new(0.0, 0.0); dim];
    for (coeff, opstring) in observable.iter() {
        let mut tmp = psi.state_vec().to_vec();
        for &(loc, op) in opstring.ops() {
            let m = crate::operator::op_matrix(&op);
            crate::instruct_qubit::instruct_1q(
                &mut tmp,
                loc,
                m[[0, 0]],
                m[[0, 1]],
                m[[1, 0]],
                m[[1, 1]],
            );
        }
        for i in 0..dim {
            lambda_state[i] += *coeff * tmp[i];
        }
    }

    let grad = reverse_unitary(circuit, &mut psi, &mut lambda_state, 2.0);
    (value, grad)
}

// Shared reversible sweep. The caller validates unitarity and dimensions.
// `scale` preserves expect_grad's H*psi seed while general VJPs use scale=1.
pub(crate) fn reverse_unitary(
    circuit: &Circuit,
    psi: &mut ArrayReg,
    lambda_state: &mut [Complex64],
    scale: f64,
) -> Vec<f64> {
    let n_params = circuit.num_params();
    let mut grad = vec![0.; n_params];
    let nbits = psi.nqubits();
    let dim = psi.state.len();
    let mut slot = n_params;
    let mut phi_scratch = vec![Complex64::new(0.0, 0.0); dim];
    for el in circuit.elements.iter().rev() {
        match el {
            CircuitElement::Annotation(_) => continue,
            CircuitElement::Channel(_) => unreachable!("rejected above"),
            CircuitElement::Gate(pg) => {
                let np = pg.gate.num_params();
                if np > 0 {
                    for i in (0..np).rev() {
                        slot -= 1;
                        phi_scratch.copy_from_slice(psi.state_vec());
                        let g = pg.gate.generator_matrix(i);
                        apply_generator(&mut phi_scratch, nbits, pg, g);

                        let mut acc = Complex64::new(0.0, 0.0);
                        for k in 0..dim {
                            acc += lambda_state[k].conj() * phi_scratch[k];
                        }
                        grad[slot] = scale * acc.re;
                    }
                }

                let dag_pg = PositionedGate {
                    gate: pg.gate.dagger(),
                    target_locs: pg.target_locs.clone(),
                    control_locs: pg.control_locs.clone(),
                    control_configs: pg.control_configs.clone(),
                };
                dispatch_arrayreg_gate(nbits, psi.state_vec_mut(), &dag_pg);
                dispatch_arrayreg_gate(nbits, lambda_state, &dag_pg);
            }
        }
    }
    debug_assert_eq!(slot, 0);

    grad
}

#[cfg(test)]
#[path = "unit_tests/ad.rs"]
mod tests;
