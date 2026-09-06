//! First-order derivatives of unitary circuits with real physical parameters.
//!
//! Cotangents use the real Hermitian pairing
//! `dL = Re(sum(conj(state_bar) * dstate)) + dot(parameter_bar, dparameters)`.
//! In particular, a Hermitian expectation loss seeds the output with `2 H psi`.
//! State inputs need not be normalized. Channels and nonunitary custom matrices
//! are rejected; exact noisy forward simulation remains available separately.

use crate::ad::{apply_generator, reverse_unitary};
use crate::apply::dispatch_arrayreg_gate;
use crate::parameters::{BoundCircuit, ParameterBinding};
use crate::{ArrayReg, Circuit, CircuitElement};
use num_complex::Complex64 as C;

/// Validated unitary structure and physical bindings, reusable at new values.
#[derive(Debug, Clone)]
pub struct DifferentiableCircuit {
    template: BoundCircuit,
}

/// A circuit pullback under the real Hermitian pairing.
#[derive(Debug, Clone)]
pub struct CircuitVjp {
    /// One real gradient per physical parameter (shared occurrences accumulate).
    pub parameters: Vec<f64>,
    /// Complex cotangent of the input state, in the register's site order.
    pub input: ArrayReg,
}

impl DifferentiableCircuit {
    /// Validate the reversible derivative domain once, before preparing a graph.
    /// Custom matrices are checked for unitarity to `1e-12` per Gram entry.
    pub fn new(template: BoundCircuit) -> Result<Self, String> {
        let circuit = template.circuit();
        if circuit.nbits >= usize::BITS as usize || circuit.dims.iter().any(|&d| d != 2) {
            return Err("Differentiable circuits require addressable qubit states".into());
        }
        if (1usize << circuit.nbits)
            .checked_mul(size_of::<C>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err("Circuit state exceeds addressable storage".into());
        }
        for element in &circuit.elements {
            match element {
                CircuitElement::Channel(_) => {
                    return Err("Reversible circuit AD does not support noise channels".into());
                }
                CircuitElement::Gate(pg) => {
                    crate::apply::validate_unitary_gate(circuit.nbits, pg)?;
                }
                CircuitElement::Annotation(_) => {}
            }
        }
        Ok(Self { template })
    }

    /// Treat each gate angle as an independent physical parameter.
    pub fn from_circuit(circuit: Circuit) -> Result<Self, String> {
        let parameters = circuit.parameters();
        let bindings = (0..parameters.len())
            .map(|index| ParameterBinding::Scaled { index, scale: 1. })
            .collect();
        Self::new(BoundCircuit::new(circuit, parameters, bindings)?)
    }

    /// Immutable initial values and parameterization.
    pub fn template(&self) -> &BoundCircuit {
        &self.template
    }
    /// Number of complex state amplitudes.
    pub fn state_len(&self) -> usize {
        1usize << self.template.circuit().nbits
    }
    /// Number of real physical parameters, including declared unused parameters.
    pub fn num_parameters(&self) -> usize {
        self.template.parameters().len()
    }

    fn bind(&self, parameters: &[f64]) -> Result<BoundCircuit, String> {
        let mut bound = self.template.clone();
        bound.dispatch(parameters)?;
        Ok(bound)
    }

    pub(crate) fn validate_state(&self, input: &ArrayReg) -> Result<(), String> {
        if input.nqubits() != self.template.circuit().nbits || input.state.len() != self.state_len()
        {
            return Err("Circuit state dimension mismatch".into());
        }
        if input
            .state
            .iter()
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err("Circuit states and cotangents must be finite".into());
        }
        Ok(())
    }

    /// Execute at explicit physical values without changing the prepared template.
    pub fn forward(&self, parameters: &[f64], input: &ArrayReg) -> Result<ArrayReg, String> {
        self.validate_state(input)?;
        self.bind(parameters)?.apply(input)
    }

    /// Apply an arbitrary output cotangent to real parameters and complex input.
    /// Uses one forward pass and a reversible sweep, with O(state length) state
    /// storage rather than one saved state per gate. Circuit storage is O(gates).
    pub fn vjp(
        &self,
        parameters: &[f64],
        input: &ArrayReg,
        cotangent: &ArrayReg,
    ) -> Result<CircuitVjp, String> {
        let output = self.forward(parameters, input)?;
        self.vjp_from_output(parameters, &output, cotangent)
    }

    /// Evaluate a real loss and both gradients with one circuit forward pass.
    /// The callback supplies the loss value and its output-state cotangent
    /// under the real Hermitian pairing. Tenferro can differentiate the loss
    /// automatically; this native entry point accepts an analytic loss seed.
    pub fn value_and_grad(
        &self,
        parameters: &[f64],
        input: &ArrayReg,
        loss: impl FnOnce(&ArrayReg) -> Result<(f64, ArrayReg), String>,
    ) -> Result<(f64, CircuitVjp), String> {
        let output = self.forward(parameters, input)?;
        let (value, seed) = loss(&output)?;
        if !value.is_finite() {
            return Err("Circuit loss must be finite".into());
        }
        Ok((value, self.vjp_from_output(parameters, &output, &seed)?))
    }

    // Tenferro retains the final state and parameters as the VJP residuals.
    pub(crate) fn vjp_from_output(
        &self,
        parameters: &[f64],
        output: &ArrayReg,
        cotangent: &ArrayReg,
    ) -> Result<CircuitVjp, String> {
        self.validate_state(output)?;
        self.validate_state(cotangent)?;
        let bound = self.bind(parameters)?;
        let mut psi = output.clone();
        let mut input = cotangent.clone();
        let angles = reverse_unitary(bound.circuit(), &mut psi, &mut input.state, 1.);
        let parameters = bound.pullback(&angles)?;
        Ok(CircuitVjp { parameters, input })
    }

    /// Evaluate the state and its directional derivative. Parameter and input
    /// tangents use the same ordering as `vjp`. This is a first-order operation.
    pub fn jvp(
        &self,
        parameters: &[f64],
        input: &ArrayReg,
        parameter_tangent: &[f64],
        input_tangent: &ArrayReg,
    ) -> Result<(ArrayReg, ArrayReg), String> {
        self.validate_state(input)?;
        self.validate_state(input_tangent)?;
        let bound = self.bind(parameters)?;
        let angles = bound.pushforward(parameter_tangent)?;
        let mut output = input.clone();
        let mut tangent = input_tangent.clone();
        let mut scratch = vec![C::new(0., 0.); self.state_len()];
        let mut slot = 0;
        let n = input.nqubits();
        for element in &bound.circuit().elements {
            if let CircuitElement::Gate(pg) = element {
                dispatch_arrayreg_gate(n, &mut output.state, pg);
                dispatch_arrayreg_gate(n, &mut tangent.state, pg);
                for i in 0..pg.gate.num_params() {
                    scratch.copy_from_slice(&output.state);
                    apply_generator(&mut scratch, n, pg, pg.gate.generator_matrix(i));
                    for (dx, g) in tangent.state.iter_mut().zip(&scratch) {
                        *dx += angles[slot] * g;
                    }
                    slot += 1;
                }
            }
        }
        Ok((output, tangent))
    }
}

#[cfg(test)]
#[path = "unit_tests/differentiable.rs"]
mod tests;
