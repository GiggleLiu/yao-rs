//! Explicit bindings from real physical parameters to circuit gate angles.
//!
//! These first-order Jacobian operations cover fixed angles, scaled parameters,
//! and products such as coupling × time. They do not construct an AD tape.

use crate::{ArrayReg, Circuit, apply};

/// One emitted angle, in [`Circuit::parameters`] order.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterBinding {
    /// An angle excluded from the physical parameter vector.
    Fixed(f64),
    /// `scale * parameters[index]`.
    Scaled { index: usize, scale: f64 },
    /// `scale * parameters[left] * parameters[right]`.
    /// Equal indices are allowed and differentiate as a square.
    Product {
        left: usize,
        right: usize,
        scale: f64,
    },
}

impl ParameterBinding {
    fn validate(&self, count: usize) -> Result<(), String> {
        let (scale, indices): (f64, &[usize]) = match self {
            Self::Fixed(value) => (*value, &[]),
            Self::Scaled { index, scale } => (*scale, std::slice::from_ref(index)),
            Self::Product { left, right, scale } => {
                if *left >= count || *right >= count {
                    return Err("Binding parameter index out of range".into());
                }
                (*scale, &[])
            }
        };
        if !scale.is_finite() {
            return Err("Binding constants must be finite".into());
        }
        if indices.iter().any(|&index| index >= count) {
            return Err("Binding parameter index out of range".into());
        }
        Ok(())
    }

    fn value(&self, parameters: &[f64]) -> f64 {
        match *self {
            Self::Fixed(value) => value,
            Self::Scaled { index, scale } => scale * parameters[index],
            Self::Product { left, right, scale } => {
                if parameters[left] == 0. || parameters[right] == 0. {
                    0.
                } else {
                    scale * parameters[left] * parameters[right]
                }
            }
        }
    }

    fn derivatives(&self, parameters: &[f64], mut visit: impl FnMut(usize, f64)) {
        match *self {
            Self::Fixed(_) => {}
            Self::Scaled { index, scale } => visit(index, scale),
            Self::Product { left, right, scale } => {
                visit(left, scale * parameters[right]);
                visit(right, scale * parameters[left]);
            }
        }
    }
}

/// A validated circuit and its physical parameterization.
///
/// The circuit is exposed immutably so bindings cannot become stale. Dispatch
/// validates all angles before changing any values. Fixed basis-change angles
/// can be represented by [`ParameterBinding::Fixed`].
#[derive(Debug, Clone)]
pub struct BoundCircuit {
    circuit: Circuit,
    parameters: Vec<f64>,
    bindings: Vec<ParameterBinding>,
    // Rotation builders record the angle slots whose simultaneous zero makes
    // their entire circuit identity. The graph is retained for differentiation.
    identity_slots: Option<Vec<usize>>,
}

impl BoundCircuit {
    /// Bind every gate parameter in a circuit to a supplied physical vector.
    pub fn new(
        circuit: Circuit,
        parameters: Vec<f64>,
        bindings: Vec<ParameterBinding>,
    ) -> Result<Self, String> {
        let circuit = Circuit::new(circuit.dims.clone(), circuit.elements.clone())
            .map_err(|e| e.to_string())?;
        if circuit.num_params() != bindings.len() {
            return Err("Binding count does not match circuit parameter count".into());
        }
        for binding in &bindings {
            binding.validate(parameters.len())?;
        }
        let mut result = Self {
            circuit,
            parameters: vec![0.; parameters.len()],
            bindings,
            identity_slots: None,
        };
        result.dispatch(&parameters)?;
        Ok(result)
    }

    /// Current lowered circuit, with expanded gate angles already dispatched.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    /// Current physical parameters. Their names/order are defined by the builder.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    /// One binding per expanded gate angle.
    pub fn bindings(&self) -> &[ParameterBinding] {
        &self.bindings
    }

    /// Update physical parameters atomically; reject invalid lengths and nonfinite values.
    pub fn dispatch(&mut self, parameters: &[f64]) -> Result<(), String> {
        if parameters.len() != self.parameters.len() {
            return Err("Physical parameter count mismatch".into());
        }
        finite(parameters, "Physical parameters")?;
        let angles = self
            .bindings
            .iter()
            .map(|b| b.value(parameters))
            .collect::<Vec<_>>();
        finite(&angles, "Bound gate angles")?;
        self.circuit.dispatch(&angles);
        self.parameters.copy_from_slice(parameters);
        Ok(())
    }

    /// Apply the binding Jacobian transpose to gradients in gate-angle order.
    /// Shared occurrences accumulate into one physical parameter gradient.
    pub fn pullback(&self, angle_gradient: &[f64]) -> Result<Vec<f64>, String> {
        if angle_gradient.len() != self.bindings.len() {
            return Err("Gate gradient count mismatch".into());
        }
        finite(angle_gradient, "Gate gradients")?;
        let mut result = vec![0.; self.parameters.len()];
        for (binding, &gradient) in self.bindings.iter().zip(angle_gradient) {
            binding.derivatives(&self.parameters, |index, derivative| {
                result[index] += gradient * derivative
            });
        }
        finite(&result, "Physical gradients")?;
        Ok(result)
    }

    /// Apply the binding Jacobian to a physical parameter tangent.
    pub fn pushforward(&self, tangent: &[f64]) -> Result<Vec<f64>, String> {
        if tangent.len() != self.parameters.len() {
            return Err("Physical tangent count mismatch".into());
        }
        finite(tangent, "Physical tangents")?;
        let result = self
            .bindings
            .iter()
            .map(|binding| {
                let mut value = 0.;
                binding.derivatives(&self.parameters, |index, derivative| {
                    value += derivative * tangent[index]
                });
                value
            })
            .collect::<Vec<_>>();
        finite(&result, "Gate tangents")?;
        Ok(result)
    }

    /// Simulate the bound qubit circuit with the native register kernels.
    /// Rotation/evolution builders return the input exactly for simultaneous
    /// zero rotation angles, while keeping their graph for derivatives at zero.
    /// The lowered circuit and tensor contraction agree up to floating-point error.
    pub fn apply(&self, input: &ArrayReg) -> Result<ArrayReg, String> {
        if input.nqubits() != self.circuit.nbits
            || self.circuit.nbits >= usize::BITS as usize
            || input.state.len() != 1usize << self.circuit.nbits
            || self.circuit.dims.iter().any(|&d| d != 2)
        {
            return Err("Bound circuit and qubit register dimensions disagree".into());
        }
        if self
            .circuit
            .elements
            .iter()
            .any(|e| matches!(e, crate::CircuitElement::Channel(_)))
        {
            return Err("State-vector application does not support channels".into());
        }
        let angles = self.circuit.parameters();
        if self
            .identity_slots
            .as_ref()
            .is_some_and(|slots| slots.iter().all(|&i| angles[i] == 0.))
        {
            return Ok(input.clone());
        }
        Ok(apply(&self.circuit, input))
    }

    pub(crate) fn with_identity_slots(mut self, slots: Vec<usize>) -> Self {
        self.identity_slots = Some(slots);
        self
    }
}

fn finite(values: &[f64], what: &str) -> Result<(), String> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(format!("{what} must be finite"))
    }
}

#[cfg(test)]
#[path = "unit_tests/parameters.rs"]
mod tests;
