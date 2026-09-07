//! Fusion for repeated execution of fixed-parameter qubit circuits.

use crate::circuit::CircuitError;
use crate::{ArrayReg, Circuit, CircuitElement, Gate, PositionedGate, apply_inplace};
use ndarray::Array2;
use num_complex::Complex64;

/// A circuit cannot be fused with the requested block size.
#[derive(Debug)]
pub enum FusionError {
    /// Fusion currently requires qubit dimensions.
    QubitsRequired,
    /// Blocks must contain between one and eight qubits.
    BlockSize(usize),
    /// The input or resulting circuit is invalid.
    InvalidCircuit(CircuitError),
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QubitsRequired => write!(f, "Circuit fusion requires qubits"),
            Self::BlockSize(n) => write!(f, "Fusion block size must be in 1..=8, got {n}"),
            Self::InvalidCircuit(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FusionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCircuit(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CircuitError> for FusionError {
    fn from(error: CircuitError) -> Self {
        Self::InvalidCircuit(error)
    }
}

impl Circuit {
    /// Fuse consecutive gates into fixed matrices on at most `max_qubits` sites.
    ///
    /// Prepare once, then reuse the returned circuit with [`crate::apply()`] or
    /// [`crate::apply_inplace`]. Two-qubit blocks are a useful starting point;
    /// larger dense blocks can be slower than the original specialized gates.
    /// Measure the preparation and execution costs for your workload.
    ///
    /// Parameter values are frozen into matrices: differentiate the original
    /// circuit, and prepare again after changing parameters. Noise channels and
    /// annotations retain their order and separate fusion blocks. A gate larger
    /// than the requested block size is retained as a separate gate.
    ///
    /// # Errors
    /// Rejects non-qubit dimensions, block sizes outside `1..=8`, and invalid
    /// circuit placements. Eight qubits bound each fused block matrix to 1 MiB.
    pub fn fused(&self, max_qubits: usize) -> Result<Self, FusionError> {
        if !(1..=8).contains(&max_qubits) {
            return Err(FusionError::BlockSize(max_qubits));
        }
        if self.dims.iter().any(|&dim| dim != 2) {
            return Err(FusionError::QubitsRequired);
        }
        let validated = Circuit::new(self.dims.clone(), self.elements.clone())?;
        let mut output = Vec::new();
        let mut pending = Vec::new();
        let mut sites = Vec::new();
        for element in validated.elements {
            let CircuitElement::Gate(gate) = element else {
                flush(&mut pending, &mut sites, &mut output)?;
                output.push(element);
                continue;
            };
            let mut combined = sites.clone();
            combined.extend(gate.all_locs());
            combined.sort_unstable();
            combined.dedup();
            if combined.len() > max_qubits {
                flush(&mut pending, &mut sites, &mut output)?;
                combined = gate.all_locs();
                combined.sort_unstable();
            }
            if combined.len() > max_qubits {
                output.push(fixed_gate(gate));
            } else {
                sites = combined;
                pending.push(gate);
            }
        }
        flush(&mut pending, &mut sites, &mut output)?;
        Ok(Circuit::new(self.dims.clone(), output)?)
    }
}

fn fixed_gate(mut gate: PositionedGate) -> CircuitElement {
    // Keep specialized permutation kernels while precomputing every parameter.
    if gate.gate.num_params() > 0 {
        gate.gate = Gate::Custom {
            matrix: gate.gate.matrix(),
            is_diagonal: gate.gate.is_diagonal(),
            label: "U".into(),
        };
    }
    CircuitElement::Gate(gate)
}

fn flush(
    pending: &mut Vec<PositionedGate>,
    sites: &mut Vec<usize>,
    output: &mut Vec<CircuitElement>,
) -> Result<(), FusionError> {
    if pending.is_empty() {
        return Ok(());
    }
    if pending.len() == 1 {
        output.push(fixed_gate(pending.pop().unwrap()));
        sites.clear();
        return Ok(());
    }
    let local_gates = pending
        .drain(..)
        .map(|mut gate| {
            for location in gate.target_locs.iter_mut().chain(&mut gate.control_locs) {
                *location = sites.binary_search(location).unwrap();
            }
            CircuitElement::Gate(gate)
        })
        .collect();
    let local = Circuit::qubits(sites.len(), local_gates)?;
    let dimension = 1usize << sites.len();
    let zero = Complex64::new(0., 0.);
    let mut matrix = Array2::zeros((dimension, dimension));
    for column in 0..dimension {
        let mut data = vec![zero; dimension];
        data[column] = Complex64::new(1., 0.);
        let mut state = ArrayReg::from_vec(sites.len(), data);
        apply_inplace(&local, &mut state);
        for (row, &value) in state.state_vec().iter().enumerate() {
            matrix[[row, column]] = value;
        }
    }
    let is_diagonal = matrix
        .indexed_iter()
        .all(|((row, column), &value)| row == column || value == zero);
    // A local ArrayReg is MSB-first; Custom gate targets enumerate matrix bits
    // from least significant to most significant.
    let targets = sites.drain(..).rev().collect::<Vec<_>>();
    output.push(crate::put(
        targets,
        Gate::Custom {
            matrix,
            is_diagonal,
            label: "U".into(),
        },
    ));
    Ok(())
}

#[cfg(test)]
#[path = "unit_tests/fusion.rs"]
mod tests;
