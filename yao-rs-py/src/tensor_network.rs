use pyo3::prelude::*;
use yao_rs::TensorNetwork;
use crate::circuit::PyCircuit;
use crate::operator::PyOperatorPolynomial;

/// A tensor network wrapper for Python.
///
/// TensorNetwork represents a quantum circuit or expectation value
/// as an einsum contraction pattern with associated tensors.
#[pyclass(name = "TensorNetwork")]
#[derive(Clone)]
pub struct PyTensorNetwork(pub TensorNetwork);

#[pymethods]
impl PyTensorNetwork {
    /// Return the number of tensors in the network.
    fn num_tensors(&self) -> usize {
        self.0.tensors.len()
    }

    fn __repr__(&self) -> String {
        format!("TensorNetwork(tensors={})", self.num_tensors())
    }
}

/// Convert circuit to tensor network for overlap <0|U|0>.
///
/// This creates a tensor network that computes the amplitude
/// of the circuit starting and ending in the |0...0> state.
///
/// Args:
///     circuit: The quantum circuit to convert
///
/// Returns:
///     A TensorNetwork representing the overlap computation
#[pyfunction]
pub fn circuit_to_overlap(circuit: &PyCircuit) -> PyTensorNetwork {
    PyTensorNetwork(yao_rs::circuit_to_overlap(&circuit.0))
}

/// Convert circuit + operator to tensor network for expectation <0|U†OU|0>.
///
/// This creates a tensor network that computes the expectation value
/// of an operator with respect to the circuit applied to |0...0>.
///
/// Args:
///     circuit: The quantum circuit U
///     operator: The operator polynomial O
///
/// Returns:
///     A TensorNetwork representing the expectation value computation
#[pyfunction]
pub fn circuit_to_expectation(circuit: &PyCircuit, operator: &PyOperatorPolynomial) -> PyTensorNetwork {
    PyTensorNetwork(yao_rs::circuit_to_expectation(&circuit.0, &operator.0))
}

// Note: contract() requires the torch feature. For now we expose the TN structure
// and let Python handle optimization/contraction via other means if needed.
// In the Rust backend, contraction can use `omeco::optimize_greedy` when available,
// but that optimization is only applied inside contract() and is not exposed here.
