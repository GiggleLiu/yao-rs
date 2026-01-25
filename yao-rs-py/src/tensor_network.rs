use pyo3::prelude::*;
#[cfg(feature = "torch")]
use pyo3::types::PyComplex;
use yao_rs::TensorNetwork;
use crate::circuit::PyCircuit;
use crate::operator::PyOperatorPolynomial;

#[cfg(feature = "torch")]
use yao_rs::torch_contractor;

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

    /// Contract the tensor network and return the result as a complex number.
    ///
    /// Args:
    ///     device: Device to run on - "cpu" or "cuda:N" for GPU N (default: "cpu")
    ///
    /// Returns:
    ///     Complex number result of the contraction
    ///
    /// Raises:
    ///     RuntimeError: If torch feature is not enabled or device is invalid
    #[pyo3(signature = (device = "cpu"))]
    fn contract<'py>(&self, py: Python<'py>, device: &str) -> PyResult<PyObject> {
        #[cfg(feature = "torch")]
        {
            let dev = parse_device(device)?;
            let result = torch_contractor::contract(&self.0, dev);

            // Extract real and imaginary parts from the scalar tensor
            let re = f64::try_from(result.real()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract real part: {}", e))
            })?;
            let im = f64::try_from(result.imag()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract imag part: {}", e))
            })?;

            Ok(PyComplex::from_doubles(py, re, im).into())
        }

        #[cfg(not(feature = "torch"))]
        {
            let _ = (py, device); // silence unused warnings
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Torch feature is not enabled. Rebuild with: maturin develop --features torch"
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("TensorNetwork(tensors={})", self.num_tensors())
    }
}

/// Parse device string to tch::Device
#[cfg(feature = "torch")]
fn parse_device(device: &str) -> PyResult<tch::Device> {
    use tch::Device;

    let device_lower = device.to_lowercase();
    if device_lower == "cpu" {
        Ok(Device::Cpu)
    } else if device_lower.starts_with("cuda") {
        // Parse "cuda" or "cuda:N"
        if device_lower == "cuda" {
            Ok(Device::Cuda(0))
        } else if let Some(idx_str) = device_lower.strip_prefix("cuda:") {
            let idx: usize = idx_str.parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid CUDA device index: {}", idx_str))
            })?;
            Ok(Device::Cuda(idx))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid device: {}", device)))
        }
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown device: {}. Use 'cpu' or 'cuda:N'", device
        )))
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
