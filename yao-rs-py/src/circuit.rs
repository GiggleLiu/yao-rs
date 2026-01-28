use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use yao_rs::{Circuit, put, control, circuit_to_json, circuit_from_json};

use crate::gate::PyGate;

/// A quantum circuit wrapper for Python.
#[pyclass(name = "Circuit")]
#[derive(Clone)]
pub struct PyCircuit(pub Circuit);

#[pymethods]
impl PyCircuit {
    /// Create a new circuit with the given qudit dimensions.
    ///
    /// Args:
    ///     dims: List of local dimensions for each site (e.g., [2, 2, 2] for 3 qubits)
    #[new]
    fn new(dims: Vec<usize>) -> PyResult<Self> {
        Circuit::new(dims, vec![])
            .map(PyCircuit)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Place a gate on target locations (no controls).
    ///
    /// Args:
    ///     gate: The gate to apply
    ///     locs: Target locations (list of site indices)
    ///
    /// Returns:
    ///     A new circuit with the gate added
    fn put(&self, gate: PyGate, locs: Vec<usize>) -> PyResult<Self> {
        let pg = put(locs, gate.0);
        let mut gates = self.0.gates.clone();
        gates.push(pg);
        Circuit::new(self.0.dims.clone(), gates)
            .map(PyCircuit)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Place a controlled gate.
    ///
    /// Args:
    ///     gate: The gate to apply
    ///     ctrl: Control locations (list of site indices)
    ///     tgt: Target locations (list of site indices)
    ///
    /// Returns:
    ///     A new circuit with the controlled gate added
    fn control(&self, gate: PyGate, ctrl: Vec<usize>, tgt: Vec<usize>) -> PyResult<Self> {
        let pg = control(ctrl, tgt, gate.0);
        let mut gates = self.0.gates.clone();
        gates.push(pg);
        Circuit::new(self.0.dims.clone(), gates)
            .map(PyCircuit)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return the adjoint (dagger) of this circuit.
    ///
    /// The dagger of a circuit has gates in reverse order,
    /// with each gate replaced by its adjoint.
    fn dagger(&self) -> PyResult<Self> {
        self.0.dagger()
            .map(PyCircuit)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return the number of sites in the circuit.
    fn num_sites(&self) -> usize {
        self.0.num_sites()
    }

    /// Return the qudit dimensions.
    fn dims(&self) -> Vec<usize> {
        self.0.dims.clone()
    }

    /// Return the number of gates in the circuit.
    fn num_gates(&self) -> usize {
        self.0.gates.len()
    }

    /// Serialize the circuit to a JSON string.
    fn to_json(&self) -> String {
        circuit_to_json(&self.0)
    }

    /// Deserialize a circuit from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        circuit_from_json(json)
            .map(PyCircuit)
            .map_err(PyValueError::new_err)
    }

    /// Create a circuit from a Python dictionary.
    ///
    /// Args:
    ///     data: A dictionary with 'registers' and 'gates' keys
    ///
    /// Returns:
    ///     A new Circuit
    #[staticmethod]
    fn from_dict(data: &pyo3::types::PyDict) -> PyResult<Self> {
        let json_str = serde_json::to_string(&pythonize::depythonize::<serde_json::Value>(data)
            .map_err(|e| PyValueError::new_err(format!("Failed to convert dict: {}", e)))?)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize: {}", e)))?;
        circuit_from_json(&json_str)
            .map(PyCircuit)
            .map_err(PyValueError::new_err)
    }

    /// Load a circuit from a JSON file.
    #[staticmethod]
    fn from_json_file(path: &str) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read file: {}", e)))?;
        circuit_from_json(&json)
            .map(PyCircuit)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }
}
