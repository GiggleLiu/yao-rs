use pyo3::prelude::*;

mod gate;
mod circuit;
mod operator;
mod tensor_network;

use gate::PyGate;
use circuit::PyCircuit;
use operator::PyOperatorPolynomial;
use tensor_network::{PyTensorNetwork, circuit_to_overlap, circuit_to_expectation};

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn _yao_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyOperatorPolynomial>()?;
    m.add_class::<PyTensorNetwork>()?;
    m.add_function(wrap_pyfunction!(circuit_to_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_to_expectation, m)?)?;
    Ok(())
}
