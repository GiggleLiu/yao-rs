// Allow PyO3-generated code patterns in Rust 2024 edition
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod circuit;
mod gate;
mod operator;
mod tensor_network;

use circuit::PyCircuit;
use gate::PyGate;
use operator::PyOperatorPolynomial;
use tensor_network::{PyTensorNetwork, circuit_to_expectation, circuit_to_overlap};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
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
