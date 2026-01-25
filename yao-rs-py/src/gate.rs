use pyo3::prelude::*;
use yao_rs::Gate;

#[pyclass(name = "Gate")]
#[derive(Clone)]
pub struct PyGate(pub Gate);

#[pymethods]
impl PyGate {
    #[staticmethod]
    fn h() -> Self {
        PyGate(Gate::H)
    }

    #[staticmethod]
    fn x() -> Self {
        PyGate(Gate::X)
    }

    #[staticmethod]
    fn y() -> Self {
        PyGate(Gate::Y)
    }

    #[staticmethod]
    fn z() -> Self {
        PyGate(Gate::Z)
    }

    #[staticmethod]
    fn s() -> Self {
        PyGate(Gate::S)
    }

    #[staticmethod]
    fn t() -> Self {
        PyGate(Gate::T)
    }

    #[staticmethod]
    fn rx(theta: f64) -> Self {
        PyGate(Gate::Rx(theta))
    }

    #[staticmethod]
    fn ry(theta: f64) -> Self {
        PyGate(Gate::Ry(theta))
    }

    #[staticmethod]
    fn rz(theta: f64) -> Self {
        PyGate(Gate::Rz(theta))
    }

    #[staticmethod]
    fn swap() -> Self {
        PyGate(Gate::SWAP)
    }

    /// Return the dagger (adjoint) of this gate
    fn dagger(&self) -> Self {
        PyGate(self.0.dagger())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
