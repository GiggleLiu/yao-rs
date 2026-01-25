use pyo3::prelude::*;
use yao_rs::Gate;

/// A quantum gate wrapper for Python.
#[pyclass(name = "Gate")]
#[derive(Clone)]
pub struct PyGate(pub Gate);

#[pymethods]
impl PyGate {
    /// Create a Hadamard gate.
    #[staticmethod]
    fn h() -> Self {
        PyGate(Gate::H)
    }

    /// Create a Pauli-X (NOT) gate.
    #[staticmethod]
    fn x() -> Self {
        PyGate(Gate::X)
    }

    /// Create a Pauli-Y gate.
    #[staticmethod]
    fn y() -> Self {
        PyGate(Gate::Y)
    }

    /// Create a Pauli-Z gate.
    #[staticmethod]
    fn z() -> Self {
        PyGate(Gate::Z)
    }

    /// Create an S gate (π/2 phase).
    #[staticmethod]
    fn s() -> Self {
        PyGate(Gate::S)
    }

    /// Create a T gate (π/4 phase).
    #[staticmethod]
    fn t() -> Self {
        PyGate(Gate::T)
    }

    /// Create a Phase gate: diag(1, e^(iθ)).
    ///
    /// Args:
    ///     theta: The phase angle in radians
    #[staticmethod]
    fn phase(theta: f64) -> Self {
        PyGate(Gate::Phase(theta))
    }

    /// Create an Rx gate (rotation around X-axis).
    ///
    /// Args:
    ///     theta: The rotation angle in radians
    #[staticmethod]
    fn rx(theta: f64) -> Self {
        PyGate(Gate::Rx(theta))
    }

    /// Create an Ry gate (rotation around Y-axis).
    ///
    /// Args:
    ///     theta: The rotation angle in radians
    #[staticmethod]
    fn ry(theta: f64) -> Self {
        PyGate(Gate::Ry(theta))
    }

    /// Create an Rz gate (rotation around Z-axis).
    ///
    /// Args:
    ///     theta: The rotation angle in radians
    #[staticmethod]
    fn rz(theta: f64) -> Self {
        PyGate(Gate::Rz(theta))
    }

    /// Create a SWAP gate (exchanges two qubits).
    #[staticmethod]
    fn swap() -> Self {
        PyGate(Gate::SWAP)
    }

    /// Create a √X gate: SqrtX² = X.
    #[staticmethod]
    fn sqrt_x() -> Self {
        PyGate(Gate::SqrtX)
    }

    /// Create a √Y gate: SqrtY² = Y.
    #[staticmethod]
    fn sqrt_y() -> Self {
        PyGate(Gate::SqrtY)
    }

    /// Create a √W gate: rotation around (X+Y)/√2 by π/2.
    #[staticmethod]
    fn sqrt_w() -> Self {
        PyGate(Gate::SqrtW)
    }

    /// Create an iSWAP gate (two-qubit).
    #[staticmethod]
    fn iswap() -> Self {
        PyGate(Gate::ISWAP)
    }

    /// Create an FSim gate (two-qubit, parameterized).
    ///
    /// Args:
    ///     theta: The swap angle
    ///     phi: The controlled-phase angle
    #[staticmethod]
    fn fsim(theta: f64, phi: f64) -> Self {
        PyGate(Gate::FSim(theta, phi))
    }

    /// Return the dagger (adjoint) of this gate.
    fn dagger(&self) -> Self {
        PyGate(self.0.dagger())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
