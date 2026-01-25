use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use num_complex::Complex64;
use yao_rs::operator::{Op, OperatorPolynomial};

#[pyclass(name = "OperatorPolynomial")]
#[derive(Clone)]
pub struct PyOperatorPolynomial(pub OperatorPolynomial);

#[pymethods]
impl PyOperatorPolynomial {
    #[staticmethod]
    fn i(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::I, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn x(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::X, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn y(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::Y, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn z(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::Z, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn p0(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::P0, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn p1(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::P1, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn pu(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::Pu, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn pd(site: usize) -> Self {
        PyOperatorPolynomial(OperatorPolynomial::single(site, Op::Pd, Complex64::new(1.0, 0.0)))
    }

    #[staticmethod]
    fn identity() -> Self {
        PyOperatorPolynomial(OperatorPolynomial::identity())
    }

    #[staticmethod]
    fn zero() -> Self {
        PyOperatorPolynomial(OperatorPolynomial::zero())
    }

    fn __add__(&self, other: &PyOperatorPolynomial) -> Self {
        PyOperatorPolynomial(&self.0 + &other.0)
    }

    fn __neg__(&self) -> Self {
        PyOperatorPolynomial(-&self.0)
    }

    fn __sub__(&self, other: &PyOperatorPolynomial) -> Self {
        PyOperatorPolynomial(&self.0 + &(-&other.0))
    }

    fn __mul__(&self, scalar: f64) -> Self {
        PyOperatorPolynomial(&self.0 * Complex64::new(scalar, 0.0))
    }

    fn __rmul__(&self, scalar: f64) -> Self {
        self.__mul__(scalar)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        serde_json::from_str(json_str)
            .map(PyOperatorPolynomial)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    fn __repr__(&self) -> String {
        format!("OperatorPolynomial(terms={})", self.0.len())
    }
}
