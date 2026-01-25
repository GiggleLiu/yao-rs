use ndarray::{array, Array2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg};

/// Single-site operator (Pauli basis + projectors)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    I,   // Identity
    X,   // Pauli X
    Y,   // Pauli Y
    Z,   // Pauli Z
    P0,  // |0><0| projector
    P1,  // |1><1| projector
    Pu,  // |0><1| raising
    Pd,  // |1><0| lowering
}

/// Product of operators at different sites: Z(0)Z(1)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperatorString {
    /// (site_index, operator) pairs, sorted by site
    ops: Vec<(usize, Op)>,
}

impl OperatorString {
    pub fn new(mut ops: Vec<(usize, Op)>) -> Self {
        // Sort by site index
        ops.sort_by_key(|(site, _)| *site);
        // Remove identity operators
        ops.retain(|(_, op)| *op != Op::I);
        Self { ops }
    }

    pub fn identity() -> Self {
        Self { ops: vec![] }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn ops(&self) -> &[(usize, Op)] {
        &self.ops
    }
}

/// Get 2x2 matrix for operator
pub fn op_matrix(op: &Op) -> Array2<Complex64> {
    let c = |r: f64, i: f64| Complex64::new(r, i);
    match op {
        Op::I => array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        Op::X => array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
        Op::Y => array![[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]],
        Op::Z => array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]],
        Op::P0 => array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]],
        Op::P1 => array![[c(0.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        Op::Pu => array![[c(0.0, 0.0), c(1.0, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]],
        Op::Pd => array![[c(0.0, 0.0), c(0.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
    }
}

/// Linear combination of operator strings: Σ cᵢ·Oᵢ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorPolynomial {
    coeffs: Vec<Complex64>,
    opstrings: Vec<OperatorString>,
}

impl OperatorPolynomial {
    pub fn new(coeffs: Vec<Complex64>, opstrings: Vec<OperatorString>) -> Self {
        assert_eq!(coeffs.len(), opstrings.len());
        Self { coeffs, opstrings }
    }

    /// Create single-term polynomial: coeff * Op(site)
    pub fn single(site: usize, op: Op, coeff: Complex64) -> Self {
        Self {
            coeffs: vec![coeff],
            opstrings: vec![OperatorString::new(vec![(site, op)])],
        }
    }

    /// Identity operator (empty polynomial = 1)
    pub fn identity() -> Self {
        Self {
            coeffs: vec![Complex64::new(1.0, 0.0)],
            opstrings: vec![OperatorString::identity()],
        }
    }

    /// Zero operator
    pub fn zero() -> Self {
        Self {
            coeffs: vec![],
            opstrings: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

    pub fn opstrings(&self) -> &[OperatorString] {
        &self.opstrings
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Complex64, &OperatorString)> {
        self.coeffs.iter().zip(self.opstrings.iter())
    }
}

impl Add for &OperatorPolynomial {
    type Output = OperatorPolynomial;

    fn add(self, other: &OperatorPolynomial) -> OperatorPolynomial {
        let mut coeffs = self.coeffs.clone();
        coeffs.extend(other.coeffs.iter().cloned());
        let mut opstrings = self.opstrings.clone();
        opstrings.extend(other.opstrings.iter().cloned());
        OperatorPolynomial::new(coeffs, opstrings)
    }
}

impl Neg for &OperatorPolynomial {
    type Output = OperatorPolynomial;

    fn neg(self) -> OperatorPolynomial {
        let coeffs = self.coeffs.iter().map(|c| -c).collect();
        OperatorPolynomial::new(coeffs, self.opstrings.clone())
    }
}

impl Mul<Complex64> for &OperatorPolynomial {
    type Output = OperatorPolynomial;

    fn mul(self, scalar: Complex64) -> OperatorPolynomial {
        let coeffs = self.coeffs.iter().map(|c| c * scalar).collect();
        OperatorPolynomial::new(coeffs, self.opstrings.clone())
    }
}
