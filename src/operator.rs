use ndarray::{array, Array2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

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
