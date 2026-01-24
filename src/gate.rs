use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_4, FRAC_1_SQRT_2};

/// Quantum gate enum supporting named qubit gates and custom gates.
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    X,
    Y,
    Z,
    H,
    S,
    T,
    SWAP,
    /// Phase gate: diag(1, e^(iθ)). Equivalent to Yao.jl's `shift(θ)`.
    Phase(f64),
    Rx(f64),
    Ry(f64),
    Rz(f64),
    Custom {
        matrix: Array2<Complex64>,
        is_diagonal: bool,
        label: String,
    },
}

impl Gate {
    /// Returns the matrix representation of the gate for local dimension `d`.
    ///
    /// # Panics
    /// Panics if `d != 2` for named (non-Custom) gate variants.
    pub fn matrix(&self, d: usize) -> Array2<Complex64> {
        match self {
            Gate::Custom { matrix, .. } => matrix.clone(),
            _ => {
                assert!(d == 2, "Named gates only support d=2, got d={}", d);
                self.qubit_matrix()
            }
        }
    }

    /// Returns the number of sites (qubits) the gate acts on.
    pub fn num_sites(&self, d: usize) -> usize {
        match self {
            Gate::SWAP => 2,
            Gate::Custom { matrix, .. } => {
                let dim = matrix.nrows();
                assert_eq!(matrix.nrows(), matrix.ncols(),
                    "Custom gate matrix must be square, got {}x{}", matrix.nrows(), matrix.ncols());
                // dim = d^n, solve for n
                let mut n = 0usize;
                let mut power = 1usize;
                while power < dim {
                    power *= d;
                    n += 1;
                }
                assert_eq!(power, dim, "Matrix dimension {} is not a power of d={}", dim, d);
                n
            }
            _ => 1,
        }
    }

    /// Returns whether the gate is diagonal.
    pub fn is_diagonal(&self) -> bool {
        match self {
            Gate::Z | Gate::S | Gate::T | Gate::Phase(_) | Gate::Rz(_) => true,
            Gate::Custom { is_diagonal, .. } => *is_diagonal,
            _ => false,
        }
    }

    /// Internal: compute the 2x2 or 4x4 matrix for named qubit gates.
    fn qubit_matrix(&self) -> Array2<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match self {
            Gate::X => {
                Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap()
            }
            Gate::Y => {
                Array2::from_shape_vec((2, 2), vec![zero, neg_i, i, zero]).unwrap()
            }
            Gate::Z => {
                Array2::from_shape_vec((2, 2), vec![one, zero, zero, neg_one]).unwrap()
            }
            Gate::H => {
                let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
                let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
                Array2::from_shape_vec((2, 2), vec![s, s, s, neg_s]).unwrap()
            }
            Gate::S => {
                Array2::from_shape_vec((2, 2), vec![one, zero, zero, i]).unwrap()
            }
            Gate::T => {
                let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
                Array2::from_shape_vec((2, 2), vec![one, zero, zero, t_phase]).unwrap()
            }
            Gate::SWAP => {
                // 4x4 matrix: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
                // Row-major: rows are |00>, |01>, |10>, |11>
                let mut m = Array2::zeros((4, 4));
                m[[0, 0]] = one; // |00> -> |00>
                m[[1, 2]] = one; // |01> -> |10>
                m[[2, 1]] = one; // |10> -> |01>
                m[[3, 3]] = one; // |11> -> |11>
                m
            }
            Gate::Phase(theta) => {
                let phase = Complex64::from_polar(1.0, *theta);
                Array2::from_shape_vec((2, 2), vec![one, zero, zero, phase]).unwrap()
            }
            Gate::Rx(theta) => {
                let cos = Complex64::new((theta / 2.0).cos(), 0.0);
                let neg_i_sin = Complex64::new(0.0, -(theta / 2.0).sin());
                Array2::from_shape_vec((2, 2), vec![cos, neg_i_sin, neg_i_sin, cos]).unwrap()
            }
            Gate::Ry(theta) => {
                let cos = Complex64::new((theta / 2.0).cos(), 0.0);
                let sin = Complex64::new((theta / 2.0).sin(), 0.0);
                let neg_sin = Complex64::new(-(theta / 2.0).sin(), 0.0);
                Array2::from_shape_vec((2, 2), vec![cos, neg_sin, sin, cos]).unwrap()
            }
            Gate::Rz(theta) => {
                let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
                let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
                Array2::from_shape_vec((2, 2), vec![phase_neg, zero, zero, phase_pos]).unwrap()
            }
            Gate::Custom { .. } => unreachable!(),
        }
    }
}
