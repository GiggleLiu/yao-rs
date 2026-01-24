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
    /// √X gate: SqrtX² = X
    SqrtX,
    /// √Y gate: SqrtY² = Y
    SqrtY,
    /// √W gate: rot((X+Y)/√2, π/2), non-Clifford
    SqrtW,
    /// iSWAP gate: two-qubit
    ISWAP,
    /// FSim gate: two-qubit, parameterized by (theta, phi)
    FSim(f64, f64),
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
            Gate::SWAP | Gate::ISWAP | Gate::FSim(_, _) => 2,
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
            Gate::SqrtX => {
                // (1+i)/2 * [[1, -i], [-i, 1]]
                let f = Complex64::new(0.5, 0.5); // (1+i)/2
                Array2::from_shape_vec((2, 2), vec![
                    f * one,
                    f * neg_i,
                    f * neg_i,
                    f * one,
                ]).unwrap()
            }
            Gate::SqrtY => {
                // (1+i)/2 * [[1, -1], [1, 1]]
                let f = Complex64::new(0.5, 0.5); // (1+i)/2
                Array2::from_shape_vec((2, 2), vec![
                    f * one,
                    f * neg_one,
                    f * one,
                    f * one,
                ]).unwrap()
            }
            Gate::SqrtW => {
                // cos(π/4)*I - i*sin(π/4)*G where G = (X+Y)/√2
                // G = [[0, (1-i)/√2], [(1+i)/√2, 0]]
                // cos(π/4) = sin(π/4) = 1/√2
                let cos_val = Complex64::new(FRAC_1_SQRT_2, 0.0);
                let neg_i_sin = Complex64::new(0.0, -FRAC_1_SQRT_2); // -i * sin(π/4)
                // G[0,1] = (1-i)/√2, G[1,0] = (1+i)/√2
                let g01 = Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
                let g10 = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
                // M = cos_val * I - i*sin_val * G
                // M[0,0] = cos_val, M[1,1] = cos_val
                // M[0,1] = neg_i_sin * G[0,1], M[1,0] = neg_i_sin * G[1,0]
                Array2::from_shape_vec((2, 2), vec![
                    cos_val,
                    neg_i_sin * g01,
                    neg_i_sin * g10,
                    cos_val,
                ]).unwrap()
            }
            Gate::ISWAP => {
                // 4x4 matrix: diag(1, 0, 0, 1) with m[1,2]=i, m[2,1]=i
                let mut m = Array2::zeros((4, 4));
                m[[0, 0]] = one;
                m[[1, 2]] = i;
                m[[2, 1]] = i;
                m[[3, 3]] = one;
                m
            }
            Gate::FSim(theta, phi) => {
                // [[1, 0, 0, 0],
                //  [0, cos(θ), -i*sin(θ), 0],
                //  [0, -i*sin(θ), cos(θ), 0],
                //  [0, 0, 0, e^(-iφ)]]
                let cos_theta = Complex64::new(theta.cos(), 0.0);
                let neg_i_sin_theta = Complex64::new(0.0, -theta.sin());
                let e_neg_i_phi = Complex64::from_polar(1.0, -phi);
                let mut m = Array2::zeros((4, 4));
                m[[0, 0]] = one;
                m[[1, 1]] = cos_theta;
                m[[1, 2]] = neg_i_sin_theta;
                m[[2, 1]] = neg_i_sin_theta;
                m[[2, 2]] = cos_theta;
                m[[3, 3]] = e_neg_i_phi;
                m
            }
            Gate::Custom { .. } => unreachable!(),
        }
    }
}
