use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

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

impl std::fmt::Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gate::X => write!(f, "X"),
            Gate::Y => write!(f, "Y"),
            Gate::Z => write!(f, "Z"),
            Gate::H => write!(f, "H"),
            Gate::S => write!(f, "S"),
            Gate::T => write!(f, "T"),
            Gate::SWAP => write!(f, "SWAP"),
            Gate::SqrtX => write!(f, "SqrtX"),
            Gate::SqrtY => write!(f, "SqrtY"),
            Gate::SqrtW => write!(f, "SqrtW"),
            Gate::ISWAP => write!(f, "ISWAP"),
            Gate::Phase(theta) => write!(f, "Phase({:.4})", theta),
            Gate::Rx(theta) => write!(f, "Rx({:.4})", theta),
            Gate::Ry(theta) => write!(f, "Ry({:.4})", theta),
            Gate::Rz(theta) => write!(f, "Rz({:.4})", theta),
            Gate::FSim(theta, phi) => write!(f, "FSim({:.4}, {:.4})", theta, phi),
            Gate::Custom { label, .. } => write!(f, "{}", label),
        }
    }
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
                assert_eq!(
                    matrix.nrows(),
                    matrix.ncols(),
                    "Custom gate matrix must be square, got {}x{}",
                    matrix.nrows(),
                    matrix.ncols()
                );
                // dim = d^n, solve for n
                let mut n = 0usize;
                let mut power = 1usize;
                while power < dim {
                    power *= d;
                    n += 1;
                }
                assert_eq!(
                    power, dim,
                    "Matrix dimension {} is not a power of d={}",
                    dim, d
                );
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

    /// Return the adjoint (conjugate transpose) of this gate.
    ///
    /// For unitary gates, the adjoint is also the inverse: U† U = I.
    pub fn dagger(&self) -> Self {
        match self {
            // Self-adjoint gates (Hermitian): H, X, Y, Z, SWAP
            Gate::H | Gate::X | Gate::Y | Gate::Z | Gate::SWAP => self.clone(),

            // S† = Rz(-π/2) equivalent, implemented as Phase(-π/2)
            Gate::S => Gate::Phase(-std::f64::consts::FRAC_PI_2),

            // T† = Rz(-π/4) equivalent, implemented as Phase(-π/4)
            Gate::T => Gate::Phase(-std::f64::consts::FRAC_PI_4),

            // Rotation gates: negate the angle
            Gate::Rx(theta) => Gate::Rx(-theta),
            Gate::Ry(theta) => Gate::Ry(-theta),
            Gate::Rz(theta) => Gate::Rz(-theta),
            Gate::Phase(theta) => Gate::Phase(-theta),

            // SqrtX† = SqrtX^(-1) - we compute the conjugate transpose
            // SqrtX = (1+i)/2 * [[1, -i], [-i, 1]]
            // SqrtX† has the same structure but with conjugated (1-i)/2 factor
            // Since SqrtX^2 = X and X is Hermitian, SqrtX† = SqrtX^(-1)
            // We represent this as a custom gate
            Gate::SqrtX => {
                let m = self.qubit_matrix();
                let dagger_matrix = conjugate_transpose(&m);
                Gate::Custom {
                    matrix: dagger_matrix,
                    is_diagonal: false,
                    label: "SqrtX†".to_string(),
                }
            }

            // SqrtY† similarly
            Gate::SqrtY => {
                let m = self.qubit_matrix();
                let dagger_matrix = conjugate_transpose(&m);
                Gate::Custom {
                    matrix: dagger_matrix,
                    is_diagonal: false,
                    label: "SqrtY†".to_string(),
                }
            }

            // SqrtW†
            Gate::SqrtW => {
                let m = self.qubit_matrix();
                let dagger_matrix = conjugate_transpose(&m);
                Gate::Custom {
                    matrix: dagger_matrix,
                    is_diagonal: false,
                    label: "SqrtW†".to_string(),
                }
            }

            // iSWAP† - conjugate transpose of iSWAP
            // iSWAP has i on off-diagonal, so iSWAP† has -i
            Gate::ISWAP => {
                let m = self.qubit_matrix();
                let dagger_matrix = conjugate_transpose(&m);
                Gate::Custom {
                    matrix: dagger_matrix,
                    is_diagonal: false,
                    label: "iSWAP†".to_string(),
                }
            }

            // FSim(θ, φ)† = FSim(-θ, -φ)
            Gate::FSim(theta, phi) => Gate::FSim(-theta, -phi),

            // Custom gate: conjugate transpose the matrix
            Gate::Custom {
                matrix,
                is_diagonal,
                label,
            } => {
                let dagger_matrix = conjugate_transpose(matrix);
                Gate::Custom {
                    matrix: dagger_matrix,
                    is_diagonal: *is_diagonal, // diagonal property preserved under transpose
                    label: format!("{}†", label),
                }
            }
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
            Gate::X => Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap(),
            Gate::Y => Array2::from_shape_vec((2, 2), vec![zero, neg_i, i, zero]).unwrap(),
            Gate::Z => Array2::from_shape_vec((2, 2), vec![one, zero, zero, neg_one]).unwrap(),
            Gate::H => {
                let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
                let neg_s = Complex64::new(-FRAC_1_SQRT_2, 0.0);
                Array2::from_shape_vec((2, 2), vec![s, s, s, neg_s]).unwrap()
            }
            Gate::S => Array2::from_shape_vec((2, 2), vec![one, zero, zero, i]).unwrap(),
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
                Array2::from_shape_vec((2, 2), vec![f * one, f * neg_i, f * neg_i, f * one])
                    .unwrap()
            }
            Gate::SqrtY => {
                // (1+i)/2 * [[1, -1], [1, 1]]
                let f = Complex64::new(0.5, 0.5); // (1+i)/2
                Array2::from_shape_vec((2, 2), vec![f * one, f * neg_one, f * one, f * one])
                    .unwrap()
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
                Array2::from_shape_vec(
                    (2, 2),
                    vec![cos_val, neg_i_sin * g01, neg_i_sin * g10, cos_val],
                )
                .unwrap()
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

/// Compute the conjugate transpose (adjoint) of a matrix.
fn conjugate_transpose(m: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = m.dim();
    let mut result = Array2::zeros((cols, rows));
    for i in 0..rows {
        for j in 0..cols {
            result[[j, i]] = m[[i, j]].conj();
        }
    }
    result
}
