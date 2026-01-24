# yao-rs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a quantum circuit description library in Rust that exports circuits as tensor networks (EinCode + tensors) via omeco.

**Architecture:** Circuits are flat lists of `PositionedGate`s (gate + target sites + optional qubit controls). Each gate is an enum with qubit-only named variants and a `Custom` fallback for arbitrary qudits. The primary output is a `TensorNetwork` (EinCode from omeco + gate tensors). A generic `apply` function provides ground truth for testing.

**Tech Stack:** Rust, omeco (crates.io), num-complex, ndarray

---

### Task 1: Project Setup

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`

**Step 1: Initialize Cargo project**

Run: `cargo init --lib /home/leo/rcode/yao-rs`

**Step 2: Set up Cargo.toml**

```toml
[package]
name = "yao-rs"
version = "0.1.0"
edition = "2021"

[dependencies]
num-complex = "0.4"
ndarray = "0.16"
omeco = "0.1"

[dev-dependencies]
approx = "0.5"
```

**Step 3: Set up lib.rs with module declarations**

```rust
pub mod gate;
pub mod circuit;
pub mod tensors;
pub mod einsum;
pub mod state;
pub mod apply;
```

**Step 4: Create empty module files**

Create `src/gate.rs`, `src/circuit.rs`, `src/tensors.rs`, `src/einsum.rs`, `src/state.rs`, `src/apply.rs` — each with just `// TODO`.

**Step 5: Verify it compiles**

Run: `cargo build`
Expected: Success (no errors)

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: initialize yao-rs project structure"
```

---

### Task 2: Gate Enum

**Files:**
- Modify: `src/gate.rs`
- Create: `tests/test_gates.rs`

**Step 1: Write failing tests for gate matrix generation**

File: `tests/test_gates.rs`
```rust
use approx::assert_abs_diff_eq;
use ndarray::Array2;
use num_complex::Complex64;
use yao_rs::gate::Gate;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_x_matrix() {
    let m = Gate::X.matrix(2);
    assert_eq!(m.shape(), &[2, 2]);
    assert_abs_diff_eq!(m[[0, 1]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 0]].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_y_matrix() {
    let m = Gate::Y.matrix(2);
    assert_abs_diff_eq!(m[[0, 1]].im, -1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 0]].im, 1.0, epsilon = 1e-10);
}

#[test]
fn test_z_matrix() {
    let m = Gate::Z.matrix(2);
    assert_abs_diff_eq!(m[[0, 0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, -1.0, epsilon = 1e-10);
}

#[test]
fn test_h_matrix() {
    let m = Gate::H.matrix(2);
    let s = 1.0 / 2.0_f64.sqrt();
    assert_abs_diff_eq!(m[[0, 0]].re, s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 1]].re, s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 0]].re, s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, -s, epsilon = 1e-10);
}

#[test]
fn test_s_matrix() {
    let m = Gate::S.matrix(2);
    assert_abs_diff_eq!(m[[0, 0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].im, 1.0, epsilon = 1e-10);
}

#[test]
fn test_t_matrix() {
    let m = Gate::T.matrix(2);
    assert_abs_diff_eq!(m[[0, 0]].re, 1.0, epsilon = 1e-10);
    let expected = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    assert_abs_diff_eq!(m[[1, 1]].re, expected.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].im, expected.im, epsilon = 1e-10);
}

#[test]
fn test_swap_matrix() {
    let m = Gate::SWAP.matrix(2);
    assert_eq!(m.shape(), &[4, 4]);
    // SWAP |01⟩ = |10⟩: row 2 (|10⟩), col 1 (|01⟩) = 1
    assert_abs_diff_eq!(m[[0, 0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 2]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[2, 1]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[3, 3]].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_rx_matrix() {
    let theta = std::f64::consts::PI / 3.0;
    let m = Gate::Rx(theta).matrix(2);
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    assert_abs_diff_eq!(m[[0, 0]].re, c, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 1]].im, -s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 0]].im, -s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, c, epsilon = 1e-10);
}

#[test]
fn test_ry_matrix() {
    let theta = std::f64::consts::PI / 3.0;
    let m = Gate::Ry(theta).matrix(2);
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    assert_abs_diff_eq!(m[[0, 0]].re, c, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 1]].re, -s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 0]].re, s, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, c, epsilon = 1e-10);
}

#[test]
fn test_rz_matrix() {
    let theta = std::f64::consts::PI / 3.0;
    let m = Gate::Rz(theta).matrix(2);
    let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
    assert_abs_diff_eq!(m[[0, 0]].re, phase_neg.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 0]].im, phase_neg.im, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, phase_pos.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].im, phase_pos.im, epsilon = 1e-10);
}

#[test]
fn test_custom_matrix() {
    let mat = Array2::from_shape_vec(
        (3, 3),
        (0..9).map(|i| c(i as f64, 0.0)).collect(),
    ).unwrap();
    let gate = Gate::Custom { matrix: mat.clone(), is_diagonal: false };
    assert_eq!(gate.matrix(3), mat);
}

#[test]
#[should_panic]
fn test_x_gate_wrong_dimension() {
    Gate::X.matrix(3); // X is qubit-only, d=3 should panic
}

#[test]
fn test_gate_num_sites() {
    assert_eq!(Gate::X.num_sites(2), 1);
    assert_eq!(Gate::SWAP.num_sites(2), 2);
    let mat = Array2::eye(9).mapv(|x| c(x, 0.0));
    let gate = Gate::Custom { matrix: mat, is_diagonal: false };
    assert_eq!(gate.num_sites(3), 2); // 3^2 = 9
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_gates`
Expected: Compilation error (Gate not defined)

**Step 3: Implement Gate enum**

File: `src/gate.rs`
```rust
use ndarray::Array2;
use num_complex::Complex64;

/// Quantum gate representation.
///
/// Named variants (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) are qubit-only (d=2).
/// Use `Custom` for arbitrary qudit gates.
#[derive(Debug, Clone)]
pub enum Gate {
    X,
    Y,
    Z,
    H,
    S,
    T,
    SWAP,
    Rx(f64),
    Ry(f64),
    Rz(f64),
    Custom {
        matrix: Array2<Complex64>,
        is_diagonal: bool,
    },
}

impl Gate {
    /// Returns the matrix representation of this gate.
    ///
    /// # Arguments
    /// * `d` - dimension per site (must be 2 for named variants)
    ///
    /// # Panics
    /// Panics if `d != 2` for named gate variants.
    pub fn matrix(&self, d: usize) -> Array2<Complex64> {
        match self {
            Gate::Custom { matrix, .. } => matrix.clone(),
            _ => {
                assert!(d == 2, "Named gate variants require d=2, got d={}", d);
                self.qubit_matrix()
            }
        }
    }

    /// Number of sites this gate acts on.
    ///
    /// # Arguments
    /// * `d` - dimension per site
    pub fn num_sites(&self, d: usize) -> usize {
        match self {
            Gate::SWAP => 2,
            Gate::Custom { matrix, .. } => {
                let n = matrix.nrows();
                let mut sites = 0;
                let mut pow = 1;
                while pow < n {
                    pow *= d;
                    sites += 1;
                }
                assert_eq!(pow, n, "Matrix size {} is not a power of d={}", n, d);
                sites
            }
            _ => 1, // X, Y, Z, H, S, T, Rx, Ry, Rz
        }
    }

    /// Whether this gate is diagonal.
    pub fn is_diagonal(&self) -> bool {
        match self {
            Gate::Z | Gate::S | Gate::T | Gate::Rz(_) => true,
            Gate::Custom { is_diagonal, .. } => *is_diagonal,
            _ => false,
        }
    }

    fn qubit_matrix(&self) -> Array2<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);

        match self {
            Gate::X => Array2::from_shape_vec((2, 2), vec![zero, one, one, zero]).unwrap(),
            Gate::Y => Array2::from_shape_vec((2, 2), vec![zero, -i, i, zero]).unwrap(),
            Gate::Z => Array2::from_shape_vec((2, 2), vec![one, zero, zero, -one]).unwrap(),
            Gate::H => {
                let s = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                Array2::from_shape_vec((2, 2), vec![s, s, s, -s]).unwrap()
            }
            Gate::S => Array2::from_shape_vec((2, 2), vec![one, zero, zero, i]).unwrap(),
            Gate::T => {
                let t = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
                Array2::from_shape_vec((2, 2), vec![one, zero, zero, t]).unwrap()
            }
            Gate::SWAP => {
                let mut m = Array2::zeros((4, 4));
                m[[0, 0]] = one;
                m[[1, 2]] = one;
                m[[2, 1]] = one;
                m[[3, 3]] = one;
                m
            }
            Gate::Rx(theta) => {
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = Complex64::new(0.0, -(theta / 2.0).sin());
                Array2::from_shape_vec((2, 2), vec![c, s, s, c]).unwrap()
            }
            Gate::Ry(theta) => {
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s_pos = Complex64::new((theta / 2.0).sin(), 0.0);
                let s_neg = Complex64::new(-(theta / 2.0).sin(), 0.0);
                Array2::from_shape_vec((2, 2), vec![c, s_neg, s_pos, c]).unwrap()
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
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_gates`
Expected: All pass

**Step 5: Commit**

```bash
git add src/gate.rs tests/test_gates.rs && git commit -m "feat: implement Gate enum with matrix generation"
```

---

### Task 3: Circuit Structure + Validation

**Files:**
- Modify: `src/circuit.rs`
- Create: `tests/test_circuit.rs`

**Step 1: Write failing tests for Circuit validation**

File: `tests/test_circuit.rs`
```rust
use ndarray::Array2;
use num_complex::Complex64;
use yao_rs::circuit::{Circuit, PositionedGate};
use yao_rs::gate::Gate;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_valid_single_qubit_gate() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![0], vec![], vec![])],
    );
    assert!(circuit.is_ok());
}

#[test]
fn test_valid_cnot() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])],
    );
    assert!(circuit.is_ok());
}

#[test]
fn test_valid_toffoli() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![PositionedGate::new(Gate::X, vec![2], vec![0, 1], vec![true, true])],
    );
    assert!(circuit.is_ok());
}

#[test]
fn test_invalid_qubit_gate_on_qutrit() {
    let result = Circuit::new(
        vec![3, 2],
        vec![PositionedGate::new(Gate::X, vec![0], vec![], vec![])],
    );
    assert!(result.is_err());
}

#[test]
fn test_invalid_control_on_qutrit() {
    let mat = Array2::eye(3).mapv(|x| c(x, 0.0));
    let result = Circuit::new(
        vec![3, 3],
        vec![PositionedGate::new(
            Gate::Custom { matrix: mat, is_diagonal: false },
            vec![1],
            vec![0], // control on d=3 site, forbidden
            vec![true],
        )],
    );
    assert!(result.is_err());
}

#[test]
fn test_invalid_overlapping_locs() {
    let result = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![0], vec![0], vec![true])],
    );
    assert!(result.is_err());
}

#[test]
fn test_invalid_loc_out_of_range() {
    let result = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![5], vec![], vec![])],
    );
    assert!(result.is_err());
}

#[test]
fn test_invalid_matrix_size_mismatch() {
    let mat = Array2::eye(2).mapv(|x| c(x, 0.0)); // 2x2 on a 3-dim site
    let result = Circuit::new(
        vec![3],
        vec![PositionedGate::new(
            Gate::Custom { matrix: mat, is_diagonal: false },
            vec![0],
            vec![],
            vec![],
        )],
    );
    assert!(result.is_err());
}

#[test]
fn test_invalid_control_config_length() {
    let result = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![])], // missing config
    );
    assert!(result.is_err());
}

#[test]
fn test_valid_custom_qudit_gate() {
    let mat = Array2::eye(9).mapv(|x| c(x, 0.0)); // 9x9 = 3^2, two qutrits
    let circuit = Circuit::new(
        vec![3, 3],
        vec![PositionedGate::new(
            Gate::Custom { matrix: mat, is_diagonal: false },
            vec![0, 1],
            vec![],
            vec![],
        )],
    );
    assert!(circuit.is_ok());
}

#[test]
fn test_multi_gate_circuit() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
            PositionedGate::new(Gate::X, vec![2], vec![1], vec![true]),
        ],
    );
    assert!(circuit.is_ok());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_circuit`
Expected: Compilation error

**Step 3: Implement Circuit and PositionedGate**

File: `src/circuit.rs`
```rust
use crate::gate::Gate;
use std::collections::HashSet;

/// A gate placed at specific sites with optional controls.
#[derive(Debug, Clone)]
pub struct PositionedGate {
    pub gate: Gate,
    pub target_locs: Vec<usize>,
    pub control_locs: Vec<usize>,
    pub control_configs: Vec<bool>,
}

impl PositionedGate {
    pub fn new(
        gate: Gate,
        target_locs: Vec<usize>,
        control_locs: Vec<usize>,
        control_configs: Vec<bool>,
    ) -> Self {
        Self {
            gate,
            target_locs,
            control_locs,
            control_configs,
        }
    }

    /// All sites this gate touches (targets + controls).
    pub fn all_locs(&self) -> Vec<usize> {
        let mut locs = self.target_locs.clone();
        locs.extend_from_slice(&self.control_locs);
        locs
    }
}

/// A quantum circuit: sequence of positioned gates on a qudit register.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub dims: Vec<usize>,
    pub gates: Vec<PositionedGate>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitError {
    /// Named gate applied to non-qubit site.
    QubitGateOnNonQubit { gate: String, site: usize, dim: usize },
    /// Control on non-qubit site.
    ControlOnNonQubit { site: usize, dim: usize },
    /// Target and control locations overlap.
    OverlappingLocs { site: usize },
    /// Location out of range.
    LocOutOfRange { site: usize, num_sites: usize },
    /// Gate matrix size doesn't match target dimensions.
    MatrixSizeMismatch { expected: usize, got: usize },
    /// Control config length doesn't match control locs.
    ControlConfigMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for CircuitError {}

impl Circuit {
    /// Create a new circuit, validating all gates against the site dimensions.
    pub fn new(dims: Vec<usize>, gates: Vec<PositionedGate>) -> Result<Self, CircuitError> {
        let n = dims.len();
        for gate in &gates {
            // Check control config length matches control locs
            if gate.control_configs.len() != gate.control_locs.len() {
                return Err(CircuitError::ControlConfigMismatch {
                    expected: gate.control_locs.len(),
                    got: gate.control_configs.len(),
                });
            }

            // Check all locs are in range
            for &loc in gate.target_locs.iter().chain(gate.control_locs.iter()) {
                if loc >= n {
                    return Err(CircuitError::LocOutOfRange { site: loc, num_sites: n });
                }
            }

            // Check no overlap between target and control
            let target_set: HashSet<usize> = gate.target_locs.iter().copied().collect();
            for &loc in &gate.control_locs {
                if target_set.contains(&loc) {
                    return Err(CircuitError::OverlappingLocs { site: loc });
                }
            }

            // Check control sites are qubits (d=2)
            for &loc in &gate.control_locs {
                if dims[loc] != 2 {
                    return Err(CircuitError::ControlOnNonQubit {
                        site: loc,
                        dim: dims[loc],
                    });
                }
            }

            // Check named gates are on qubit sites
            let requires_qubit = !matches!(gate.gate, Gate::Custom { .. });
            if requires_qubit {
                for &loc in &gate.target_locs {
                    if dims[loc] != 2 {
                        return Err(CircuitError::QubitGateOnNonQubit {
                            gate: format!("{:?}", gate.gate),
                            site: loc,
                            dim: dims[loc],
                        });
                    }
                }
            }

            // Check matrix size matches target dimensions
            let target_dim: usize = gate.target_locs.iter().map(|&l| dims[l]).product();
            let gate_dim = gate.gate.matrix(
                if gate.target_locs.is_empty() { 2 } else { dims[gate.target_locs[0]] }
            ).nrows();
            if gate_dim != target_dim {
                return Err(CircuitError::MatrixSizeMismatch {
                    expected: target_dim,
                    got: gate_dim,
                });
            }
        }

        Ok(Self { dims, gates })
    }

    /// Number of sites in the circuit.
    pub fn num_sites(&self) -> usize {
        self.dims.len()
    }

    /// Total Hilbert space dimension.
    pub fn total_dim(&self) -> usize {
        self.dims.iter().product()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_circuit`
Expected: All pass

**Step 5: Commit**

```bash
git add src/circuit.rs tests/test_circuit.rs && git commit -m "feat: implement Circuit with validation"
```

---

### Task 4: State Struct

**Files:**
- Modify: `src/state.rs`
- Create: `tests/test_state.rs`

**Step 1: Write failing tests**

File: `tests/test_state.rs`
```rust
use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use yao_rs::state::State;

#[test]
fn test_zero_state_qubits() {
    let s = State::zero_state(&[2, 2]);
    assert_eq!(s.data.len(), 4);
    assert_abs_diff_eq!(s.data[0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(s.data[1].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_zero_state_qutrits() {
    let s = State::zero_state(&[3, 3]);
    assert_eq!(s.data.len(), 9);
    assert_abs_diff_eq!(s.data[0].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_product_state() {
    // |1,0⟩ on 2 qubits = index 1*2 + 0 = 2... wait
    // Ordering: |i,j⟩ -> index i*d_j + j (row-major)
    // |1,0⟩ -> index 1*2 + 0 = 2
    let s = State::product_state(&[2, 2], &[1, 0]);
    assert_abs_diff_eq!(s.data[2].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_product_state_qutrit() {
    // |2,1⟩ on 2 qutrits -> index 2*3 + 1 = 7
    let s = State::product_state(&[3, 3], &[2, 1]);
    assert_abs_diff_eq!(s.data[7].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_state_norm() {
    let s = State::zero_state(&[2, 2]);
    assert_abs_diff_eq!(s.norm(), 1.0, epsilon = 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_state`
Expected: Compilation error

**Step 3: Implement State**

File: `src/state.rs`
```rust
use ndarray::Array1;
use num_complex::Complex64;

/// Quantum state as a dense vector.
#[derive(Debug, Clone)]
pub struct State {
    pub dims: Vec<usize>,
    pub data: Array1<Complex64>,
}

impl State {
    /// Create the all-zeros computational basis state |0,0,...,0⟩.
    pub fn zero_state(dims: &[usize]) -> Self {
        let total: usize = dims.iter().product();
        let mut data = Array1::zeros(total);
        data[0] = Complex64::new(1.0, 0.0);
        Self {
            dims: dims.to_vec(),
            data,
        }
    }

    /// Create a product state |i_0, i_1, ..., i_{n-1}⟩.
    ///
    /// `levels[k]` specifies the level of site k (0-indexed, must be < dims[k]).
    pub fn product_state(dims: &[usize], levels: &[usize]) -> Self {
        assert_eq!(dims.len(), levels.len());
        let total: usize = dims.iter().product();
        let mut data = Array1::zeros(total);
        let mut index = 0;
        for (k, &level) in levels.iter().enumerate() {
            assert!(level < dims[k], "level {} >= dim {} at site {}", level, dims[k], k);
            index = index * dims[k] + level;
        }
        data[index] = Complex64::new(1.0, 0.0);
        Self {
            dims: dims.to_vec(),
            data,
        }
    }

    /// Compute the L2 norm of the state.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Total Hilbert space dimension.
    pub fn total_dim(&self) -> usize {
        self.data.len()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_state`
Expected: All pass

**Step 5: Commit**

```bash
git add src/state.rs tests/test_state.rs && git commit -m "feat: implement State struct"
```

---

### Task 5: Gate-to-Tensor Conversion

**Files:**
- Modify: `src/tensors.rs`
- Create: `tests/test_tensors.rs`

This module converts a `PositionedGate` into a tensor (multi-dimensional array) for the tensor network. The key insight: a controlled gate becomes a single tensor with legs for all involved sites (controls + targets), with input and output legs for non-diagonal sites.

**Step 1: Write failing tests**

File: `tests/test_tensors.rs`
```rust
use approx::assert_abs_diff_eq;
use ndarray::{Array2, ArrayD};
use num_complex::Complex64;
use yao_rs::circuit::PositionedGate;
use yao_rs::gate::Gate;
use yao_rs::tensors::gate_to_tensor;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_single_qubit_gate_tensor() {
    // X gate on site 0, no controls
    let pg = PositionedGate::new(Gate::X, vec![0], vec![], vec![]);
    let (tensor, legs) = gate_to_tensor(&pg, &[2]);
    // Non-diagonal: tensor has shape (2, 2) = (out, in)
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(legs.len(), 2); // [out_label, in_label]
}

#[test]
fn test_diagonal_gate_tensor() {
    // Z gate is diagonal, single leg per site
    let pg = PositionedGate::new(Gate::Z, vec![0], vec![], vec![]);
    let (tensor, legs) = gate_to_tensor(&pg, &[2]);
    // Diagonal: tensor has shape (2,) = one leg
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(legs.len(), 1);
}

#[test]
fn test_controlled_gate_tensor() {
    // CNOT: control on site 0, X on site 1
    let pg = PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]);
    let (tensor, legs) = gate_to_tensor(&pg, &[2, 2]);
    // Control (non-diagonal) + target (non-diagonal) = 4 legs
    // Shape: (2, 2, 2, 2) = (ctrl_out, tgt_out, ctrl_in, tgt_in)
    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(legs.len(), 4);

    // Verify CNOT tensor values:
    // |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    // tensor[c_out, t_out, c_in, t_in]
    assert_abs_diff_eq!(tensor[[0, 0, 0, 0]].re, 1.0, epsilon = 1e-10); // |00⟩→|00⟩
    assert_abs_diff_eq!(tensor[[0, 1, 0, 1]].re, 1.0, epsilon = 1e-10); // |01⟩→|01⟩
    assert_abs_diff_eq!(tensor[[1, 1, 1, 0]].re, 1.0, epsilon = 1e-10); // |10⟩→|11⟩
    assert_abs_diff_eq!(tensor[[1, 0, 1, 1]].re, 1.0, epsilon = 1e-10); // |11⟩→|10⟩
}

#[test]
fn test_controlled_diagonal_gate_tensor() {
    // CZ: control on site 0, Z on site 1 (Z is diagonal)
    // But the control itself is non-diagonal, so the whole tensor is non-diagonal
    let pg = PositionedGate::new(Gate::Z, vec![1], vec![0], vec![true]);
    let (tensor, legs) = gate_to_tensor(&pg, &[2, 2]);
    // Control is non-diagonal: (ctrl_out, tgt_out, ctrl_in, tgt_in) = (2,2,2,2)
    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
}

#[test]
fn test_swap_gate_tensor() {
    let pg = PositionedGate::new(Gate::SWAP, vec![0, 1], vec![], vec![]);
    let (tensor, legs) = gate_to_tensor(&pg, &[2, 2]);
    // Non-diagonal 2-site gate: shape (2, 2, 2, 2)
    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(legs.len(), 4);
}

#[test]
fn test_custom_diagonal_qudit_tensor() {
    // Diagonal 3x3 gate on a qutrit
    let diag = vec![c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0)];
    let mut mat = Array2::zeros((3, 3));
    for i in 0..3 {
        mat[[i, i]] = diag[i];
    }
    let pg = PositionedGate::new(
        Gate::Custom { matrix: mat, is_diagonal: true },
        vec![0],
        vec![],
        vec![],
    );
    let (tensor, legs) = gate_to_tensor(&pg, &[3]);
    assert_eq!(tensor.shape(), &[3]); // diagonal: one leg
    assert_eq!(legs.len(), 1);
    // Values are the diagonal
    assert_abs_diff_eq!(tensor[[0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(tensor[[1]].im, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(tensor[[2]].re, -1.0, epsilon = 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_tensors`
Expected: Compilation error

**Step 3: Implement gate_to_tensor**

File: `src/tensors.rs`
```rust
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;

use crate::circuit::PositionedGate;
use crate::gate::Gate;

/// Leg descriptor for a tensor in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    /// Output leg for site at given index in all_locs.
    Out(usize),
    /// Input leg for site at given index in all_locs.
    In(usize),
    /// Shared (diagonal) leg for site at given index in all_locs.
    Diag(usize),
}

/// Convert a PositionedGate to a tensor and its leg descriptors.
///
/// Returns (tensor, legs) where legs describe the meaning of each axis.
///
/// For non-diagonal gates: tensor has shape (d_0_out, d_1_out, ..., d_0_in, d_1_in, ...)
/// For diagonal gates with no controls: tensor has shape (d_0, d_1, ...)
/// For controlled gates: always non-diagonal (controls are never diagonal).
///
/// # Arguments
/// * `pg` - The positioned gate
/// * `dims` - Dimensions of all sites in the circuit
pub fn gate_to_tensor(pg: &PositionedGate, dims: &[usize]) -> (ArrayD<Complex64>, Vec<Leg>) {
    let has_controls = !pg.control_locs.is_empty();
    let is_diag = pg.gate.is_diagonal() && !has_controls;

    // Collect all involved sites: controls first, then targets
    let ctrl_dims: Vec<usize> = pg.control_locs.iter().map(|&l| dims[l]).collect();
    let tgt_dims: Vec<usize> = pg.target_locs.iter().map(|&l| dims[l]).collect();

    if is_diag {
        // Diagonal gate, no controls: extract diagonal as 1D/nD tensor
        let d = dims[pg.target_locs[0]];
        let mat = pg.gate.matrix(d);
        let total: usize = tgt_dims.iter().product();
        let mut data = vec![Complex64::new(0.0, 0.0); total];
        for i in 0..total {
            data[i] = mat[[i, i]];
        }
        let shape: Vec<usize> = tgt_dims.clone();
        let tensor = ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap();
        let legs: Vec<Leg> = (0..tgt_dims.len()).map(|i| Leg::Diag(i)).collect();
        (tensor, legs)
    } else {
        // Non-diagonal gate (or has controls): build full tensor
        // Shape: (ctrl_out_dims..., tgt_out_dims..., ctrl_in_dims..., tgt_in_dims...)
        let all_dims: Vec<usize> = ctrl_dims.iter().chain(tgt_dims.iter()).copied().collect();
        let n_sites = all_dims.len();
        let total_dim: usize = all_dims.iter().product();

        // Build the full unitary matrix including controls
        let full_matrix = build_controlled_matrix(pg, dims);

        // Reshape matrix (total_dim x total_dim) into tensor (out_dims..., in_dims...)
        let mut shape: Vec<usize> = all_dims.clone(); // out dims
        shape.extend_from_slice(&all_dims);           // in dims
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&shape),
            full_matrix.into_raw_vec(),
        ).unwrap();

        // Build leg descriptors
        let mut legs = Vec::new();
        for i in 0..n_sites {
            legs.push(Leg::Out(i));
        }
        for i in 0..n_sites {
            legs.push(Leg::In(i));
        }
        (tensor, legs)
    }
}

/// Build the full controlled unitary matrix.
///
/// For a gate U with controls, the matrix is:
/// - Identity on all states where controls don't match config
/// - U on target when controls match config
fn build_controlled_matrix(pg: &PositionedGate, dims: &[usize]) -> ndarray::Array2<Complex64> {
    let ctrl_dims: Vec<usize> = pg.control_locs.iter().map(|&l| dims[l]).collect();
    let tgt_dims: Vec<usize> = pg.target_locs.iter().map(|&l| dims[l]).collect();
    let all_dims: Vec<usize> = ctrl_dims.iter().chain(tgt_dims.iter()).copied().collect();
    let total_dim: usize = all_dims.iter().product();
    let ctrl_total: usize = ctrl_dims.iter().product();
    let tgt_total: usize = tgt_dims.iter().product();

    let tgt_d = if pg.target_locs.is_empty() { 2 } else { dims[pg.target_locs[0]] };
    let gate_mat = pg.gate.matrix(tgt_d);

    let mut full = ndarray::Array2::zeros((total_dim, total_dim));
    let one = Complex64::new(1.0, 0.0);

    if pg.control_locs.is_empty() {
        // No controls: just the gate matrix
        return gate_mat;
    }

    // Compute the control index that triggers the gate
    let trigger_ctrl_idx = control_trigger_index(&pg.control_configs, &ctrl_dims);

    for ctrl_in in 0..ctrl_total {
        for tgt_in in 0..tgt_total {
            let row_base = ctrl_in * tgt_total;
            if ctrl_in == trigger_ctrl_idx {
                // Apply gate
                for tgt_out in 0..tgt_total {
                    let row = ctrl_in * tgt_total + tgt_out;
                    let col = ctrl_in * tgt_total + tgt_in;
                    full[[row, col]] = gate_mat[[tgt_out, tgt_in]];
                }
            } else {
                // Identity
                let idx = ctrl_in * tgt_total + tgt_in;
                full[[idx, idx]] = one;
            }
        }
    }

    full
}

/// Compute the linear index of the control configuration that triggers the gate.
fn control_trigger_index(configs: &[bool], ctrl_dims: &[usize]) -> usize {
    let mut idx = 0;
    for (i, &config) in configs.iter().enumerate() {
        idx = idx * ctrl_dims[i] + if config { 1 } else { 0 };
    }
    idx
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_tensors`
Expected: All pass

**Step 5: Commit**

```bash
git add src/tensors.rs tests/test_tensors.rs && git commit -m "feat: implement gate-to-tensor conversion"
```

---

### Task 6: Einsum Export

**Files:**
- Modify: `src/einsum.rs`
- Create: `tests/test_einsum.rs`

**Step 1: Write failing tests**

File: `tests/test_einsum.rs`
```rust
use yao_rs::circuit::{Circuit, PositionedGate};
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;

#[test]
fn test_single_gate_einsum() {
    // H on qubit 0 in a 2-qubit circuit
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::H, vec![0], vec![], vec![])],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    // Input labels: [0, 1]
    // H on site 0 (non-diagonal): new label 2
    // Tensor legs: [2, 0] (out, in)
    // Output labels: [2, 1]
    assert_eq!(tn.code.ixs.len(), 1); // one tensor
    assert_eq!(tn.code.ixs[0], vec![2, 0]); // out=2, in=0
    assert_eq!(tn.code.iy, vec![2, 1]); // output state
    assert_eq!(tn.tensors.len(), 1);
    assert_eq!(tn.tensors[0].shape(), &[2, 2]);
}

#[test]
fn test_diagonal_gate_einsum() {
    // Z on qubit 0 (diagonal)
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::Z, vec![0], vec![], vec![])],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    // Diagonal: no new label, tensor leg is just [0]
    assert_eq!(tn.code.ixs[0], vec![0]);
    assert_eq!(tn.code.iy, vec![0]); // same label in output
    assert_eq!(tn.tensors[0].shape(), &[2]);
}

#[test]
fn test_cnot_einsum() {
    // CNOT: control=0, target=1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    // Input labels: [0, 1]
    // CNOT is non-diagonal on both sites: new labels 2 (ctrl out), 3 (tgt out)
    // Tensor legs: [2, 3, 0, 1] (ctrl_out, tgt_out, ctrl_in, tgt_in)
    assert_eq!(tn.code.ixs[0], vec![2, 3, 0, 1]);
    assert_eq!(tn.code.iy, vec![2, 3]);
}

#[test]
fn test_two_gate_circuit_einsum() {
    // H on qubit 0, then CNOT(0→1)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
        ],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    // Gate 1 (H on site 0): input label 0, output label 2. Legs: [2, 0]
    // Gate 2 (CNOT): ctrl input is now 2, tgt input is 1
    //   ctrl out: 3, tgt out: 4. Legs: [3, 4, 2, 1]
    // Output: [3, 4]
    assert_eq!(tn.code.ixs.len(), 2);
    assert_eq!(tn.code.ixs[0], vec![2, 0]);
    assert_eq!(tn.code.ixs[1], vec![3, 4, 2, 1]);
    assert_eq!(tn.code.iy, vec![3, 4]);
}

#[test]
fn test_size_dict() {
    let circuit = Circuit::new(
        vec![2, 3],
        vec![],
    ).unwrap();
    let tn = circuit_to_einsum(&circuit);
    // Empty circuit: no tensors, output = input labels
    assert_eq!(tn.code.ixs.len(), 0);
    assert_eq!(tn.code.iy, vec![0, 1]);
    assert_eq!(*tn.size_dict.get(&0).unwrap(), 2);
    assert_eq!(*tn.size_dict.get(&1).unwrap(), 3);
}

#[test]
fn test_mixed_diagonal_nondiagonal() {
    // Rz (diagonal) then H (non-diagonal) on same qubit
    let circuit = Circuit::new(
        vec![2],
        vec![
            PositionedGate::new(Gate::Rz(1.0), vec![0], vec![], vec![]),
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
        ],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    // Rz is diagonal: no new label, legs [0]
    // H is non-diagonal: new label 1, legs [1, 0]
    // Output: [1]
    assert_eq!(tn.code.ixs[0], vec![0]);   // Rz diagonal
    assert_eq!(tn.code.ixs[1], vec![1, 0]); // H non-diagonal
    assert_eq!(tn.code.iy, vec![1]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_einsum`
Expected: Compilation error

**Step 3: Implement circuit_to_einsum**

File: `src/einsum.rs`
```rust
use std::collections::HashMap;

use ndarray::ArrayD;
use num_complex::Complex64;
use omeco::EinCode;

use crate::circuit::Circuit;
use crate::tensors::{gate_to_tensor, Leg};

/// Result of converting a circuit to a tensor network.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    /// The einsum contraction pattern.
    pub code: EinCode<usize>,
    /// Gate tensors in matching order.
    pub tensors: Vec<ArrayD<Complex64>>,
    /// Size dictionary mapping labels to dimensions.
    pub size_dict: HashMap<usize, usize>,
}

/// Convert a circuit to a tensor network (EinCode + tensors).
///
/// Each gate becomes a tensor. Wire labels are assigned sequentially:
/// - Labels 0..n-1 are the initial state (input) of each site
/// - Each non-diagonal interaction on a site allocates a new output label
/// - Diagonal gates reuse the current label (no new index)
pub fn circuit_to_einsum(circuit: &Circuit) -> TensorNetwork {
    let n = circuit.num_sites();
    let mut next_label: usize = n;

    // Current label for each site (starts as 0..n-1)
    let mut current_labels: Vec<usize> = (0..n).collect();

    // Size dictionary: label -> dimension
    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for (i, &d) in circuit.dims.iter().enumerate() {
        size_dict.insert(i, d);
    }

    let mut all_ixs: Vec<Vec<usize>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    for pg in &circuit.gates {
        let (tensor, legs) = gate_to_tensor(pg, &circuit.dims);

        // Determine which sites are involved: controls first, then targets
        let all_locs: Vec<usize> = pg.control_locs.iter()
            .chain(pg.target_locs.iter())
            .copied()
            .collect();

        let has_controls = !pg.control_locs.is_empty();
        let is_diag = pg.gate.is_diagonal() && !has_controls;

        // Build the index list for this tensor
        let mut tensor_ixs: Vec<usize> = Vec::new();

        if is_diag {
            // Diagonal: each leg is the current label (shared in/out)
            for &loc in &pg.target_locs {
                tensor_ixs.push(current_labels[loc]);
            }
            // Labels don't change
        } else {
            // Non-diagonal: allocate new output labels, then input labels
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }

            // Tensor legs are: [out_0, out_1, ..., in_0, in_1, ...]
            // Out labels (new)
            for &nl in &new_labels {
                tensor_ixs.push(nl);
            }
            // In labels (current)
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }

            // Update current labels
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }
        }

        all_ixs.push(tensor_ixs);
        all_tensors.push(tensor);
    }

    // Output labels are the final current labels
    let iy: Vec<usize> = current_labels;

    TensorNetwork {
        code: EinCode::new(all_ixs, iy),
        tensors: all_tensors,
        size_dict,
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_einsum`
Expected: All pass

**Step 5: Commit**

```bash
git add src/einsum.rs tests/test_einsum.rs && git commit -m "feat: implement circuit-to-einsum conversion"
```

---

### Task 7: Generic Apply

**Files:**
- Modify: `src/apply.rs`
- Create: `tests/test_apply.rs`

**Step 1: Write failing tests**

File: `tests/test_apply.rs`
```rust
use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, PositionedGate};
use yao_rs::gate::Gate;
use yao_rs::state::State;

#[test]
fn test_apply_x_gate() {
    // X|0⟩ = |1⟩
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::X, vec![0], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.data[0].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.data[1].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_h_gate() {
    // H|0⟩ = (|0⟩+|1⟩)/√2
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::H, vec![0], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2]);
    let result = apply(&circuit, &state);
    let s = 1.0 / 2.0_f64.sqrt();
    assert_abs_diff_eq!(result.data[0].re, s, epsilon = 1e-10);
    assert_abs_diff_eq!(result.data[1].re, s, epsilon = 1e-10);
}

#[test]
fn test_apply_cnot() {
    // CNOT|10⟩ = |11⟩
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])],
    ).unwrap();
    let state = State::product_state(&[2, 2], &[1, 0]);
    let result = apply(&circuit, &state);
    // |11⟩ = index 3
    assert_abs_diff_eq!(result.data[3].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_cnot_no_trigger() {
    // CNOT|00⟩ = |00⟩ (control not set)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])],
    ).unwrap();
    let state = State::product_state(&[2, 2], &[0, 0]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.data[0].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_bell_state() {
    // H on qubit 0, then CNOT(0→1): creates Bell state (|00⟩+|11⟩)/√2
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
        ],
    ).unwrap();
    let state = State::zero_state(&[2, 2]);
    let result = apply(&circuit, &state);
    let s = 1.0 / 2.0_f64.sqrt();
    assert_abs_diff_eq!(result.data[0].re, s, epsilon = 1e-10); // |00⟩
    assert_abs_diff_eq!(result.data[3].re, s, epsilon = 1e-10); // |11⟩
    assert_abs_diff_eq!(result.data[1].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.data[2].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_apply_preserves_norm() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
            PositionedGate::new(Gate::Ry(1.2), vec![2], vec![], vec![]),
        ],
    ).unwrap();
    let state = State::zero_state(&[2, 2, 2]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_apply_qutrit() {
    // Custom gate on a qutrit: cyclic permutation |0⟩→|1⟩→|2⟩→|0⟩
    use ndarray::Array2;
    let mut mat = Array2::zeros((3, 3));
    mat[[1, 0]] = Complex64::new(1.0, 0.0); // |0⟩→|1⟩
    mat[[2, 1]] = Complex64::new(1.0, 0.0); // |1⟩→|2⟩
    mat[[0, 2]] = Complex64::new(1.0, 0.0); // |2⟩→|0⟩

    let circuit = Circuit::new(
        vec![3],
        vec![PositionedGate::new(
            Gate::Custom { matrix: mat, is_diagonal: false },
            vec![0],
            vec![],
            vec![],
        )],
    ).unwrap();
    let state = State::zero_state(&[3]); // |0⟩
    let result = apply(&circuit, &state);
    // Should be |1⟩
    assert_abs_diff_eq!(result.data[0].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.data[1].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.data[2].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_apply_gate_on_second_qubit() {
    // X on qubit 1 of a 2-qubit system: |00⟩ → |01⟩
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2, 2]);
    let result = apply(&circuit, &state);
    // |01⟩ = index 1
    assert_abs_diff_eq!(result.data[1].re, 1.0, epsilon = 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_apply`
Expected: Compilation error

**Step 3: Implement apply**

File: `src/apply.rs`
```rust
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::circuit::Circuit;
use crate::state::State;

/// Apply a circuit to a quantum state.
///
/// This is a generic, unoptimized implementation for testing.
/// For each gate, it builds the full-space matrix and multiplies the state vector.
pub fn apply(circuit: &Circuit, state: &State) -> State {
    let mut data = state.data.clone();
    let total_dim = circuit.total_dim();

    for pg in &circuit.gates {
        // Build the full-space matrix for this gate
        let full_matrix = build_full_matrix(pg, &circuit.dims);
        // Apply: new_state = full_matrix * state
        let new_data = matrix_vector_mul(&full_matrix, &data);
        data = new_data;
    }

    State {
        dims: state.dims.clone(),
        data,
    }
}

/// Build the full Hilbert space matrix for a PositionedGate.
///
/// The gate acts on specific sites; we tensor-product with identity on others.
fn build_full_matrix(
    pg: &crate::circuit::PositionedGate,
    dims: &[usize],
) -> Array2<Complex64> {
    let n = dims.len();
    let total_dim: usize = dims.iter().product();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    // Get the gate's local matrix (on target sites only)
    let tgt_d = if pg.target_locs.is_empty() { 2 } else { dims[pg.target_locs[0]] };
    let gate_mat = pg.gate.matrix(tgt_d);
    let tgt_dim: usize = pg.target_locs.iter().map(|&l| dims[l]).product();

    // All involved sites (controls + targets)
    let all_locs: Vec<usize> = pg.control_locs.iter()
        .chain(pg.target_locs.iter())
        .copied()
        .collect();
    let all_dims: Vec<usize> = all_locs.iter().map(|&l| dims[l]).collect();
    let involved_dim: usize = all_dims.iter().product();

    // Build the controlled matrix on involved sites
    let local_mat = build_local_controlled_matrix(&gate_mat, &pg.control_configs, &all_dims, &pg.control_locs, &pg.target_locs, dims);

    // Now embed local_mat into full Hilbert space
    let mut full = Array2::zeros((total_dim, total_dim));

    // For each pair of full-space indices (row, col), decompose into
    // (involved_sites_index, other_sites_index)
    for row in 0..total_dim {
        for col in 0..total_dim {
            let row_indices = linear_to_multi(row, dims);
            let col_indices = linear_to_multi(col, dims);

            // Check that non-involved sites match (identity)
            let mut other_match = true;
            for site in 0..n {
                if !all_locs.contains(&site) && row_indices[site] != col_indices[site] {
                    other_match = false;
                    break;
                }
            }

            if other_match {
                // Extract involved site indices
                let row_involved: Vec<usize> = all_locs.iter().map(|&l| row_indices[l]).collect();
                let col_involved: Vec<usize> = all_locs.iter().map(|&l| col_indices[l]).collect();
                let ri = multi_to_linear(&row_involved, &all_dims);
                let ci = multi_to_linear(&col_involved, &all_dims);
                full[[row, col]] = local_mat[[ri, ci]];
            }
        }
    }

    full
}

/// Build the local controlled matrix on the involved sites.
fn build_local_controlled_matrix(
    gate_mat: &Array2<Complex64>,
    control_configs: &[bool],
    all_dims: &[usize],
    control_locs: &[usize],
    _target_locs: &[usize],
    _dims: &[usize],
) -> Array2<Complex64> {
    let involved_dim: usize = all_dims.iter().product();
    let n_ctrls = control_locs.len();
    let ctrl_dims: Vec<usize> = all_dims[..n_ctrls].to_vec();
    let tgt_dims: Vec<usize> = all_dims[n_ctrls..].to_vec();
    let ctrl_dim: usize = ctrl_dims.iter().product::<usize>().max(1);
    let tgt_dim: usize = tgt_dims.iter().product::<usize>().max(1);

    let one = Complex64::new(1.0, 0.0);
    let mut mat = Array2::zeros((involved_dim, involved_dim));

    if n_ctrls == 0 {
        return gate_mat.clone();
    }

    // Compute trigger index
    let trigger = control_trigger_index(control_configs, &ctrl_dims);

    for c in 0..ctrl_dim {
        for t_row in 0..tgt_dim {
            for t_col in 0..tgt_dim {
                let row = c * tgt_dim + t_row;
                let col = c * tgt_dim + t_col;
                if c == trigger {
                    mat[[row, col]] = gate_mat[[t_row, t_col]];
                } else if t_row == t_col {
                    mat[[row, col]] = one;
                }
            }
        }
    }

    mat
}

fn control_trigger_index(configs: &[bool], ctrl_dims: &[usize]) -> usize {
    let mut idx = 0;
    for (i, &config) in configs.iter().enumerate() {
        idx = idx * ctrl_dims[i] + if config { 1 } else { 0 };
    }
    idx
}

fn linear_to_multi(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut result = vec![0; n];
    for i in (0..n).rev() {
        result[i] = index % dims[i];
        index /= dims[i];
    }
    result
}

fn multi_to_linear(indices: &[usize], dims: &[usize]) -> usize {
    let mut index = 0;
    for (i, &idx) in indices.iter().enumerate() {
        index = index * dims[i] + idx;
    }
    index
}

fn matrix_vector_mul(mat: &Array2<Complex64>, vec: &Array1<Complex64>) -> Array1<Complex64> {
    let n = vec.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n {
            result[i] += mat[[i, j]] * vec[j];
        }
    }
    result
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --test test_apply`
Expected: All pass

**Step 5: Commit**

```bash
git add src/apply.rs tests/test_apply.rs && git commit -m "feat: implement generic apply for testing"
```

---

### Task 8: Integration Tests (Apply vs Einsum)

**Files:**
- Create: `tests/test_integration.rs`

This is the critical test: verify that contracting the tensor network produces the same result as direct apply.

**Step 1: Write the integration test**

File: `tests/test_integration.rs`
```rust
use approx::assert_abs_diff_eq;
use ndarray::{Array1, ArrayD, IxDyn};
use num_complex::Complex64;
use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, PositionedGate};
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;
use yao_rs::state::State;

/// Naive tensor network contraction: contract tensors with input state
/// according to the einsum code.
///
/// This does a simple sequential pairwise contraction (no optimization).
fn naive_contract(
    tn: &yao_rs::einsum::TensorNetwork,
    input_state: &State,
) -> Array1<Complex64> {
    let n = input_state.dims.len();

    // Start with the state as a tensor with shape (d0, d1, ..., d_{n-1})
    // and indices [0, 1, ..., n-1]
    let state_shape: Vec<usize> = input_state.dims.clone();
    let state_tensor = ArrayD::from_shape_vec(
        IxDyn(&state_shape),
        input_state.data.to_vec(),
    ).unwrap();

    // Accumulate by contracting each gate tensor with the running state
    let mut current = state_tensor;
    let mut current_indices: Vec<usize> = (0..n).collect();

    for (i, gate_tensor) in tn.tensors.iter().enumerate() {
        let gate_indices = &tn.code.ixs[i];

        // Find shared indices between current and gate tensor
        let shared: Vec<usize> = current_indices.iter()
            .filter(|idx| gate_indices.contains(idx))
            .copied()
            .collect();

        // Contract: sum over shared indices
        current = contract_tensors(
            &current,
            &current_indices,
            gate_tensor,
            gate_indices,
            &shared,
        );

        // Update current indices: remove shared (contracted), add new from gate
        let mut new_indices: Vec<usize> = current_indices.iter()
            .filter(|idx| !shared.contains(idx))
            .copied()
            .collect();
        for idx in gate_indices {
            if !shared.contains(idx) && !new_indices.contains(idx) {
                new_indices.push(*idx);
            }
        }
        current_indices = new_indices;
    }

    // Permute to match output order
    let output_indices = &tn.code.iy;
    let perm: Vec<usize> = output_indices.iter()
        .map(|idx| current_indices.iter().position(|x| x == idx).unwrap())
        .collect();

    let permuted = current.permuted_axes(IxDyn(&perm));
    let output_shape: Vec<usize> = output_indices.iter()
        .map(|idx| *tn.size_dict.get(idx).unwrap())
        .collect();
    let total: usize = output_shape.iter().product();

    // Flatten to vector
    let flat: Vec<Complex64> = permuted.iter().copied().collect();
    Array1::from_vec(flat)
}

/// Contract two tensors over shared indices.
fn contract_tensors(
    a: &ArrayD<Complex64>,
    a_indices: &[usize],
    b: &ArrayD<Complex64>,
    b_indices: &[usize],
    shared: &[usize],
) -> ArrayD<Complex64> {
    // Determine output indices (free indices from both tensors)
    let a_free: Vec<(usize, usize)> = a_indices.iter().enumerate()
        .filter(|(_, idx)| !shared.contains(idx))
        .map(|(pos, idx)| (pos, *idx))
        .collect();
    let b_free: Vec<(usize, usize)> = b_indices.iter().enumerate()
        .filter(|(_, idx)| !shared.contains(idx))
        .map(|(pos, idx)| (pos, *idx))
        .collect();

    let a_shared_pos: Vec<usize> = shared.iter()
        .map(|s| a_indices.iter().position(|x| x == s).unwrap())
        .collect();
    let b_shared_pos: Vec<usize> = shared.iter()
        .map(|s| b_indices.iter().position(|x| x == s).unwrap())
        .collect();

    // Compute output shape
    let mut out_shape: Vec<usize> = Vec::new();
    for &(pos, _) in &a_free {
        out_shape.push(a.shape()[pos]);
    }
    for &(pos, _) in &b_free {
        out_shape.push(b.shape()[pos]);
    }

    if out_shape.is_empty() {
        // Scalar result
        out_shape.push(1);
    }

    let mut result = ArrayD::zeros(IxDyn(&out_shape));

    // Brute force contraction
    let a_shape: Vec<usize> = a.shape().to_vec();
    let b_shape: Vec<usize> = b.shape().to_vec();

    let a_total: usize = a_shape.iter().product();
    let b_total: usize = b_shape.iter().product();

    for a_flat in 0..a_total {
        let a_multi = flat_to_multi(a_flat, &a_shape);
        let a_val = a[IxDyn(&a_multi)];
        if a_val == Complex64::new(0.0, 0.0) {
            continue;
        }

        for b_flat in 0..b_total {
            let b_multi = flat_to_multi(b_flat, &b_shape);

            // Check shared indices match
            let mut matches = true;
            for k in 0..shared.len() {
                if a_multi[a_shared_pos[k]] != b_multi[b_shared_pos[k]] {
                    matches = false;
                    break;
                }
            }
            if !matches {
                continue;
            }

            let b_val = b[IxDyn(&b_multi)];

            // Compute output index
            let mut out_multi: Vec<usize> = Vec::new();
            for &(pos, _) in &a_free {
                out_multi.push(a_multi[pos]);
            }
            for &(pos, _) in &b_free {
                out_multi.push(b_multi[pos]);
            }
            if out_multi.is_empty() {
                out_multi.push(0);
            }

            result[IxDyn(&out_multi)] += a_val * b_val;
        }
    }

    result
}

fn flat_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut result = vec![0; n];
    for i in (0..n).rev() {
        result[i] = index % shape[i];
        index /= shape[i];
    }
    result
}

fn assert_states_close(a: &Array1<Complex64>, b: &Array1<Complex64>) {
    assert_eq!(a.len(), b.len(), "State vectors have different lengths");
    for i in 0..a.len() {
        assert_abs_diff_eq!(a[i].re, b[i].re, epsilon = 1e-9);
        assert_abs_diff_eq!(a[i].im, b[i].im, epsilon = 1e-9);
    }
}

#[test]
fn test_x_gate_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::X, vec![0], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_h_gate_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::H, vec![0], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_cnot_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])],
    ).unwrap();

    for levels in &[[0, 0], [0, 1], [1, 0], [1, 1]] {
        let state = State::product_state(&[2, 2], levels);
        let result_apply = apply(&circuit, &state);
        let tn = circuit_to_einsum(&circuit);
        let result_einsum = naive_contract(&tn, &state);
        assert_states_close(&result_apply.data, &result_einsum);
    }
}

#[test]
fn test_bell_state_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
        ],
    ).unwrap();
    let state = State::zero_state(&[2, 2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_diagonal_gate_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2],
        vec![PositionedGate::new(Gate::Rz(1.5), vec![0], vec![], vec![])],
    ).unwrap();
    let state = State::zero_state(&[2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_three_qubit_circuit_apply_vs_einsum() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
            PositionedGate::new(Gate::Ry(0.7), vec![2], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![2], vec![1], vec![true]),
        ],
    ).unwrap();
    let state = State::zero_state(&[2, 2, 2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_qutrit_apply_vs_einsum() {
    use ndarray::Array2;
    // Cyclic permutation on a qutrit
    let mut mat = Array2::zeros((3, 3));
    mat[[1, 0]] = Complex64::new(1.0, 0.0);
    mat[[2, 1]] = Complex64::new(1.0, 0.0);
    mat[[0, 2]] = Complex64::new(1.0, 0.0);

    let circuit = Circuit::new(
        vec![3],
        vec![PositionedGate::new(
            Gate::Custom { matrix: mat, is_diagonal: false },
            vec![0],
            vec![],
            vec![],
        )],
    ).unwrap();
    let state = State::zero_state(&[3]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}

#[test]
fn test_mixed_dims_apply_vs_einsum() {
    use ndarray::Array2;
    // 2-qubit + 1-qutrit system, custom gate on the qutrit
    let mut mat = Array2::zeros((3, 3));
    mat[[0, 0]] = Complex64::new(0.0, 1.0);
    mat[[1, 1]] = Complex64::new(1.0, 0.0);
    mat[[2, 2]] = Complex64::new(0.0, -1.0);

    let circuit = Circuit::new(
        vec![2, 3, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(
                Gate::Custom { matrix: mat, is_diagonal: true },
                vec![1],
                vec![],
                vec![],
            ),
            PositionedGate::new(Gate::X, vec![2], vec![0], vec![true]),
        ],
    ).unwrap();
    let state = State::zero_state(&[2, 3, 2]);

    let result_apply = apply(&circuit, &state);
    let tn = circuit_to_einsum(&circuit);
    let result_einsum = naive_contract(&tn, &state);

    assert_states_close(&result_apply.data, &result_einsum);
}
```

**Step 2: Run tests**

Run: `cargo test --test test_integration`
Expected: All pass (if previous tasks are correct)

**Step 3: Commit**

```bash
git add tests/test_integration.rs && git commit -m "test: add integration tests comparing apply vs einsum contraction"
```

---

### Task 9: Verify omeco Integration

**Files:**
- Create: `tests/test_omeco.rs`

Verify that the generated EinCode works with omeco's optimizer.

**Step 1: Write test**

File: `tests/test_omeco.rs`
```rust
use omeco::{optimize_code, GreedyMethod, contraction_complexity};
use yao_rs::circuit::{Circuit, PositionedGate};
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;

#[test]
fn test_optimize_bell_circuit() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
            PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
        ],
    ).unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    let complexity = contraction_complexity(&nested, &tn.size_dict, &tn.code.ixs);
    assert!(complexity.tc > 0.0);
}

#[test]
fn test_optimize_ghz_circuit() {
    // 5-qubit GHZ preparation: H on 0, then CNOT chain
    let mut gates = vec![PositionedGate::new(Gate::H, vec![0], vec![], vec![])];
    for i in 0..4 {
        gates.push(PositionedGate::new(Gate::X, vec![i + 1], vec![i], vec![true]));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    assert!(nested.is_binary());
    let complexity = contraction_complexity(&nested, &tn.size_dict, &tn.code.ixs);
    println!("GHZ-5 contraction: tc=2^{:.2}, sc=2^{:.2}", complexity.tc, complexity.sc);
}

#[test]
fn test_optimize_larger_circuit() {
    // 10-qubit random-ish circuit
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(PositionedGate::new(Gate::H, vec![i], vec![], vec![]));
    }
    for i in 0..9 {
        gates.push(PositionedGate::new(Gate::X, vec![i + 1], vec![i], vec![true]));
    }
    for i in 0..10 {
        gates.push(PositionedGate::new(Gate::Rz(0.5 * i as f64), vec![i], vec![], vec![]));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let complexity = contraction_complexity(
        &optimized.unwrap(),
        &tn.size_dict,
        &tn.code.ixs,
    );
    println!("10-qubit circuit: tc=2^{:.2}, sc=2^{:.2}", complexity.tc, complexity.sc);
}
```

**Step 2: Run tests**

Run: `cargo test --test test_omeco`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_omeco.rs && git commit -m "test: verify omeco integration with circuit tensor networks"
```

---

### Task 10: Final Cleanup

**Step 1: Run all tests**

Run: `cargo test`
Expected: All pass

**Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

**Step 3: Fix any clippy issues found**

**Step 4: Add .gitignore**

File: `.gitignore`
```
/target
Cargo.lock
```

**Step 5: Final commit**

```bash
git add .gitignore && git commit -m "chore: add .gitignore"
```

---

## Dependency Graph

```
Task 1 (setup)
  └── Task 2 (gate enum)
        ├── Task 3 (circuit + validation)
        │     ├── Task 5 (tensors)
        │     │     └── Task 6 (einsum)
        │     │           ├── Task 8 (integration tests)
        │     │           └── Task 9 (omeco integration)
        │     └── Task 7 (apply)
        │           └── Task 8 (integration tests)
        └── Task 4 (state)
              ├── Task 7 (apply)
              └── Task 8 (integration tests)
Task 10 (cleanup) — after all others
```
