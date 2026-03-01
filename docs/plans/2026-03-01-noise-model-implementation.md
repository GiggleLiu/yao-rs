# Noise Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add noise channel support with full Yao.jl parity — 10 error types + custom Kraus, circuit integration, density matrix einsum export.

**Architecture:** `NoiseChannel` enum in `src/noise.rs` provides Kraus operators and superoperator matrices. `CircuitElement::Channel` wraps positioned channels. `circuit_to_einsum_dm()` in `src/einsum.rs` converts circuits (with noise) to tensor networks using `EinCode<i32>` with negated labels for bra indices.

**Tech Stack:** Rust, ndarray, num-complex, omeco (EinCode<i32>)

**Design doc:** `docs/plans/2026-03-01-noise-model-design.md`

**Julia reference:** `~/.julia/dev/Yao/lib/YaoBlocks/src/channel/` and `~/.julia/dev/Yao/lib/YaoToEinsum/src/`

---

### Task 1: NoiseChannel enum — PhaseAmplitudeDamping core

This is the foundation: `PhaseAmplitudeDamping` is the base case for 3 other channel types.

**Files:**
- Create: `src/noise.rs`
- Modify: `src/lib.rs`
- Create: `tests/noise.rs`

**Step 1: Write failing tests for PhaseAmplitudeDamping Kraus operators**

Create `tests/noise.rs`:

```rust
use ndarray::Array2;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn assert_matrix_approx(a: &Array2<Complex64>, b: &Array2<Complex64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < tol, "Element mismatch: {} vs {}", x, y);
    }
}

#[test]
fn test_phase_amplitude_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    // PhaseAmplitudeDamping(a=0.3, b=0.2, p1=0.0)
    // Julia ref: errortypes.jl:271-296
    // Expected: A0, A1, A2 only (p1=0)
    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();

    // A0 = sqrt(1-p1) * [[1, 0], [0, sqrt(1-a-b)]] = [[1,0],[0,sqrt(0.5)]]
    let expected_a0 = Array2::from_shape_vec(
        (2, 2),
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5_f64.sqrt(), 0.0)],
    ).unwrap();
    assert_matrix_approx(&kraus[0], &expected_a0, 1e-10);

    // A1 = sqrt(1-p1) * [[0, sqrt(a)], [0, 0]] = [[0, sqrt(0.3)], [0, 0]]
    let expected_a1 = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(0.3_f64.sqrt(), 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    ).unwrap();
    assert_matrix_approx(&kraus[1], &expected_a1, 1e-10);

    // A2 = sqrt(1-p1) * [[0, 0], [0, sqrt(b)]] = [[0, 0], [0, sqrt(0.2)]]
    let expected_a2 = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.2_f64.sqrt(), 0.0)],
    ).unwrap();
    assert_matrix_approx(&kraus[2], &expected_a2, 1e-10);

    assert_eq!(kraus.len(), 3);
}

#[test]
fn test_phase_amplitude_damping_with_excited_pop() {
    use yao_rs::noise::NoiseChannel;

    // PhaseAmplitudeDamping(a=0.3, b=0.2, p1=0.4)
    // Should produce 6 Kraus operators (A0,A1,A2,B0,B1,B2)
    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.4,
    };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 6);

    // Verify completeness: sum_i K_i^dag K_i = I
    let mut sum = Array2::<Complex64>::zeros((2, 2));
    for k in &kraus {
        let kdag = k.t().mapv(|c| c.conj());
        sum = sum + kdag.dot(k);
    }
    let eye = Array2::from_diag(&ndarray::arr1(&[c(1.0, 0.0), c(1.0, 0.0)]));
    assert_matrix_approx(&sum, &eye, 1e-10);
}

#[test]
fn test_noise_channel_num_qubits() {
    use yao_rs::noise::NoiseChannel;

    assert_eq!(
        NoiseChannel::PhaseAmplitudeDamping { amplitude: 0.1, phase: 0.1, excited_population: 0.0 }.num_qubits(),
        1
    );
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features test_phase_amplitude_damping -v`
Expected: Compilation error — `yao_rs::noise` doesn't exist.

**Step 3: Create `src/noise.rs` with `NoiseChannel` enum and `PhaseAmplitudeDamping`**

Create `src/noise.rs`:

```rust
use ndarray::Array2;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Quantum noise channel types.
///
/// Each variant represents a different physical noise process.
/// All channels can produce their Kraus operator representation
/// and superoperator matrix for tensor network export.
///
/// Julia ref: `~/.julia/dev/Yao/lib/YaoBlocks/src/channel/errortypes.jl`
#[derive(Debug, Clone)]
pub enum NoiseChannel {
    BitFlip { p: f64 },
    PhaseFlip { p: f64 },
    Depolarizing { n: usize, p: f64 },
    PauliChannel { px: f64, py: f64, pz: f64 },
    Reset { p0: f64, p1: f64 },
    AmplitudeDamping { gamma: f64, excited_population: f64 },
    PhaseDamping { gamma: f64 },
    PhaseAmplitudeDamping { amplitude: f64, phase: f64, excited_population: f64 },
    ThermalRelaxation { t1: f64, t2: f64, time: f64, excited_population: f64 },
    Coherent { matrix: Array2<Complex64> },
    Custom { kraus_ops: Vec<Array2<Complex64>> },
}

impl NoiseChannel {
    /// Number of qubits this channel acts on.
    pub fn num_qubits(&self) -> usize {
        match self {
            NoiseChannel::Depolarizing { n, .. } => *n,
            NoiseChannel::Custom { kraus_ops } => {
                let d = kraus_ops[0].nrows();
                (d as f64).log2() as usize
            }
            NoiseChannel::Coherent { matrix } => {
                let d = matrix.nrows();
                (d as f64).log2() as usize
            }
            _ => 1, // All other single-qubit channels
        }
    }

    /// Construct Kraus operators for this channel.
    ///
    /// Returns matrices K_i such that E(rho) = sum_i K_i rho K_i^dag.
    ///
    /// Julia ref: errortypes.jl KrausChannel() conversions
    pub fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        match self {
            NoiseChannel::PhaseAmplitudeDamping { amplitude, phase, excited_population } => {
                phase_amplitude_damping_kraus(*amplitude, *phase, *excited_population)
            }
            NoiseChannel::AmplitudeDamping { gamma, excited_population } => {
                // Julia ref: errortypes.jl:362
                phase_amplitude_damping_kraus(*gamma, 0.0, *excited_population)
            }
            NoiseChannel::PhaseDamping { gamma } => {
                // Julia ref: errortypes.jl:327
                phase_amplitude_damping_kraus(0.0, *gamma, 0.0)
            }
            NoiseChannel::ThermalRelaxation { t1, t2, time, excited_population } => {
                // Julia ref: errortypes.jl:223-229
                let t_phi = (t1 * t2) / (2.0 * t1 - t2);
                let a = 1.0 - (-time / t1).exp();
                let b = 1.0 - (-time / t_phi).exp();
                phase_amplitude_damping_kraus(a, b, *excited_population)
            }
            NoiseChannel::BitFlip { p } => {
                // Julia ref: errortypes.jl:38 → MixedUnitaryChannel([I2, X], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*X
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(s1, 0.0), c(s1, 0.0), c(0.0, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::PhaseFlip { p } => {
                // Julia ref: errortypes.jl:56 → MixedUnitaryChannel([I2, Z], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*Z
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(s1, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-s1, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::PauliChannel { px, py, pz } => {
                // Julia ref: errortypes.jl:120
                // K0 = sqrt(1-px-py-pz)*I, K1 = sqrt(px)*X, K2 = sqrt(py)*Y, K3 = sqrt(pz)*Z
                let s0 = (1.0 - px - py - pz).sqrt();
                let sx = px.sqrt();
                let sy = py.sqrt();
                let sz = pz.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sx, 0.0), c(sx, 0.0), c(0.0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, -sy), c(0.0, sy), c(0.0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(sz, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-sz, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::Depolarizing { n, p } => {
                // Julia ref: errortypes.jl:82 single-qubit = PauliChannel(p/4, p/4, p/4)
                // Multi-qubit: all n-qubit Pauli products
                if *n == 1 {
                    let q = p / 4.0;
                    NoiseChannel::PauliChannel { px: q, py: q, pz: q }.kraus_operators()
                } else {
                    depolarizing_multi_qubit_kraus(*n, *p)
                }
            }
            NoiseChannel::Reset { p0, p1 } => {
                // Julia ref: errortypes.jl:168-181
                let s = (1.0 - p0 - p1).sqrt();
                let mut ops = vec![
                    // K0 = sqrt(1-p0-p1) * I
                    Array2::from_shape_vec((2, 2), vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)]).unwrap(),
                ];
                if *p0 > 0.0 {
                    let sp0 = p0.sqrt();
                    // sqrt(p0) * P0 = sqrt(p0) * |0><0|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
                    // sqrt(p0) * Pd = sqrt(p0) * |0><1|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
                }
                if *p1 > 0.0 {
                    let sp1 = p1.sqrt();
                    // sqrt(p1) * P1 = sqrt(p1) * |1><1|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0)]).unwrap());
                    // sqrt(p1) * Pu = sqrt(p1) * |1><0|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0), c(0.0, 0.0)]).unwrap());
                }
                ops
            }
            NoiseChannel::Coherent { matrix } => {
                // Julia ref: errortypes.jl:19 — single Kraus op = the unitary
                vec![matrix.clone()]
            }
            NoiseChannel::Custom { kraus_ops } => {
                kraus_ops.clone()
            }
        }
    }

    /// Build the superoperator matrix for this channel.
    ///
    /// S = sum_i kron(conj(K_i), K_i)
    ///
    /// Julia ref: kraus.jl:73-78
    pub fn superop(&self) -> Array2<Complex64> {
        let kraus = self.kraus_operators();
        let d = kraus[0].nrows();
        let d2 = d * d;
        let mut superop = Array2::<Complex64>::zeros((d2, d2));
        for k in &kraus {
            let k_conj = k.mapv(|c| c.conj());
            // kron(conj(K), K)
            for i in 0..d {
                for j in 0..d {
                    for k_row in 0..d {
                        for k_col in 0..d {
                            superop[[i * d + k_row, j * d + k_col]] += k_conj[[i, j]] * k[[k_row, k_col]];
                        }
                    }
                }
            }
        }
        superop
    }
}

/// Julia ref: errortypes.jl:271-296 KrausChannel(err::PhaseAmplitudeDampingError)
fn phase_amplitude_damping_kraus(a: f64, b: f64, p1: f64) -> Vec<Array2<Complex64>> {
    let mut ops = Vec::new();
    let rest = (1.0 - a - b).sqrt();

    if (p1 - 1.0).abs() > f64::EPSILON {
        // Damping to ground state
        let s = (1.0 - p1).sqrt();
        // A0
        ops.push(Array2::from_shape_vec((2, 2), vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s * rest, 0.0)]).unwrap());
        // A1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
        }
        // A2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sb, 0.0)]).unwrap());
        }
    }

    if p1.abs() > f64::EPSILON {
        // Damping to excited state
        let s = p1.sqrt();
        // B0
        ops.push(Array2::from_shape_vec((2, 2), vec![c(s * rest, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)]).unwrap());
        // B1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0)]).unwrap());
        }
        // B2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(sb, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
        }
    }

    ops
}

/// Multi-qubit depolarizing Kraus operators.
/// Julia ref: mixed_unitary_channel.jl:144-152
fn depolarizing_multi_qubit_kraus(n: usize, p: f64) -> Vec<Array2<Complex64>> {
    // Pauli matrices
    let eye = Array2::from_shape_vec((2, 2), vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]).unwrap();
    let px = Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]).unwrap();
    let py = Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]).unwrap();
    let pz = Array2::from_shape_vec((2, 2), vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]).unwrap();
    let paulis = [eye, px, py, pz];

    let dim = 1usize << (2 * n); // 4^n
    let mut ops = Vec::with_capacity(dim);

    // Generate all n-qubit Pauli products
    for idx in 0..dim {
        let mut mat = Array2::from_shape_vec((1, 1), vec![c(1.0, 0.0)]).unwrap();
        let mut tmp = idx;
        for _ in 0..n {
            let pauli_idx = tmp % 4;
            tmp /= 4;
            mat = kron(&mat, &paulis[pauli_idx]);
        }
        // Weight: p/4^n for all, identity gets extra 1-p
        let weight = if idx == 0 { 1.0 - p + p / dim as f64 } else { p / dim as f64 };
        let s = weight.sqrt();
        ops.push(mat.mapv(|v| v * c(s, 0.0)));
    }
    ops
}

/// Kronecker product of two matrices.
fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (br, bc) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            for k in 0..br {
                for l in 0..bc {
                    result[[i * br + k, j * bc + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    result
}
```

Also add `pub mod noise;` to `src/lib.rs` (after `pub mod measure;`) and re-export:
```rust
pub use noise::NoiseChannel;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features test_phase_amplitude_damping -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/noise.rs src/lib.rs tests/noise.rs
git commit -m "feat: add NoiseChannel enum with all 10 error types + Custom"
```

---

### Task 2: Remaining Kraus operator tests

**Files:**
- Modify: `tests/noise.rs`

**Step 1: Add tests for all remaining channel types**

Add to `tests/noise.rs`:

```rust
#[test]
fn test_bit_flip_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2);

    // K0 = sqrt(0.9)*I
    assert!((kraus[0][[0, 0]].re - 0.9_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[0][[1, 1]].re - 0.9_f64.sqrt()).abs() < 1e-10);
    // K1 = sqrt(0.1)*X
    assert!((kraus[1][[0, 1]].re - 0.1_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[1][[1, 0]].re - 0.1_f64.sqrt()).abs() < 1e-10);

    verify_completeness(&kraus);
}

#[test]
fn test_phase_flip_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PhaseFlip { p: 0.2 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2);
    verify_completeness(&kraus);

    // K1 = sqrt(0.2)*Z → [[sqrt(0.2), 0], [0, -sqrt(0.2)]]
    assert!((kraus[1][[0, 0]].re - 0.2_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[1][[1, 1]].re + 0.2_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_pauli_channel_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PauliChannel { px: 0.1, py: 0.2, pz: 0.05 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 4);
    verify_completeness(&kraus);
}

#[test]
fn test_depolarizing_single_qubit_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Depolarizing { n: 1, p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 4);
    verify_completeness(&kraus);
}

#[test]
fn test_depolarizing_two_qubit_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Depolarizing { n: 2, p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 16);

    // Verify each is 4x4
    for k in &kraus {
        assert_eq!(k.shape(), &[4, 4]);
    }
    verify_completeness(&kraus);
}

#[test]
fn test_reset_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Reset { p0: 0.1, p1: 0.05 };
    let kraus = ch.kraus_operators();
    // 1 (identity) + 2 (p0>0) + 2 (p1>0) = 5
    assert_eq!(kraus.len(), 5);
    verify_completeness(&kraus);
}

#[test]
fn test_reset_only_p0() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Reset { p0: 0.3, p1: 0.0 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 3); // identity + P0 + Pd
    verify_completeness(&kraus);
}

#[test]
fn test_amplitude_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::AmplitudeDamping { gamma: 0.3, excited_population: 0.0 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2); // A0 and A1
    verify_completeness(&kraus);
}

#[test]
fn test_phase_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PhaseDamping { gamma: 0.2 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2); // A0 and A2
    verify_completeness(&kraus);
}

#[test]
fn test_thermal_relaxation_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::ThermalRelaxation { t1: 100.0, t2: 80.0, time: 10.0, excited_population: 0.0 };
    let kraus = ch.kraus_operators();
    assert!(kraus.len() >= 2);
    verify_completeness(&kraus);
}

#[test]
fn test_coherent_kraus() {
    use yao_rs::noise::NoiseChannel;

    // Coherent error = single unitary (e.g., small rotation)
    let theta = 0.01_f64;
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![c(theta.cos(), 0.0), c(0.0, -theta.sin()), c(0.0, -theta.sin()), c(theta.cos(), 0.0)],
    ).unwrap();
    let ch = NoiseChannel::Coherent { matrix: matrix.clone() };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 1);
    assert_matrix_approx(&kraus[0], &matrix, 1e-10);
}

#[test]
fn test_superop_from_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let superop = ch.superop();
    assert_eq!(superop.shape(), &[4, 4]);

    // Verify: superop = sum_i kron(conj(K_i), K_i)
    let kraus = ch.kraus_operators();
    let mut expected = Array2::<Complex64>::zeros((4, 4));
    for k in &kraus {
        let kc = k.mapv(|c| c.conj());
        for i in 0..2 {
            for j in 0..2 {
                for m in 0..2 {
                    for n in 0..2 {
                        expected[[i * 2 + m, j * 2 + n]] += kc[[i, j]] * k[[m, n]];
                    }
                }
            }
        }
    }
    assert_matrix_approx(&superop, &expected, 1e-10);
}

/// Verify Kraus completeness: sum_i K_i^dag K_i = I
fn verify_completeness(kraus: &[Array2<Complex64>]) {
    let d = kraus[0].nrows();
    let mut sum = Array2::<Complex64>::zeros((d, d));
    for k in kraus {
        let kdag = k.t().mapv(|c| c.conj());
        sum = sum + kdag.dot(k);
    }
    let eye = Array2::from_diag(&ndarray::Array1::from_vec(vec![c(1.0, 0.0); d]));
    assert_matrix_approx(&sum, &eye, 1e-10);
}
```

**Step 2: Run tests**

Run: `cargo test --all-features noise -v`
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add tests/noise.rs
git commit -m "test: comprehensive Kraus operator and superop tests for all 10 noise types"
```

---

### Task 3: Circuit integration — `CircuitElement::Channel`

**Files:**
- Modify: `src/circuit.rs:25-28` (CircuitElement enum)
- Modify: `src/circuit.rs:134-254` (Circuit::new validation, dagger, Display)
- Modify: `src/apply.rs:44-48` (skip Channel in apply)
- Modify: `src/einsum.rs:51-54` (skip Channel in existing einsum — will be handled by DM path)
- Modify: `src/lib.rs:21-26` (re-exports)
- Create: `tests/circuit_noise.rs`

**Step 1: Write failing test**

Create `tests/circuit_noise.rs`:

```rust
use yao_rs::circuit::{Circuit, CircuitElement, channel, put};
use yao_rs::gate::Gate;
use yao_rs::noise::NoiseChannel;

#[test]
fn test_circuit_with_noise_channel() {
    let elements = vec![
        put(vec![0], Gate::H),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();
    assert_eq!(circuit.elements.len(), 2);
}

#[test]
fn test_circuit_channel_loc_validation() {
    let elements = vec![
        channel(vec![5], NoiseChannel::BitFlip { p: 0.1 }), // loc out of range
    ];
    assert!(Circuit::new(vec![2, 2], elements).is_err());
}

#[test]
fn test_apply_skips_channels() {
    use yao_rs::{apply, State};
    use yao_rs::circuit::channel;

    let elements = vec![
        put(vec![0], Gate::X),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let state = State::zero_state(&[2]);
    let result = apply(&circuit, &state);
    // Channel is skipped; only X is applied: |0⟩ → |1⟩
    assert!((result.data[0].norm() - 0.0).abs() < 1e-10);
    assert!((result.data[1].norm() - 1.0).abs() < 1e-10);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features test_circuit_with_noise -v`
Expected: Compilation error — `channel` function doesn't exist.

**Step 3: Implement circuit integration**

Modify `src/circuit.rs`:

1. Add import at top: `use crate::noise::NoiseChannel;`

2. Add `PositionedChannel` struct and extend `CircuitElement`:

```rust
/// A noise channel placed at specific qubit locations.
#[derive(Debug, Clone)]
pub struct PositionedChannel {
    pub channel: NoiseChannel,
    pub locs: Vec<usize>,
}

pub enum CircuitElement {
    Gate(PositionedGate),
    Annotation(PositionedAnnotation),
    Channel(PositionedChannel),
}
```

3. In `Circuit::new()` validation, add a match arm for `CircuitElement::Channel`:

```rust
CircuitElement::Channel(pc) => {
    // Validate locs are in range
    for &loc in &pc.locs {
        if loc >= num_sites {
            return Err(CircuitError::LocOutOfRange { loc, num_sites });
        }
    }
    // Validate qubit count matches channel
    let expected_qubits = pc.channel.num_qubits();
    if pc.locs.len() != expected_qubits {
        return Err(CircuitError::MatrixSizeMismatch {
            expected: expected_qubits,
            actual: pc.locs.len(),
        });
    }
}
```

4. In `Circuit::dagger()`, channels are not daggerable (noise is irreversible). Add `Channel` arm that clones as-is (or skip — for now clone):

```rust
CircuitElement::Channel(pc) => CircuitElement::Channel(pc.clone()),
```

5. In `Circuit::Display`, add channel formatting:

```rust
CircuitElement::Channel(pc) => {
    writeln!(f, "  {:?} @ q[{}]", pc.channel, format_locs(&pc.locs))?;
}
```

6. Add `channel()` helper function:

```rust
/// Place a noise channel at specific qubit locations.
pub fn channel(locs: Vec<usize>, noise: NoiseChannel) -> CircuitElement {
    CircuitElement::Channel(PositionedChannel { channel: noise, locs })
}
```

Modify `src/apply.rs:44-48`: add Channel arm:
```rust
CircuitElement::Channel(_) => continue, // Skip noise in apply path
```

Modify `src/einsum.rs:51-54` (and similar in other einsum functions): add Channel arm:
```rust
CircuitElement::Channel(_) => continue, // Handled by DM path only
```

Update `src/lib.rs` re-exports to include `channel` and `PositionedChannel`.

**Step 4: Run tests**

Run: `cargo test --all-features test_circuit -v`
Expected: All circuit noise tests PASS. Existing tests still pass.

**Step 5: Run full check**

Run: `make check-all`
Expected: All green.

**Step 6: Commit**

```bash
git add src/circuit.rs src/apply.rs src/einsum.rs src/lib.rs tests/circuit_noise.rs
git commit -m "feat: add CircuitElement::Channel for noise in circuits"
```

---

### Task 4: Density matrix einsum — `TensorNetworkDM` and `circuit_to_einsum_dm()`

This is the core tensor network conversion for noisy circuits.

**Files:**
- Modify: `src/einsum.rs` (add TensorNetworkDM, circuit_to_einsum_dm)
- Modify: `src/lib.rs` (re-exports)
- Create: `tests/einsum_dm.rs`

**Step 1: Write failing test — pure circuit in DM mode equals |psi|^2**

Create `tests/einsum_dm.rs`:

```rust
use ndarray::ArrayD;
use num_complex::Complex64;
use yao_rs::circuit::{Circuit, put, control};
use yao_rs::gate::Gate;
use yao_rs::einsum::{circuit_to_einsum, circuit_to_einsum_dm};

mod common;
use common::naive_contract;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_dm_single_hadamard() {
    // H|0⟩ = (|0⟩+|1⟩)/sqrt(2)
    // rho = |psi><psi| = [[0.5, 0.5], [0.5, 0.5]]
    let elements = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], elements).unwrap();

    let tn = circuit_to_einsum_dm(&circuit);

    // Contract the DM tensor network
    let rho = naive_contract_dm(&tn);

    // Expected: 2x2 density matrix
    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[0, 1]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.5).abs() < 1e-10);
}
```

Note: `naive_contract_dm` needs to be implemented — a simple einsum contraction for `EinCode<i32>`.

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features test_dm_single_hadamard -v`
Expected: Compilation error — `circuit_to_einsum_dm` doesn't exist.

**Step 3: Implement `circuit_to_einsum_dm()`**

Add to `src/einsum.rs`:

```rust
use omeco::EinCode;

/// Tensor network with i32 labels for density matrix mode.
/// Positive labels = ket (forward), negative = bra (conjugate).
///
/// Julia ref: YaoToEinsum/src/Core.jl TensorNetwork
#[derive(Debug, Clone)]
pub struct TensorNetworkDM {
    pub code: EinCode<i32>,
    pub tensors: Vec<ArrayD<Complex64>>,
    pub size_dict: HashMap<i32, usize>,
}

/// Convert a circuit to a density matrix tensor network.
///
/// Pure gates are doubled (ket + bra copies). Noise channels are
/// converted to superoperator tensors.
///
/// Initial state: |0...0><0...0|
///
/// Julia ref: YaoToEinsum/src/circuitmap.jl:353-381 yao2einsum(; mode=DensityMatrixMode())
pub fn circuit_to_einsum_dm(circuit: &Circuit) -> TensorNetworkDM {
    let n = circuit.num_sites();

    // Labels 1..n for ket, -1..-n for bra (matching Yao.jl)
    let mut slots: Vec<i32> = (1..=n as i32).collect();
    let mut next_label: i32 = n as i32 + 1;

    let mut size_dict: HashMap<i32, usize> = HashMap::new();
    for i in 0..n {
        let label = (i + 1) as i32;
        size_dict.insert(label, circuit.dims[i]);
        size_dict.insert(-label, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<i32>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    // Initial state: |0><0| boundary tensors on each qubit
    for i in 0..n {
        let d = circuit.dims[i];
        let mut data = vec![Complex64::new(0.0, 0.0); d];
        data[0] = Complex64::new(1.0, 0.0);
        let tensor = ArrayD::from_shape_vec(IxDyn(&[d]), data.clone()).unwrap();
        let tensor_conj = tensor.clone();

        // Ket boundary: label = i+1
        all_ixs.push(vec![slots[i]]);
        all_tensors.push(tensor);
        // Bra boundary: label = -(i+1)
        all_ixs.push(vec![-slots[i]]);
        all_tensors.push(tensor_conj);
    }

    for element in &circuit.elements {
        match element {
            CircuitElement::Gate(pg) => {
                let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);

                let all_locs = pg.all_locs();
                let has_controls = !pg.control_locs.is_empty();
                let is_diag = pg.gate.is_diagonal() && !has_controls;

                if is_diag {
                    // Diagonal: reuse labels, add ket and bra copies
                    // Julia ref: circuitmap.jl:83-84 (push_normal_tensor! with diagonal)
                    let ket_ixs: Vec<i32> = pg.target_locs.iter().map(|&loc| slots[loc]).collect();
                    all_ixs.push(ket_ixs.clone());
                    all_tensors.push(tensor.clone());

                    // Bra copy: conj tensor, negated labels
                    let bra_ixs: Vec<i32> = ket_ixs.iter().map(|&l| -l).collect();
                    all_ixs.push(bra_ixs);
                    all_tensors.push(tensor.mapv(|c| c.conj()));
                } else {
                    // Non-diagonal: allocate new output labels
                    // Julia ref: circuitmap.jl:99-101
                    let mut new_labels: Vec<i32> = Vec::new();
                    for &loc in &all_locs {
                        let nl = next_label;
                        next_label += 1;
                        size_dict.insert(nl, circuit.dims[loc]);
                        size_dict.insert(-nl, circuit.dims[loc]);
                        new_labels.push(nl);
                    }

                    // Ket tensor: [new_out..., current_in...]
                    let mut ket_ixs: Vec<i32> = new_labels.clone();
                    for &loc in &all_locs {
                        ket_ixs.push(slots[loc]);
                    }
                    all_ixs.push(ket_ixs);
                    all_tensors.push(tensor.clone());

                    // Bra tensor: conj, negated labels
                    // Julia ref: circuitmap.jl:42-43
                    let mut bra_ixs: Vec<i32> = new_labels.iter().map(|&l| -l).collect();
                    for &loc in &all_locs {
                        bra_ixs.push(-slots[loc]);
                    }
                    all_ixs.push(bra_ixs);
                    all_tensors.push(tensor.mapv(|c| c.conj()));

                    // Update slots
                    for (i, &loc) in all_locs.iter().enumerate() {
                        slots[loc] = new_labels[i];
                    }
                }
            }
            CircuitElement::Channel(pc) => {
                // Convert to superoperator tensor
                // Julia ref: circuitmap.jl:229-239 add_channel!
                let superop = pc.channel.superop();
                let k = pc.locs.len();
                let d = 2_usize; // qubit dimension

                // Reshape to D^(4k) tensor
                let shape: Vec<usize> = vec![d; 4 * k];
                let tensor = ArrayD::from_shape_vec(IxDyn(&shape), superop.into_raw_vec()).unwrap();

                // Allocate new output labels
                let mut new_labels: Vec<i32> = Vec::new();
                for &loc in &pc.locs {
                    let nl = next_label;
                    next_label += 1;
                    size_dict.insert(nl, circuit.dims[loc]);
                    size_dict.insert(-nl, circuit.dims[loc]);
                    new_labels.push(nl);
                }

                // Labels: [out, -out, in, -in]
                // Julia ref: circuitmap.jl:236
                let mut ixs: Vec<i32> = Vec::new();
                ixs.extend(&new_labels);                           // out
                ixs.extend(new_labels.iter().map(|&l| -l));        // -out
                for &loc in &pc.locs {
                    ixs.push(slots[loc]);                           // in
                }
                for &loc in &pc.locs {
                    ixs.push(-slots[loc]);                          // -in
                }

                all_ixs.push(ixs);
                all_tensors.push(tensor);

                // Update slots
                for (i, &loc) in pc.locs.iter().enumerate() {
                    slots[loc] = new_labels[i];
                }
            }
            CircuitElement::Annotation(_) => continue,
        }
    }

    // Output: [slots, -slots] for full density matrix
    let mut output_labels: Vec<i32> = slots.clone();
    output_labels.extend(slots.iter().map(|&l| -l));

    TensorNetworkDM {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}
```

Also add a `naive_contract_dm` helper in `tests/common/mod.rs` for testing (contract `TensorNetworkDM` by brute-force iteration).

**Step 4: Run tests**

Run: `cargo test --all-features test_dm -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/einsum.rs src/lib.rs tests/einsum_dm.rs tests/common/mod.rs
git commit -m "feat: add circuit_to_einsum_dm() for density matrix tensor networks"
```

---

### Task 5: DM einsum tests — noisy circuits

**Files:**
- Modify: `tests/einsum_dm.rs`

**Step 1: Add tests for noisy circuit DM einsum**

```rust
#[test]
fn test_dm_with_bit_flip_noise() {
    // H|0⟩ then BitFlip(p=0.1)
    // rho_noisy = (1-p)*|+><+| + p*X|+><+|X
    //           = (1-p)*[[0.5,0.5],[0.5,0.5]] + p*[[0.5,-0.5],[-0.5,0.5]]
    //           = [[0.5, 0.5-p], [0.5-p, 0.5]]  (for p=0.1: [[0.5, 0.4], [0.4, 0.5]])
    let elements = vec![
        put(vec![0], Gate::H),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = naive_contract_dm(&tn);

    assert!((rho[[0, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[0, 1]].re - 0.4).abs() < 1e-10);
    assert!((rho[[1, 0]].re - 0.4).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.5).abs() < 1e-10);
}

#[test]
fn test_dm_bell_state_with_depolarizing() {
    // Bell state: H on q0, CNOT(q0, q1)
    // Then depolarizing(p=0.1) on q0
    let elements = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
        channel(vec![0], NoiseChannel::Depolarizing { n: 1, p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = naive_contract_dm(&tn);

    // Verify trace = 1
    let trace = rho[[0, 0]] + rho[[1, 1]] + rho[[2, 2]] + rho[[3, 3]];
    assert!((trace.re - 1.0).abs() < 1e-10);
    assert!(trace.im.abs() < 1e-10);

    // Verify rho is positive semidefinite (all eigenvalues >= 0)
    // and Hermitian
    for i in 0..4 {
        for j in 0..4 {
            assert!((rho[[i, j]] - rho[[j, i]].conj()).norm() < 1e-10);
        }
    }
}

#[test]
fn test_dm_pure_matches_vector_mode() {
    // Pure circuit (no noise): DM mode should give |psi><psi|
    // which matches vector mode result
    let elements = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
    ];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();

    // Vector mode
    let tn_vec = circuit_to_einsum(&circuit);
    let psi = naive_contract_vec(&tn_vec);  // 4-element state vector

    // DM mode
    let tn_dm = circuit_to_einsum_dm(&circuit);
    let rho = naive_contract_dm(&tn_dm);

    // Verify: rho[i,j] = psi[i] * conj(psi[j])
    for i in 0..4 {
        for j in 0..4 {
            let expected = psi[i] * psi[j].conj();
            assert!((rho[[i, j]] - expected).norm() < 1e-10,
                "Mismatch at [{},{}]: {} vs {}", i, j, rho[[i, j]], expected);
        }
    }
}
```

**Step 2: Run tests**

Run: `cargo test --all-features test_dm -v`
Expected: All PASS.

**Step 3: Commit**

```bash
git add tests/einsum_dm.rs
git commit -m "test: noisy circuit DM einsum tests with analytical verification"
```

---

### Task 6: `circuit_to_expectation_dm()` — expectation values in DM mode

**Files:**
- Modify: `src/einsum.rs`
- Modify: `tests/einsum_dm.rs`

**Step 1: Write failing test**

Add to `tests/einsum_dm.rs`:

```rust
#[test]
fn test_expectation_dm_hadamard_z() {
    use yao_rs::operator::{Op, OperatorString, OperatorPolynomial};

    // <Z> for H|0⟩ = 0 (|+⟩ state)
    let elements = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], elements).unwrap();

    let op = OperatorPolynomial::from_single(
        Complex64::new(1.0, 0.0),
        OperatorString::new(vec![(0, Op::Z)]),
    );

    let tn = circuit_to_expectation_dm(&circuit, &op);
    let result = naive_contract_scalar_dm(&tn);
    assert!((result.re - 0.0).abs() < 1e-10);
}

#[test]
fn test_expectation_dm_noisy_hadamard_x() {
    use yao_rs::operator::{Op, OperatorString, OperatorPolynomial};

    // <X> for H|0⟩ with depolarizing(p=0.1)
    // Pure: <X> = 1.0
    // With depolarizing: <X> = (1-p)*1 = 0.9  (for single-qubit dep)
    // Actually: E(rho) = (1-p)*rho + (p/4)*(X rho X + Y rho Y + Z rho Z + I rho I)
    // <X>_noisy = tr(X * E(rho)) where rho = |+><+|
    let elements = vec![
        put(vec![0], Gate::H),
        channel(vec![0], NoiseChannel::Depolarizing { n: 1, p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();

    let op = OperatorPolynomial::from_single(
        Complex64::new(1.0, 0.0),
        OperatorString::new(vec![(0, Op::X)]),
    );

    let tn = circuit_to_expectation_dm(&circuit, &op);
    let result = naive_contract_scalar_dm(&tn);

    // Analytical: <X> = (1 - p/2 - p/2) * 1 = 1 - p = 0.9...
    // Actually for depolarizing: <O>_noisy = (1-p)*<O>_pure + p/4*(tr(O)*2)
    // For X: tr(X)=0, so <X>_noisy = (1-p)*1 + p/4*0 = 0.9
    // But depolarizing uses the channel formula:
    // E(rho) = (1-p)rho + p*I/2
    // <X>_noisy = (1-p)*tr(X*rho) + p*tr(X*I/2) = (1-p)*1 + p*0 = 0.9
    assert!((result.re - 0.9).abs() < 1e-8);
}
```

**Step 2: Implement `circuit_to_expectation_dm()`**

This traces the density matrix with the observable:
```
result = tr(rho * O) = contract TN with observable inserted and dual indices traced
```

Julia ref: `circuitmap.jl:252-258 eat_observable!` — adds observable to ket side, then traces ket with bra.

**Step 3: Run tests**

Run: `cargo test --all-features test_expectation_dm -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/einsum.rs tests/einsum_dm.rs
git commit -m "feat: add circuit_to_expectation_dm() for noisy expectation values"
```

---

### Task 7: Julia fixture generation and cross-validation

**Files:**
- Create: `tests/fixtures/generate_noise.jl` (Julia script)
- Create: `tests/data/noise.json`
- Modify: `tests/noise.rs`

**Step 1: Write Julia fixture generation script**

Create `tests/fixtures/generate_noise.jl`:

```julia
using Yao, JSON

# Generate Kraus operators for each noise type
noise_data = Dict()

# 1. BitFlip
ch = quantum_channel(BitFlipError(0.1))
noise_data["bit_flip_0.1"] = Dict(
    "kraus" => [collect(mat(ComplexF64, op)') for op in KrausChannel(ch).operators],
    "superop" => collect(SuperOp(ComplexF64, ch).superop')
)

# ... (generate for each type)

# 2. DM einsum test: H + BitFlip
circuit = chain(1, put(1, 1=>H), put(1, 1=>quantum_channel(BitFlipError(0.1))))
tn = yao2einsum(circuit, mode=DensityMatrixMode())
rho = reshape(contract(tn), 2, 2)
noise_data["dm_h_bitflip_0.1"] = Dict(
    "rho_re" => real.(rho),
    "rho_im" => imag.(rho)
)

# Write JSON
open("tests/data/noise.json", "w") do f
    JSON.print(f, noise_data, 4)
end
```

**Step 2: Run the script and generate fixtures**

Run: `cd /path/to/yao-rs && julia tests/fixtures/generate_noise.jl`

**Step 3: Add fixture-based tests to `tests/noise.rs`**

```rust
#[test]
fn test_kraus_matches_julia_fixture() {
    // Load noise.json and compare Kraus matrices
    let data: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("tests/data/noise.json").unwrap()
    ).unwrap();

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let kraus = ch.kraus_operators();
    let fixture = &data["bit_flip_0.1"]["kraus"];
    // Compare each matrix...
}
```

**Step 4: Run tests**

Run: `cargo test --all-features test_kraus_matches -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/fixtures/generate_noise.jl tests/data/noise.json tests/noise.rs
git commit -m "test: add Julia fixtures and cross-validation for noise channels"
```

---

### Task 8: Final integration — `make check-all` and cleanup

**Files:**
- Modify: `src/lib.rs` (final re-exports)
- Modify: `src/json.rs` (optional: add noise channel serialization)

**Step 1: Run full check suite**

Run: `make check-all`
Expected: All green — fmt, clippy, tests pass.

**Step 2: Fix any clippy warnings or formatting issues**

Run: `make fmt && make clippy`

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: cleanup and final integration for noise model support"
```

---

## Summary

| Task | Component | Files | Key Julia Ref |
|------|-----------|-------|--------------|
| 1 | NoiseChannel enum + PhaseAmplitudeDamping | `src/noise.rs`, `tests/noise.rs` | `errortypes.jl:256-296` |
| 2 | All Kraus operator tests | `tests/noise.rs` | `errortypes.jl` |
| 3 | CircuitElement::Channel | `src/circuit.rs`, `src/apply.rs` | `circuitmap.jl:107-117, 221-226` |
| 4 | circuit_to_einsum_dm() | `src/einsum.rs`, `tests/einsum_dm.rs` | `circuitmap.jl:353-381, 229-239` |
| 5 | Noisy DM einsum tests | `tests/einsum_dm.rs` | `densitymatrix.jl` tests |
| 6 | circuit_to_expectation_dm() | `src/einsum.rs` | `circuitmap.jl:252-258` |
| 7 | Julia fixtures + cross-validation | `tests/data/noise.json` | All noise channel types |
| 8 | Full integration check | All | — |
