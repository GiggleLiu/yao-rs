# Issues #2, #4, #5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement EasyBuild prebuilt circuits, JSON visualization pipeline with quill/typst, and libtorch contraction backend with boundary conditions.

**Architecture:** Gate enum is extended with new variants and a required label on Custom. A single `easybuild.rs` module provides circuit builders. JSON serialization via serde enables a typst script to render circuits. Boundary conditions extend the einsum export, and a feature-gated tch-rs contractor executes contractions on CPU/GPU.

**Tech Stack:** Rust (edition 2024), ndarray, omeco, serde/serde_json, rand, tch-rs (optional), typst + quill

---

## Dependency Order

1. **Task 1-3**: Gate changes (Custom label, new variants) — foundation for everything
2. **Task 4-5**: EasyBuild circuits — uses new gates
3. **Task 6-7**: JSON serialization — uses Custom label
4. **Task 8**: Typst visualization script
5. **Task 9-10**: Boundary conditions + torch contractor

---

### Task 1: Add `label` field to Custom gate

**Files:**
- Modify: `src/gate.rs:20-23`
- Modify: `src/circuit.rs:168` (is_named check)
- Modify: `tests/test_gates.rs` (all Custom gate constructions)
- Modify: `tests/test_circuit.rs` (all Custom gate constructions)
- Modify: `tests/test_tensors.rs` (all Custom gate constructions)
- Modify: `tests/test_einsum.rs` (all Custom gate constructions)
- Modify: `tests/test_integration.rs` (all Custom gate constructions)
- Modify: `tests/test_apply.rs` (all Custom gate constructions)

**Step 1: Update Gate enum**

In `src/gate.rs:20-23`, change:
```rust
Custom {
    matrix: Array2<Complex64>,
    is_diagonal: bool,
    label: String,
},
```

**Step 2: Fix all compilation errors**

Find every `Gate::Custom { matrix, is_diagonal }` pattern in tests and source, add `label: "...".to_string()`. Use descriptive labels matching the test context (e.g., `"test_2x2"`, `"qutrit_gate"`, `"diagonal_phase"`).

Also update `src/circuit.rs:168`:
```rust
let is_named = !matches!(pg.gate, Gate::Custom { .. });
```
This already works — no change needed since we just added a field.

**Step 3: Run tests**

Run: `cargo test`
Expected: All existing tests pass with the new label field.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: add required label field to Custom gate variant"
```

---

### Task 2: Add new named gate variants (SqrtX, SqrtY, SqrtW, ISWAP)

**Files:**
- Modify: `src/gate.rs:7-24` (enum variants)
- Modify: `src/gate.rs:42-60` (num_sites)
- Modify: `src/gate.rs:64-69` (is_diagonal)
- Modify: `src/gate.rs:80-133` (qubit_matrix)
- Create: `tests/test_new_gates.rs`

**Step 1: Write failing tests**

Create `tests/test_new_gates.rs`:
```rust
use approx::assert_abs_diff_eq;
use ndarray::Array2;
use num_complex::Complex64;
use yao_rs::Gate;

#[test]
fn test_sqrt_x_matrix() {
    let gate = Gate::SqrtX;
    let m = gate.matrix(2);
    // SqrtX = (1+i)/2 * [[1, -i], [-i, 1]]
    let half = Complex64::new(0.5, 0.5);
    let expected = Array2::from_shape_vec((2, 2), vec![
        half * Complex64::new(1.0, 0.0),
        half * Complex64::new(0.0, -1.0),
        half * Complex64::new(0.0, -1.0),
        half * Complex64::new(1.0, 0.0),
    ]).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(m[[i, j]].re, expected[[i, j]].re, epsilon = 1e-10);
            assert_abs_diff_eq!(m[[i, j]].im, expected[[i, j]].im, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_sqrt_x_squared_is_x() {
    let m = Gate::SqrtX.matrix(2);
    let x = Gate::X.matrix(2);
    let m2 = m.dot(&m);
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(m2[[i, j]].re, x[[i, j]].re, epsilon = 1e-10);
            assert_abs_diff_eq!(m2[[i, j]].im, x[[i, j]].im, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_sqrt_y_squared_is_y() {
    let m = Gate::SqrtY.matrix(2);
    let y = Gate::Y.matrix(2);
    let m2 = m.dot(&m);
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(m2[[i, j]].re, y[[i, j]].re, epsilon = 1e-10);
            assert_abs_diff_eq!(m2[[i, j]].im, y[[i, j]].im, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_sqrt_w_is_unitary() {
    let m = Gate::SqrtW.matrix(2);
    let mh = m.t().mapv(|x| x.conj());
    let id = m.dot(&mh);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(id[[i, j]].re, expected, epsilon = 1e-10);
            assert_abs_diff_eq!(id[[i, j]].im, 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_iswap_matrix() {
    let gate = Gate::ISWAP;
    let m = gate.matrix(2);
    assert_eq!(gate.num_sites(2), 2);
    // ISWAP: |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_abs_diff_eq!(m[[0, 0]].re, one.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 2]].re, i.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 2]].im, i.im, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[2, 1]].re, i.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[2, 1]].im, i.im, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[3, 3]].re, one.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[0, 1]].re, zero.re, epsilon = 1e-10);
}

#[test]
fn test_iswap_is_unitary() {
    let m = Gate::ISWAP.matrix(2);
    let mh = m.t().mapv(|x| x.conj());
    let id = m.dot(&mh);
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(id[[i, j]].re, expected, epsilon = 1e-10);
            assert_abs_diff_eq!(id[[i, j]].im, 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_num_sites() {
    assert_eq!(Gate::SqrtX.num_sites(2), 1);
    assert_eq!(Gate::SqrtY.num_sites(2), 1);
    assert_eq!(Gate::SqrtW.num_sites(2), 1);
    assert_eq!(Gate::ISWAP.num_sites(2), 2);
}

#[test]
fn test_is_not_diagonal() {
    assert!(!Gate::SqrtX.is_diagonal());
    assert!(!Gate::SqrtY.is_diagonal());
    assert!(!Gate::SqrtW.is_diagonal());
    assert!(!Gate::ISWAP.is_diagonal());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test test_new_gates`
Expected: FAIL — variants don't exist yet.

**Step 3: Implement gate variants**

In `src/gate.rs`, add after `SWAP` (line 14):
```rust
SqrtX,
SqrtY,
SqrtW,
ISWAP,
```

In `num_sites()`, update the match:
```rust
Gate::SWAP | Gate::ISWAP => 2,
```

In `is_diagonal()`, no change needed (these are all non-diagonal, fall through to `_ => false`).

In `qubit_matrix()`, add cases before `Gate::Custom`:
```rust
Gate::SqrtX => {
    // (1+i)/2 * [[1, -i], [-i, 1]]
    let half_plus = Complex64::new(0.5, 0.5);
    let a = half_plus * one;
    let b = half_plus * neg_i;
    Array2::from_shape_vec((2, 2), vec![a, b, b, a]).unwrap()
}
Gate::SqrtY => {
    // (1+i)/2 * [[1, -1], [1, 1]]
    let half_plus = Complex64::new(0.5, 0.5);
    let a = half_plus * one;
    let b = half_plus * neg_one;
    Array2::from_shape_vec((2, 2), vec![a, b, a, a]).unwrap()
}
Gate::SqrtW => {
    // rot((X+Y)/sqrt(2), pi/2) = exp(-i * pi/4 * (X+Y)/sqrt(2))
    // = cos(pi/4)*I - i*sin(pi/4)*(X+Y)/sqrt(2)
    // = (1/sqrt(2)) * I - i*(1/sqrt(2)) * (X+Y)/sqrt(2)
    // = (1/sqrt(2)) * [[1, -(1+i)/2], [(1-i)/2, ... ]]
    // Exact: [[1+i, -1-i], [1-i, 1+i]] / 2  ... let me compute properly
    // (X+Y)/sqrt(2) has eigenvalues +-1. Matrix = [[0, (1-i)/sqrt(2)], [(1+i)/sqrt(2), 0]]
    // rot(G, theta) = cos(theta/2)*I - i*sin(theta/2)*G
    // For theta=pi/2: cos(pi/4)*I - i*sin(pi/4)*G
    let c = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let ic = Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
    // G = (X+Y)/sqrt(2) = [[0, (1-i)/sqrt(2)], [(1+i)/sqrt(2), 0]]
    let g01 = Complex64::new(0.5, -0.5) * Complex64::new(std::f64::consts::SQRT_2, 0.0);
    let g10 = Complex64::new(0.5, 0.5) * Complex64::new(std::f64::consts::SQRT_2, 0.0);
    // Result: cos(pi/4)*I - i*sin(pi/4)*G
    let m00 = c;
    let m01 = ic * g01;
    let m10 = ic * g10;
    let m11 = c;
    Array2::from_shape_vec((2, 2), vec![m00, m01, m10, m11]).unwrap()
}
Gate::ISWAP => {
    // |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
    let mut m = Array2::zeros((4, 4));
    m[[0, 0]] = one;
    m[[1, 2]] = i;
    m[[2, 1]] = i;
    m[[3, 3]] = one;
    m
}
```

**Step 4: Run tests**

Run: `cargo test --test test_new_gates`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add SqrtX, SqrtY, SqrtW, ISWAP gate variants"
```

---

### Task 3: Add FSim gate variant

**Files:**
- Modify: `src/gate.rs` (add FSim variant + matrix)
- Add tests to: `tests/test_new_gates.rs`

**Step 1: Write failing tests**

Append to `tests/test_new_gates.rs`:
```rust
#[test]
fn test_fsim_matrix() {
    use std::f64::consts::PI;
    let gate = Gate::FSim(PI / 2.0, PI / 6.0);
    let m = gate.matrix(2);
    assert_eq!(gate.num_sites(2), 2);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    // FSim(pi/2, pi/6):
    // [[1, 0, 0, 0], [0, cos(pi/2), -i*sin(pi/2), 0], [0, -i*sin(pi/2), cos(pi/2), 0], [0, 0, 0, e^(-i*pi/6)]]
    assert_abs_diff_eq!(m[[0, 0]].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[1, 1]].re, 0.0, epsilon = 1e-10); // cos(pi/2) = 0
    assert_abs_diff_eq!(m[[1, 2]].im, -1.0, epsilon = 1e-10); // -i*sin(pi/2) = -i
    assert_abs_diff_eq!(m[[2, 1]].im, -1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[2, 2]].re, 0.0, epsilon = 1e-10);
    let exp_phi = Complex64::from_polar(1.0, -PI / 6.0);
    assert_abs_diff_eq!(m[[3, 3]].re, exp_phi.re, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[3, 3]].im, exp_phi.im, epsilon = 1e-10);
    // Off-diagonals should be zero
    assert_abs_diff_eq!(m[[0, 1]].norm(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(m[[3, 0]].norm(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_fsim_is_unitary() {
    use std::f64::consts::PI;
    let m = Gate::FSim(PI / 3.0, PI / 4.0).matrix(2);
    let mh = m.t().mapv(|x| x.conj());
    let id = m.dot(&mh);
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(id[[i, j]].re, expected, epsilon = 1e-10);
            assert_abs_diff_eq!(id[[i, j]].im, 0.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_fsim_num_sites() {
    use std::f64::consts::PI;
    assert_eq!(Gate::FSim(PI, PI).num_sites(2), 2);
}

#[test]
fn test_fsim_not_diagonal() {
    use std::f64::consts::PI;
    assert!(!Gate::FSim(PI / 2.0, PI / 6.0).is_diagonal());
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test --test test_new_gates`
Expected: FAIL — FSim variant doesn't exist.

**Step 3: Implement FSim**

In `src/gate.rs`, add to enum after `ISWAP`:
```rust
FSim(f64, f64),
```

In `num_sites()`:
```rust
Gate::SWAP | Gate::ISWAP | Gate::FSim(_, _) => 2,
```

In `qubit_matrix()`, add:
```rust
Gate::FSim(theta, phi) => {
    let cos_t = Complex64::new(theta.cos(), 0.0);
    let neg_i_sin_t = Complex64::new(0.0, -theta.sin());
    let exp_neg_i_phi = Complex64::from_polar(1.0, -phi);
    let mut m = Array2::zeros((4, 4));
    m[[0, 0]] = one;
    m[[1, 1]] = cos_t;
    m[[1, 2]] = neg_i_sin_t;
    m[[2, 1]] = neg_i_sin_t;
    m[[2, 2]] = cos_t;
    m[[3, 3]] = exp_neg_i_phi;
    m
}
```

**Step 4: Run tests**

Run: `cargo test --test test_new_gates`
Expected: All pass.

**Step 5: Run full test suite**

Run: `cargo test`
Expected: All pass.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add FSim(theta, phi) two-qubit gate variant"
```

---

### Task 4: Create `src/easybuild.rs` — entanglement layouts and deterministic circuits

**Files:**
- Create: `src/easybuild.rs`
- Modify: `src/lib.rs` (add module + exports)
- Create: `tests/test_easybuild.rs`

**Step 1: Write failing tests for layouts and QFT**

Create `tests/test_easybuild.rs`:
```rust
use yao_rs::easybuild::*;
use yao_rs::{Gate, apply, State, circuit_to_einsum};
use approx::assert_abs_diff_eq;
use num_complex::Complex64;

#[test]
fn test_pair_ring() {
    assert_eq!(pair_ring(4), vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    assert_eq!(pair_ring(2), vec![(0, 1), (1, 0)]);
}

#[test]
fn test_pair_square_non_periodic() {
    let pairs = pair_square(2, 2, false);
    // 2x2 grid, non-periodic: should have 4 edges
    assert_eq!(pairs.len(), 4);
}

#[test]
fn test_pair_square_periodic() {
    let pairs = pair_square(2, 2, true);
    // 2x2 grid, periodic: should have 8 edges
    assert_eq!(pairs.len(), 8);
}

#[test]
fn test_qft_circuit_uniform_superposition() {
    let n = 3;
    let circuit = qft_circuit(n);
    let state = State::zero_state(&vec![2; n]);
    let result = apply(&circuit, &state);
    let expected_amp = 1.0 / (8.0_f64).sqrt();
    for i in 0..8 {
        assert_abs_diff_eq!(result.data[i].re, expected_amp, epsilon = 1e-10);
        assert_abs_diff_eq!(result.data[i].im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_qft_circuit_norm_preserved() {
    let n = 4;
    let circuit = qft_circuit(n);
    let state = State::product_state(&vec![2; n], &[0, 1, 0, 1]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_general_u2_is_unitary() {
    let gates = general_u2(0, 1.0, 0.5, 0.3);
    // Build a 1-qubit circuit from it
    let circuit = yao_rs::Circuit::new(vec![2], gates).unwrap();
    let state = State::zero_state(&vec![2]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_general_u4_is_unitary() {
    let params: [f64; 15] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    let gates = general_u4(0, &params);
    let circuit = yao_rs::Circuit::new(vec![2, 2], gates).unwrap();
    let state = State::zero_state(&vec![2, 2]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_variational_circuit_structure() {
    let n = 4;
    let nlayer = 2;
    let pairs = pair_ring(n);
    let circuit = variational_circuit(n, nlayer, &pairs);
    assert_eq!(circuit.num_sites(), n);
    // All zero angles: output should equal input
    let state = State::zero_state(&vec![2; n]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_hadamard_test_circuit() {
    let n_u = 1;  // 1-qubit unitary
    // Use X gate as the unitary (custom matrix)
    let x_matrix = Gate::X.matrix(2);
    let u = Gate::Custom { matrix: x_matrix, is_diagonal: false, label: "X".to_string() };
    let circuit = hadamard_test_circuit(u, 0.0);
    assert_eq!(circuit.num_sites(), n_u + 1); // ancilla + unitary qubits
    let state = State::zero_state(&vec![2; n_u + 1]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_swap_test_identical_states() {
    // For identical input states, ancilla should measure |0⟩ with probability 1
    let circuit = swap_test_circuit(1, 2, 0.0);
    assert_eq!(circuit.num_sites(), 3); // 1 ancilla + 2*1 qubits
    let state = State::zero_state(&vec![2; 3]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test --test test_easybuild`
Expected: FAIL — module doesn't exist.

**Step 3: Implement easybuild module**

Create `src/easybuild.rs`:
```rust
use std::f64::consts::PI;
use crate::gate::Gate;
use crate::circuit::{Circuit, PositionedGate, put, control};

/// Ring entanglement layout: [(0,1), (1,2), ..., (n-1,0)]
pub fn pair_ring(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i, (i + 1) % n)).collect()
}

/// Square lattice entanglement layout on an m×n grid.
pub fn pair_square(m: usize, n: usize, periodic: bool) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let idx = |i: usize, j: usize| -> usize { i * n + j };

    // Horizontal edges
    for i in 0..m {
        for j in 0..n {
            if periodic || j + 1 < n {
                pairs.push((idx(i, j), idx(i, (j + 1) % n)));
            }
        }
    }
    // Vertical edges
    for i in 0..m {
        for j in 0..n {
            if periodic || i + 1 < m {
                pairs.push((idx(i, j), idx((i + 1) % m, j)));
            }
        }
    }
    pairs
}

/// Build an n-qubit QFT circuit.
pub fn qft_circuit(n: usize) -> Circuit {
    let mut gates: Vec<PositionedGate> = Vec::new();
    for i in 0..n {
        gates.push(put(vec![i], Gate::H));
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1u64 << (j + 1)) as f64;
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }
    for i in 0..(n / 2) {
        gates.push(put(vec![i, n - 1 - i], Gate::SWAP));
    }
    Circuit::new(vec![2; n], gates).unwrap()
}

/// General single-qubit gate: Rz(θ3) * Ry(θ2) * Rz(θ1)
/// Returns gates positioned on qubit `qubit`.
pub fn general_u2(qubit: usize, theta1: f64, theta2: f64, theta3: f64) -> Vec<PositionedGate> {
    vec![
        put(vec![qubit], Gate::Rz(theta1)),
        put(vec![qubit], Gate::Ry(theta2)),
        put(vec![qubit], Gate::Rz(theta3)),
    ]
}

/// General two-qubit gate (SU(4) decomposition with 15 parameters).
/// Gates are positioned on qubits `qubit0` and `qubit0+1`.
pub fn general_u4(qubit0: usize, params: &[f64; 15]) -> Vec<PositionedGate> {
    let q0 = qubit0;
    let q1 = qubit0 + 1;
    let mut gates = Vec::new();
    gates.extend(general_u2(q0, params[0], params[1], params[2]));
    gates.extend(general_u2(q1, params[3], params[4], params[5]));
    gates.push(control(vec![q1], vec![q0], Gate::X)); // CNOT(q1→q0)
    gates.push(put(vec![q0], Gate::Rz(params[6])));
    gates.push(put(vec![q1], Gate::Ry(params[7])));
    gates.push(control(vec![q0], vec![q1], Gate::X)); // CNOT(q0→q1)
    gates.push(put(vec![q1], Gate::Ry(params[8])));
    gates.push(control(vec![q1], vec![q0], Gate::X)); // CNOT(q1→q0)
    gates.extend(general_u2(q0, params[9], params[10], params[11]));
    gates.extend(general_u2(q1, params[12], params[13], params[14]));
    gates
}

/// Hardware-efficient variational circuit with angles initialized to zero.
pub fn variational_circuit(n: usize, nlayer: usize, pairs: &[(usize, usize)]) -> Circuit {
    let mut gates: Vec<PositionedGate> = Vec::new();

    // First rotor layer: Rx, Rz (no leading Rz)
    for q in 0..n {
        gates.push(put(vec![q], Gate::Rx(0.0)));
        gates.push(put(vec![q], Gate::Rz(0.0)));
    }

    for _ in 0..nlayer {
        // CNOT entangler layer
        for &(ctrl, tgt) in pairs {
            gates.push(control(vec![ctrl], vec![tgt], Gate::X));
        }
        // Full rotor: Rz, Rx, Rz
        for q in 0..n {
            gates.push(put(vec![q], Gate::Rz(0.0)));
            gates.push(put(vec![q], Gate::Rx(0.0)));
            gates.push(put(vec![q], Gate::Rz(0.0)));
        }
    }

    // Final rotor layer: Rz, Rx (no trailing Rz)
    // Note: the last layer above already added Rz, Rx, Rz.
    // We need to remove the trailing Rz from the last layer and not add extra.
    // Actually, re-reading the Julia code: the loop is 1..(nlayer+1).
    // Layer i=1: no entangler, rotorset(noleading=true, notrailing=false) = Rx, Rz
    // Layer i=2..nlayer: entangler + rotorset(noleading=false, notrailing=false) = Rz, Rx, Rz
    // Layer i=nlayer+1: entangler + rotorset(noleading=false, notrailing=true) = Rz, Rx
    // So the structure is correct. Let me redo:
    let mut gates: Vec<PositionedGate> = Vec::new();

    for layer in 0..=(nlayer) {
        // Entangler (skip first layer)
        if layer > 0 {
            for &(ctrl, tgt) in pairs {
                gates.push(control(vec![ctrl], vec![tgt], Gate::X));
            }
        }
        // Rotor set
        let noleading = layer == 0;
        let notrailing = layer == nlayer;
        for q in 0..n {
            if !noleading {
                gates.push(put(vec![q], Gate::Rz(0.0)));
            }
            gates.push(put(vec![q], Gate::Rx(0.0)));
            if !notrailing {
                gates.push(put(vec![q], Gate::Rz(0.0)));
            }
        }
    }

    Circuit::new(vec![2; n], gates).unwrap()
}

/// Hadamard test circuit. Takes a unitary gate and phase φ.
/// Returns a circuit on nqubits(U)+1 qubits (qubit 0 is ancilla).
pub fn hadamard_test_circuit(unitary: Gate, phi: f64) -> Circuit {
    let n_u = unitary.num_sites(2);
    let n = n_u + 1;
    let mut gates: Vec<PositionedGate> = Vec::new();

    gates.push(put(vec![0], Gate::H));
    gates.push(put(vec![0], Gate::Rz(phi)));
    // Controlled-U: control on qubit 0, targets on 1..n
    let targets: Vec<usize> = (1..n).collect();
    gates.push(PositionedGate::new(unitary, targets, vec![0], vec![true]));
    gates.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n], gates).unwrap()
}

/// Swap test circuit for computing overlap between nstate states of nbit qubits each.
/// Total qubits = nstate*nbit + 1 (qubit 0 is ancilla).
pub fn swap_test_circuit(nbit: usize, nstate: usize, phi: f64) -> Circuit {
    let n = nstate * nbit + 1;
    let mut gates: Vec<PositionedGate> = Vec::new();

    gates.push(put(vec![0], Gate::H));
    gates.push(put(vec![0], Gate::Rz(phi)));

    // Controlled-SWAP between consecutive state registers
    for k in 0..(nstate - 1) {
        for i in 0..nbit {
            let q1 = 1 + k * nbit + i;
            let q2 = 1 + (k + 1) * nbit + i;
            gates.push(PositionedGate::new(
                Gate::SWAP,
                vec![q1, q2],
                vec![0],
                vec![true],
            ));
        }
    }

    gates.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n], gates).unwrap()
}

/// Phase estimation circuit.
/// Takes a unitary gate, n_reg register qubits, n_b target qubits.
/// Total qubits = n_reg + n_b.
pub fn phase_estimation_circuit(unitary: Gate, n_reg: usize, n_b: usize) -> Circuit {
    use ndarray::Array2;
    let n = n_reg + n_b;
    let mut gates: Vec<PositionedGate> = Vec::new();

    // H on all register qubits
    for i in 0..n_reg {
        gates.push(put(vec![i], Gate::H));
    }

    // Controlled-U^(2^i) for each register qubit
    let targets: Vec<usize> = (n_reg..n).collect();
    let mut u_matrix = unitary.matrix(2);
    for i in 0..n_reg {
        let u_gate = Gate::Custom {
            matrix: u_matrix.clone(),
            is_diagonal: false,
            label: format!("U^{}", 1 << i),
        };
        gates.push(PositionedGate::new(u_gate, targets.clone(), vec![i], vec![true]));
        // Square the matrix for next iteration
        if i + 1 < n_reg {
            u_matrix = u_matrix.dot(&u_matrix);
        }
    }

    // Inverse QFT on register qubits (reversed order, negative phases)
    for i in (0..n_reg).rev() {
        for j in ((i + 1)..n_reg).rev() {
            let theta = -2.0 * PI / (1u64 << (j - i + 1)) as f64;
            gates.push(control(vec![j], vec![i], Gate::Phase(theta)));
        }
        gates.push(put(vec![i], Gate::H));
    }

    // Reverse register qubit order with SWAPs
    for i in 0..(n_reg / 2) {
        gates.push(put(vec![i, n_reg - 1 - i], Gate::SWAP));
    }

    Circuit::new(vec![2; n], gates).unwrap()
}
```

Add to `src/lib.rs`:
```rust
pub mod easybuild;
```

**Step 4: Run tests**

Run: `cargo test --test test_easybuild`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add easybuild module with deterministic circuit builders"
```

---

### Task 5: Add random circuit builders (rand_supremacy2d, rand_google53)

**Files:**
- Modify: `Cargo.toml` (add `rand` dependency)
- Modify: `src/easybuild.rs` (add random circuit functions)
- Add tests to: `tests/test_easybuild.rs`

**Step 1: Add rand dependency**

In `Cargo.toml`, add:
```toml
[dependencies]
rand = "0.8"
```

**Step 2: Write failing tests**

Append to `tests/test_easybuild.rs`:
```rust
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn test_rand_supremacy2d_structure() {
    let mut rng = StdRng::seed_from_u64(42);
    let circuit = rand_supremacy2d(3, 3, 5, &mut rng);
    assert_eq!(circuit.num_sites(), 9);
    let state = State::zero_state(&vec![2; 9]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rand_google53_structure() {
    let mut rng = StdRng::seed_from_u64(42);
    let circuit = rand_google53(4, 10, &mut rng);
    assert_eq!(circuit.num_sites(), 10);
    let state = State::zero_state(&vec![2; 10]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rand_supremacy2d_deterministic_with_seed() {
    let mut rng1 = StdRng::seed_from_u64(123);
    let mut rng2 = StdRng::seed_from_u64(123);
    let c1 = rand_supremacy2d(2, 2, 3, &mut rng1);
    let c2 = rand_supremacy2d(2, 2, 3, &mut rng2);
    assert_eq!(c1.gates.len(), c2.gates.len());
}
```

**Step 3: Implement random circuits**

Append to `src/easybuild.rs`:
```rust
use rand::Rng;

/// Pair pattern for 2D supremacy circuits (CZ entangler layout).
fn pair_supremacy(nx: usize, ny: usize) -> Vec<Vec<(usize, usize)>> {
    let idx = |i: usize, j: usize| -> usize { i * ny + j };
    let mut patterns: Vec<Vec<(usize, usize)>> = Vec::new();

    // 8 patterns: alternating horizontal and vertical CZ layers
    // Horizontal even columns
    let mut p = Vec::new();
    for i in 0..nx {
        for j in (0..ny - 1).step_by(2) {
            p.push((idx(i, j), idx(i, j + 1)));
        }
    }
    patterns.push(p);

    // Horizontal odd columns
    let mut p = Vec::new();
    for i in 0..nx {
        for j in (1..ny - 1).step_by(2) {
            p.push((idx(i, j), idx(i, j + 1)));
        }
    }
    patterns.push(p);

    // Vertical even rows
    let mut p = Vec::new();
    for i in (0..nx - 1).step_by(2) {
        for j in 0..ny {
            p.push((idx(i, j), idx(i + 1, j)));
        }
    }
    patterns.push(p);

    // Vertical odd rows
    let mut p = Vec::new();
    for i in (1..nx - 1).step_by(2) {
        for j in 0..ny {
            p.push((idx(i, j), idx(i + 1, j)));
        }
    }
    patterns.push(p);

    patterns
}

/// Random 2D quantum supremacy circuit.
pub fn rand_supremacy2d(nx: usize, ny: usize, depth: usize, rng: &mut impl Rng) -> Circuit {
    let nbits = nx * ny;
    let mut gates: Vec<PositionedGate> = Vec::new();
    let patterns = pair_supremacy(nx, ny);
    let gateset = [Gate::T, Gate::SqrtX, Gate::SqrtY];

    // Initial layer of H gates
    for i in 0..nbits {
        gates.push(put(vec![i], Gate::H));
    }

    let mut prev_gates: Vec<Option<usize>> = vec![None; nbits]; // index into gateset
    let mut has_t: Vec<bool> = vec![false; nbits];

    for layer in 0..depth.saturating_sub(2) {
        let pattern = &patterns[layer % patterns.len()];

        // CZ entangler
        for &(i, j) in pattern {
            gates.push(control(vec![i], vec![j], Gate::Z));
        }

        // Single-qubit gates on non-entangled qubits
        let mut entangled: Vec<bool> = vec![false; nbits];
        for &(i, j) in pattern {
            entangled[i] = true;
            entangled[j] = true;
        }

        for q in 0..nbits {
            if !entangled[q] {
                let gate_idx = if !has_t[q] {
                    has_t[q] = true;
                    0 // T gate first
                } else {
                    // Pick random gate different from previous
                    loop {
                        let idx = rng.gen_range(0..gateset.len());
                        if Some(idx) != prev_gates[q] {
                            break idx;
                        }
                    }
                };
                prev_gates[q] = Some(gate_idx);
                gates.push(put(vec![q], gateset[gate_idx].clone()));
            }
        }
    }

    // Final layer of H gates
    if depth > 1 {
        for i in 0..nbits {
            gates.push(put(vec![i], Gate::H));
        }
    }

    Circuit::new(vec![2; nbits], gates).unwrap()
}

/// Lattice53 structure for Google Sycamore topology.
struct Lattice53 {
    labels: Vec<Vec<usize>>, // 5x12 grid
    nbits: usize,
}

impl Lattice53 {
    fn new(nbits: usize) -> Self {
        let mut config = vec![vec![true; 12]; 5];
        // Remove specific sites to match Sycamore topology
        for j in (1..12).step_by(2) {
            config[4][j] = false;
        }
        config[0][6] = false;

        let mut labels = vec![vec![0usize; 12]; 5];
        let mut k = 0;
        for j in 0..12 {
            for i in 0..5 {
                if config[i][j] && k < nbits {
                    k += 1;
                    labels[i][j] = k;
                }
            }
        }

        Lattice53 { labels, nbits: k }
    }

    fn get(&self, i: isize, j: isize) -> usize {
        if i >= 0 && i < 5 && j >= 0 && j < 12 {
            self.labels[i as usize][j as usize]
        } else {
            0
        }
    }

    fn upperright(&self, i: isize, j: isize) -> usize {
        self.get(i - (j as usize % 2) as isize, j + 1)
    }

    fn lowerright(&self, i: isize, j: isize) -> usize {
        self.get(i + ((j - 1).rem_euclid(2)) as isize, j + 1)
    }

    fn pattern(&self, chr: char) -> Vec<(usize, usize)> {
        let mut res = Vec::new();
        let di: usize = if chr > 'D' { 2 } else { 1 };
        let dj: usize = if chr > 'D' { 1 } else { 2 };
        let j0: usize = 1 + dj.min(((chr as u8 - b'A') % 2) as usize).min(dj - 1);
        let use_lowerright = 'C' <= chr && chr <= 'F';

        let mut j = j0;
        while j <= 12 {
            let i0 = if chr > 'D' {
                ((chr as u8 - b'D') as usize + (j - (if chr >= 'G' { 1 } else { 0 })) / 2) % 2
            } else {
                1
            };
            let mut i = i0;
            while i <= 5 {
                let src = self.get(i as isize - 1, j as isize - 1);
                let dest = if use_lowerright {
                    self.lowerright(i as isize - 1, j as isize - 1)
                } else {
                    self.upperright(i as isize - 1, j as isize - 1)
                };
                if src != 0 && dest != 0 {
                    res.push((src - 1, dest - 1)); // 0-indexed
                }
                i += di;
            }
            j += dj;
        }
        res
    }
}

/// Random Google 53-qubit (Sycamore) circuit.
pub fn rand_google53(depth: usize, nbits: usize, rng: &mut impl Rng) -> Circuit {
    let lattice = Lattice53::new(nbits);
    let actual_nbits = lattice.nbits;
    let mut gates: Vec<PositionedGate> = Vec::new();
    let pattern_cycle = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B'];
    let single_gates = [Gate::SqrtX, Gate::SqrtY, Gate::SqrtW];

    for layer in 0..depth {
        let pattern_char = pattern_cycle[layer % pattern_cycle.len()];

        // Random single-qubit gates on all qubits
        for q in 0..actual_nbits {
            let idx = rng.gen_range(0..single_gates.len());
            gates.push(put(vec![q], single_gates[idx].clone()));
        }

        // FSim entanglers on pattern
        let pairs = lattice.pattern(pattern_char);
        for (i, j) in pairs {
            if i < actual_nbits && j < actual_nbits {
                gates.push(put(
                    vec![i, j],
                    Gate::FSim(PI / 2.0, PI / 6.0),
                ));
            }
        }
    }

    Circuit::new(vec![2; actual_nbits], gates).unwrap()
}
```

**Step 4: Run tests**

Run: `cargo test --test test_easybuild`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add random supremacy and Google53 circuit builders"
```

---

### Task 6: Add JSON serialization (serde)

**Files:**
- Modify: `Cargo.toml` (add serde, serde_json)
- Create: `src/json.rs`
- Modify: `src/lib.rs` (add module + exports)
- Create: `tests/test_json.rs`

**Step 1: Add dependencies**

In `Cargo.toml`:
```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Step 2: Write failing tests**

Create `tests/test_json.rs`:
```rust
use yao_rs::{Gate, Circuit, put, control};
use yao_rs::json::{circuit_to_json, circuit_from_json};
use yao_rs::circuit::PositionedGate;
use ndarray::Array2;
use num_complex::Complex64;
use approx::assert_abs_diff_eq;

#[test]
fn test_roundtrip_named_gates() {
    let gates = vec![
        put(vec![0], Gate::H),
        put(vec![1], Gate::X),
        put(vec![0], Gate::Phase(1.5)),
        put(vec![1], Gate::Rx(0.5)),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.num_sites(), 2);
    assert_eq!(restored.gates.len(), 4);
}

#[test]
fn test_roundtrip_controlled_gate() {
    let gates = vec![
        control(vec![0], vec![1], Gate::X),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.gates[0].control_locs, vec![0]);
    assert_eq!(restored.gates[0].target_locs, vec![1]);
    assert_eq!(restored.gates[0].control_configs, vec![true]);
}

#[test]
fn test_roundtrip_custom_gate() {
    let matrix = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
    ]).unwrap();
    let gates = vec![
        put(vec![0], Gate::Custom {
            matrix: matrix.clone(),
            is_diagonal: false,
            label: "MyGate".to_string(),
        }),
    ];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();

    if let Gate::Custom { matrix: m, is_diagonal, label } = &restored.gates[0].gate {
        assert_eq!(label, "MyGate");
        assert_eq!(*is_diagonal, false);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(m[[i,j]].re, matrix[[i,j]].re, epsilon = 1e-15);
                assert_abs_diff_eq!(m[[i,j]].im, matrix[[i,j]].im, epsilon = 1e-15);
            }
        }
    } else {
        panic!("Expected Custom gate");
    }
}

#[test]
fn test_roundtrip_fsim() {
    let gates = vec![
        put(vec![0, 1], Gate::FSim(1.0, 0.5)),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    if let Gate::FSim(theta, phi) = &restored.gates[0].gate {
        assert_abs_diff_eq!(*theta, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(*phi, 0.5, epsilon = 1e-15);
    } else {
        panic!("Expected FSim gate");
    }
}

#[test]
fn test_json_structure() {
    let gates = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["num_qubits"], 1);
    assert_eq!(parsed["gates"][0]["gate"], "H");
    assert_eq!(parsed["gates"][0]["targets"][0], 0);
}
```

**Step 3: Implement json module**

Create `src/json.rs`:
```rust
use serde::{Serialize, Deserialize};
use ndarray::Array2;
use num_complex::Complex64;

use crate::gate::Gate;
use crate::circuit::{Circuit, PositionedGate};

#[derive(Serialize, Deserialize)]
struct CircuitJson {
    num_qubits: usize,
    gates: Vec<GateJson>,
}

#[derive(Serialize, Deserialize)]
struct GateJson {
    gate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Vec<f64>>,
    targets: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    controls: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    control_configs: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matrix: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_diagonal: Option<bool>,
}

/// Serialize a Circuit to a JSON string.
pub fn circuit_to_json(circuit: &Circuit) -> String {
    let gates: Vec<GateJson> = circuit.gates.iter().map(|pg| {
        let (gate_name, params, label, matrix, is_diagonal) = match &pg.gate {
            Gate::X => ("X".to_string(), None, None, None, None),
            Gate::Y => ("Y".to_string(), None, None, None, None),
            Gate::Z => ("Z".to_string(), None, None, None, None),
            Gate::H => ("H".to_string(), None, None, None, None),
            Gate::S => ("S".to_string(), None, None, None, None),
            Gate::T => ("T".to_string(), None, None, None, None),
            Gate::SWAP => ("SWAP".to_string(), None, None, None, None),
            Gate::SqrtX => ("SqrtX".to_string(), None, None, None, None),
            Gate::SqrtY => ("SqrtY".to_string(), None, None, None, None),
            Gate::SqrtW => ("SqrtW".to_string(), None, None, None, None),
            Gate::ISWAP => ("ISWAP".to_string(), None, None, None, None),
            Gate::Phase(theta) => ("Phase".to_string(), Some(vec![*theta]), None, None, None),
            Gate::Rx(theta) => ("Rx".to_string(), Some(vec![*theta]), None, None, None),
            Gate::Ry(theta) => ("Ry".to_string(), Some(vec![*theta]), None, None, None),
            Gate::Rz(theta) => ("Rz".to_string(), Some(vec![*theta]), None, None, None),
            Gate::FSim(theta, phi) => ("FSim".to_string(), Some(vec![*theta, *phi]), None, None, None),
            Gate::Custom { matrix: m, is_diagonal: diag, label: lbl } => {
                let mat: Vec<Vec<[f64; 2]>> = (0..m.nrows())
                    .map(|i| (0..m.ncols()).map(|j| [m[[i, j]].re, m[[i, j]].im]).collect())
                    .collect();
                ("Custom".to_string(), None, Some(lbl.clone()), Some(mat), Some(*diag))
            }
        };

        let controls = if pg.control_locs.is_empty() { None } else { Some(pg.control_locs.clone()) };
        let control_configs = if pg.control_configs.is_empty() { None } else { Some(pg.control_configs.clone()) };

        GateJson {
            gate: gate_name,
            params,
            targets: pg.target_locs.clone(),
            controls,
            control_configs,
            label,
            matrix,
            is_diagonal,
        }
    }).collect();

    let cj = CircuitJson {
        num_qubits: circuit.num_sites(),
        gates,
    };

    serde_json::to_string_pretty(&cj).unwrap()
}

/// Deserialize a Circuit from a JSON string.
pub fn circuit_from_json(json: &str) -> Result<Circuit, String> {
    let cj: CircuitJson = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let n = cj.num_qubits;

    let mut gates: Vec<PositionedGate> = Vec::new();
    for gj in &cj.gates {
        let gate = match gj.gate.as_str() {
            "X" => Gate::X,
            "Y" => Gate::Y,
            "Z" => Gate::Z,
            "H" => Gate::H,
            "S" => Gate::S,
            "T" => Gate::T,
            "SWAP" => Gate::SWAP,
            "SqrtX" => Gate::SqrtX,
            "SqrtY" => Gate::SqrtY,
            "SqrtW" => Gate::SqrtW,
            "ISWAP" => Gate::ISWAP,
            "Phase" => {
                let p = gj.params.as_ref().ok_or("Phase gate missing params")?;
                Gate::Phase(p[0])
            }
            "Rx" => {
                let p = gj.params.as_ref().ok_or("Rx gate missing params")?;
                Gate::Rx(p[0])
            }
            "Ry" => {
                let p = gj.params.as_ref().ok_or("Ry gate missing params")?;
                Gate::Ry(p[0])
            }
            "Rz" => {
                let p = gj.params.as_ref().ok_or("Rz gate missing params")?;
                Gate::Rz(p[0])
            }
            "FSim" => {
                let p = gj.params.as_ref().ok_or("FSim gate missing params")?;
                Gate::FSim(p[0], p[1])
            }
            "Custom" => {
                let mat_data = gj.matrix.as_ref().ok_or("Custom gate missing matrix")?;
                let nrows = mat_data.len();
                let ncols = if nrows > 0 { mat_data[0].len() } else { 0 };
                let mut matrix = Array2::zeros((nrows, ncols));
                for i in 0..nrows {
                    for j in 0..ncols {
                        matrix[[i, j]] = Complex64::new(mat_data[i][j][0], mat_data[i][j][1]);
                    }
                }
                Gate::Custom {
                    matrix,
                    is_diagonal: gj.is_diagonal.unwrap_or(false),
                    label: gj.label.clone().unwrap_or_default(),
                }
            }
            other => return Err(format!("Unknown gate type: {}", other)),
        };

        let control_locs = gj.controls.clone().unwrap_or_default();
        let control_configs = gj.control_configs.clone().unwrap_or_default();

        gates.push(PositionedGate::new(gate, gj.targets.clone(), control_locs, control_configs));
    }

    Circuit::new(vec![2; n], gates).map_err(|e| e.to_string())
}
```

Add to `src/lib.rs`:
```rust
pub mod json;
pub use json::{circuit_to_json, circuit_from_json};
```

**Step 4: Run tests**

Run: `cargo test --test test_json`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add JSON serialization for circuits (serde)"
```

---

### Task 7: Run full test suite and fix any issues

**Step 1: Run everything**

Run: `cargo test`
Expected: All pass.

**Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

**Step 3: Commit if fixes needed**

```bash
git add -A
git commit -m "fix: resolve clippy warnings and test issues"
```

---

### Task 8: Create typst visualization script

**Files:**
- Create: `visualization/circuit.typ`

**Step 1: Create visualization directory and script**

Create `visualization/circuit.typ`:
```typst
#import "@preview/quill:0.5.0": *

/// Render a quantum circuit from parsed JSON data.
///
/// Parameters:
/// - data: Parsed JSON object with `num_qubits` and `gates` fields
/// - gate-style: Function (gate-name: string) → dictionary with optional keys:
///   - label: content to display (default: gate name)
///   - fill: fill color (default: none)
///   - stroke: stroke style (default: none)
#let render-circuit-impl(data, gate-style: gate-name => (label: gate-name)) = {
  let n = data.num_qubits
  let ops = ()

  for g in data.gates {
    let name = g.gate
    let key = if name == "Custom" { g.at("label", default: "?") } else { name }
    let style = gate-style(key)
    let lbl = style.at("label", default: key)
    let fill = style.at("fill", default: none)

    let targets = g.targets
    let controls = g.at("controls", default: ())
    let params = g.at("params", default: ())

    // Format parametric label
    let display-label = if params != () and name != "Custom" {
      let param-str = params.map(p => str(calc.round(p, digits: 3))).join(", ")
      [#lbl (#param-str)]
    } else {
      lbl
    }

    if controls != () and controls.len() > 0 {
      // Controlled gate
      if name == "X" and targets.len() == 1 {
        // CNOT: use tq.cx for each control
        for ctrl in controls {
          ops += (tq.cx(ctrl, targets.at(0)),)
        }
      } else if name == "Z" and targets.len() == 1 {
        ops += (tq.cz(controls.at(0), targets.at(0)),)
      } else if name == "SWAP" and targets.len() == 2 {
        // Controlled-SWAP (Fredkin)
        ops += (tq.gate($"SWAP"$, targets.at(0), targets.at(1), ctrl: controls.at(0), fill: fill),)
      } else {
        // Generic controlled gate
        let all-targets = targets
        ops += (tq.gate(display-label, ..all-targets, ctrl: controls.at(0), fill: fill),)
      }
    } else {
      // Non-controlled gates
      if name == "H" and targets.len() == 1 { ops += (tq.h(targets.at(0)),) }
      else if name == "X" and targets.len() == 1 { ops += (tq.x(targets.at(0)),) }
      else if name == "Y" and targets.len() == 1 { ops += (tq.y(targets.at(0)),) }
      else if name == "Z" and targets.len() == 1 { ops += (tq.z(targets.at(0)),) }
      else if name == "S" and targets.len() == 1 { ops += (tq.s(targets.at(0)),) }
      else if name == "T" and targets.len() == 1 { ops += (tq.t(targets.at(0)),) }
      else if name == "SWAP" and targets.len() == 2 { ops += (tq.swap(targets.at(0), targets.at(1)),) }
      else if name == "Rx" { ops += (tq.rx(params.at(0), targets.at(0)),) }
      else if name == "Ry" { ops += (tq.ry(params.at(0), targets.at(0)),) }
      else if name == "Rz" { ops += (tq.rz(params.at(0), targets.at(0)),) }
      else if name == "Phase" { ops += (tq.p(params.at(0), targets.at(0)),) }
      else if targets.len() == 1 {
        ops += (tq.gate(display-label, targets.at(0), fill: fill),)
      } else {
        ops += (tq.mqgate(display-label, ..targets, fill: fill),)
      }
    }
  }

  quantum-circuit(..tq.build(n, ..ops))
}

/// Render a quantum circuit from a JSON file.
///
/// Parameters:
/// - filename: Path to the JSON file
/// - gate-style: Styling function (see render-circuit-impl)
#let render-circuit(filename, gate-style: gate-name => (label: gate-name)) = {
  let data = json(filename)
  render-circuit-impl(data, gate-style: gate-style)
}
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: add typst visualization script using quill tequila"
```

---

### Task 9: Add boundary conditions to einsum

**Files:**
- Modify: `src/einsum.rs` (add `circuit_to_einsum_with_boundary`)
- Modify: `src/lib.rs` (export)
- Create: `tests/test_boundary.rs`

**Step 1: Write failing tests**

Create `tests/test_boundary.rs`:
```rust
use yao_rs::{Gate, Circuit, State, put, control, apply};
use yao_rs::einsum::{circuit_to_einsum_with_boundary};
use approx::assert_abs_diff_eq;
use num_complex::Complex64;

#[test]
fn test_all_pinned_amplitude() {
    // <0|H|0> = 1/sqrt(2)
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    // Contract manually: should be a scalar
    assert!(tn.code.get_output().is_empty());
}

#[test]
fn test_none_pinned_matches_apply() {
    // No final state pinned = full output state
    let circuit = Circuit::new(vec![2, 2], vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
    ]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    // Output should have 2 open legs
    assert_eq!(tn.code.get_output().len(), 2);
}

#[test]
fn test_partial_pinning() {
    // Pin qubit 0 to |0⟩, leave qubit 1 open
    let circuit = Circuit::new(vec![2, 2], vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
    ]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    // Output should have 1 open leg (qubit 1)
    assert_eq!(tn.code.get_output().len(), 1);
}

#[test]
fn test_boundary_vs_apply_identity() {
    // Identity circuit: <0|I|0> = 1
    let circuit = Circuit::new(vec![2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    assert!(tn.code.get_output().is_empty());
    // The tensors are just two [1, 0] vectors, contracting to 1.0
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test --test test_boundary`
Expected: FAIL — function doesn't exist.

**Step 3: Implement boundary conditions**

In `src/einsum.rs`, add:
```rust
use ndarray::{Array1, ArrayD, IxDyn};

/// Convert a quantum circuit to a tensor network with boundary conditions.
///
/// Initial state is always |0...0⟩ (each qubit gets a [1, 0, ...] tensor).
/// Qubits listed in `final_state` are pinned to |0⟩ on output.
/// Unpinned output qubits remain as open legs.
pub fn circuit_to_einsum_with_boundary(circuit: &Circuit, final_state: &[usize]) -> TensorNetwork {
    let n = circuit.num_sites();

    // Start with the base tensor network
    let mut current_labels: Vec<usize> = (0..n).collect();
    let mut next_label: usize = n;
    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        size_dict.insert(i, circuit.dims[i]);
    }

    let mut all_ixs: Vec<Vec<usize>> = Vec::new();
    let mut all_tensors: Vec<ArrayD<Complex64>> = Vec::new();

    // Add initial state tensors: [1, 0, ..., 0] for each qubit
    for i in 0..n {
        let d = circuit.dims[i];
        let mut init_tensor = Array1::<Complex64>::zeros(d);
        init_tensor[0] = Complex64::new(1.0, 0.0);
        all_tensors.push(init_tensor.into_shape_with_order(IxDyn(&[d])).unwrap());
        all_ixs.push(vec![i]); // connects to input label i
    }

    // Process gates (same as circuit_to_einsum)
    for pg in &circuit.gates {
        let (tensor, _legs) = gate_to_tensor(pg, &circuit.dims);
        let all_locs = pg.all_locs();
        let has_controls = !pg.control_locs.is_empty();
        let is_diagonal = pg.gate.is_diagonal() && !has_controls;

        if is_diagonal {
            let tensor_ixs: Vec<usize> = pg.target_locs.iter()
                .map(|&loc| current_labels[loc])
                .collect();
            all_ixs.push(tensor_ixs);
        } else {
            let mut tensor_ixs: Vec<usize> = Vec::new();
            let mut new_labels: Vec<usize> = Vec::new();
            for &loc in &all_locs {
                let new_label = next_label;
                next_label += 1;
                size_dict.insert(new_label, circuit.dims[loc]);
                new_labels.push(new_label);
            }
            tensor_ixs.extend(&new_labels);
            for &loc in &all_locs {
                tensor_ixs.push(current_labels[loc]);
            }
            for (i, &loc) in all_locs.iter().enumerate() {
                current_labels[loc] = new_labels[i];
            }
            all_ixs.push(tensor_ixs);
        }
        all_tensors.push(tensor);
    }

    // Add final state tensors for pinned qubits
    let final_set: std::collections::HashSet<usize> = final_state.iter().copied().collect();
    for &q in final_state {
        let d = circuit.dims[q];
        let mut final_tensor = Array1::<Complex64>::zeros(d);
        final_tensor[0] = Complex64::new(1.0, 0.0);
        all_tensors.push(final_tensor.into_shape_with_order(IxDyn(&[d])).unwrap());
        all_ixs.push(vec![current_labels[q]]); // connects to output label of qubit q
    }

    // Output labels = unpinned qubits' final labels
    let output_labels: Vec<usize> = (0..n)
        .filter(|q| !final_set.contains(q))
        .map(|q| current_labels[q])
        .collect();

    TensorNetwork {
        code: EinCode::new(all_ixs, output_labels),
        tensors: all_tensors,
        size_dict,
    }
}
```

Add to `src/lib.rs` exports:
```rust
pub use einsum::{circuit_to_einsum, circuit_to_einsum_with_boundary, TensorNetwork};
```

**Step 4: Run tests**

Run: `cargo test --test test_boundary`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add circuit_to_einsum_with_boundary for initial/final state pinning"
```

---

### Task 10: Add feature-gated torch contractor

**Files:**
- Modify: `Cargo.toml` (add tch optional dependency + feature)
- Create: `src/torch_contractor.rs`
- Modify: `src/lib.rs` (conditional module)
- Modify: `docs/src/SUMMARY.md` (add torch chapter)
- Create: `docs/src/torch-interop.md`

**Step 1: Update Cargo.toml**

```toml
[features]
default = []
torch = ["dep:tch"]

[dependencies]
tch = { version = "0.17", optional = true }
```

**Step 2: Implement torch contractor**

Create `src/torch_contractor.rs`:
```rust
//! Feature-gated tensor network contractor using libtorch (tch-rs).
//!
//! Enable with `cargo build --features torch`.

use tch::{Tensor, Device, Kind};
use num_complex::Complex64;
use crate::einsum::TensorNetwork;

/// Convert an ndarray complex tensor to a tch::Tensor on the given device.
fn ndarray_to_tch(tensor: &ndarray::ArrayD<Complex64>, device: Device) -> Tensor {
    let shape: Vec<i64> = tensor.shape().iter().map(|&s| s as i64).collect();
    // Flatten to contiguous slice of [re, im] pairs
    let data: Vec<f64> = tensor.iter()
        .flat_map(|c| vec![c.re, c.im])
        .collect();
    Tensor::from_slice(&data)
        .reshape(&[&shape[..], &[2]].concat())
        .view_as_complex()
        .to_device(device)
}

/// Contract a tensor network using libtorch.
///
/// # Arguments
/// * `tn` - The tensor network to contract (from `circuit_to_einsum` or `circuit_to_einsum_with_boundary`)
/// * `device` - The device to perform computation on (`Device::Cpu` or `Device::Cuda(0)`)
///
/// # Returns
/// The contracted result as a `tch::Tensor` (complex64).
pub fn contract(tn: &TensorNetwork, device: Device) -> Tensor {
    // Convert all tensors to tch
    let tch_tensors: Vec<Tensor> = tn.tensors.iter()
        .map(|t| ndarray_to_tch(t, device))
        .collect();

    // Build einsum string from the EinCode
    let ixs = tn.code.get_ixs();
    let output = tn.code.get_output();

    // Map usize labels to char labels for einsum
    let mut label_map: std::collections::HashMap<usize, char> = std::collections::HashMap::new();
    let mut next_char = b'a';
    let mut get_char = |label: usize, map: &mut std::collections::HashMap<usize, char>| -> char {
        *map.entry(label).or_insert_with(|| {
            let c = next_char as char;
            next_char += 1;
            c
        })
    };

    let mut einsum_parts: Vec<String> = Vec::new();
    for ix in &ixs {
        let s: String = ix.iter().map(|&l| get_char(l, &mut label_map)).collect();
        einsum_parts.push(s);
    }
    let output_str: String = output.iter().map(|&l| get_char(l, &mut label_map)).collect();
    let einsum_str = format!("{}->{}", einsum_parts.join(","), output_str);

    // For large networks, we need to contract pairwise following the optimized order.
    // For now, use tch's einsum directly (works for small networks).
    // TODO: Walk the contraction tree for large networks.
    Tensor::einsum(&einsum_str, &tch_tensors, None::<&[i64]>)
}
```

Add to `src/lib.rs`:
```rust
#[cfg(feature = "torch")]
pub mod torch_contractor;
```

**Step 3: Write mdBook chapter**

Create `docs/src/torch-interop.md`:
```markdown
# PyTorch Interop via libtorch

yao-rs supports tensor network contraction using libtorch (PyTorch's C++ backend) for CPU and GPU acceleration.

## Setup

1. Install libtorch (see [tch-rs documentation](https://github.com/LaurentMazare/tch-rs))
2. Build with the `torch` feature:

```bash
cargo build --features torch
```

## Usage

```rust
use yao_rs::{Gate, Circuit, put, control};
use yao_rs::einsum::circuit_to_einsum_with_boundary;
use yao_rs::torch_contractor::contract;
use tch::Device;

// Build a circuit
let circuit = Circuit::new(vec![2, 2, 2], vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
    control(vec![1], vec![2], Gate::X),
]).unwrap();

// Export to tensor network with all qubits pinned to |0⟩
let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1, 2]);

// Contract on CPU
let result = contract(&tn, Device::Cpu);
println!("Amplitude <000|C|000>: {:?}", result);

// Contract on GPU (if available)
let result_gpu = contract(&tn, Device::Cuda(0));
```

## How It Works

1. Each tensor in the network is converted to a `tch::Tensor` (complex64) on the target device
2. The einsum contraction specification from `omeco` defines how tensors are contracted
3. `tch::Tensor::einsum` executes the contraction using libtorch's optimized backend

## Notes

- GPU acceleration is most beneficial for large circuits (>20 qubits)
- The `tch` crate requires libtorch to be installed on the system
- Complex64 tensors are represented as real tensors with a trailing dimension of 2
```

Update `docs/src/SUMMARY.md` to include the new chapter.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add feature-gated libtorch contractor and mdBook docs"
```

---

### Task 11: Final integration test and cleanup

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All pass.

**Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

**Step 3: Run the QFT example**

Run: `cargo run --example qft`
Expected: Runs successfully.

**Step 4: Test JSON → typst pipeline manually**

Run a small test:
```bash
cargo test --test test_json -- --nocapture
```
Verify the JSON output can be saved and rendered by typst (manual check).

**Step 5: Final commit if any fixes**

```bash
git add -A
git commit -m "fix: final integration fixes"
```
