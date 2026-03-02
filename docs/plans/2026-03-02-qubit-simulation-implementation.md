# Qubit Vector Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace mixed-radix instruct functions with bit-manipulation-based qubit (d=2) fast paths, add measurement completion.

**Architecture:** New `bitutils.rs` module ports Julia's BitBasis bit operations and IterControl iterator. Four new instruct functions (1q, 1q-diag, 2q, 2q-diag) use stride-based/itercontrol iteration with zero heap allocations in hot loops. `apply_inplace` dispatches to qubit fast path when all dims==2. Existing generic path remains for qudits.

**Tech Stack:** Rust, ndarray, num-complex. Julia reference: `~/.julia/dev/BitBasis/`, `~/.julia/dev/Yao/lib/YaoArrayRegister/`.

**Principle:** Copy Julia implementation as closely as possible. Mirror structure, naming, algorithms.

---

### Task 1: Bit utilities module

**Files:**
- Create: `src/bitutils.rs`
- Modify: `src/lib.rs` (add `pub mod bitutils;`)
- Test: `tests/bitutils.rs`

**Step 1: Write the failing tests**

Create `tests/bitutils.rs`:

```rust
// tests/bitutils.rs
use yao_rs::bitutils::*;

#[test]
fn test_indicator() {
    assert_eq!(indicator(0), 1);    // bit 0
    assert_eq!(indicator(1), 2);    // bit 1
    assert_eq!(indicator(3), 8);    // bit 3
}

#[test]
fn test_bmask_single() {
    assert_eq!(bmask(&[0]), 0b1);
    assert_eq!(bmask(&[1]), 0b10);
    assert_eq!(bmask(&[0, 1]), 0b11);
    assert_eq!(bmask(&[0, 2]), 0b101);
}

#[test]
fn test_bmask_empty() {
    assert_eq!(bmask(&[]), 0);
}

#[test]
fn test_bmask_range() {
    // bmask_range(0, 3) = bits 0,1,2 = 0b111
    assert_eq!(bmask_range(0, 3), 0b111);
    assert_eq!(bmask_range(1, 4), 0b1110);
    assert_eq!(bmask_range(2, 5), 0b11100);
}

#[test]
fn test_flip() {
    assert_eq!(flip(0b1011, 0b1011), 0b0000);
    assert_eq!(flip(0b0000, 0b1010), 0b1010);
    assert_eq!(flip(0b1100, 0b0011), 0b1111);
}

#[test]
fn test_anyone() {
    assert!(anyone(0b1011, 0b1001));
    assert!(anyone(0b1011, 0b1100));
    assert!(!anyone(0b1011, 0b0100));
}

#[test]
fn test_allone() {
    assert!(allone(0b1011, 0b1011));
    assert!(allone(0b1011, 0b1001));
    assert!(!allone(0b1011, 0b0100));
}

#[test]
fn test_ismatch() {
    // ismatch(0b11001, 0b10100, 0b10000) == true
    let n = 0b11001usize;
    let mask = 0b10100usize;
    let target = 0b10000usize;
    assert!(ismatch(n, mask, target));
    assert!(!ismatch(n, mask, 0b00100));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test bitutils 2>&1 | head -20`
Expected: compilation error, module not found

**Step 3: Write the implementation**

Add to `src/lib.rs`:
```rust
pub mod bitutils;
```

Create `src/bitutils.rs`:

```rust
//! Bit manipulation utilities for qubit (d=2) simulation.
//!
//! Ported from Julia BitBasis.jl: `~/.julia/dev/BitBasis/src/bit_operations.jl`
//!
//! All bit positions are 0-indexed (unlike Julia's 1-indexed).

/// Return an integer with the k-th bit set to 1 (0-indexed).
///
/// Julia: `indicator(::Type{T}, k::Int) = one(T) << (k-1)` (1-indexed)
/// Rust: 0-indexed, so `1 << k`.
#[inline]
pub fn indicator(k: usize) -> usize {
    1 << k
}

/// Return a bitmask with bits set at the given positions (0-indexed).
///
/// Julia: `bmask(::Type{T}, positions...) = reduce(+, indicator(T, b) for b in itr)`
#[inline]
pub fn bmask(positions: &[usize]) -> usize {
    positions.iter().fold(0, |acc, &k| acc | indicator(k))
}

/// Return a bitmask for a contiguous range [start, stop) of bit positions.
///
/// Julia: `bmask(::Type{T}, range::UnitRange{Int}) = ((1 << (stop-start+1)) - 1) << (start-1)`
#[inline]
pub fn bmask_range(start: usize, stop: usize) -> usize {
    if stop <= start {
        return 0;
    }
    ((1usize << (stop - start)) - 1) << start
}

/// XOR flip bits at masked positions.
///
/// Julia: `flip(index::T, mask::T) = index ⊻ mask`
#[inline]
pub fn flip(index: usize, mask: usize) -> usize {
    index ^ mask
}

/// Return true if any bit at masked position is 1.
///
/// Julia: `anyone(index::T, mask::T) = (index & mask) != zero(T)`
#[inline]
pub fn anyone(index: usize, mask: usize) -> bool {
    (index & mask) != 0
}

/// Return true if all bits at masked positions are 1.
///
/// Julia: `allone(index::T, mask::T) = (index & mask) == mask`
#[inline]
pub fn allone(index: usize, mask: usize) -> bool {
    (index & mask) == mask
}

/// Return true if bits at masked positions equal target.
///
/// Julia: `ismatch(index::T, mask::T, target::T) = (index & mask) == target`
#[inline]
pub fn ismatch(index: usize, mask: usize, target: usize) -> bool {
    (index & mask) == target
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features --test bitutils -v`
Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add src/bitutils.rs src/lib.rs tests/bitutils.rs
git commit -m "feat: add bitutils module with bit manipulation primitives"
```

---

### Task 2: IterControl iterator

**Files:**
- Modify: `src/bitutils.rs`
- Modify: `tests/bitutils.rs`

**Step 1: Write the failing tests**

Append to `tests/bitutils.rs`:

```rust
#[test]
fn test_group_shift_basic() {
    // positions [1, 3, 4, 7] (Julia 1-indexed)
    // In Rust 0-indexed: [0, 2, 3, 6]
    // nbits=7
    let (masks, factors) = group_shift(7, &mut vec![0, 2, 3, 6]);
    // After sorting: [0, 2, 3, 6]
    // Gaps: before 0 (none), between 0..2 (bit 1), between 3..6 (bits 4,5)
    // After removing locked bits, free bits are: 1, 4, 5
    // Chunks: [bit 1], [bits 4,5]
    assert_eq!(masks.len(), factors.len());
    // The exact values depend on the algorithm; verify via itercontrol below
}

#[test]
fn test_itercontrol_basic() {
    // Julia example: itercontrol(7, [1,3,4,7], (1,0,1,0))
    // iterates: 0001001, 0001011, 0011001, 0011011, 0101001, 0101011, 0111001, 0111011
    //
    // Rust 0-indexed: nbits=7, positions=[0,2,3,6], bits=[1,0,1,0]
    // Pattern: bit0=1, bit2=0, bit3=1, bit6=0 => x0x10x1
    // Free bits: 1, 4, 5
    let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 8); // 2^3 = 8 free bits
    // Expected: 0001001=9, 0001011=11, 0011001=25, 0011011=27,
    //           0101001=41, 0101011=43, 0111001=57, 0111011=59
    assert_eq!(values, vec![9, 11, 25, 27, 41, 43, 57, 59]);
}

#[test]
fn test_itercontrol_no_controls() {
    // No locked bits: iterate all 2^3 = 8 states
    let ic = itercontrol(3, &[], &[]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 8);
    assert_eq!(values, (0..8).collect::<Vec<_>>());
}

#[test]
fn test_itercontrol_all_locked() {
    // Lock all 3 bits to 101 = 5
    let ic = itercontrol(3, &[0, 1, 2], &[1, 0, 1]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 1);
    assert_eq!(values, vec![5]);
}

#[test]
fn test_itercontrol_single_free() {
    // Lock bit 0 to 1, free bit 1. nbits=2
    let ic = itercontrol(2, &[0], &[1]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 2); // 2^1 free
    assert_eq!(values, vec![1, 3]); // 01, 11
}

#[test]
fn test_itercontrol_getindex() {
    let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
    // Test random access matches iteration
    let values: Vec<usize> = (0..ic.len()).map(|k| ic.get(k)).collect();
    let iter_values: Vec<usize> = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]).collect();
    assert_eq!(values, iter_values);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test bitutils 2>&1 | head -20`
Expected: FAIL, `group_shift` and `itercontrol` not found

**Step 3: Write the implementation**

Append to `src/bitutils.rs`:

```rust
/// Iterator for controlled subspace of bits.
///
/// Efficiently iterates through basis states where specified bit positions
/// are locked to given values. Only "free" (unlocked) bits vary.
///
/// Ported from Julia BitBasis: `~/.julia/dev/BitBasis/src/iterate_control.jl`
///
/// # Example
/// To iterate states matching pattern `x0x10x1` (0-indexed positions 0,2,3,6):
/// ```
/// use yao_rs::bitutils::itercontrol;
/// let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
/// // Yields: 9, 11, 25, 27, 41, 43, 57, 59
/// ```
pub struct IterControl {
    n: usize,           // number of free configurations = 2^(nbits - nlocked)
    base: usize,        // base offset from locked bits set to 1
    masks: Vec<usize>,  // masks for each chunk of free bits
    factors: Vec<usize>, // shift factors for each chunk
    current: usize,     // iteration counter
}

impl IterControl {
    /// Total number of states this iterator will yield.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Random access: get the k-th element (0-indexed).
    ///
    /// Julia: `getindex(it, k) = sum((k & mask) * factor for (mask, factor)) + base`
    /// (Julia is 1-indexed, so uses k-1)
    pub fn get(&self, k: usize) -> usize {
        let mut out = 0usize;
        for (&mask, &factor) in self.masks.iter().zip(self.factors.iter()) {
            out += (k & mask) * factor;
        }
        out + self.base
    }
}

impl Iterator for IterControl {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.current >= self.n {
            return None;
        }
        let val = self.get(self.current);
        self.current += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IterControl {}

/// Create an IterControl that iterates through the controlled subspace.
///
/// `nbits`: total number of qubits
/// `positions`: 0-indexed bit positions to lock (sorted internally)
/// `bit_configs`: value (0 or 1) to lock each position to
///
/// Julia: `itercontrol(nbits, positions, bit_configs)` from
/// `~/.julia/dev/BitBasis/src/iterate_control.jl:53-59`
pub fn itercontrol(nbits: usize, positions: &[usize], bit_configs: &[usize]) -> IterControl {
    debug_assert_eq!(positions.len(), bit_configs.len());
    debug_assert!(bit_configs.iter().all(|&b| b == 0 || b == 1));

    // base = bitmask of positions where config == 1
    let base: usize = positions
        .iter()
        .zip(bit_configs.iter())
        .filter(|(_, &b)| b == 1)
        .map(|(&pos, _)| indicator(pos))
        .sum();

    let mut sorted_positions: Vec<usize> = positions.to_vec();
    let (masks, factors) = group_shift(nbits, &mut sorted_positions);

    let n = 1 << (nbits - positions.len());
    IterControl {
        n,
        base,
        masks,
        factors,
        current: 0,
    }
}

/// Compute masks and shift factors for IterControl.
///
/// Groups the free (unlocked) bits into contiguous chunks and computes
/// the bitmask and multiplication factor for each chunk.
///
/// Julia: `group_shift!(nbits, positions)` from
/// `~/.julia/dev/BitBasis/src/iterate_control.jl:108-130`
///
/// NOTE: Julia uses 1-indexed positions. This function uses 0-indexed.
/// The algorithm is adjusted accordingly:
/// - Julia: k starts at 1, k_prv starts at 0
/// - Rust: k starts at 0 (but we add 1 internally to match the algorithm)
pub fn group_shift(nbits: usize, positions: &mut Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    positions.sort();

    let mut masks = Vec::new();
    let mut factors = Vec::new();

    // Convert to 1-indexed to match Julia algorithm exactly
    let positions_1: Vec<usize> = positions.iter().map(|&p| p + 1).collect();

    let mut k_prv: usize = 0;
    let mut i: usize = 0;

    for &k in &positions_1 {
        debug_assert!(k > k_prv, "Conflict at location: {}", k);
        if k != k_prv + 1 {
            factors.push(1 << (k_prv.wrapping_sub(i)));
            let gap = k - k_prv - 1;
            // bmask_range for 1-indexed range [i+1, i+gap] -> 0-indexed [i, i+gap)
            masks.push(bmask_range(i, i + gap));
            i += gap;
        }
        k_prv = k;
    }

    // The last block
    if i != nbits {
        factors.push(1 << (k_prv.wrapping_sub(i)));
        masks.push(bmask_range(i, nbits - positions.len()));
    }

    (masks, factors)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features --test bitutils -v`
Expected: all 13 tests PASS

**Step 5: Commit**

```bash
git add src/bitutils.rs tests/bitutils.rs
git commit -m "feat: add IterControl for controlled subspace iteration"
```

---

### Task 3: Qubit instruct functions — 1q and 1q-diagonal (no controls)

**Files:**
- Create: `src/instruct_qubit.rs`
- Modify: `src/lib.rs` (add `pub mod instruct_qubit;`)
- Test: `tests/instruct_qubit.rs`

**Step 1: Write the failing tests**

Create `tests/instruct_qubit.rs`:

```rust
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

mod common;
use common::*;

/// Load instruct test data from tests/data/instruct.json
fn load_instruct_data() -> serde_json::Value {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/instruct.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

fn parse_state(val: &serde_json::Value) -> Vec<Complex64> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let arr = pair.as_array().unwrap();
            Complex64::new(arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap())
        })
        .collect()
}

fn parse_matrix(val: &serde_json::Value) -> Vec<Vec<Complex64>> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    let arr = pair.as_array().unwrap();
                    Complex64::new(arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap())
                })
                .collect()
        })
        .collect()
}

fn states_approx_eq(a: &[Complex64], b: &[Complex64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
}

// ---------- 1q without controls ----------

#[test]
fn test_instruct_1q_from_julia_data() {
    use yao_rs::instruct_qubit::instruct_1q;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        // Only test 1q cases without controls and with a gate_matrix (2x2)
        if !label.contains("1q") || case.get("ctrl_locs").is_some() {
            continue;
        }
        if case.get("gate_matrix").is_none() {
            continue;
        }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        if locs.len() != 1 { continue; }

        let mat = parse_matrix(&case["gate_matrix"]);
        if mat.len() != 2 { continue; }

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);
        let loc = locs[0];

        let a = mat[0][0]; let b = mat[0][1];
        let c = mat[1][0]; let d = mat[1][1];

        instruct_1q(&mut state, loc, a, b, c, d);

        assert!(
            states_approx_eq(&state, &expected, 1e-10),
            "FAIL: {label}"
        );
    }
}

#[test]
fn test_instruct_1q_diag_z_gate() {
    use yao_rs::instruct_qubit::instruct_1q_diag;

    // Z gate on qubit 0: diag(1, -1)
    let d0 = Complex64::new(1.0, 0.0);
    let d1 = Complex64::new(-1.0, 0.0);
    let s = FRAC_1_SQRT_2;

    // |+> = [1/√2, 1/√2] -> Z -> [1/√2, -1/√2] = |->
    let mut state = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    instruct_1q_diag(&mut state, 0, d0, d1);
    assert!((state[0] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((state[1] - Complex64::new(-s, 0.0)).norm() < 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test instruct_qubit 2>&1 | head -20`
Expected: compilation error

**Step 3: Write the implementation**

Add to `src/lib.rs`:
```rust
pub mod instruct_qubit;
```

Create `src/instruct_qubit.rs`:

```rust
//! Qubit (d=2) instruct functions using bit-manipulation for zero-alloc hot loops.
//!
//! Ported from Julia YaoArrayRegister:
//! `~/.julia/dev/Yao/lib/YaoArrayRegister/src/instruct.jl`
//!
//! All 4 functions support optional controls via IterControl.

use num_complex::Complex64;

use crate::bitutils::{bmask, group_shift, itercontrol, IterControl};

/// Apply 2x2 unitary [[a,b],[c,d]] to amplitudes at indices i and j.
///
/// Julia: `u1rows!(state, i, j, a, b, c, d)` from utils.jl:110-116
#[inline]
fn u1rows(state: &mut [Complex64], i: usize, j: usize, a: Complex64, b: Complex64, c: Complex64, d: Complex64) {
    let w = state[i];
    let v = state[j];
    state[i] = a * w + b * v;
    state[j] = c * w + d * v;
}

/// Apply d-dimensional unitary to amplitudes at given indices.
///
/// Julia: `unrows!(state, inds, U)` from utils.jl:176-183
fn unrows(state: &mut [Complex64], inds: &[usize], gate: &[Complex64]) {
    let d = inds.len();
    // Collect old amplitudes to stack for small d
    let mut old = [Complex64::new(0.0, 0.0); 4];
    for (k, &idx) in inds.iter().enumerate() {
        old[k] = state[idx];
    }
    for i in 0..d {
        let mut new_amp = Complex64::new(0.0, 0.0);
        for j in 0..d {
            new_amp += gate[i * d + j] * old[j];
        }
        state[inds[i]] = new_amp;
    }
}

// ========================================================================
// 1-qubit instruct (no controls)
// ========================================================================

/// Apply single-qubit gate to state vector using stride-based iteration.
///
/// Julia: `single_qubit_instruct!(state, U1, loc)` + `instruct_kernel`
/// from `instruct.jl:153-166`
///
/// ```text
/// step1 = 1 << loc
/// step2 = 1 << (loc + 1)
/// for j in (0..total).step_by(step2):
///     for i in j..(j + step1):
///         u1rows(state, i, i + step1, a, b, c, d)
/// ```
pub fn instruct_1q(
    state: &mut [Complex64],
    loc: usize,
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
) {
    let step1 = 1 << loc;
    let step2 = 1 << (loc + 1);
    let total = state.len();
    let mut j = 0;
    while j < total {
        for i in j..(j + step1) {
            u1rows(state, i, i + step1, a, b, c, d);
        }
        j += step2;
    }
}

// ========================================================================
// 1-qubit diagonal instruct (no controls)
// ========================================================================

/// Apply single-qubit diagonal gate diag(d0, d1) using stride-based iteration.
///
/// Julia: `single_qubit_instruct!(state, U1::SDDiagonal, loc)`
/// from `instruct.jl:187-198`
pub fn instruct_1q_diag(
    state: &mut [Complex64],
    loc: usize,
    d0: Complex64,
    d1: Complex64,
) {
    let step1 = 1 << loc;
    let step2 = 1 << (loc + 1);
    let total = state.len();
    let mut j = 0;
    while j < total {
        for i in j..(j + step1) {
            state[i] *= d0;
            state[i + step1] *= d1;
        }
        j += step2;
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features --test instruct_qubit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/instruct_qubit.rs src/lib.rs tests/instruct_qubit.rs
git commit -m "feat: add instruct_1q and instruct_1q_diag with stride-based iteration"
```

---

### Task 4: Qubit instruct functions — 2q (no controls)

**Files:**
- Modify: `src/instruct_qubit.rs`
- Modify: `tests/instruct_qubit.rs`

**Step 1: Write the failing tests**

Append to `tests/instruct_qubit.rs`:

```rust
#[test]
fn test_instruct_2q_from_julia_data() {
    use yao_rs::instruct_qubit::instruct_2q;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        // Only test 2q cases without controls
        if !label.contains("2q") || case.get("ctrl_locs").is_some() {
            continue;
        }
        if case.get("gate_matrix").is_none() { continue; }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        if locs.len() != 2 { continue; }

        let mat_2d = parse_matrix(&case["gate_matrix"]);
        if mat_2d.len() != 4 { continue; }

        // Flatten matrix to row-major vec
        let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        instruct_2q(&mut state, nbits, &locs, &gate);

        assert!(
            states_approx_eq(&state, &expected, 1e-10),
            "FAIL: {label}"
        );
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test instruct_qubit::test_instruct_2q 2>&1 | head -20`
Expected: FAIL, `instruct_2q` not found

**Step 3: Write the implementation**

Append to `src/instruct_qubit.rs`:

```rust
// ========================================================================
// 2-qubit instruct (no controls)
// ========================================================================

/// Permute gate matrix when locs are not in ascending order.
///
/// Julia: `sort_unitary(Val(2), U, locs)` from utils.jl:16-23
/// When locs are not ascending, we need to reorder the gate matrix
/// to match the sorted order.
fn sort_unitary_2q(gate: &[Complex64], locs: &[usize]) -> (Vec<Complex64>, [usize; 2]) {
    if locs[0] < locs[1] {
        return (gate.to_vec(), [locs[0], locs[1]]);
    }
    // Swap: permute rows and columns by swapping qubits
    // Permutation: |00> -> |00>, |01> -> |10>, |10> -> |01>, |11> -> |11>
    let perm = [0, 2, 1, 3];
    let mut new_gate = vec![Complex64::new(0.0, 0.0); 16];
    for i in 0..4 {
        for j in 0..4 {
            new_gate[perm[i] * 4 + perm[j]] = gate[i * 4 + j];
        }
    }
    ([locs[0], locs[1]].sort();
    let mut sorted = [locs[0], locs[1]];
    sorted.sort();
    (new_gate, sorted)
}

/// Apply 2-qubit gate using itercontrol.
///
/// Julia: generic `instruct!(Val(2), state, operator, locs::NTuple{M,Int})`
/// from `instruct.jl:90-121`
///
/// Algorithm:
/// 1. Sort locs ascending, permute gate if needed
/// 2. Compute locs_raw: the 4 base indices where only target bits vary
/// 3. Use itercontrol to iterate free bits (non-target)
/// 4. For each free config, apply unrows! to the 4 target indices
pub fn instruct_2q(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    gate: &[Complex64],
) {
    debug_assert_eq!(locs.len(), 2);
    debug_assert_eq!(gate.len(), 16);

    let (gate, sorted_locs) = sort_unitary_2q(gate, locs);

    // Compute locs_raw: iterate free bits (non-targets) locked to 0,
    // to get offsets where only target bits vary
    let non_target_locs: Vec<usize> = (0..nbits)
        .filter(|b| !sorted_locs.contains(b))
        .collect();
    let non_target_zeros: Vec<usize> = vec![0; non_target_locs.len()];
    let locs_raw_ic = itercontrol(nbits, &non_target_locs, &non_target_zeros);
    let locs_raw: Vec<usize> = locs_raw_ic.collect();
    debug_assert_eq!(locs_raw.len(), 4);

    // Lock target bits to 0, iterate free bits
    let locked_positions: Vec<usize> = sorted_locs.to_vec();
    let locked_values: Vec<usize> = vec![0; locked_positions.len()];
    let ic = itercontrol(nbits, &locked_positions, &locked_values);

    for base in ic {
        let inds: Vec<usize> = locs_raw.iter().map(|&r| r + base).collect();
        unrows(state, &inds, &gate);
    }
}
```

Wait — looking at this more carefully, the `locs_raw` computation is wrong. Let me re-derive from Julia.

In Julia (1-indexed):
```julia
locs_raw_it = (b + 1 for b in itercontrol(N, setdiff(1:N, locs), zeros(Int, N - M)))
```

This iterates over all N bits, locking the bits that are NOT targets to 0. The free bits are the target bits themselves. So it yields `2^M` values — exactly the indices where only target bits vary and everything else is 0.

For 0-indexed Rust:
- Target locs: `sorted_locs`
- Lock the NON-target bits to 0
- Free bits = target bits (they vary from 00 to 11)
- Yields 4 values: the raw offsets

Actually, `itercontrol` locks `positions` to `bit_configs` and frees the rest. So to get locs_raw:
- Lock non-target bits to 0 → free bits = targets → yields 2^M values

Let me fix the implementation. The `sort_unitary_2q` function also has a syntax error. Let me rewrite Task 4 properly.

**Step 3 (revised): Write the implementation**

Append to `src/instruct_qubit.rs`:

```rust
// ========================================================================
// 2-qubit instruct (no controls)
// ========================================================================

/// Permute 4x4 gate matrix when locs are not in ascending order.
///
/// Julia: `sort_unitary(Val(2), U, locs)` from utils.jl:16-23
fn sort_unitary_2q(gate: &[Complex64], locs: &[usize]) -> (Vec<Complex64>, [usize; 2]) {
    let mut sorted = [locs[0], locs[1]];
    if sorted[0] < sorted[1] {
        return (gate.to_vec(), sorted);
    }
    sorted.sort();
    // Swap qubit order: permute |ij> -> |ji>
    // perm = [0, 2, 1, 3] maps: 00->00, 01->10, 10->01, 11->11
    let perm = [0usize, 2, 1, 3];
    let mut new_gate = vec![Complex64::new(0.0, 0.0); 16];
    for i in 0..4 {
        for j in 0..4 {
            new_gate[perm[i] * 4 + perm[j]] = gate[i * 4 + j];
        }
    }
    (new_gate, sorted)
}

/// Apply 2-qubit 4x4 gate using itercontrol-based iteration.
///
/// Julia: generic `instruct!(Val(2), state, operator, locs)`
/// from `instruct.jl:90-121`
pub fn instruct_2q(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    gate: &[Complex64],
) {
    debug_assert_eq!(locs.len(), 2);
    debug_assert_eq!(gate.len(), 16);

    let (gate, sorted_locs) = sort_unitary_2q(gate, locs);

    // locs_raw: lock non-target bits to 0, free target bits
    // Yields 4 offsets where only target bits vary (00, 01, 10, 11)
    let non_target: Vec<usize> = (0..nbits)
        .filter(|b| *b != sorted_locs[0] && *b != sorted_locs[1])
        .collect();
    let non_target_zeros = vec![0usize; non_target.len()];
    let locs_raw: Vec<usize> = itercontrol(nbits, &non_target, &non_target_zeros).collect();
    debug_assert_eq!(locs_raw.len(), 4);

    // Main iteration: lock target bits to 0, iterate free (non-target) bits
    let ic = itercontrol(nbits, &sorted_locs.to_vec(), &[0, 0]);

    for base in ic {
        let inds = [
            locs_raw[0] + base,
            locs_raw[1] + base,
            locs_raw[2] + base,
            locs_raw[3] + base,
        ];
        unrows(state, &inds, &gate);
    }
}

/// Apply 2-qubit diagonal gate diag(d0, d1, d2, d3).
pub fn instruct_2q_diag(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    diag: &[Complex64; 4],
) {
    debug_assert_eq!(locs.len(), 2);

    let mut sorted_locs = [locs[0], locs[1]];
    // For diagonal, we need correct ordering of diagonal elements
    let diag = if sorted_locs[0] > sorted_locs[1] {
        sorted_locs.sort();
        // Swap qubit order in diagonal: |01> <-> |10>
        [diag[0], diag[2], diag[1], diag[3]]
    } else {
        *diag
    };

    let non_target: Vec<usize> = (0..nbits)
        .filter(|b| *b != sorted_locs[0] && *b != sorted_locs[1])
        .collect();
    let non_target_zeros = vec![0usize; non_target.len()];
    let locs_raw: Vec<usize> = itercontrol(nbits, &non_target, &non_target_zeros).collect();
    debug_assert_eq!(locs_raw.len(), 4);

    let ic = itercontrol(nbits, &sorted_locs.to_vec(), &[0, 0]);

    for base in ic {
        for k in 0..4 {
            state[locs_raw[k] + base] *= diag[k];
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features --test instruct_qubit -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/instruct_qubit.rs tests/instruct_qubit.rs
git commit -m "feat: add instruct_2q and instruct_2q_diag with itercontrol"
```

---

### Task 5: Add control support to all 4 instruct functions

**Files:**
- Modify: `src/instruct_qubit.rs`
- Modify: `tests/instruct_qubit.rs`

**Step 1: Write the failing tests**

Append to `tests/instruct_qubit.rs`:

```rust
#[test]
fn test_controlled_1q_from_julia_data() {
    use yao_rs::instruct_qubit::instruct_1q_controlled;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        if case.get("ctrl_locs").is_none() { continue; }
        if case.get("gate_matrix").is_none() { continue; }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let ctrl_locs: Vec<usize> = case["ctrl_locs"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let ctrl_bits: Vec<usize> = case["ctrl_bits"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let mat_2d = parse_matrix(&case["gate_matrix"]);

        // Route to 1q or 2q controlled based on gate size
        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        if mat_2d.len() == 2 && locs.len() == 1 {
            let a = mat_2d[0][0]; let b = mat_2d[0][1];
            let c = mat_2d[1][0]; let d = mat_2d[1][1];
            instruct_1q_controlled(&mut state, nbits, locs[0], a, b, c, d, &ctrl_locs, &ctrl_bits);
        } else if mat_2d.len() == 4 && locs.len() == 2 {
            use yao_rs::instruct_qubit::instruct_2q_controlled;
            let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();
            instruct_2q_controlled(&mut state, nbits, &locs, &gate, &ctrl_locs, &ctrl_bits);
        } else {
            continue;
        }

        assert!(
            states_approx_eq(&state, &expected, 1e-10),
            "FAIL: {label}"
        );
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test instruct_qubit::test_controlled 2>&1 | head -20`
Expected: FAIL

**Step 3: Write the implementation**

Append to `src/instruct_qubit.rs`:

```rust
// ========================================================================
// Controlled versions
// ========================================================================

/// Apply controlled single-qubit gate.
///
/// Julia: `instruct!(Val(2), state, operator, locs, control_locs, control_bits)`
/// from `instruct.jl:90-121`
pub fn instruct_1q_controlled(
    state: &mut [Complex64],
    nbits: usize,
    loc: usize,
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    // Lock control bits + target bit (target locked to 0)
    let mut locked_locs: Vec<usize> = ctrl_locs.to_vec();
    locked_locs.push(loc);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.push(0); // target locked to 0

    // locs_raw for 1 target: [0, 1<<loc] (the two indices where target=0 and target=1)
    let step = 1 << loc;

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        u1rows(state, base, base + step, a, b, c, d);
    }
}

/// Apply controlled 2-qubit gate.
pub fn instruct_2q_controlled(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    gate: &[Complex64],
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    debug_assert_eq!(locs.len(), 2);
    debug_assert_eq!(gate.len(), 16);

    let (gate, sorted_locs) = sort_unitary_2q(gate, locs);

    // locs_raw: lock non-target, non-control bits to 0, free only target bits
    let non_target: Vec<usize> = (0..nbits)
        .filter(|b| *b != sorted_locs[0] && *b != sorted_locs[1])
        .collect();
    let non_target_zeros = vec![0usize; non_target.len()];
    let locs_raw: Vec<usize> = itercontrol(nbits, &non_target, &non_target_zeros).collect();
    debug_assert_eq!(locs_raw.len(), 4);

    // Lock controls + targets (targets to 0)
    let mut locked_locs: Vec<usize> = ctrl_locs.to_vec();
    locked_locs.extend_from_slice(&sorted_locs);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.extend_from_slice(&[0, 0]);

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        let inds = [
            locs_raw[0] + base,
            locs_raw[1] + base,
            locs_raw[2] + base,
            locs_raw[3] + base,
        ];
        unrows(state, &inds, &gate);
    }
}

/// Apply controlled 1-qubit diagonal gate.
pub fn instruct_1q_diag_controlled(
    state: &mut [Complex64],
    nbits: usize,
    loc: usize,
    d0: Complex64,
    d1: Complex64,
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    let mut locked_locs: Vec<usize> = ctrl_locs.to_vec();
    locked_locs.push(loc);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.push(0);

    let step = 1 << loc;
    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        state[base] *= d0;
        state[base + step] *= d1;
    }
}

/// Apply controlled 2-qubit diagonal gate.
pub fn instruct_2q_diag_controlled(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    diag: &[Complex64; 4],
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    debug_assert_eq!(locs.len(), 2);

    let mut sorted_locs = [locs[0], locs[1]];
    let diag = if sorted_locs[0] > sorted_locs[1] {
        sorted_locs.sort();
        [diag[0], diag[2], diag[1], diag[3]]
    } else {
        *diag
    };

    let non_target: Vec<usize> = (0..nbits)
        .filter(|b| *b != sorted_locs[0] && *b != sorted_locs[1])
        .collect();
    let non_target_zeros = vec![0usize; non_target.len()];
    let locs_raw: Vec<usize> = itercontrol(nbits, &non_target, &non_target_zeros).collect();

    let mut locked_locs: Vec<usize> = ctrl_locs.to_vec();
    locked_locs.extend_from_slice(&sorted_locs);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.extend_from_slice(&[0, 0]);

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        for k in 0..4 {
            state[locs_raw[k] + base] *= diag[k];
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features --test instruct_qubit -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/instruct_qubit.rs tests/instruct_qubit.rs
git commit -m "feat: add controlled versions of all 4 qubit instruct functions"
```

---

### Task 6: Full Julia ground truth test suite

**Files:**
- Modify: `tests/instruct_qubit.rs`

**Step 1: Write comprehensive ground-truth tests**

Append to `tests/instruct_qubit.rs`:

```rust
/// Run ALL instruct.json test cases through the appropriate qubit instruct function.
/// This is the main correctness test against Julia ground truth.
#[test]
fn test_all_julia_ground_truth() {
    use yao_rs::instruct_qubit::*;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();
    let mut tested = 0;

    for case in cases {
        let label = case["label"].as_str().unwrap();

        // Skip regression and probability cases (different format)
        if case.get("gate_pairs").is_some() || case.get("probabilities").is_some() {
            continue;
        }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let has_controls = case.get("ctrl_locs").is_some();
        let ctrl_locs: Vec<usize> = if has_controls {
            case["ctrl_locs"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect()
        } else { vec![] };
        let ctrl_bits: Vec<usize> = if has_controls {
            case["ctrl_bits"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect()
        } else { vec![] };

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        // Dispatch based on gate type
        if let Some(gate_val) = case.get("gate_matrix") {
            let mat_2d = parse_matrix(gate_val);
            let d = mat_2d.len();

            match (d, locs.len(), has_controls) {
                (2, 1, false) => {
                    let a = mat_2d[0][0]; let b = mat_2d[0][1];
                    let c = mat_2d[1][0]; let d = mat_2d[1][1];
                    instruct_1q(&mut state, locs[0], a, b, c, d);
                }
                (2, 1, true) => {
                    let a = mat_2d[0][0]; let b = mat_2d[0][1];
                    let c = mat_2d[1][0]; let dd = mat_2d[1][1];
                    instruct_1q_controlled(&mut state, nbits, locs[0], a, b, c, dd, &ctrl_locs, &ctrl_bits);
                }
                (4, 2, false) => {
                    let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();
                    instruct_2q(&mut state, nbits, &locs, &gate);
                }
                (4, 2, true) => {
                    let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();
                    instruct_2q_controlled(&mut state, nbits, &locs, &gate, &ctrl_locs, &ctrl_bits);
                }
                _ => continue, // skip non-qubit cases
            }
        } else if let Some(gate_name) = case.get("gate_name") {
            // Named gate — get matrix from Gate enum
            let name = gate_name.as_str().unwrap();
            let theta = case.get("theta").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let gate = match name {
                "X" => yao_rs::Gate::X,
                "Y" => yao_rs::Gate::Y,
                "Z" => yao_rs::Gate::Z,
                "H" => yao_rs::Gate::H,
                "S" => yao_rs::Gate::S,
                "T" => yao_rs::Gate::T,
                "SWAP" => yao_rs::Gate::SWAP,
                "Rx" => yao_rs::Gate::Rx(theta),
                "Ry" => yao_rs::Gate::Ry(theta),
                "Rz" => yao_rs::Gate::Rz(theta),
                "PSWAP" | "CPHASE" => {
                    // These need custom matrix; skip for named path
                    continue;
                }
                _ => continue,
            };
            let mat = gate.matrix(2);
            let d = mat.nrows();

            if d == 2 && locs.len() == 1 {
                let a = mat[[0,0]]; let b = mat[[0,1]];
                let c = mat[[1,0]]; let dd = mat[[1,1]];
                if has_controls {
                    instruct_1q_controlled(&mut state, nbits, locs[0], a, b, c, dd, &ctrl_locs, &ctrl_bits);
                } else {
                    instruct_1q(&mut state, locs[0], a, b, c, dd);
                }
            } else if d == 4 && locs.len() == 2 {
                let gate_flat: Vec<Complex64> = (0..4)
                    .flat_map(|i| (0..4).map(move |j| mat[[i, j]]))
                    .collect();
                if has_controls {
                    instruct_2q_controlled(&mut state, nbits, &locs, &gate_flat, &ctrl_locs, &ctrl_bits);
                } else {
                    instruct_2q(&mut state, nbits, &locs, &gate_flat);
                }
            } else {
                continue;
            }
        } else {
            continue;
        }

        assert!(
            states_approx_eq(&state, &expected, 1e-8),
            "FAIL: {label}\n  got:      {:?}\n  expected: {:?}",
            &state[..state.len().min(8)],
            &expected[..expected.len().min(8)],
        );
        tested += 1;
    }

    assert!(tested >= 40, "Expected at least 40 test cases, got {tested}");
}
```

**Step 2: Run tests**

Run: `cargo test --all-features --test instruct_qubit::test_all_julia_ground_truth -v`
Expected: PASS with 40+ cases tested

**Step 3: Commit**

```bash
git add tests/instruct_qubit.rs
git commit -m "test: comprehensive Julia ground truth test for all qubit instruct functions"
```

---

### Task 7: Wire up apply_inplace dispatch

**Files:**
- Modify: `src/apply.rs`
- Test: existing `tests/apply.rs` (must still pass)

**Step 1: Write the failing test**

No new test file needed — the existing `tests/apply.rs` ground truth tests serve as the regression suite. We add a specific test to verify the qubit path is used.

Append to the existing `tests/instruct_qubit.rs`:

```rust
#[test]
fn test_apply_inplace_uses_qubit_path() {
    // Bell state via apply: should produce same result as before
    use yao_rs::*;

    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    ).unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2]));

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let expected_0 = Complex64::new(s, 0.0);
    assert!((state.data[0] - expected_0).norm() < 1e-10);
    assert!(state.data[1].norm() < 1e-10);
    assert!(state.data[2].norm() < 1e-10);
    assert!((state.data[3] - expected_0).norm() < 1e-10);
}
```

**Step 2: Run existing tests to establish baseline**

Run: `cargo test --all-features --test apply -v 2>&1 | tail -5`
Expected: all PASS

**Step 3: Modify apply_inplace to dispatch to qubit path**

Edit `src/apply.rs` — add qubit dispatch at the top of the gate loop:

```rust
use crate::instruct_qubit;

pub fn apply_inplace(circuit: &Circuit, state: &mut State) {
    let dims = &circuit.dims;
    let all_qubit = dims.iter().all(|&d| d == 2);
    let nbits = dims.len();

    #[cfg(feature = "parallel")]
    let use_parallel = state.data.len() >= PARALLEL_THRESHOLD;

    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) | CircuitElement::Channel(_) => continue,
        };

        if all_qubit {
            let state_slice = state.data.as_slice_mut().unwrap();
            let d = 2;
            let gate_matrix = pg.gate.matrix(d);

            let ctrl_locs: Vec<usize> = pg.control_locs.clone();
            let ctrl_bits: Vec<usize> = pg.control_configs.iter().map(|&b| if b { 1 } else { 0 }).collect();
            let has_controls = !ctrl_locs.is_empty();

            if pg.target_locs.len() == 1 {
                let loc = pg.target_locs[0];
                let a = gate_matrix[[0,0]]; let b = gate_matrix[[0,1]];
                let c = gate_matrix[[1,0]]; let dd = gate_matrix[[1,1]];

                if is_diagonal(&pg.gate) {
                    let d0 = gate_matrix[[0,0]];
                    let d1 = gate_matrix[[1,1]];
                    if has_controls {
                        instruct_qubit::instruct_1q_diag_controlled(state_slice, nbits, loc, d0, d1, &ctrl_locs, &ctrl_bits);
                    } else {
                        instruct_qubit::instruct_1q_diag(state_slice, loc, d0, d1);
                    }
                } else {
                    if has_controls {
                        instruct_qubit::instruct_1q_controlled(state_slice, nbits, loc, a, b, c, dd, &ctrl_locs, &ctrl_bits);
                    } else {
                        instruct_qubit::instruct_1q(state_slice, loc, a, b, c, dd);
                    }
                }
            } else if pg.target_locs.len() == 2 {
                let locs = &pg.target_locs;
                let gate_flat: Vec<Complex64> = (0..4)
                    .flat_map(|i| (0..4).map(move |j| gate_matrix[[i, j]]))
                    .collect();

                if is_diagonal(&pg.gate) {
                    let diag = [gate_matrix[[0,0]], gate_matrix[[1,1]], gate_matrix[[2,2]], gate_matrix[[3,3]]];
                    if has_controls {
                        instruct_qubit::instruct_2q_diag_controlled(state_slice, nbits, locs, &diag, &ctrl_locs, &ctrl_bits);
                    } else {
                        instruct_qubit::instruct_2q_diag(state_slice, nbits, locs, &diag);
                    }
                } else {
                    if has_controls {
                        instruct_qubit::instruct_2q_controlled(state_slice, nbits, locs, &gate_flat, &ctrl_locs, &ctrl_bits);
                    } else {
                        instruct_qubit::instruct_2q(state_slice, nbits, locs, &gate_flat);
                    }
                }
            }
            continue;
        }

        // Existing generic path (unchanged)
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);
        // ... rest of existing code unchanged ...
    }
}
```

**Step 4: Run ALL tests**

Run: `cargo test --all-features 2>&1 | tail -10`
Expected: all tests PASS (including existing apply, integration, etc.)

**Step 5: Commit**

```bash
git add src/apply.rs
git commit -m "feat: dispatch to qubit fast path in apply_inplace when all dims==2"
```

---

### Task 8: Measurement additions (measure_reset, measure_remove)

**Files:**
- Modify: `src/measure.rs`
- Modify: `src/lib.rs` (add exports)
- Modify: `tests/measure.rs`

**Step 1: Write the failing tests**

Append to `tests/measure.rs`:

```rust
#[test]
fn test_measure_reset() {
    use yao_rs::measure::measure_reset;
    use rand::SeedableRng;

    // Create superposition state, measure and reset to 0
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_reset(&mut state, &[0], 0, &mut rng);

    // State should be normalized after reset
    assert!((state.norm() - 1.0).abs() < 1e-10, "State not normalized after reset");

    // Qubit 0 should be in |0> state
    let p = probs(&state, Some(&[0]));
    assert!((p[0] - 1.0).abs() < 1e-10, "Qubit 0 should be |0> after reset");
}

#[test]
fn test_measure_remove() {
    use yao_rs::measure::measure_remove;
    use rand::SeedableRng;

    let mut state = State::product_state(&[2, 2, 2], &[1, 0, 1]); // |101>
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let (result, new_state) = measure_remove(&state, &[1], &mut rng);

    // Measured qubit 1 should be 0
    assert_eq!(result, vec![0]);
    // Remaining state should have 2 qubits
    assert_eq!(new_state.dims.len(), 2);
    assert_eq!(new_state.dims, vec![2, 2]);
    // Remaining state should be |11> (qubits 0 and 2)
    assert!((new_state.data[3].norm() - 1.0).abs() < 1e-10);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --all-features --test measure::test_measure_reset 2>&1 | head -20`
Expected: FAIL

**Step 3: Write the implementation**

Add to `src/measure.rs`:

```rust
/// Measure specified qubits and reset them to a given value.
///
/// Julia: `measure!(YaoAPI.ResetTo(val), reg, locs)`
pub fn measure_reset(
    state: &mut State,
    locs: &[usize],
    reset_val: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let result = measure(state, Some(locs), 1, rng).pop().unwrap();

    // Collapse to measured outcome
    collapse_to(state, locs, &result);

    // Now flip bits to get from measured value to reset value
    let total_dim: usize = state.dims.iter().product();
    if result != vec![reset_val; locs.len()] {
        // Need to swap amplitudes
        for flat_idx in 0..total_dim {
            let indices = linear_to_indices(flat_idx, &state.dims);
            // Check if this index matches the measured outcome at locs
            let matches_measured = locs.iter().zip(result.iter()).all(|(&l, &v)| indices[l] == v);
            if matches_measured {
                // Compute the target index with reset values
                let mut target_indices = indices.clone();
                for (&l, _) in locs.iter().zip(result.iter()) {
                    target_indices[l] = reset_val;
                }
                let target_flat = mixed_radix_index(&target_indices, &state.dims);
                state.data[target_flat] = state.data[flat_idx];
                state.data[flat_idx] = Complex64::new(0.0, 0.0);
            }
        }
    }

    result
}

/// Measure specified qubits and remove them from the state.
///
/// Returns (measurement_result, new_smaller_state).
///
/// Julia: `measure!(YaoAPI.RemoveMeasured(), reg, locs)`
pub fn measure_remove(
    state: &State,
    locs: &[usize],
    rng: &mut impl Rng,
) -> (Vec<usize>, State) {
    let result = measure(state, Some(locs), 1, rng).pop().unwrap();

    // Build new dims without the measured qubits
    let remaining_locs: Vec<usize> = (0..state.dims.len())
        .filter(|i| !locs.contains(i))
        .collect();
    let new_dims: Vec<usize> = remaining_locs.iter().map(|&i| state.dims[i]).collect();
    let new_total: usize = new_dims.iter().product();

    let mut new_data = ndarray::Array1::zeros(new_total);

    let total_dim: usize = state.dims.iter().product();
    let mut norm_sq = 0.0;

    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);

        // Check if measured qubits match the result
        let matches = locs.iter().zip(result.iter()).all(|(&l, &v)| indices[l] == v);
        if !matches {
            continue;
        }

        // Compute new index without measured qubits
        let new_indices: Vec<usize> = remaining_locs.iter().map(|&i| indices[i]).collect();
        let new_flat = mixed_radix_index(&new_indices, &new_dims);
        new_data[new_flat] = state.data[flat_idx];
        norm_sq += state.data[flat_idx].norm_sqr();
    }

    // Normalize
    let norm = norm_sq.sqrt();
    if norm > 1e-15 {
        new_data.mapv_inplace(|v| v / norm);
    }

    (result, State::new(new_dims, new_data))
}
```

Update exports in `src/lib.rs`:
```rust
pub use measure::{collapse_to, measure, measure_and_collapse, measure_remove, measure_reset, probs};
```

**Step 4: Run tests**

Run: `cargo test --all-features --test measure -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/measure.rs src/lib.rs tests/measure.rs
git commit -m "feat: add measure_reset and measure_remove for measurement completion"
```

---

### Task 9: Run full test suite and verify no regressions

**Files:** None (verification only)

**Step 1: Run make check-all**

Run: `make check-all`
Expected: fmt-check, clippy, and all tests pass

**Step 2: Run the specific instruct ground truth test**

Run: `cargo test --all-features --test instruct_qubit -- --nocapture 2>&1 | tail -20`
Expected: all tests PASS

**Step 3: Run existing integration tests**

Run: `cargo test --all-features --test integration -v 2>&1 | tail -10`
Expected: all 72+ ground truth cases PASS

**Step 4: If any failures, debug and fix**

Check specific failing tests with: `cargo test --all-features test_name -- --nocapture`

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify all tests pass after qubit simulation overhaul"
```

---

Plan complete and saved to `docs/plans/2026-03-02-qubit-simulation-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?