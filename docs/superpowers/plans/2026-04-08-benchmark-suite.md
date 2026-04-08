# Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correctness + performance benchmark suite comparing yao-rs against Yao.jl across single gates, QFT, and noisy density-matrix circuits.

**Architecture:** Julia script generates ground-truth JSON + timing data. Rust integration tests validate correctness against JSON. Criterion benchmarks measure Rust performance. A Python script compares Julia vs Rust timings.

**Tech Stack:** Criterion (Rust benchmarks), BenchmarkTools.jl (Julia benchmarks), serde_json (JSON loading), approx (tolerance comparisons)

**Spec:** `docs/superpowers/specs/2026-04-08-benchmark-suite-design.md`

---

### Task 1: Directory Structure and Cargo Config

**Files:**
- Create: `benchmarks/julia/Project.toml`
- Create: `benchmarks/data/.gitkeep`
- Modify: `Cargo.toml:40-43` (add new bench targets)

- [ ] **Step 1: Create benchmarks directory structure**

```bash
mkdir -p benchmarks/julia benchmarks/data
touch benchmarks/data/.gitkeep
```

- [ ] **Step 2: Create Julia Project.toml**

```toml
[deps]
Yao = "0.9"
BenchmarkTools = "1"
JSON = "0.21"
```

Write this to `benchmarks/julia/Project.toml`.

- [ ] **Step 3: Add Criterion bench targets to Cargo.toml**

Add after the existing `[[bench]]` block in `Cargo.toml`:

```toml
[[bench]]
name = "gates"
harness = false

[[bench]]
name = "qft"
harness = false

[[bench]]
name = "density"
harness = false
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/ Cargo.toml
git commit -m "chore: add benchmark directory structure and Criterion targets"
```

---

### Task 2: Julia Ground-Truth + Timing Script

**Files:**
- Create: `benchmarks/julia/generate_ground_truth.jl`

This is the Julia script the developer will run manually. It generates all 5 JSON files + 1 timings file.

- [ ] **Step 1: Write the Julia script**

Write `benchmarks/julia/generate_ground_truth.jl`:

```julia
using Yao
using BenchmarkTools
using JSON

const OUTPUT_DIR = joinpath(@__DIR__, "..", "data")
mkpath(OUTPUT_DIR)

# ============================================================================
# Helper: deterministic initial state
# state[k] = Complex(cos(0.1*k), sin(0.2*k)), normalized
# k is 0-indexed (Julia arrays are 1-indexed, so k = idx - 1)
# ============================================================================
function deterministic_state(n::Int)
    dim = 1 << n
    state = [cos(0.1 * (k-1)) + sin(0.2 * (k-1)) * im for k in 1:dim]
    state ./ norm(state)
end

function state_to_interleaved(state::Vector{ComplexF64})
    result = Float64[]
    for c in state
        push!(result, real(c))
        push!(result, imag(c))
    end
    result
end

function dm_to_interleaved(dm::DensityMatrix)
    state = statevec(dm)
    result = Float64[]
    for c in state
        push!(result, real(c))
        push!(result, imag(c))
    end
    result
end

# ============================================================================
# Task 1: Single Gate Benchmarks
# ============================================================================
println("=== Task 1: Single Gates ===")

timings = Dict{String, Any}()

# 1q gates: applied to qubit 3 (1-indexed = qubit 2 in 0-indexed)
# Yao uses 1-indexed qubits
function single_gate_1q_data()
    gates = Dict(
        "X" => put(3 => X),
        "H" => put(3 => H),
        "T" => put(3 => T),
        "Rx_0.5" => put(3 => Rx(0.5)),
        "Rz_0.5" => put(3 => Rz(0.5)),
    )
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()
    for (name, gate_factory) in gates
        data[name] = Dict{String, Any}()
        timing_data[name] = Dict{String, Any}()
        for nq in 4:25
            println("  1q gate $name, $nq qubits")
            init = deterministic_state(nq)
            reg = ArrayReg(init)
            gate = put(nq, gate_factory)
            result = copy(reg) |> gate
            data[name]["$nq"] = state_to_interleaved(statevec(result))

            # Benchmark
            bench = @benchmark (copy($reg) |> $gate) samples=100 evals=10
            timing_data[name]["$nq"] = minimum(bench).time
        end
    end
    data, timing_data
end

# 2q controlled gates: control=3, target=4 (1-indexed)
function single_gate_2q_data()
    gates = Dict(
        "CNOT" => control(3, 4 => X),
        "CRx_0.5" => control(3, 4 => Rx(0.5)),
    )
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()
    for (name, gate_factory) in gates
        data[name] = Dict{String, Any}()
        timing_data[name] = Dict{String, Any}()
        for nq in 4:25
            println("  2q gate $name, $nq qubits")
            init = deterministic_state(nq)
            reg = ArrayReg(init)
            gate = put(nq, gate_factory)
            result = copy(reg) |> gate
            data[name]["$nq"] = state_to_interleaved(statevec(result))

            bench = @benchmark (copy($reg) |> $gate) samples=100 evals=10
            timing_data[name]["$nq"] = minimum(bench).time
        end
    end
    data, timing_data
end

# Multi-controlled: Toffoli = controls=(3,4), target=2 (1-indexed)
function single_gate_multi_data()
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()
    data["Toffoli"] = Dict{String, Any}()
    timing_data["Toffoli"] = Dict{String, Any}()
    for nq in 4:25
        println("  Toffoli, $nq qubits")
        init = deterministic_state(nq)
        reg = ArrayReg(init)
        gate = control(nq, (3, 4), 2 => X)
        result = copy(reg) |> gate
        data["Toffoli"]["$nq"] = state_to_interleaved(statevec(result))

        bench = @benchmark (copy($reg) |> $gate) samples=100 evals=10
        timing_data["Toffoli"]["$nq"] = minimum(bench).time
    end
    data, timing_data
end

data_1q, timing_1q = single_gate_1q_data()
data_2q, timing_2q = single_gate_2q_data()
data_multi, timing_multi = single_gate_multi_data()

open(joinpath(OUTPUT_DIR, "single_gate_1q.json"), "w") do f
    JSON.print(f, data_1q, 2)
end
open(joinpath(OUTPUT_DIR, "single_gate_2q.json"), "w") do f
    JSON.print(f, data_2q, 2)
end
open(joinpath(OUTPUT_DIR, "single_gate_multi.json"), "w") do f
    JSON.print(f, data_multi, 2)
end

timings["single_gate_1q"] = timing_1q
timings["single_gate_2q"] = timing_2q
timings["single_gate_multi"] = timing_multi

# ============================================================================
# Task 2: QFT Circuit
# ============================================================================
println("\n=== Task 2: QFT ===")

function qft_data()
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()
    for nq in 4:25
        println("  QFT, $nq qubits")
        # |1> = |000...01> (qubit 1 is set in 1-indexed)
        reg = product_state(bit"0"^(nq-1) * bit"1")
        circuit = chain(nq, EasyBuild.qft_circuit(nq)...)
        result = copy(reg) |> circuit
        data["$nq"] = state_to_interleaved(statevec(result))

        bench = @benchmark (copy($reg) |> $circuit) samples=100 evals=10
        timing_data["$nq"] = minimum(bench).time
    end
    data, timing_data
end

data_qft, timing_qft = qft_data()

open(joinpath(OUTPUT_DIR, "qft.json"), "w") do f
    JSON.print(f, data_qft, 2)
end

timings["qft"] = timing_qft

# ============================================================================
# Task 3: Noisy Circuit — Density Matrix
# ============================================================================
println("\n=== Task 3: Noisy DM ===")

function noisy_dm_data()
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()
    for nq in 4:10
        println("  Noisy DM, $nq qubits")
        # Build circuit: H all, CNOT chain, depolarizing, Rz, amplitude damping
        circ = chain(nq)

        # H on all qubits
        for q in 1:nq
            push!(circ, put(nq, q => H))
        end

        # CNOT chain
        for q in 1:(nq-1)
            push!(circ, control(nq, q, q+1 => X))
        end

        # Depolarizing noise p=0.01 on each qubit
        for q in 1:nq
            push!(circ, put(nq, q => DepolarizingChannel(0.01)))
        end

        # Rz(0.3) on all qubits
        for q in 1:nq
            push!(circ, put(nq, q => Rz(0.3)))
        end

        # Amplitude damping gamma=0.05 on each qubit
        for q in 1:nq
            push!(circ, put(nq, q => AmplitudeDampingChannel(0.05)))
        end

        # Evolve density matrix
        dm = density_matrix(zero_state(nq))
        dm_result = copy(dm) |> circ

        entry = Dict{String, Any}()
        entry["trace"] = real(tr(dm_result))
        entry["purity"] = real(purity(dm_result))

        # Partial trace over last qubit
        reduced = partial_tr(dm_result, nq)
        entry["entropy"] = von_neumann_entropy(reduced)

        # Expectation value: H_ising = sum Z_i Z_{i+1} + 0.5 sum X_i
        h_ising = sum([put(nq, i => Z) * put(nq, i+1 => Z) for i in 1:(nq-1)]) +
                  0.5 * sum([put(nq, i => X) for i in 1:nq])
        exp_val = expect(h_ising, dm_result)
        entry["expect_ising"] = Dict("re" => real(exp_val), "im" => imag(exp_val))

        # Full DM and reduced DM for small circuits
        if nq <= 6
            entry["density_matrix"] = dm_to_interleaved(dm_result)
            entry["reduced_dm"] = dm_to_interleaved(reduced)
        end

        data["$nq"] = entry

        # Benchmark
        bench = @benchmark (copy($dm) |> $circ) samples=50 evals=5
        timing_data["$nq"] = minimum(bench).time
    end
    data, timing_data
end

data_noisy, timing_noisy = noisy_dm_data()

open(joinpath(OUTPUT_DIR, "noisy_circuit.json"), "w") do f
    JSON.print(f, data_noisy, 2)
end

timings["noisy_dm"] = timing_noisy

# ============================================================================
# Write timings
# ============================================================================
open(joinpath(OUTPUT_DIR, "timings.json"), "w") do f
    JSON.print(f, timings, 2)
end

println("\nDone! Generated files in $OUTPUT_DIR")
```

**Note:** This Julia script is a best-effort draft. The developer may need to adjust Yao.jl API calls (e.g., `partial_tr`, `von_neumann_entropy`, `dm_to_interleaved`) to match the actual Yao.jl API. The qubit indexing convention (1-indexed in Julia vs 0-indexed in Rust) needs careful attention — Julia's qubit 3 corresponds to Rust's qubit 2.

- [ ] **Step 2: Commit**

```bash
git add benchmarks/julia/generate_ground_truth.jl
git commit -m "feat: add Julia ground-truth generation script for benchmarks"
```

---

### Task 3: Benchmark Data Loading Helpers

**Files:**
- Create: `tests/common/benchmark_data.rs`
- Modify: `tests/common/mod.rs` (add module)

- [ ] **Step 1: Write the benchmark data deserialization structs and loaders**

Create `tests/common/benchmark_data.rs`:

```rust
//! Deserialization helpers for benchmark ground-truth JSON data.

use num_complex::Complex64;
use serde::Deserialize;
use std::collections::HashMap;

/// State vectors stored as interleaved [re0, im0, re1, im1, ...].
/// Outer key = gate name, inner key = nqubits (as string).
pub type GateData = HashMap<String, HashMap<String, Vec<f64>>>;

/// QFT data: key = nqubits (as string), value = interleaved state vector.
pub type QftData = HashMap<String, Vec<f64>>;

/// A single noisy-circuit entry for one qubit count.
#[derive(Deserialize)]
pub struct NoisyEntry {
    pub trace: f64,
    pub purity: f64,
    pub entropy: f64,
    pub expect_ising: ExpectComplex,
    #[serde(default)]
    pub density_matrix: Option<Vec<f64>>,
    #[serde(default)]
    pub reduced_dm: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct ExpectComplex {
    pub re: f64,
    pub im: f64,
}

/// Noisy circuit data: key = nqubits (as string).
pub type NoisyData = HashMap<String, NoisyEntry>;

/// Convert interleaved [re0, im0, re1, im1, ...] to Vec<Complex64>.
pub fn interleaved_to_complex(data: &[f64]) -> Vec<Complex64> {
    data.chunks_exact(2)
        .map(|chunk| Complex64::new(chunk[0], chunk[1]))
        .collect()
}

/// Load JSON file from benchmarks/data/ by filename.
/// Returns None if the file doesn't exist (benchmark data not yet generated).
fn try_load_benchmark_json(filename: &str) -> Option<String> {
    let path = format!(
        "{}/benchmarks/data/{}",
        env!("CARGO_MANIFEST_DIR"),
        filename
    );
    std::fs::read_to_string(&path).ok()
}

pub fn load_single_gate_1q() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_1q.json")?).unwrap())
}

pub fn load_single_gate_2q() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_2q.json")?).unwrap())
}

pub fn load_single_gate_multi() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_multi.json")?).unwrap())
}

pub fn load_qft() -> Option<QftData> {
    Some(serde_json::from_str(&try_load_benchmark_json("qft.json")?).unwrap())
}

pub fn load_noisy_circuit() -> Option<NoisyData> {
    Some(serde_json::from_str(&try_load_benchmark_json("noisy_circuit.json")?).unwrap())
}
```

- [ ] **Step 2: Add module to tests/common/mod.rs**

Add at the top of `tests/common/mod.rs`:

```rust
pub mod benchmark_data;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo test --all-features --no-run 2>&1 | head -20`
Expected: compiles without errors (tests won't run since JSON files don't exist yet).

- [ ] **Step 4: Commit**

```bash
git add tests/common/benchmark_data.rs tests/common/mod.rs
git commit -m "feat: add benchmark JSON data loading helpers"
```

---

### Task 4: Deterministic State Helper

**Files:**
- Modify: `src/register.rs` (add `deterministic_state` constructor)

This is the shared deterministic initial state used by tasks 1 and the Rust benchmarks. Both Julia and Rust must produce the exact same state from the index formula.

- [ ] **Step 1: Write the failing test**

Add to `src/register.rs` inside the `mod tests` block:

```rust
    #[test]
    fn test_deterministic_state() {
        let reg = ArrayReg::deterministic_state(4);
        assert_eq!(reg.nqubits(), 4);
        assert_eq!(reg.state.len(), 16);
        assert_abs_diff_eq!(reg.norm(), 1.0, epsilon = 1e-12);
        // state[0] = Complex64(cos(0.0), sin(0.0)) = (1.0, 0.0), normalized
        // state[1] = Complex64(cos(0.1), sin(0.2)), normalized
        // Just check normalization and that state[0] != state[1]
        assert!((reg.state[0] - reg.state[1]).norm() > 1e-6);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib test_deterministic_state`
Expected: FAIL — method `deterministic_state` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `impl ArrayReg` in `src/register.rs`, after the `from_vec` method:

```rust
    /// Deterministic state for benchmarking: state[k] = Complex64(cos(0.1*k), sin(0.2*k)), normalized.
    /// Matches the Julia benchmark script formula exactly.
    pub fn deterministic_state(nbits: usize) -> Self {
        let dim = 1usize << nbits;
        let mut state: Vec<Complex64> = (0..dim)
            .map(|k| {
                let kf = k as f64;
                Complex64::new(kf.mul_add(0.1, 0.0).cos(), (0.2 * kf).sin())
            })
            .collect();
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in &mut state {
            *c /= norm;
        }
        Self { nbits, state }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib test_deterministic_state`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/register.rs
git commit -m "feat: add ArrayReg::deterministic_state for benchmark initial states"
```

---

### Task 5: Benchmark Validation Tests — Single Gates (Task 1 correctness)

**Files:**
- Create: `tests/suites/benchmark_validation.rs`
- Modify: `tests/main.rs` (add module)

- [ ] **Step 1: Write the validation test file**

Create `tests/suites/benchmark_validation.rs`:

```rust
//! Correctness validation against Julia (Yao.jl) ground-truth data.
//!
//! These tests require JSON files in benchmarks/data/ generated by:
//!   cd benchmarks/julia && julia --project=. generate_ground_truth.jl

use crate::common::benchmark_data::*;
use num_complex::Complex64;
use yao_rs::{ArrayReg, Circuit, Gate, apply, control, put};

const ATOL_STATE: f64 = 1e-10;
const ATOL_SCALAR: f64 = 1e-8;
const ATOL_ENTROPY: f64 = 1e-6;

fn assert_state_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).norm();
        assert!(
            diff < ATOL_STATE,
            "{label}[{i}]: got {a:?}, expected {e:?}, diff={diff}"
        );
    }
}

// =====================================================================
// Task 1: Single Gate Validation
// =====================================================================

fn validate_gate(
    data: &GateData,
    gate_name: &str,
    build_circuit: impl Fn(usize) -> Circuit,
) {
    let gate_data = &data[gate_name];
    for nq in 4..=25 {
        let key = nq.to_string();
        if !gate_data.contains_key(&key) {
            continue;
        }
        let expected = interleaved_to_complex(&gate_data[&key]);
        let reg = ArrayReg::deterministic_state(nq);
        let circuit = build_circuit(nq);
        let result = apply(&circuit, &reg);
        assert_state_close(
            result.state_vec(),
            &expected,
            &format!("{gate_name}/{nq}q"),
        );
    }
}

#[test]
fn test_single_gate_x() {
    let Some(data) = load_single_gate_1q() else {
        eprintln!("Skipping: single_gate_1q.json not found");
        return;
    };
    validate_gate(&data, "X", |nq| {
        Circuit::qubits(nq, vec![put(vec![2], Gate::X)]).unwrap()
    });
}

#[test]
fn test_single_gate_h() {
    let Some(data) = load_single_gate_1q() else {
        eprintln!("Skipping: single_gate_1q.json not found");
        return;
    };
    validate_gate(&data, "H", |nq| {
        Circuit::qubits(nq, vec![put(vec![2], Gate::H)]).unwrap()
    });
}

#[test]
fn test_single_gate_t() {
    let Some(data) = load_single_gate_1q() else {
        eprintln!("Skipping: single_gate_1q.json not found");
        return;
    };
    validate_gate(&data, "T", |nq| {
        Circuit::qubits(nq, vec![put(vec![2], Gate::T)]).unwrap()
    });
}

#[test]
fn test_single_gate_rx() {
    let Some(data) = load_single_gate_1q() else {
        eprintln!("Skipping: single_gate_1q.json not found");
        return;
    };
    validate_gate(&data, "Rx_0.5", |nq| {
        Circuit::qubits(nq, vec![put(vec![2], Gate::Rx(0.5))]).unwrap()
    });
}

#[test]
fn test_single_gate_rz() {
    let Some(data) = load_single_gate_1q() else {
        eprintln!("Skipping: single_gate_1q.json not found");
        return;
    };
    validate_gate(&data, "Rz_0.5", |nq| {
        Circuit::qubits(nq, vec![put(vec![2], Gate::Rz(0.5))]).unwrap()
    });
}

#[test]
fn test_single_gate_cnot() {
    let Some(data) = load_single_gate_2q() else {
        eprintln!("Skipping: single_gate_2q.json not found");
        return;
    };
    validate_gate(&data, "CNOT", |nq| {
        Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::X)]).unwrap()
    });
}

#[test]
fn test_single_gate_crx() {
    let Some(data) = load_single_gate_2q() else {
        eprintln!("Skipping: single_gate_2q.json not found");
        return;
    };
    validate_gate(&data, "CRx_0.5", |nq| {
        Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::Rx(0.5))]).unwrap()
    });
}

#[test]
fn test_single_gate_toffoli() {
    let Some(data) = load_single_gate_multi() else {
        eprintln!("Skipping: single_gate_multi.json not found");
        return;
    };
    validate_gate(&data, "Toffoli", |nq| {
        Circuit::qubits(nq, vec![control(vec![2, 3], vec![1], Gate::X)]).unwrap()
    });
}

// =====================================================================
// Task 2: QFT Validation
// =====================================================================

#[test]
fn test_qft_vs_julia() {
    let Some(data) = load_qft() else {
        eprintln!("Skipping: qft.json not found");
        return;
    };
    for nq in 4..=25 {
        let key = nq.to_string();
        if !data.contains_key(&key) {
            continue;
        }
        let expected = interleaved_to_complex(&data[&key]);

        // Build |1> = |000...01> (qubit 0 set)
        let mut init = vec![Complex64::new(0.0, 0.0); 1 << nq];
        init[1] = Complex64::new(1.0, 0.0);
        let reg = ArrayReg::from_vec(nq, init);

        let circuit = yao_rs::easybuild::qft_circuit(nq);
        let result = apply(&circuit, &reg);
        assert_state_close(
            result.state_vec(),
            &expected,
            &format!("QFT/{nq}q"),
        );
    }
}

#[test]
fn test_qft_analytical() {
    // Secondary verification: QFT|1> has analytically known output
    use std::f64::consts::PI;
    for nq in 4..=16 {
        let dim = 1usize << nq;
        let mut init = vec![Complex64::new(0.0, 0.0); dim];
        init[1] = Complex64::new(1.0, 0.0);
        let reg = ArrayReg::from_vec(nq, init);

        let circuit = yao_rs::easybuild::qft_circuit(nq);
        let result = apply(&circuit, &reg);

        let amp = 1.0 / (dim as f64).sqrt();
        for k in 0..dim {
            let phase = 2.0 * PI * (k as f64) / (dim as f64);
            let expected = Complex64::new(amp * phase.cos(), amp * phase.sin());
            let diff = (result.state_vec()[k] - expected).norm();
            assert!(
                diff < ATOL_STATE,
                "QFT analytical/{nq}q[{k}]: diff={diff}"
            );
        }
    }
}
```

- [ ] **Step 2: Add module to tests/main.rs**

Add to `tests/main.rs`:

```rust
#[path = "suites/benchmark_validation.rs"]
mod benchmark_validation;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo test --all-features --no-run 2>&1 | head -20`
Expected: compiles (tests will fail at runtime if JSON files are missing — that's expected).

- [ ] **Step 4: Commit**

```bash
git add tests/suites/benchmark_validation.rs tests/main.rs
git commit -m "feat: add benchmark validation tests for single gates and QFT"
```

---

### Task 6: Density Matrix Kraus Channel Evolution (Feature Gap)

**Files:**
- Modify: `src/density_matrix.rs:184-212` (extend `Register::apply`)

Currently `DensityMatrix::apply` builds unitary columns and computes `U rho U†`. It processes the entire circuit as one unitary, which means noise channels (non-unitary) are skipped. We need to process elements one at a time: gates as `U rho U†`, channels as `sum_i K_i rho K_i†`.

- [ ] **Step 1: Write the failing test**

Add to `src/density_matrix.rs` inside the `mod tests` block:

```rust
    #[test]
    fn test_dm_apply_with_noise_channel() {
        use crate::circuit::{Circuit, channel, put};
        use crate::gate::Gate;
        use crate::noise::NoiseChannel;

        // Apply H then bit-flip(0.1) on 1 qubit
        let circ = Circuit::qubits(
            1,
            vec![
                put(vec![0], Gate::H),
                channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
            ],
        )
        .unwrap();

        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(1));
        dm.apply(&circ);

        // After H: rho = |+><+| = [[0.5, 0.5], [0.5, 0.5]]
        // After BitFlip(0.1): rho' = 0.9*rho + 0.1*X*rho*X
        // X|+><+|X = |+><+|, so rho' = |+><+| (bit flip on |+> is a no-op)
        assert_abs_diff_eq!(dm.trace().re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm.purity(), 1.0, epsilon = 1e-10);

        // Now test with Z state where bit flip actually changes the state
        let circ2 = Circuit::qubits(
            1,
            vec![
                channel(vec![0], NoiseChannel::BitFlip { p: 0.5 }),
            ],
        )
        .unwrap();
        let mut dm2 = DensityMatrix::from_reg(&ArrayReg::zero_state(1));
        dm2.apply(&circ2);

        // BitFlip(0.5) on |0><0|: rho' = 0.5*|0><0| + 0.5*|1><1| = I/2
        assert_abs_diff_eq!(dm2.trace().re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm2.purity(), 0.5, epsilon = 1e-10);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib test_dm_apply_with_noise_channel`
Expected: FAIL — the channel is skipped, purity stays 1.0 instead of 0.5.

- [ ] **Step 3: Rewrite DensityMatrix::apply to process elements individually**

Replace the `impl Register for DensityMatrix` block in `src/density_matrix.rs` (lines 179-217):

```rust
impl Register for DensityMatrix {
    fn nbits(&self) -> usize {
        self.nbits
    }

    fn apply(&mut self, circuit: &Circuit) {
        use crate::circuit::CircuitElement;

        for element in &circuit.elements {
            match element {
                CircuitElement::Gate(pg) => {
                    self.apply_gate(pg);
                }
                CircuitElement::Channel(pc) => {
                    self.apply_channel(&pc.channel, &pc.locs);
                }
                CircuitElement::Annotation(_) => {}
            }
        }
    }

    fn state_data(&self) -> &[Complex64] {
        &self.state
    }
}
```

- [ ] **Step 4: Add the `apply_gate` helper method**

Add to `impl DensityMatrix` (before the `Register` impl):

```rust
    /// Apply a single gate to the density matrix: rho -> U rho U†
    /// where U is the full-space unitary for this positioned gate.
    fn apply_gate(&mut self, pg: &crate::circuit::PositionedGate) {
        let dim = self.dim();

        // Build a single-gate circuit to reuse apply_inplace
        let single_circuit = crate::circuit::Circuit::qubits(
            self.nbits,
            vec![crate::circuit::CircuitElement::Gate(pg.clone())],
        )
        .unwrap();

        // Compute U by applying the gate to each basis vector
        let mut columns = Vec::with_capacity(dim);
        for basis_state in 0..dim {
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[basis_state] = Complex64::new(1.0, 0.0);
            let mut reg = crate::register::ArrayReg::from_vec(self.nbits, state);
            crate::apply::apply_inplace(&single_circuit, &mut reg);
            columns.push(reg.state);
        }

        // rho' = U rho U†
        let mut transformed = vec![Complex64::new(0.0, 0.0); dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                let mut acc = Complex64::new(0.0, 0.0);
                for left in 0..dim {
                    for right in 0..dim {
                        acc += columns[left][row]
                            * self.state[left * dim + right]
                            * columns[right][col].conj();
                    }
                }
                transformed[row * dim + col] = acc;
            }
        }
        self.state = transformed;
    }

    /// Apply a noise channel to the density matrix via Kraus operators.
    /// rho -> sum_i K_i rho K_i†
    /// The Kraus operators act on the subsystem specified by `locs`.
    fn apply_channel(&mut self, channel: &crate::noise::NoiseChannel, locs: &[usize]) {
        let kraus_ops = channel.kraus_operators();
        let dim = self.dim();
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim * dim];

        for kraus_op in &kraus_ops {
            // Build full-space Kraus operator by embedding into the subsystem
            let full_k = self.embed_operator(kraus_op, locs);

            // new_state += K rho K†
            for row in 0..dim {
                for col in 0..dim {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for left in 0..dim {
                        for right in 0..dim {
                            acc += full_k[row * dim + left]
                                * self.state[left * dim + right]
                                * full_k[col * dim + right].conj();
                        }
                    }
                    new_state[row * dim + col] += acc;
                }
            }
        }
        self.state = new_state;
    }

    /// Embed a local operator (acting on `locs`) into the full Hilbert space.
    /// Returns a flat Vec<Complex64> of size dim*dim representing the full operator.
    fn embed_operator(
        &self,
        local_op: &ndarray::Array2<Complex64>,
        locs: &[usize],
    ) -> Vec<Complex64> {
        let dim = self.dim();
        let local_dim = local_op.nrows();
        let mut full = vec![Complex64::new(0.0, 0.0); dim * dim];

        for row in 0..dim {
            for col in 0..dim {
                // Non-involved qubits must match
                let mut non_involved_match = true;
                for q in 0..self.nbits {
                    if !locs.contains(&q) && ((row >> q) & 1) != ((col >> q) & 1) {
                        non_involved_match = false;
                        break;
                    }
                }
                if !non_involved_match {
                    continue;
                }

                // Extract local indices for the involved qubits
                let mut local_row = 0usize;
                let mut local_col = 0usize;
                for (i, &loc) in locs.iter().enumerate() {
                    local_row |= ((row >> loc) & 1) << i;
                    local_col |= ((col >> loc) & 1) << i;
                }

                if local_row < local_dim && local_col < local_dim {
                    full[row * dim + col] = local_op[[local_row, local_col]];
                }
            }
        }
        full
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib test_dm_apply_with_noise_channel`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `make check-all`
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/density_matrix.rs
git commit -m "feat: extend DensityMatrix to apply noise channels via Kraus operators"
```

---

### Task 7: Benchmark Validation Tests — Noisy DM (Task 3 correctness)

**Files:**
- Modify: `tests/suites/benchmark_validation.rs` (add noisy DM tests)

- [ ] **Step 1: Add noisy DM validation tests**

Append to `tests/suites/benchmark_validation.rs`:

```rust
// =====================================================================
// Task 3: Noisy Density Matrix Validation
// =====================================================================

use yao_rs::{
    Circuit, DensityMatrix, NoiseChannel, OperatorPolynomial, OperatorString, Op,
    channel, expect_dm,
};

fn build_noisy_circuit(nq: usize) -> Circuit {
    use yao_rs::circuit::CircuitElement;
    let mut elements: Vec<CircuitElement> = Vec::new();

    // H on all qubits
    for q in 0..nq {
        elements.push(put(vec![q], Gate::H));
    }

    // CNOT chain
    for q in 0..(nq - 1) {
        elements.push(control(vec![q], vec![q + 1], Gate::X));
    }

    // Depolarizing noise p=0.01 on each qubit
    for q in 0..nq {
        elements.push(channel(vec![q], NoiseChannel::Depolarizing { n: 1, p: 0.01 }));
    }

    // Rz(0.3) on all qubits
    for q in 0..nq {
        elements.push(put(vec![q], Gate::Rz(0.3)));
    }

    // Amplitude damping gamma=0.05 on each qubit
    for q in 0..nq {
        elements.push(channel(
            vec![q],
            NoiseChannel::AmplitudeDamping {
                gamma: 0.05,
                excited_population: 0.0,
            },
        ));
    }

    Circuit::qubits(nq, elements).unwrap()
}

fn build_ising_hamiltonian(nq: usize) -> OperatorPolynomial {
    let mut coeffs = Vec::new();
    let mut strings = Vec::new();

    // ZZ terms
    for i in 0..(nq - 1) {
        coeffs.push(Complex64::new(1.0, 0.0));
        strings.push(OperatorString::new(vec![(i, Op::Z), (i + 1, Op::Z)]));
    }

    // 0.5 * X terms
    for i in 0..nq {
        coeffs.push(Complex64::new(0.5, 0.0));
        strings.push(OperatorString::new(vec![(i, Op::X)]));
    }

    OperatorPolynomial::new(coeffs, strings)
}

#[test]
fn test_noisy_dm_trace_and_purity() {
    let Some(data) = load_noisy_circuit() else {
        eprintln!("Skipping: noisy_circuit.json not found");
        return;
    };
    for nq in 4..=10 {
        let key = nq.to_string();
        if !data.contains_key(&key) {
            continue;
        }
        let entry = &data[&key];

        let circuit = build_noisy_circuit(nq);
        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nq));
        dm.apply(&circuit);

        let trace = dm.trace().re;
        assert!(
            (trace - entry.trace).abs() < 1e-12,
            "noisy_dm/{nq}q trace: got {trace}, expected {}",
            entry.trace
        );

        let purity = dm.purity();
        assert!(
            (purity - entry.purity).abs() < ATOL_SCALAR,
            "noisy_dm/{nq}q purity: got {purity}, expected {}",
            entry.purity
        );
    }
}

#[test]
fn test_noisy_dm_entropy() {
    let Some(data) = load_noisy_circuit() else {
        eprintln!("Skipping: noisy_circuit.json not found");
        return;
    };
    for nq in 4..=10 {
        let key = nq.to_string();
        if !data.contains_key(&key) {
            continue;
        }
        let entry = &data[&key];

        let circuit = build_noisy_circuit(nq);
        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nq));
        dm.apply(&circuit);

        let reduced = dm.partial_tr(&[nq - 1]);
        let entropy = reduced.von_neumann_entropy();
        assert!(
            (entropy - entry.entropy).abs() < ATOL_ENTROPY,
            "noisy_dm/{nq}q entropy: got {entropy}, expected {}",
            entry.entropy
        );
    }
}

#[test]
fn test_noisy_dm_expectation() {
    let Some(data) = load_noisy_circuit() else {
        eprintln!("Skipping: noisy_circuit.json not found");
        return;
    };
    for nq in 4..=10 {
        let key = nq.to_string();
        if !data.contains_key(&key) {
            continue;
        }
        let entry = &data[&key];

        let circuit = build_noisy_circuit(nq);
        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nq));
        dm.apply(&circuit);

        let h_ising = build_ising_hamiltonian(nq);
        let exp_val = expect_dm(&dm, &h_ising);

        assert!(
            (exp_val.re - entry.expect_ising.re).abs() < ATOL_SCALAR,
            "noisy_dm/{nq}q expect re: got {}, expected {}",
            exp_val.re,
            entry.expect_ising.re
        );
        assert!(
            (exp_val.im - entry.expect_ising.im).abs() < ATOL_SCALAR,
            "noisy_dm/{nq}q expect im: got {}, expected {}",
            exp_val.im,
            entry.expect_ising.im
        );
    }
}

#[test]
fn test_noisy_dm_full_matrix() {
    let Some(data) = load_noisy_circuit() else {
        eprintln!("Skipping: noisy_circuit.json not found");
        return;
    };
    for nq in 4..=6 {
        let key = nq.to_string();
        if !data.contains_key(&key) {
            continue;
        }
        let entry = &data[&key];
        if entry.density_matrix.is_none() {
            continue;
        }
        let expected_dm = interleaved_to_complex(entry.density_matrix.as_ref().unwrap());

        let circuit = build_noisy_circuit(nq);
        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nq));
        dm.apply(&circuit);

        let actual = dm.state_data();
        assert_eq!(actual.len(), expected_dm.len());
        for (i, (a, e)) in actual.iter().zip(expected_dm.iter()).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < ATOL_STATE,
                "noisy_dm/{nq}q dm[{i}]: got {a:?}, expected {e:?}, diff={diff}"
            );
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo test --all-features --no-run 2>&1 | head -20`
Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add tests/suites/benchmark_validation.rs
git commit -m "feat: add noisy density matrix benchmark validation tests"
```

---

### Task 8: Criterion Benchmarks — Single Gates

**Files:**
- Create: `benches/gates.rs`

- [ ] **Step 1: Write the benchmark file**

Create `benches/gates.rs`:

```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use yao_rs::{ArrayReg, Circuit, Gate, apply, control, put};

fn bench_gates_1q(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_1q");

    let gates: Vec<(&str, Gate)> = vec![
        ("X", Gate::X),
        ("H", Gate::H),
        ("T", Gate::T),
        ("Rx_0.5", Gate::Rx(0.5)),
        ("Rz_0.5", Gate::Rz(0.5)),
    ];

    for (name, gate) in &gates {
        for nq in [4, 8, 12, 16, 20, 25] {
            let circuit = Circuit::qubits(nq, vec![put(vec![2], gate.clone())]).unwrap();
            let reg = ArrayReg::deterministic_state(nq);
            group.bench_with_input(
                BenchmarkId::new(*name, nq),
                &nq,
                |b, _| b.iter(|| apply(black_box(&circuit), black_box(&reg))),
            );
        }
    }

    group.finish();
}

fn bench_gates_2q(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_2q");

    let gates: Vec<(&str, Box<dyn Fn(usize) -> Circuit>)> = vec![
        (
            "CNOT",
            Box::new(|nq| {
                Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::X)]).unwrap()
            }),
        ),
        (
            "CRx_0.5",
            Box::new(|nq| {
                Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::Rx(0.5))]).unwrap()
            }),
        ),
    ];

    for (name, builder) in &gates {
        for nq in [4, 8, 12, 16, 20, 25] {
            let circuit = builder(nq);
            let reg = ArrayReg::deterministic_state(nq);
            group.bench_with_input(
                BenchmarkId::new(*name, nq),
                &nq,
                |b, _| b.iter(|| apply(black_box(&circuit), black_box(&reg))),
            );
        }
    }

    group.finish();
}

fn bench_gates_multi(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_multi");

    for nq in [4, 8, 12, 16, 20, 25] {
        let circuit =
            Circuit::qubits(nq, vec![control(vec![2, 3], vec![1], Gate::X)]).unwrap();
        let reg = ArrayReg::deterministic_state(nq);
        group.bench_with_input(
            BenchmarkId::new("Toffoli", nq),
            &nq,
            |b, _| b.iter(|| apply(black_box(&circuit), black_box(&reg))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gates_1q, bench_gates_2q, bench_gates_multi);
criterion_main!(benches);
```

- [ ] **Step 2: Verify it compiles and runs**

Run: `cargo bench --bench gates -- --test`
Expected: quick test run, no errors.

- [ ] **Step 3: Commit**

```bash
git add benches/gates.rs
git commit -m "feat: add Criterion benchmarks for single gate performance"
```

---

### Task 9: Criterion Benchmarks — QFT

**Files:**
- Create: `benches/qft.rs`

- [ ] **Step 1: Write the benchmark file**

Create `benches/qft.rs`:

```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_complex::Complex64;
use yao_rs::{ArrayReg, apply};

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");

    for nq in [4, 8, 12, 16, 20, 25] {
        let circuit = yao_rs::easybuild::qft_circuit(nq);
        // |1> = |000...01>
        let mut init = vec![Complex64::new(0.0, 0.0); 1 << nq];
        init[1] = Complex64::new(1.0, 0.0);
        let reg = ArrayReg::from_vec(nq, init);

        group.bench_with_input(
            BenchmarkId::new("QFT", nq),
            &nq,
            |b, _| b.iter(|| apply(black_box(&circuit), black_box(&reg))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_qft);
criterion_main!(benches);
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo bench --bench qft -- --test`
Expected: quick test run, no errors.

- [ ] **Step 3: Commit**

```bash
git add benches/qft.rs
git commit -m "feat: add Criterion benchmarks for QFT circuit performance"
```

---

### Task 10: Criterion Benchmarks — Noisy Density Matrix

**Files:**
- Create: `benches/density.rs`

- [ ] **Step 1: Write the benchmark file**

Create `benches/density.rs`:

```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use yao_rs::circuit::CircuitElement;
use yao_rs::register::Register;
use yao_rs::{ArrayReg, Circuit, DensityMatrix, Gate, NoiseChannel, channel, control, put};

fn build_noisy_circuit(nq: usize) -> Circuit {
    let mut elements: Vec<CircuitElement> = Vec::new();

    for q in 0..nq {
        elements.push(put(vec![q], Gate::H));
    }
    for q in 0..(nq - 1) {
        elements.push(control(vec![q], vec![q + 1], Gate::X));
    }
    for q in 0..nq {
        elements.push(channel(vec![q], NoiseChannel::Depolarizing { n: 1, p: 0.01 }));
    }
    for q in 0..nq {
        elements.push(put(vec![q], Gate::Rz(0.3)));
    }
    for q in 0..nq {
        elements.push(channel(
            vec![q],
            NoiseChannel::AmplitudeDamping {
                gamma: 0.05,
                excited_population: 0.0,
            },
        ));
    }

    Circuit::qubits(nq, elements).unwrap()
}

fn bench_noisy_dm(c: &mut Criterion) {
    let mut group = c.benchmark_group("noisy_dm");
    group.sample_size(20);

    for nq in [4, 6, 8, 10] {
        let circuit = build_noisy_circuit(nq);

        group.bench_with_input(
            BenchmarkId::new("noisy_dm", nq),
            &nq,
            |b, _| {
                b.iter(|| {
                    let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nq));
                    dm.apply(black_box(&circuit));
                    dm
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_noisy_dm);
criterion_main!(benches);
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo bench --bench density -- --test`
Expected: quick test run, no errors.

- [ ] **Step 3: Commit**

```bash
git add benches/density.rs
git commit -m "feat: add Criterion benchmarks for noisy density matrix performance"
```

---

### Task 11: Performance Comparison Script

**Files:**
- Create: `benchmarks/compare.py`

- [ ] **Step 1: Write the comparison script**

Create `benchmarks/compare.py`:

```python
#!/usr/bin/env python3
"""Compare Julia (Yao.jl) and Rust (yao-rs) benchmark timings.

Usage: python3 benchmarks/compare.py
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "benchmarks" / "data"
CRITERION_DIR = ROOT / "target" / "criterion"


def load_julia_timings():
    path = DATA_DIR / "timings.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run Julia script first.")
        return None
    with open(path) as f:
        return json.load(f)


def load_criterion_estimate(group: str, name: str, param: str):
    """Load Criterion estimate for a benchmark.
    Criterion stores results at: target/criterion/{group}/{name} {param}/new/estimates.json
    """
    bench_dir = CRITERION_DIR / group / f"{name} {param}" / "new"
    est_path = bench_dir / "estimates.json"
    if not est_path.exists():
        # Try without space
        bench_dir = CRITERION_DIR / group / f"{name}/{param}" / "new"
        est_path = bench_dir / "estimates.json"
    if not est_path.exists():
        return None
    with open(est_path) as f:
        data = json.load(f)
    # Criterion stores point_estimate in nanoseconds
    return data.get("median", {}).get("point_estimate")


def main():
    julia = load_julia_timings()
    if julia is None:
        return

    rows = []

    # Single gates
    for group_key, criterion_group in [
        ("single_gate_1q", "gates_1q"),
        ("single_gate_2q", "gates_2q"),
        ("single_gate_multi", "gates_multi"),
    ]:
        if group_key not in julia:
            continue
        for gate_name, nq_data in julia[group_key].items():
            for nq_str, julia_ns in nq_data.items():
                rust_ns = load_criterion_estimate(criterion_group, gate_name, nq_str)
                rows.append(("single_gate", gate_name, nq_str, julia_ns, rust_ns))

    # QFT
    if "qft" in julia:
        for nq_str, julia_ns in julia["qft"].items():
            rust_ns = load_criterion_estimate("qft", "QFT", nq_str)
            rows.append(("qft", "QFT", nq_str, julia_ns, rust_ns))

    # Noisy DM
    if "noisy_dm" in julia:
        for nq_str, julia_ns in julia["noisy_dm"].items():
            rust_ns = load_criterion_estimate("noisy_dm", "noisy_dm", nq_str)
            rows.append(("noisy_dm", "full", nq_str, julia_ns, rust_ns))

    # Print table
    print(f"| {'Task':<14} | {'Gate/Circuit':<12} | {'Qubits':>6} | {'Julia (ns)':>12} | {'Rust (ns)':>12} | {'Speedup':>8} |")
    print(f"|{'-'*16}|{'-'*14}|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*10}|")

    for task, name, nq, j_ns, r_ns in rows:
        j_str = f"{j_ns:>12.0f}" if j_ns else "         N/A"
        if r_ns:
            r_str = f"{r_ns:>12.0f}"
            speedup = f"{j_ns / r_ns:>7.1f}x" if j_ns else "     N/A"
        else:
            r_str = "         N/A"
            speedup = "     N/A"
        print(f"| {task:<14} | {name:<12} | {nq:>6} | {j_str} | {r_str} | {speedup} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable**

Run: `chmod +x benchmarks/compare.py`

- [ ] **Step 3: Commit**

```bash
git add benchmarks/compare.py
git commit -m "feat: add performance comparison script (Julia vs Rust)"
```

---

### Task 12: Final Integration Test

**Files:** None (uses existing files)

- [ ] **Step 1: Run fmt + clippy**

Run: `make fmt && make clippy`
Expected: no warnings, no errors.

- [ ] **Step 2: Run test suite (without benchmark data)**

Run: `cargo test --all-features 2>&1 | tail -5`
Expected: all tests pass (benchmark_validation tests will be skipped/fail gracefully if JSON files are missing — verify this is the case or add a guard).

- [ ] **Step 3: Run make check-all**

Run: `make check-all`
Expected: all checks pass.

- [ ] **Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: final integration fixes for benchmark suite"
```
