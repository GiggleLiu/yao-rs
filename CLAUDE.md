# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

yao-rs is a Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description, qubit simulation, and tensor network export. Circuits support qudit dimensions for tensor network export, but state-vector simulation is qubit-only via `ArrayReg`.

## Philosophy
- **Simple logic, maximum reuse.** Prefer straightforward code with fewer branches (less if-else).
  Try best to reuse existing logic rather than adding ad hoc special cases.
- **Root-cause fixes over patches.** When a bug surfaces, trace it to its origin.
  A fix that prevents a class of bugs is better than one that handles a single case.
- **Tests over implementation.** Spend more time designing tests than implementing code.
  Well-designed tests catch bugs early and document intended behavior.

## Build and Test Commands

```bash
make check-all     # Run fmt-check, clippy, and test (use before commits)
make test          # Run test suite with all features
make clippy        # Run clippy with -D warnings
make fmt           # Format code
make fmt-check     # Check formatting without modifying

# Run a single test
cargo test --all-features test_name

# Documentation
make doc           # Build mdBook documentation
make rustdoc       # Build and open Rust API docs
```

## Architecture

### Core Data Flow

```
Gate → PositionedGate → Circuit → TensorNetwork / ArrayReg
```

- **Gate** (`gate.rs`): Enum of quantum gates (X, Y, Z, H, S, T, SWAP, Phase, Rx, Ry, Rz, SqrtX, SqrtY, SqrtW, ISWAP, FSim) with matrix representations. Named gates require d=2 (qubits); `Gate::Custom` supports arbitrary qudit matrices.

- **PositionedGate** (`circuit.rs`): Gate + placement info (`target_locs`, `control_locs`, `control_configs`). Built via `put()` for uncontrolled gates or `control()` for controlled gates.

- **CircuitElement** (`circuit.rs`): Enum wrapping `Gate(PositionedGate)`, `Annotation(PositionedAnnotation)`, or `Channel(PositionedChannel)`. Annotations are visual-only labels (no-op in execution, rendered in PDF diagrams). Channels represent noise (used by density-matrix/tensor-network paths only).

- **Circuit** (`circuit.rs`): Validated sequence of `CircuitElement`s on a register with specified dimensions (`dims: Vec<usize>`). Constructor validates: control sites must be qubits, no overlapping control/target locs, matrix size matches target dimensions.

- **ArrayReg** (`register.rs`): Dense qubit-only state vector (`Vec<Complex64>`). The sole register type for state-vector simulation. Constructed via `zero_state(n)`, `product_state(BitStr)`, `from_vec(n, data)`, etc.

### Two Execution Paths

1. **Direct simulation** (`apply.rs`, `instruct_qubit.rs`): Qubit state-vector simulation via `apply(&circuit, &reg)` / `apply_inplace(&circuit, &mut reg)`. Uses bit-manipulation-based instruct functions from the `bitbasis` crate for zero-allocation hot loops. Dispatches by gate type: X fast-path, SWAP fast-path, 1q/2q diagonal, 1q/2q general, nq fallback.

2. **Tensor network export & contraction** (`einsum.rs`, `tensors.rs`, `contractor.rs`): Convert circuit to einsum via `circuit_to_einsum()`. Returns `TensorNetwork` with `EinCode<usize>` (from omeco crate), tensors, and size dictionary. Supports arbitrary qudit dimensions. Diagonal gates without controls share input/output legs (optimization). Native contraction via `omeinsum` submodule (`contractor.rs`, feature-gated).

### Key Design Patterns

- **Diagonal gate optimization**: `Gate::is_diagonal()` enables tensor leg reuse in einsum and specialized stride-based application paths.
- **Control configurations**: `control_configs: Vec<bool>` allows active-low controls (trigger on |0⟩), though the `control()` helper defaults to active-high.
- **Qubit-only simulation**: `ArrayReg` and `instruct_qubit` use bit manipulation (via `bitbasis` crate) for O(1) index computation. Qudit circuits are supported for tensor network export only.
- **bitbasis crate** (`bitbasis/`): Workspace member providing `BitStr<N>`, `IterControl`, bit operations (`bmask`, `flip`, `indicator`, etc.). Ported from Julia's BitBasis.jl.

### Module Overview

- `register.rs`: `ArrayReg` qubit register and `Register` trait
- `instruct_qubit.rs`: Bit-manipulation-based qubit instruct functions (1q, 2q, nq, diagonal, controlled, X/SWAP fast paths)
- `apply.rs`: Circuit application to `ArrayReg`
- `measure.rs`: Measurement operations (`probs`, `measure_with_postprocess` with `PostProcess::NoPostProcess/ResetTo/RemoveMeasured`)
- `density_matrix.rs`: Density matrix representation for mixed states
- `expect.rs`: Expectation values for `ArrayReg` and `DensityMatrix`
- `easybuild.rs`: Pre-built circuits (QFT, variational ansatz, phase estimation, Google supremacy)
- `operator.rs`: Pauli operators and polynomials for expectation values
- `json.rs`: Circuit serialization
- `einsum.rs`, `tensors.rs`: Tensor network export (supports qudits)
- `svg.rs`: Built-in SVG circuit renderer for `Circuit::to_svg()`
- `noise.rs`: Noise channel definitions
- `qasm.rs` (feature-gated): OpenQASM 2.0 import/export via the `openqasm` crate. Import decomposes all gates to U+CX primitives. Export maps Gate variants to standard qelib1.inc names.
- `contractor.rs` (feature-gated `omeinsum`): Native tensor network contraction via [omeinsum-rs](https://github.com/tensor4all/omeinsum-rs) submodule. Converts `TensorNetwork` ndarray tensors to omeinsum column-major format, contracts with greedy optimization, returns column-major `ArrayD`.
- `torch_contractor.rs` (feature-gated `torch`): libtorch-based tensor network contraction via omeco greedy optimizer
- SVG rendering is built in and available through `Circuit::to_svg()`.

## Test Layout

- **Unit tests** (`src/unit_tests/`): Each source file owns its tests via `#[cfg(test)] #[path = "unit_tests/xxx.rs"] mod tests;`. These compile with the lib crate for fast incremental builds.
- **Integration tests** (`tests/suites/`): Cross-module tests that depend on shared helpers (`tests/common/`) and JSON ground-truth data (`tests/data/`). Single entry point at `tests/main.rs`. Qudit-specific ground-truth cases are skipped (qubit-only simulation).

## CLI Tool (`yao-cli/`)

The `yao` CLI wraps the library for command-line use. Build with `cargo install --path yao-cli`.

```bash
yao example bell              # Print example circuit JSON (bell, ghz, qft)
yao example qft --nqubits 6   # QFT with custom qubit count
yao inspect circuit.json       # Display circuit info
yao run circuit.json --shots 1024   # Simulate and measure
yao run circuit.json --op "Z(0)Z(1)"  # Expectation value
yao simulate circuit.json | yao probs -   # Pipeline workflow
yao toeinsum circuit.json      # Export as tensor network (pure state)
yao toeinsum circuit.json --mode overlap  # Overlap ⟨0|U|0⟩ TN
yao toeinsum circuit.json --mode state    # State vector TN with |0⟩ boundary
yao toeinsum circuit.json --op "Z(0)Z(1)" # Expectation value TN
yao optimize tn.json           # Optimize contraction order (greedy default)
yao optimize tn.json --method treesa --ntrials 20  # TreeSA optimizer
yao contract tn.json           # Contract pre-optimized tensor network
yao toeinsum circuit.json --mode state | yao optimize - | yao contract -  # Full pipeline
yao fromqasm circuit.qasm      # Import OpenQASM 2.0 circuit
yao toqasm circuit.json        # Export circuit as OpenQASM 2.0
yao fetch qasmbench list       # List QASMBench benchmark circuits
yao fetch qasmbench grover     # Download circuit (auto-detect scale)
yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100
```

All commands output human-readable text in a terminal, JSON when piped. Use `--json` to force JSON. Use `-` for stdin input.

## Submodules

- **`omeinsum-rs/`**: [omeinsum-rs](https://github.com/tensor4all/omeinsum-rs) — native Rust tensor network contraction. Initialize with `git submodule update --init`. Required for the `omeinsum` feature.

## Feature Flags

- `omeinsum`: Native tensor network contraction via omeinsum (enabled by default in CLI, requires submodule)
- `parallel`: Enable rayon for parallel operations
- `qasm`: OpenQASM 2.0 import/export (enabled by default in CLI)
- `torch`: PyTorch tensor contraction via tch (requires libtorch)

## Claude Skills

- `/issue-to-pr <number>` — Convert a GitHub issue into a PR with a brainstormed plan that triggers automated execution
- `/fix-pr` — Address PR review comments, fix CI failures, and resolve codecov coverage gaps for the current branch
- `/review-implementation` — Dispatch review subagents to check code quality, test coverage, and correctness before committing
- `/plan-review` — Execute a plan end-to-end: implement, review, Copilot review, fix CI loop, and squash merge

## Automation

```bash
make run-plan                          # Execute latest plan in docs/plans/ with Claude
make run-plan PLAN_FILE=docs/plans/foo.md  # Execute a specific plan
```
