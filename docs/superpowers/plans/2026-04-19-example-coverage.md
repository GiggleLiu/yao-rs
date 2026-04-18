# Example Coverage Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cover Yao.jl documentation examples and `/tmp/QuAlgorithmZoo.jl` examples through `yao` CLI examples where feasible, and record unsupported algorithm families as GitHub issues.

**Architecture:** Keep `yao example <name>` as the circuit-producing entry point. Reuse `src/easybuild.rs` for reusable circuit builders, keep CLI-only parameter parsing in `yao-cli/src/commands/example.rs`, and document all supported and deferred examples in a catalog page.

**Tech Stack:** Rust 2024, clap, serde_json, yao-rs circuit JSON, existing `ArrayReg` simulation, mdBook docs, GitHub CLI for Tier 3 issues.

---

## Source Inventory

Primary spec: `docs/superpowers/specs/2026-04-19-example-coverage-design.md`

Local source inventory:

- Yao docs examples: GHZ, QFT, phase estimation, QCBM.
- `/tmp/QuAlgorithmZoo.jl`: QFT, phase estimation, hadamard test, swap test, VQE, QSVD, GroundStateSolvers, HHL, QAOA, QCBM, QuGAN, Shor, Grover, GateLearning, QMPS, PortZygote, PortQuantumInformation, DiffEq, ErrorCorrection/Shor code, Mermin Magic Square, Bernstein-Vazirani.

Audit result for first implementation wave:

- Implement now: `phase-estimation`, `hadamard-test`, `swap-test`, `bernstein-vazirani`, `grover`, `qaoa-maxcut`, `qcbm`.
- Keep existing and document better: `bell`, `ghz`, `qft`.
- Defer with issues: VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, GroundStateSolvers, DiffEq, HHL, QSVD, Shor, Grover inference/variational variants, Mermin Magic Square, PortZygote, PortQuantumInformation, ErrorCorrection/Shor code.

Mermin Magic Square is deferred because the source is a measurement-game workflow over commuting Pauli observables, not a circuit-generation example. Current `yao example` emits circuit JSON and the CLI does not yet model multi-observable game rounds.

## File Map

- Modify `yao-cli/src/cli.rs`: add example-specific CLI options to `Commands::Example`.
- Modify `yao-cli/src/main.rs`: pass all example options into `commands::example`.
- Modify `yao-cli/src/commands/example.rs`: add `ExampleOptions`, dispatch names, option validation, unitary presets, and CLI-only circuit builders where appropriate.
- Modify `src/easybuild.rs`: add reusable builders for Bernstein-Vazirani, marked-state Grover, QAOA MaxCut ansatz, and QCBM ansatz if they are useful outside the CLI.
- Modify `src/unit_tests/easybuild.rs`: add focused tests for new public builders.
- Modify `yao-cli/tests/integration.rs`: add CLI integration tests for new examples and invalid parameters.
- Modify `docs/src/SUMMARY.md`: add the example catalog page.
- Create `docs/src/examples/catalog.md`: source coverage matrix, commands, expected results, and deferred issue links.
- Modify `docs/src/cli.md`: update `yao example` docs and available examples.
- Create GitHub issues for Tier 3 groups using `gh issue create`.

## Chunk 1: Tier 3 Issues And CLI Option Scaffolding

### Task 1: Record Tier 3 Deferred Coverage As Issues

**Files:**
- No repo files are required.
- External state: GitHub issues in `GiggleLiu/yao-rs`.

- [ ] **Step 1: Check existing deferred coverage issues**

Run:

```bash
gh issue list --repo GiggleLiu/yao-rs --state all --search "algorithm-zoo deferred" --json number,title,state,labels
```

Expected: identify whether matching issues already exist. If they do, update them instead of creating duplicates.

- [ ] **Step 2: Ensure labels exist**

Run:

```bash
gh label create examples --repo GiggleLiu/yao-rs --color 0e8a16 --description "Example coverage and documentation" || true
gh label create algorithm-zoo --repo GiggleLiu/yao-rs --color 5319e7 --description "QuAlgorithmZoo and Yao.jl example coverage" || true
gh label create deferred --repo GiggleLiu/yao-rs --color cfd3d7 --description "Deferred follow-up work" || true
```

Expected: labels are created or already exist. If `gh label create` is unavailable, create the issues without labels and add labels manually afterward.

- [ ] **Step 3: Create issue for optimization and training loops**

Run:

```bash
gh issue create --repo GiggleLiu/yao-rs \
  --title "Track deferred QuAlgorithmZoo coverage: optimization and training loops" \
  --label examples --label algorithm-zoo --label deferred \
  --body "Source examples: VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, PortZygote.

Missing yao-rs capability:
- Parameter storage and dispatch for trainable circuits.
- Optimizers or optimizer integration.
- Gradient, parameter-shift, or autodiff workflow.
- Training-loop examples with stable validation.

Minimal implementation target:
- One trainable ansatz with explicit parameters.
- One optimizer-backed CLI or example workflow.
- A deterministic small test that verifies loss decreases or reaches a known optimum.

Expected CLI surface after capability exists:
- A command or example workflow that can run a short training loop and emit metrics.

Validation strategy:
- Compare small examples with Yao.jl where possible.
- Prefer deterministic seeds and analytical sanity checks."
```

Expected: issue URL printed.

- [ ] **Step 4: Create issue for Hamiltonian and time-evolution algorithms**

Run:

```bash
gh issue create --repo GiggleLiu/yao-rs \
  --title "Track deferred QuAlgorithmZoo coverage: Hamiltonian and time-evolution algorithms" \
  --label examples --label algorithm-zoo --label deferred \
  --body "Source examples: GroundStateSolvers, DiffEq, HHL, QSVD.

Missing yao-rs capability:
- Hamiltonian abstractions beyond the current Pauli expectation DSL.
- Controlled time evolution or block encoding builders.
- Linear-system and singular-value workflows.

Minimal implementation target:
- A small Hamiltonian API or adapter.
- A controlled time-evolution circuit example.
- A validation case with known analytical behavior.

Expected CLI surface after capability exists:
- Example workflows for HHL-like and time-evolution demos.

Validation strategy:
- Compare small circuits against direct dense simulation and Yao.jl reference outputs."
```

Expected: issue URL printed.

- [ ] **Step 5: Create issue for Shor arithmetic and number-theory oracles**

Run:

```bash
gh issue create --repo GiggleLiu/yao-rs \
  --title "Track deferred QuAlgorithmZoo coverage: arithmetic and Shor oracles" \
  --label examples --label algorithm-zoo --label deferred \
  --body "Source examples: Shor, ErrorCorrection/Shor code if it needs related reusable circuits.

Missing yao-rs capability:
- Modular multiplication and modular exponentiation circuit builders.
- Classical preprocessing and continued-fraction postprocessing.
- Scalable arithmetic oracle construction.

Minimal implementation target:
- A small order-finding circuit for a toy modulus.
- Tests for modular arithmetic builders.
- A CLI example that factors 15 or records why the circuit is only illustrative.

Expected CLI surface after capability exists:
- A Shor/order-finding example with clear size limits.

Validation strategy:
- Compare with classical order finding and Yao.jl toy cases."
```

Expected: issue URL printed.

- [ ] **Step 6: Create issue for richer Grover oracle families**

Run:

```bash
gh issue create --repo GiggleLiu/yao-rs \
  --title "Track deferred QuAlgorithmZoo coverage: richer Grover oracles" \
  --label examples --label algorithm-zoo --label deferred \
  --body "Source examples: Grover inference oracle, Grover variational-generator amplitude amplification.

Missing yao-rs capability:
- Oracle constructors from signed evidence constraints.
- Subspace predicates and validation helpers.
- Non-Hadamard state-preparation generators.

Minimal implementation target:
- A reusable predicate-oracle builder.
- A small inference-oracle example.
- Tests proving the amplified subspace probability increases.

Expected CLI surface after capability exists:
- Extended Grover examples beyond a single marked basis state.

Validation strategy:
- Compare small predicate cases against brute-force expected subspaces."
```

Expected: issue URL printed.

- [ ] **Step 7: Create issue for measurement-game and external workflow examples**

Run:

```bash
gh issue create --repo GiggleLiu/yao-rs \
  --title "Track deferred QuAlgorithmZoo coverage: measurement games and external workflows" \
  --label examples --label algorithm-zoo --label deferred \
  --body "Source examples: Mermin Magic Square, PortQuantumInformation, external chemistry workflows from VQE OpenFermion.

Missing yao-rs capability:
- Multi-observable measurement game workflow.
- Higher-level information-theory examples.
- External data import and operator conversion strategy.

Minimal implementation target:
- A Mermin-Peres Magic Square validation helper or CLI workflow.
- A documented import strategy for external data examples.
- Tests for the game consistency or imported operator conversion.

Expected CLI surface after capability exists:
- Non-circuit-only example workflows that can emit structured results.

Validation strategy:
- Compare Mermin game consistency with the Julia source behavior.
- Use small fixed imported operators for external workflows."
```

Expected: issue URL printed.

- [ ] **Step 8: Verify Tier 3 issues exist**

Run:

```bash
gh issue list --repo GiggleLiu/yao-rs --state open --label algorithm-zoo --label deferred
```

Expected: all new or updated Tier 3 issue titles are listed.

### Task 2: Add CLI Option Scaffolding

**Files:**
- Modify: `yao-cli/src/cli.rs`
- Modify: `yao-cli/src/main.rs`
- Modify: `yao-cli/src/commands/example.rs`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Write failing CLI tests for a new example option**

Add this test helper and test to `yao-cli/tests/integration.rs`:

```rust
fn run_yao_json(args: &[&str]) -> Value {
    let output = run_yao(args);
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn example_phase_estimation_accepts_preset_and_register_size() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "phase-estimation",
        "--nqubits",
        "3",
        "--preset",
        "z",
    ]);

    assert_eq!(json["num_qubits"].as_u64().unwrap(), 4);
    assert!(json["elements"].as_array().unwrap().len() > 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p yao-cli example_phase_estimation_accepts_preset_and_register_size
```

Expected: FAIL because `phase-estimation` is unknown or `--preset` is not accepted.

- [ ] **Step 3: Add `ExampleOptions` and CLI fields**

In `yao-cli/src/cli.rs`, replace the `Example` variant fields with:

```rust
Example {
    /// Example name: bell, ghz, qft, phase-estimation, hadamard-test, swap-test,
    /// bernstein-vazirani, grover, qaoa-maxcut, qcbm
    name: String,
    /// Number of qubits or register qubits, depending on the example.
    #[arg(long)]
    nqubits: Option<usize>,
    /// Preset name for examples that need a small built-in unitary or graph.
    #[arg(long)]
    preset: Option<String>,
    /// Secret bit string for Bernstein-Vazirani, e.g. 10101.
    #[arg(long)]
    secret: Option<String>,
    /// Marked basis-state index for Grover.
    #[arg(long)]
    marked: Option<usize>,
    /// Grover iteration count, or "auto".
    #[arg(long)]
    iterations: Option<String>,
    /// Ansatz depth for QAOA and QCBM.
    #[arg(long)]
    depth: Option<usize>,
    /// Phase parameter for phase-estimation, hadamard-test, and swap-test presets.
    #[arg(long)]
    phase: Option<f64>,
    /// Number of qubits per compared state for swap-test.
    #[arg(long)]
    nqubits_per_state: Option<usize>,
    /// Number of states for swap-test.
    #[arg(long)]
    nstates: Option<usize>,
}
```

In `yao-cli/src/commands/example.rs`, add:

```rust
#[derive(Debug, Default, Clone)]
pub struct ExampleOptions {
    pub nqubits: Option<usize>,
    pub preset: Option<String>,
    pub secret: Option<String>,
    pub marked: Option<usize>,
    pub iterations: Option<String>,
    pub depth: Option<usize>,
    pub phase: Option<f64>,
    pub nqubits_per_state: Option<usize>,
    pub nstates: Option<usize>,
}
```

Update `example` signature:

```rust
pub fn example(name: &str, opts: ExampleOptions, out: &OutputConfig) -> Result<()>
```

In `yao-cli/src/main.rs`, construct and pass `ExampleOptions` from `Commands::Example`.

- [ ] **Step 4: Keep existing examples working**

Update the existing dispatch to use `opts.nqubits`:

```rust
let circuit = match name {
    "bell" => bell(opts.nqubits.unwrap_or(2))?,
    "ghz" => ghz(opts.nqubits.unwrap_or(3))?,
    "qft" => qft(opts.nqubits.unwrap_or(4))?,
    _ => bail!("Unknown example: '{name}'\n\nAvailable examples: bell, ghz, qft"),
};
```

Do not implement `phase-estimation` in this task yet.

- [ ] **Step 5: Run tests to verify scaffolding preserves existing behavior**

Run:

```bash
cargo test -p yao-cli example_phase_estimation_accepts_preset_and_register_size
cargo test -p yao-cli visualize_writes_svg_in_default_build
```

Expected: phase-estimation test still FAILS because behavior is not implemented; existing visualization test PASSES.

- [ ] **Step 6: Commit scaffolding**

Run:

```bash
git add yao-cli/src/cli.rs yao-cli/src/main.rs yao-cli/src/commands/example.rs yao-cli/tests/integration.rs
git commit -m "test(cli): scaffold algorithm example options"
```

## Chunk 2: Existing Easybuild Examples Exposed In CLI

### Task 3: Add Phase Estimation, Hadamard Test, And Swap Test CLI Examples

**Files:**
- Modify: `yao-cli/src/commands/example.rs`
- Modify: `yao-cli/tests/integration.rs`
- Modify: `yao-cli/src/cli.rs`

- [ ] **Step 1: Add failing tests for the three existing builders**

Add tests:

```rust
#[test]
fn example_hadamard_test_uses_existing_builder() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "hadamard-test",
        "--preset",
        "z",
        "--phase",
        "0.3",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 2);
    assert!(json["elements"].as_array().unwrap().len() >= 4);
}

#[test]
fn example_swap_test_uses_existing_builder() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "swap-test",
        "--nqubits-per-state",
        "2",
        "--nstates",
        "2",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 5);
    assert!(json["elements"].as_array().unwrap().len() >= 4);
}
```

The phase-estimation test from Task 2 remains the failing test for that example.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cargo test -p yao-cli example_
```

Expected: FAIL for all three new examples.

- [ ] **Step 3: Implement preset unitary helper**

Add helper:

```rust
fn preset_unitary(name: Option<&str>, phase: Option<f64>) -> Result<Gate> {
    match name.unwrap_or("z") {
        "z" => Ok(Gate::Z),
        "x" => Ok(Gate::X),
        "t" => Ok(Gate::T),
        "phase" => Ok(Gate::Phase(phase.unwrap_or(std::f64::consts::PI / 4.0))),
        other => bail!("Unknown unitary preset: '{other}' (available: z, x, t, phase)"),
    }
}
```

If `phase_estimation_circuit` needs a custom gate for matrix powers, pass `Gate::Z`, `Gate::X`, `Gate::T`, or `Gate::Phase`.

- [ ] **Step 4: Implement existing builder dispatch**

Add match arms:

```rust
"phase-estimation" => {
    let n_reg = opts.nqubits.unwrap_or(3);
    let unitary = preset_unitary(opts.preset.as_deref(), opts.phase)?;
    yao_rs::easybuild::phase_estimation_circuit(unitary, n_reg, 1)
}
"hadamard-test" => {
    let unitary = preset_unitary(opts.preset.as_deref(), opts.phase)?;
    yao_rs::easybuild::hadamard_test_circuit(unitary, opts.phase.unwrap_or(0.0))
}
"swap-test" => {
    let nbit = opts.nqubits_per_state.unwrap_or(1);
    let nstate = opts.nstates.unwrap_or(2);
    if nbit == 0 {
        bail!("swap-test requires --nqubits-per-state >= 1");
    }
    if nstate < 2 {
        bail!("swap-test requires --nstates >= 2");
    }
    yao_rs::easybuild::swap_test_circuit(nbit, nstate, opts.phase.unwrap_or(0.0))
}
```

Update unknown-example help to list all supported names implemented so far.

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
cargo test -p yao-cli example_
```

Expected: PASS.

- [ ] **Step 6: Add invalid parameter test**

Add:

```rust
#[test]
fn example_swap_test_rejects_one_state() {
    let output = run_yao(&[
        "example",
        "swap-test",
        "--nqubits-per-state",
        "1",
        "--nstates",
        "1",
    ]);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("swap-test requires --nstates >= 2"), "{stderr}");
}
```

- [ ] **Step 7: Run targeted CLI tests**

Run:

```bash
cargo test -p yao-cli example_
```

Expected: all example-related tests pass.

- [ ] **Step 8: Commit existing-builder examples**

Run:

```bash
git add yao-cli/src/commands/example.rs yao-cli/src/cli.rs yao-cli/tests/integration.rs
git commit -m "feat(cli): expose easybuild examples"
```

## Chunk 3: Bernstein-Vazirani And Marked-State Grover

### Task 4: Add Bernstein-Vazirani Builder And CLI Example

**Files:**
- Modify: `src/easybuild.rs`
- Modify: `src/unit_tests/easybuild.rs`
- Modify: `yao-cli/src/commands/example.rs`
- Modify: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Write failing library test**

Add to `src/unit_tests/easybuild.rs`:

```rust
use crate::measure::probs;

fn msb_bits_to_index(bits: &[bool]) -> usize {
    bits.iter()
        .fold(0usize, |acc, &bit| (acc << 1) | usize::from(bit))
}

#[test]
fn test_bernstein_vazirani_recovers_secret() {
    let secret = [true, false, true, true];
    let circuit = crate::easybuild::bernstein_vazirani_circuit(&secret);
    let result = apply(&circuit, &ArrayReg::zero_state(secret.len()));
    let probabilities = probs(&result, None);
    let expected = msb_bits_to_index(&secret);

    assert_abs_diff_eq!(probabilities[expected], 1.0, epsilon = 1e-10);
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cargo test -p yao-rs test_bernstein_vazirani_recovers_secret
```

Expected: FAIL because `bernstein_vazirani_circuit` does not exist.

- [ ] **Step 3: Implement builder**

Add to `src/easybuild.rs`:

```rust
/// Build a Bernstein-Vazirani circuit for a phase oracle defined by `secret`.
///
/// The circuit is H on every qubit, Z on each secret bit set to 1, then H on
/// every qubit. Starting from |0...0>, measurement returns `secret`.
pub fn bernstein_vazirani_circuit(secret: &[bool]) -> Circuit {
    assert!(!secret.is_empty(), "secret must not be empty");

    let n = secret.len();
    let mut elements: Vec<CircuitElement> = Vec::new();

    for q in 0..n {
        elements.push(put(vec![q], Gate::H));
    }
    for (q, &bit) in secret.iter().enumerate() {
        if bit {
            elements.push(put(vec![q], Gate::Z));
        }
    }
    for q in 0..n {
        elements.push(put(vec![q], Gate::H));
    }

    Circuit::qubits(n, elements).unwrap()
}
```

- [ ] **Step 4: Run library test to verify pass**

Run:

```bash
cargo test -p yao-rs test_bernstein_vazirani_recovers_secret
```

Expected: PASS.

- [ ] **Step 5: Write failing CLI test**

Add to `yao-cli/tests/integration.rs`:

```rust
#[test]
fn example_bernstein_vazirani_accepts_secret() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "bernstein-vazirani",
        "--secret",
        "1011",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 4);
    assert!(json["elements"].as_array().unwrap().len() >= 8);
}
```

- [ ] **Step 6: Run CLI test to verify failure**

Run:

```bash
cargo test -p yao-cli example_bernstein_vazirani_accepts_secret
```

Expected: FAIL because CLI dispatch is missing.

- [ ] **Step 7: Implement CLI parsing**

Add helper in `yao-cli/src/commands/example.rs`:

```rust
fn parse_secret_bits(secret: Option<&str>) -> Result<Vec<bool>> {
    let secret = secret.ok_or_else(|| anyhow::anyhow!("bernstein-vazirani requires --secret"))?;
    if secret.is_empty() {
        bail!("bernstein-vazirani requires a non-empty --secret");
    }
    secret
        .chars()
        .map(|ch| match ch {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => bail!("secret must contain only 0 and 1"),
        })
        .collect()
}
```

Add dispatch:

```rust
"bernstein-vazirani" => {
    let secret = parse_secret_bits(opts.secret.as_deref())?;
    yao_rs::easybuild::bernstein_vazirani_circuit(&secret)
}
```

- [ ] **Step 8: Run CLI test to verify pass**

Run:

```bash
cargo test -p yao-cli example_bernstein_vazirani_accepts_secret
```

Expected: PASS.

- [ ] **Step 9: Commit Bernstein-Vazirani**

Run:

```bash
git add src/easybuild.rs src/unit_tests/easybuild.rs yao-cli/src/commands/example.rs yao-cli/tests/integration.rs
git commit -m "feat: add Bernstein-Vazirani example"
```

### Task 5: Add Marked-State Grover Builder And CLI Example

**Files:**
- Modify: `src/easybuild.rs`
- Modify: `src/unit_tests/easybuild.rs`
- Modify: `yao-cli/src/commands/example.rs`
- Modify: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Write failing Grover library test**

Add:

```rust
#[test]
fn test_marked_state_grover_amplifies_marked_state() {
    let n = 3;
    let marked = 5;
    let circuit = crate::easybuild::marked_state_grover_circuit(n, marked, 2);
    let result = apply(&circuit, &ArrayReg::zero_state(n));
    let probabilities = probs(&result, None);

    assert!(probabilities[marked] > 0.9, "marked probability = {}", probabilities[marked]);
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cargo test -p yao-rs test_marked_state_grover_amplifies_marked_state
```

Expected: FAIL because builder does not exist.

- [ ] **Step 3: Implement Grover helper builders**

Add to `src/easybuild.rs`:

```rust
fn marked_oracle_gate(n: usize, marked: usize) -> Gate {
    let dim = 1usize << n;
    let mut matrix = Array2::zeros((dim, dim));
    for i in 0..dim {
        matrix[[i, i]] = if i == marked {
            Complex64::new(-1.0, 0.0)
        } else {
            Complex64::new(1.0, 0.0)
        };
    }
    Gate::Custom {
        matrix,
        is_diagonal: true,
        label: format!("Oracle({marked})"),
    }
}

fn diffusion_gate(n: usize) -> Gate {
    let dim = 1usize << n;
    let fill = 2.0 / dim as f64;
    let mut matrix = Array2::from_elem((dim, dim), Complex64::new(fill, 0.0));
    for i in 0..dim {
        matrix[[i, i]] -= Complex64::new(1.0, 0.0);
    }
    Gate::Custom {
        matrix,
        is_diagonal: false,
        label: "Diffusion".to_string(),
    }
}

/// Build a marked-basis-state Grover search circuit.
pub fn marked_state_grover_circuit(n: usize, marked: usize, iterations: usize) -> Circuit {
    assert!(n > 0, "Grover requires at least one qubit");
    assert!(n <= 8, "Grover example uses dense custom gates and is limited to 8 qubits");
    assert!(marked < (1usize << n), "marked state out of range");

    let targets: Vec<usize> = (0..n).collect();
    let mut elements: Vec<CircuitElement> = Vec::new();

    for q in 0..n {
        elements.push(put(vec![q], Gate::H));
    }
    for _ in 0..iterations {
        elements.push(put(targets.clone(), marked_oracle_gate(n, marked)));
        elements.push(put(targets.clone(), diffusion_gate(n)));
    }

    Circuit::qubits(n, elements).unwrap()
}

pub fn grover_auto_iterations(n: usize, marked_count: usize) -> usize {
    assert!(marked_count > 0, "marked_count must be positive");
    let dim = 1usize << n;
    ((std::f64::consts::PI / 4.0) * ((dim as f64) / (marked_count as f64)).sqrt()).round() as usize
}
```

- [ ] **Step 4: Run Grover library test**

Run:

```bash
cargo test -p yao-rs test_marked_state_grover_amplifies_marked_state
```

Expected: PASS.

- [ ] **Step 5: Add failing CLI tests**

Add:

```rust
#[test]
fn example_grover_accepts_marked_state() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "grover",
        "--nqubits",
        "3",
        "--marked",
        "5",
        "--iterations",
        "auto",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 3);
    assert!(json["elements"].as_array().unwrap().len() >= 3);
}

#[test]
fn example_grover_rejects_out_of_range_marked_state() {
    let output = run_yao(&["example", "grover", "--nqubits", "3", "--marked", "8"]);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("marked state out of range"), "{stderr}");
}
```

- [ ] **Step 6: Implement CLI Grover dispatch**

Add helper:

```rust
fn grover_iterations(n: usize, iterations: Option<&str>) -> Result<usize> {
    match iterations.unwrap_or("auto") {
        "auto" => Ok(yao_rs::easybuild::grover_auto_iterations(n, 1)),
        value => value
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("--iterations must be a non-negative integer or auto")),
    }
}
```

Add dispatch:

```rust
"grover" => {
    let n = opts.nqubits.unwrap_or(3);
    if n == 0 || n > 8 {
        bail!("grover requires 1 <= --nqubits <= 8");
    }
    let marked = opts.marked.unwrap_or((1usize << n) - 1);
    if marked >= (1usize << n) {
        bail!("marked state out of range for {n} qubits");
    }
    let iterations = grover_iterations(n, opts.iterations.as_deref())?;
    yao_rs::easybuild::marked_state_grover_circuit(n, marked, iterations)
}
```

- [ ] **Step 7: Run targeted tests**

Run:

```bash
cargo test -p yao-rs grover
cargo test -p yao-cli example_grover
```

Expected: PASS.

- [ ] **Step 8: Commit Grover**

Run:

```bash
git add src/easybuild.rs src/unit_tests/easybuild.rs yao-cli/src/commands/example.rs yao-cli/tests/integration.rs
git commit -m "feat: add marked-state Grover example"
```

## Chunk 4: Static QAOA And QCBM Circuit Examples

### Task 6: Add QAOA MaxCut Static Ansatz

**Files:**
- Modify: `src/easybuild.rs`
- Modify: `src/unit_tests/easybuild.rs`
- Modify: `yao-cli/src/commands/example.rs`
- Modify: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Write failing QAOA builder test**

Add:

```rust
#[test]
fn test_qaoa_maxcut_ansatz_builds_and_preserves_norm() {
    let edges = [(0usize, 1usize, 1.0f64), (1, 2, 1.0), (2, 3, 1.0)];
    let circuit = crate::easybuild::qaoa_maxcut_circuit(4, &edges, &[0.2], &[0.3]);
    let result = apply(&circuit, &ArrayReg::zero_state(4));

    assert_eq!(circuit.num_sites(), 4);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cargo test -p yao-rs test_qaoa_maxcut_ansatz_builds_and_preserves_norm
```

Expected: FAIL because builder does not exist.

- [ ] **Step 3: Implement QAOA builder**

Add:

```rust
/// Build a static QAOA MaxCut ansatz.
///
/// This emits the circuit only. It does not optimize parameters.
pub fn qaoa_maxcut_circuit(
    n: usize,
    edges: &[(usize, usize, f64)],
    gammas: &[f64],
    betas: &[f64],
) -> Circuit {
    assert!(n > 0, "QAOA requires at least one qubit");
    assert_eq!(gammas.len(), betas.len(), "gammas and betas must have equal length");

    let mut elements: Vec<CircuitElement> = Vec::new();
    for q in 0..n {
        elements.push(put(vec![q], Gate::H));
    }

    for (&gamma, &beta) in gammas.iter().zip(betas.iter()) {
        for &(u, v, weight) in edges {
            assert!(u < n && v < n && u != v, "invalid MaxCut edge");
            elements.push(control(vec![u], vec![v], Gate::X));
            elements.push(put(vec![v], Gate::Rz(gamma * weight)));
            elements.push(control(vec![u], vec![v], Gate::X));
        }
        for q in 0..n {
            elements.push(put(vec![q], Gate::Rx(2.0 * beta)));
        }
    }

    Circuit::qubits(n, elements).unwrap()
}
```

- [ ] **Step 4: Add CLI preset and tests**

Add test:

```rust
#[test]
fn example_qaoa_maxcut_uses_line4_preset() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "qaoa-maxcut",
        "--preset",
        "line4",
        "--depth",
        "2",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 4);
    assert!(json["elements"].as_array().unwrap().len() > 10);
}
```

In `example.rs`, add helper:

```rust
fn qaoa_preset_edges(preset: Option<&str>) -> Result<(usize, Vec<(usize, usize, f64)>)> {
    match preset.unwrap_or("line4") {
        "line4" => Ok((4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])),
        "triangle" => Ok((3, vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])),
        other => bail!("Unknown qaoa-maxcut preset: '{other}' (available: line4, triangle)"),
    }
}
```

Add dispatch:

```rust
"qaoa-maxcut" => {
    let depth = opts.depth.unwrap_or(1);
    if depth == 0 {
        bail!("qaoa-maxcut requires --depth >= 1");
    }
    let (n, edges) = qaoa_preset_edges(opts.preset.as_deref())?;
    let gammas = vec![0.2; depth];
    let betas = vec![0.3; depth];
    yao_rs::easybuild::qaoa_maxcut_circuit(n, &edges, &gammas, &betas)
}
```

- [ ] **Step 5: Run targeted QAOA tests**

Run:

```bash
cargo test -p yao-rs qaoa
cargo test -p yao-cli example_qaoa_maxcut
```

Expected: PASS.

- [ ] **Step 6: Commit QAOA**

Run:

```bash
git add src/easybuild.rs src/unit_tests/easybuild.rs yao-cli/src/commands/example.rs yao-cli/tests/integration.rs
git commit -m "feat: add static QAOA example"
```

### Task 7: Add QCBM Static Ansatz CLI Example

**Files:**
- Modify: `yao-cli/src/commands/example.rs`
- Modify: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Write failing CLI test**

Add:

```rust
#[test]
fn example_qcbm_builds_variational_ansatz() {
    let json = run_yao_json(&[
        "--json",
        "example",
        "qcbm",
        "--nqubits",
        "6",
        "--depth",
        "2",
    ]);
    assert_eq!(json["num_qubits"].as_u64().unwrap(), 6);
    assert!(json["elements"].as_array().unwrap().len() > 20);
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cargo test -p yao-cli example_qcbm_builds_variational_ansatz
```

Expected: FAIL because `qcbm` is not implemented.

- [ ] **Step 3: Implement CLI dispatch using existing variational builder**

Add dispatch:

```rust
"qcbm" => {
    let n = opts.nqubits.unwrap_or(6);
    let depth = opts.depth.unwrap_or(6);
    if n == 0 {
        bail!("qcbm requires --nqubits >= 1");
    }
    if depth == 0 {
        bail!("qcbm requires --depth >= 1");
    }
    let pairs = yao_rs::easybuild::pair_ring(n);
    yao_rs::easybuild::variational_circuit(n, depth, &pairs)
}
```

- [ ] **Step 4: Run QCBM CLI test**

Run:

```bash
cargo test -p yao-cli example_qcbm_builds_variational_ansatz
```

Expected: PASS.

- [ ] **Step 5: Commit QCBM**

Run:

```bash
git add yao-cli/src/commands/example.rs yao-cli/tests/integration.rs
git commit -m "feat(cli): add static QCBM example"
```

## Chunk 5: Documentation, Coverage Matrix, And Final Verification

### Task 8: Add Example Catalog Documentation

**Files:**
- Modify: `docs/src/SUMMARY.md`
- Modify: `docs/src/cli.md`
- Create: `docs/src/examples/catalog.md`

- [ ] **Step 1: Write documentation page**

Create `docs/src/examples/catalog.md` with:

```markdown
# Example Catalog

This page maps Yao.jl documentation and QuAlgorithmZoo.jl examples to yao-rs CLI workflows.

## Supported CLI Examples

| Example | Source | Command | Notes |
|---------|--------|---------|-------|
| Bell | yao-rs starter example | `yao example bell` | Produces a 2-qubit Bell circuit. |
| GHZ | Yao docs GHZ | `yao example ghz --nqubits 5` | Produces GHZ entanglement. |
| QFT | Yao docs QFT | `yao example qft --nqubits 6` | Matches Yao EasyBuild QFT without final swaps. |
| Phase estimation | Yao docs / QuAlgorithmZoo | `yao example phase-estimation --nqubits 3 --preset z` | Emits circuit JSON for a small preset unitary. |
| Hadamard test | QuAlgorithmZoo README | `yao example hadamard-test --preset z` | Uses the existing easybuild circuit. |
| Swap test | QuAlgorithmZoo README | `yao example swap-test --nqubits-per-state 2 --nstates 2` | Uses the existing easybuild circuit. |
| Bernstein-Vazirani | QuAlgorithmZoo | `yao example bernstein-vazirani --secret 1011` | Measurement concentrates on the secret. |
| Grover | QuAlgorithmZoo | `yao example grover --nqubits 3 --marked 5 --iterations auto` | Dense custom-gate marked-state demo, limited to small sizes. |
| QAOA MaxCut | QuAlgorithmZoo | `yao example qaoa-maxcut --preset line4 --depth 2` | Static ansatz only, no optimizer. |
| QCBM | Yao docs / QuAlgorithmZoo | `yao example qcbm --nqubits 6 --depth 2` | Static variational ansatz only, no training. |

## Typical Workflows

```bash
yao example bernstein-vazirani --secret 1011 | yao run - --shots 128
yao example grover --nqubits 3 --marked 5 --iterations auto | yao probs -
yao example qft --nqubits 4 | yao visualize - --output qft.svg
yao example qaoa-maxcut --preset line4 --depth 1 | yao toeinsum - --mode overlap | yao optimize - | yao contract -
```

## Deferred Coverage

The following examples are tracked by GitHub issues because they require capabilities beyond circuit JSON generation:

| Group | Source examples | Issue |
|-------|-----------------|-------|
| Optimization and training | VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, PortZygote | TODO: link issue |
| Hamiltonian and time evolution | GroundStateSolvers, DiffEq, HHL, QSVD | TODO: link issue |
| Arithmetic oracles | Shor | TODO: link issue |
| Rich Grover oracles | Grover inference and variational-generator variants | TODO: link issue |
| Measurement games and external workflows | Mermin Magic Square, PortQuantumInformation, external chemistry import | TODO: link issue |
```

Replace TODO issue links with the issue URLs created in Task 1.

- [ ] **Step 2: Link page from summary**

Add to `docs/src/SUMMARY.md` under examples:

```markdown
- [Example Catalog](./examples/catalog.md)
```

- [ ] **Step 3: Update CLI docs**

In `docs/src/cli.md`, update `yao example` section:

```markdown
Available examples: `bell`, `ghz`, `qft`, `phase-estimation`, `hadamard-test`, `swap-test`, `bernstein-vazirani`, `grover`, `qaoa-maxcut`, `qcbm`.
```

Add option rows for `--preset`, `--secret`, `--marked`, `--iterations`, `--depth`, `--phase`, `--nqubits-per-state`, and `--nstates`.

- [ ] **Step 4: Build docs**

Run:

```bash
mdbook build docs
```

Expected: PASS.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add docs/src/SUMMARY.md docs/src/cli.md docs/src/examples/catalog.md
git commit -m "docs: add example catalog"
```

### Task 9: Final Verification

**Files:**
- All modified files from prior tasks.

- [ ] **Step 1: Run formatting**

Run:

```bash
cargo fmt
```

Expected: no errors.

- [ ] **Step 2: Run targeted library tests**

Run:

```bash
cargo test -p yao-rs easybuild
```

Expected: all easybuild tests pass.

- [ ] **Step 3: Run CLI tests**

Run:

```bash
cargo test -p yao-cli
```

Expected: all CLI unit and integration tests pass.

- [ ] **Step 4: Run docs build**

Run:

```bash
mdbook build docs
```

Expected: docs build succeeds.

- [ ] **Step 5: Optional full check**

Run only if local `torch`/libtorch setup is available:

```bash
make check-all
```

Expected: fmt-check, clippy, and all tests pass.

- [ ] **Step 6: Inspect final diff and status**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: only intended commits on the branch. Note any pre-existing unrelated changes, especially the `omeinsum-rs` submodule pointer if still present.

## Execution Notes

- Follow TDD for every code-producing task: failing test first, run it, implement, run it again.
- Keep dense custom gates size-limited. Grover uses dense custom oracle and diffusion matrices, so the CLI should reject `--nqubits > 8`.
- Do not implement optimizers, autodiff, chemistry imports, HHL, Shor, or Mermin Magic Square game logic in this plan. They are issue-tracked follow-ups.
- Keep CLI output behavior unchanged: human-readable in terminal, JSON when piped or with `--json`, JSON to file when `--output` is used.
- Do not stage or commit unrelated `omeinsum-rs` submodule pointer changes unless the user explicitly asks for it.
