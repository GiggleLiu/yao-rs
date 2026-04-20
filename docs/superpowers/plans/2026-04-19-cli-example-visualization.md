# CLI Example Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bash-reproducible mdBook visualization page for Yao.jl and QuAlgorithmZoo-style CLI examples, using generated circuit JSON, SVG, and result data.

**Architecture:** Existing `examples/cli/*.sh` scripts remain the source of runnable examples. A new artifact generator drives built-in examples and scripts into `docs/src/examples/generated/`, producing a manifest, circuits, SVGs, and results. mdBook pages then embed those generated artifacts and show exact copy-paste commands.

**Tech Stack:** Bash, Rust integration tests with `assert_cmd`-style process execution, `serde_json`, mdBook Markdown, existing `yao` CLI.

---

## Scope Check

This plan covers one cohesive subsystem: CLI example artifact generation plus mdBook documentation that consumes those artifacts. It does not implement missing quantum algorithms, training loops, richer oracles, or renderer changes.

Use the approved spec:

- `docs/superpowers/specs/2026-04-19-cli-example-visualization-design.md`

Use TDD for every behavior change:

- @test-driven-development
- @verification-before-completion

Plan review note: this harness only allows subagents when the user explicitly asks for delegated agent work, so the plan-review subagent loop has not been run.

## File Structure

- Modify `examples/cli/lib.sh`: add optional artifact-writing helpers while preserving existing stdout behavior.
- Modify `examples/cli/phase_estimation_z.sh`: call the new probability artifact helper.
- Modify `examples/cli/hadamard_test_z.sh`: call the new probability artifact helper.
- Modify `examples/cli/swap_test.sh`: call the new probability artifact helper.
- Modify `examples/cli/bernstein_vazirani.sh`: call the new probability artifact helper.
- Modify `examples/cli/grover_marked_state.sh`: call the new probability artifact helper.
- Modify `examples/cli/qaoa_maxcut_line4.sh`: call the new expectation artifact helper.
- Modify `examples/cli/qcbm_static.sh`: call the new probability artifact helper.
- Create `examples/cli/generate_artifacts.sh`: generate all checked-in docs artifacts and manifest.
- Modify `yao-cli/tests/integration.rs`: add generator and documentation regression tests.
- Create `docs/src/examples/cli-visualization.md`: mdBook-native visualization report.
- Create `docs/src/examples/generated/circuits/*.json`: generated circuit JSON artifacts.
- Create `docs/src/examples/generated/results/*.json`: generated result JSON artifacts.
- Create `docs/src/examples/generated/svg/*.svg`: generated circuit SVG artifacts.
- Create `docs/src/examples/generated/manifest.md`: generated trace table and key evidence.
- Modify `docs/src/examples/catalog.md`: link to the visualization report and mention generated artifacts.
- Modify `docs/src/SUMMARY.md`: add `CLI Example Visualization` to the Examples section.

## Chunk 1: Generator Contract

### Task 1: Add a Failing Generator Integration Test

**Files:**

- Modify: `yao-cli/tests/integration.rs`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Add helper functions for script output directories**

Add these helpers near the existing `run_cli_script_json` helper:

```rust
fn temp_dir_path(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{unique}"))
}

fn run_cli_artifact_generator(output_dir: &Path) -> Output {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    Command::new("bash")
        .arg("examples/cli/generate_artifacts.sh")
        .arg(output_dir)
        .current_dir(repo_root)
        .env("YAO_BIN", env!("CARGO_BIN_EXE_yao"))
        .output()
        .unwrap()
}
```

- [ ] **Step 2: Add the failing test**

Add this test below the existing CLI script tests:

```rust
#[test]
fn cli_artifact_generator_writes_manifest_svg_and_results() {
    let output_dir = temp_dir_path("yao-cli-artifacts");
    let output = run_cli_artifact_generator(&output_dir);
    assert!(output.status.success(), "{output:?}");

    let manifest = fs::read_to_string(output_dir.join("manifest.md")).unwrap();
    assert!(manifest.contains("YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011"));
    assert!(manifest.contains("grover-marked-5"));
    assert!(manifest.contains("qaoa-maxcut-line4-depth2"));

    let qft_svg = fs::read_to_string(output_dir.join("svg/qft4.svg")).unwrap();
    assert!(qft_svg.starts_with("<svg"));
    assert!(qft_svg.contains("</svg>"));

    let bv_result: Value =
        serde_json::from_str(&fs::read_to_string(output_dir.join("results/bernstein-vazirani-1011-probs.json")).unwrap()).unwrap();
    let probabilities = bv_result["probabilities"].as_array().unwrap();
    assert!((probabilities[11].as_f64().unwrap() - 1.0).abs() < 1e-10);

    let grover_result: Value =
        serde_json::from_str(&fs::read_to_string(output_dir.join("results/grover-marked-5-probs.json")).unwrap()).unwrap();
    let probabilities = grover_result["probabilities"].as_array().unwrap();
    assert!(probabilities[5].as_f64().unwrap() > 0.9);

    let _ = fs::remove_dir_all(output_dir);
}
```

- [ ] **Step 3: Run the test to verify RED**

Run:

```bash
cargo test -p yao-cli --test integration cli_artifact_generator_writes_manifest_svg_and_results --no-default-features
```

Expected: FAIL because `examples/cli/generate_artifacts.sh` does not exist.

### Task 2: Implement Artifact Helpers and Generator

**Files:**

- Modify: `examples/cli/lib.sh`
- Modify: `examples/cli/phase_estimation_z.sh`
- Modify: `examples/cli/hadamard_test_z.sh`
- Modify: `examples/cli/swap_test.sh`
- Modify: `examples/cli/bernstein_vazirani.sh`
- Modify: `examples/cli/grover_marked_state.sh`
- Modify: `examples/cli/qaoa_maxcut_line4.sh`
- Modify: `examples/cli/qcbm_static.sh`
- Create: `examples/cli/generate_artifacts.sh`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Extend `examples/cli/lib.sh`**

Add these helpers after `finish_circuit()`:

```bash
artifact_dir() {
  printf '%s' "${YAO_ARTIFACT_DIR:-}"
}

prepare_artifact_dirs() {
  local dir
  dir="$(artifact_dir)"
  [ -n "$dir" ] || return 0
  mkdir -p "$dir/circuits" "$dir/results" "$dir/svg"
}

write_artifacts() {
  local name="$1"
  local result_file="$2"
  local dir
  dir="$(artifact_dir)"
  [ -n "$dir" ] || return 0
  prepare_artifact_dirs
  cp "$circuit" "$dir/circuits/$name.json"
  "$YAO_BIN" visualize "$circuit" --output "$dir/svg/$name.svg"
  cp "$result_file" "$dir/results/$(basename "$result_file")"
}

finish_example_probs() {
  local name="$1"
  local result_file="${tmpdir:-$(make_example_tmpdir "$name")}/$name-probs.json"
  "$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs - | tee "$result_file"
  write_artifacts "$name" "$result_file"
}

finish_example_expect() {
  local name="$1"
  local op="$2"
  local result_file="${tmpdir:-$(make_example_tmpdir "$name")}/$name-expect.json"
  "$YAO_BIN" run "$circuit" --op "$op" | tee "$result_file"
  write_artifacts "$name" "$result_file"
}
```

Keep `simulate_probs()` unchanged for compatibility.

- [ ] **Step 2: Update probability scripts**

In each probability script, replace the final `simulate_probs` call with:

```bash
finish_example_probs "<artifact-name>"
```

Use these names:

- `phase-estimation-z`
- `hadamard-test-z`
- `swap-test`
- `bernstein-vazirani-${secret}`
- `grover-marked-${marked_value}`
- `qcbm-static-depth${depth_value}`

- [ ] **Step 3: Update QAOA script**

In `examples/cli/qaoa_maxcut_line4.sh`, replace:

```bash
"$YAO_BIN" run "$circuit" --op "Z(0)Z(1)"
```

with:

```bash
finish_example_expect "qaoa-maxcut-line4-depth${depth_value}" "Z(0)Z(1)"
```

- [ ] **Step 4: Create `examples/cli/generate_artifacts.sh`**

Create the script with this behavior:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib.sh"

out="${1:-$REPO_ROOT/docs/src/examples/generated}"
mkdir -p "$out/circuits" "$out/results" "$out/svg"

run_builtin_probs() {
  local name="$1"
  shift
  "$YAO_BIN" example "$@" --json --output "$out/circuits/$name.json"
  "$YAO_BIN" visualize "$out/circuits/$name.json" --output "$out/svg/$name.svg"
  "$YAO_BIN" simulate "$out/circuits/$name.json" | "$YAO_BIN" probs - > "$out/results/$name-probs.json"
}

run_script() {
  local script="$1"
  shift
  YAO_ARTIFACT_DIR="$out" YAO_BIN="$YAO_BIN" bash "$SCRIPT_DIR/$script" "$@" >/dev/null
}

run_builtin_probs bell bell
run_builtin_probs ghz4 ghz --nqubits 4
run_builtin_probs qft4 qft --nqubits 4

run_script phase_estimation_z.sh
run_script hadamard_test_z.sh
run_script swap_test.sh
run_script bernstein_vazirani.sh 1011
run_script grover_marked_state.sh 5
run_script qaoa_maxcut_line4.sh 2
run_script qcbm_static.sh 2
```

Append a `manifest.md` writer at the end. It should write a Markdown table with these columns: Example, Source, Command, Circuit, SVG, Result, Evidence. Include exact copy-paste commands using `YAO_BIN=target/debug/yao`.

Use this exact manifest shape, adjusting evidence only if generated outputs differ:

````bash
cat > "$out/manifest.md" <<'MANIFEST'
# Generated CLI Example Artifacts

Regenerate from the repository root:

```bash
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
```

| Example | Source | Command | Circuit | SVG | Result | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Bell | yao-rs starter / Yao basics | `target/debug/yao example bell` | [`bell.json`](./circuits/bell.json) | [`bell.svg`](./svg/bell.svg) | [`bell-probs.json`](./results/bell-probs.json) | `00` and `11` each have probability `0.5` |
| GHZ 4 | Yao GHZ pattern | `target/debug/yao example ghz --nqubits 4` | [`ghz4.json`](./circuits/ghz4.json) | [`ghz4.svg`](./svg/ghz4.svg) | [`ghz4-probs.json`](./results/ghz4-probs.json) | `0000` and `1111` each have probability `0.5` |
| QFT 4 | Yao EasyBuild QFT | `target/debug/yao example qft --nqubits 4` | [`qft4.json`](./circuits/qft4.json) | [`qft4.svg`](./svg/qft4.svg) | [`qft4-probs.json`](./results/qft4-probs.json) | Uniform distribution from `|0000>` |
| Phase estimation Z | Yao / QuAlgorithmZoo phase estimation | `YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh` | [`phase-estimation-z.json`](./circuits/phase-estimation-z.json) | [`phase-estimation-z.svg`](./svg/phase-estimation-z.svg) | [`phase-estimation-z-probs.json`](./results/phase-estimation-z-probs.json) | State `3` has probability `1.0` |
| Hadamard test Z | QuAlgorithmZoo README class | `YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh` | [`hadamard-test-z.json`](./circuits/hadamard-test-z.json) | [`hadamard-test-z.svg`](./svg/hadamard-test-z.svg) | [`hadamard-test-z-probs.json`](./results/hadamard-test-z-probs.json) | Probability output over two qubits |
| Swap test | QuAlgorithmZoo README class | `YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh` | [`swap-test.json`](./circuits/swap-test.json) | [`swap-test.svg`](./svg/swap-test.svg) | [`swap-test-probs.json`](./results/swap-test-probs.json) | Probability output over three qubits |
| Bernstein-Vazirani `1011` | QuAlgorithmZoo example | `YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011` | [`bernstein-vazirani-1011.json`](./circuits/bernstein-vazirani-1011.json) | [`bernstein-vazirani-1011.svg`](./svg/bernstein-vazirani-1011.svg) | [`bernstein-vazirani-1011-probs.json`](./results/bernstein-vazirani-1011-probs.json) | Secret state index `11` has probability `1.0` |
| Grover marked state `5` | QuAlgorithmZoo Grover class | `YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5` | [`grover-marked-5.json`](./circuits/grover-marked-5.json) | [`grover-marked-5.svg`](./svg/grover-marked-5.svg) | [`grover-marked-5-probs.json`](./results/grover-marked-5-probs.json) | Marked state index `5` has probability greater than `0.9` |
| QAOA MaxCut line4 depth2 | QuAlgorithmZoo QAOA class | `YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2` | [`qaoa-maxcut-line4-depth2.json`](./circuits/qaoa-maxcut-line4-depth2.json) | [`qaoa-maxcut-line4-depth2.svg`](./svg/qaoa-maxcut-line4-depth2.svg) | [`qaoa-maxcut-line4-depth2-expect.json`](./results/qaoa-maxcut-line4-depth2-expect.json) | `Z(0)Z(1)` expectation from `yao run --op` |
| QCBM static depth2 | Yao / QuAlgorithmZoo QCBM class | `YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2` | [`qcbm-static-depth2.json`](./circuits/qcbm-static-depth2.json) | [`qcbm-static-depth2.svg`](./svg/qcbm-static-depth2.svg) | [`qcbm-static-depth2-probs.json`](./results/qcbm-static-depth2-probs.json) | Static zero-parameter circuit; not full training |
MANIFEST
````

- [ ] **Step 5: Run the generator test to verify GREEN**

Run:

```bash
cargo test -p yao-cli --test integration cli_artifact_generator_writes_manifest_svg_and_results --no-default-features
```

Expected: PASS.

- [ ] **Step 6: Commit Chunk 1**

```bash
git add examples/cli yao-cli/tests/integration.rs
git commit -m "feat: generate CLI example artifacts"
```

## Chunk 2: Checked-In Generated Data

### Task 3: Generate Documentation Artifacts

**Files:**

- Create: `docs/src/examples/generated/circuits/*.json`
- Create: `docs/src/examples/generated/results/*.json`
- Create: `docs/src/examples/generated/svg/*.svg`
- Create: `docs/src/examples/generated/manifest.md`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Run the generator into docs**

Run:

```bash
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
```

Expected: directories `circuits`, `results`, `svg`, and file `manifest.md` exist under `docs/src/examples/generated`.

- [ ] **Step 2: Inspect generated key evidence**

Run:

```bash
jq -r '.probabilities[11]' docs/src/examples/generated/results/bernstein-vazirani-1011-probs.json
jq -r '.probabilities[5]' docs/src/examples/generated/results/grover-marked-5-probs.json
jq -r '.expectation_value.re' docs/src/examples/generated/results/qaoa-maxcut-line4-depth2-expect.json
```

Expected:

- Bernstein-Vazirani is approximately `1.0`.
- Grover is greater than `0.9`.
- QAOA expectation is a number.

- [ ] **Step 3: Run the generator test again**

Run:

```bash
cargo test -p yao-cli --test integration cli_artifact_generator_writes_manifest_svg_and_results --no-default-features
```

Expected: PASS.

- [ ] **Step 4: Commit Chunk 2**

```bash
git add docs/src/examples/generated
git commit -m "docs: add generated CLI example artifacts"
```

## Chunk 3: mdBook Page and Links

### Task 4: Add Failing Docs Regression Test

**Files:**

- Modify: `yao-cli/tests/integration.rs`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Add the docs regression test**

Add this test near the other documentation-adjacent CLI tests:

```rust
#[test]
fn cli_visualization_docs_reference_commands_and_generated_artifacts() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let page_path = repo_root.join("docs/src/examples/cli-visualization.md");
    let page = fs::read_to_string(page_path).unwrap();

    assert!(page.contains("YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated"));
    assert!(page.contains("YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011"));
    assert!(page.contains("generated/svg/qft4.svg"));
    assert!(page.contains("generated/results/grover-marked-5-probs.json"));
    assert!(page.contains("generated/manifest.md"));

    let summary = fs::read_to_string(repo_root.join("docs/src/SUMMARY.md")).unwrap();
    assert!(summary.contains("[CLI Example Visualization](./examples/cli-visualization.md)"));
}
```

- [ ] **Step 2: Run the docs test to verify RED**

Run:

```bash
cargo test -p yao-cli --test integration cli_visualization_docs_reference_commands_and_generated_artifacts --no-default-features
```

Expected: FAIL because `docs/src/examples/cli-visualization.md` is missing.

### Task 5: Write mdBook Documentation

**Files:**

- Create: `docs/src/examples/cli-visualization.md`
- Modify: `docs/src/examples/catalog.md`
- Modify: `docs/src/SUMMARY.md`
- Test: `yao-cli/tests/integration.rs`

- [ ] **Step 1: Create `docs/src/examples/cli-visualization.md`**

Use this structure:

````markdown
# CLI Example Visualization

This page shows reproducible CLI examples generated from bash workflows. The generated files live under `docs/src/examples/generated/`.

## Regenerate All Artifacts

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
```

## Run Individual Examples

```bash
YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

## Generated Trace

Include the generated manifest:

[Generated manifest](./generated/manifest.md)

## Circuit Gallery

Embed representative SVGs:

![QFT 4](./generated/svg/qft4.svg)
![Grover marked state 5](./generated/svg/grover-marked-5.svg)
![QAOA MaxCut line4 depth2](./generated/svg/qaoa-maxcut-line4-depth2.svg)
![QCBM static depth2](./generated/svg/qcbm-static-depth2.svg)

## Generated Results

Use Markdown tables that reference generated result files. Keep values consistent with generated JSON:

| Example | Result file | Key result |
| --- | --- | --- |
| Bernstein-Vazirani `1011` | [`bernstein-vazirani-1011-probs.json`](./generated/results/bernstein-vazirani-1011-probs.json) | state `1011` has probability `1.0` |
| Grover marked state `5` | [`grover-marked-5-probs.json`](./generated/results/grover-marked-5-probs.json) | state `101` has probability greater than `0.9` |
| QAOA MaxCut line4 depth2 | [`qaoa-maxcut-line4-depth2-expect.json`](./generated/results/qaoa-maxcut-line4-depth2-expect.json) | `Z(0)Z(1)` expectation real part is generated by `yao run --op` |
````

Adjust prose for polish, but preserve the command strings required by the test.

- [ ] **Step 2: Update `docs/src/SUMMARY.md`**

Add this line in the Examples section, after Example Catalog:

```markdown
- [CLI Example Visualization](./examples/cli-visualization.md)
```

- [ ] **Step 3: Update `docs/src/examples/catalog.md`**

Add a short section after `Typical Workflows`:

```markdown
## Generated Visualization

For copy-paste commands, generated circuit SVGs, and generated result summaries, see [CLI Example Visualization](./cli-visualization.md).
```

- [ ] **Step 4: Run the docs regression test to verify GREEN**

Run:

```bash
cargo test -p yao-cli --test integration cli_visualization_docs_reference_commands_and_generated_artifacts --no-default-features
```

Expected: PASS.

- [ ] **Step 5: Commit Chunk 3**

```bash
git add docs/src/SUMMARY.md docs/src/examples/catalog.md docs/src/examples/cli-visualization.md yao-cli/tests/integration.rs
git commit -m "docs: add CLI example visualization page"
```

## Chunk 4: Verification

### Task 6: Full Verification

**Files:**

- Verify all changed files.

- [ ] **Step 1: Run focused CLI integration tests**

Run:

```bash
cargo test -p yao-cli --test integration --no-default-features
```

Expected: PASS.

- [ ] **Step 2: Build mdBook**

Run:

```bash
mdbook build docs
```

Expected: PASS and generated book in `docs/book`.

- [ ] **Step 3: Run formatting check**

Run:

```bash
cargo fmt -- --check
```

Expected: PASS.

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional tracked changes remain, or a clean tree after commits. Ignore `.superpowers/` unless explicitly cleaning local brainstorming artifacts.

- [ ] **Step 5: Final commit if verification required changes**

If verification required small fixes:

```bash
git add <fixed-files>
git commit -m "chore: verify CLI example visualization docs"
```

Expected: final branch contains focused commits for generator, generated docs assets, and docs page.
