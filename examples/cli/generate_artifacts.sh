#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib.sh"

out="${1:-$REPO_ROOT/docs/src/examples/generated}"
rm -rf "$out/circuits" "$out/results" "$out/svg" "$out/plots"
rm -f "$out/manifest.md"
mkdir -p "$out/circuits" "$out/results" "$out/svg" "$out/plots"

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

python3 "$SCRIPT_DIR/plot_results.py" "$out/results" "$out/plots"

cat > "$out/manifest.md" <<'MANIFEST'
# Generated CLI Example Artifacts

This page indexes the generated circuit JSON, SVG diagrams, result JSON, and
result plots used by the CLI example visualization docs.

Regenerate these artifacts from the repository root:

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
python3 examples/cli/plot_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

Built-in sources use `target/debug/yao example ...` directly. Script sources
use `YAO_BIN=target/debug/yao bash examples/cli/...` so the shell workflows run
against the freshly built CLI. The generator also invokes
`python3 examples/cli/plot_results.py docs/src/examples/generated/results docs/src/examples/generated/plots`
to render the result plots from the generated JSON.

| Example | Source | Command | Circuit | SVG | Result | Plot | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bell | Built-in CLI | `target/debug/yao example bell --json --output docs/src/examples/generated/circuits/bell.json` | [bell.json](./circuits/bell.json) | [bell.svg](./svg/bell.svg) | [bell-probs.json](./results/bell-probs.json) | [bell-probs.svg](./plots/bell-probs.svg) | States `00` and `11` each have probability `0.5`. |
| ghz4 | Built-in CLI | `target/debug/yao example ghz --nqubits 4 --json --output docs/src/examples/generated/circuits/ghz4.json` | [ghz4.json](./circuits/ghz4.json) | [ghz4.svg](./svg/ghz4.svg) | [ghz4-probs.json](./results/ghz4-probs.json) | [ghz4-probs.svg](./plots/ghz4-probs.svg) | States `0000` and `1111` each have probability `0.5`. |
| qft4 | Built-in CLI | `target/debug/yao example qft --nqubits 4 --json --output docs/src/examples/generated/circuits/qft4.json` | [qft4.json](./circuits/qft4.json) | [qft4.svg](./svg/qft4.svg) | [qft4-probs.json](./results/qft4-probs.json) | [qft4-probs.svg](./plots/qft4-probs.svg) | Uniform 16-state distribution with probability `0.0625` per state. |
| phase-estimation-z | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh` | [phase-estimation-z.json](./circuits/phase-estimation-z.json) | [phase-estimation-z.svg](./svg/phase-estimation-z.svg) | [phase-estimation-z-probs.json](./results/phase-estimation-z-probs.json) | [phase-estimation-z-probs.svg](./plots/phase-estimation-z-probs.svg) | State `11` / index `3` has probability `1.0`. |
| hadamard-test-z | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh` | [hadamard-test-z.json](./circuits/hadamard-test-z.json) | [hadamard-test-z.svg](./svg/hadamard-test-z.svg) | [hadamard-test-z-probs.json](./results/hadamard-test-z-probs.json) | [hadamard-test-z-probs.svg](./plots/hadamard-test-z-probs.svg) | Minimal Z Hadamard-test circuit intentionally matches the phase-estimation Z demo. |
| swap-test | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh` | [swap-test.json](./circuits/swap-test.json) | [swap-test.svg](./svg/swap-test.svg) | [swap-test-probs.json](./results/swap-test-probs.json) | [swap-test-probs.svg](./plots/swap-test-probs.svg) | Nonzero states are `001`, `010`, `101`, and `110`, each with probability `0.25`. |
| bernstein-vazirani-1011 | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011` | [bernstein-vazirani-1011.json](./circuits/bernstein-vazirani-1011.json) | [bernstein-vazirani-1011.svg](./svg/bernstein-vazirani-1011.svg) | [bernstein-vazirani-1011-probs.json](./results/bernstein-vazirani-1011-probs.json) | [bernstein-vazirani-1011-probs.svg](./plots/bernstein-vazirani-1011-probs.svg) | Secret state `1011` / index `11` has probability `1.0`. |
| grover-marked-5 | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5` | [grover-marked-5.json](./circuits/grover-marked-5.json) | [grover-marked-5.svg](./svg/grover-marked-5.svg) | [grover-marked-5-probs.json](./results/grover-marked-5-probs.json) | [grover-marked-5-probs.svg](./plots/grover-marked-5-probs.svg) | Marked state `101` / index `5` has probability about `0.9453`. |
| qaoa-maxcut-line4-depth2 | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2` | [qaoa-maxcut-line4-depth2.json](./circuits/qaoa-maxcut-line4-depth2.json) | [qaoa-maxcut-line4-depth2.svg](./svg/qaoa-maxcut-line4-depth2.svg) | [qaoa-maxcut-line4-depth2-expect.json](./results/qaoa-maxcut-line4-depth2-expect.json) | [qaoa-maxcut-line4-depth2-expect.svg](./plots/qaoa-maxcut-line4-depth2-expect.svg) | `Z(0)Z(1)` expectation real part is about `0.3074`. |
| qcbm-static-depth2 | Script workflow | `YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2` | [qcbm-static-depth2.json](./circuits/qcbm-static-depth2.json) | [qcbm-static-depth2.svg](./svg/qcbm-static-depth2.svg) | [qcbm-static-depth2-probs.json](./results/qcbm-static-depth2-probs.json) | [qcbm-static-depth2-probs.svg](./plots/qcbm-static-depth2-probs.svg) | static zero-parameter demo; not full training. |
MANIFEST
