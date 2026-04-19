#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib.sh"

out="${1:-$REPO_ROOT/docs/src/examples/generated}"
rm -rf "$out/circuits" "$out/results" "$out/svg"
rm -f "$out/manifest.md"
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

cat > "$out/manifest.md" <<'MANIFEST'
| Example | Source | Command | Circuit | SVG | Result | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| bell | built-in | `YAO_BIN=target/debug/yao target/debug/yao example bell --json --output docs/src/examples/generated/circuits/bell.json` | `circuits/bell.json` | `svg/bell.svg` | `results/bell-probs.json` | Bell probabilities split between 00 and 11. |
| ghz4 | built-in | `YAO_BIN=target/debug/yao target/debug/yao example ghz --nqubits 4 --json --output docs/src/examples/generated/circuits/ghz4.json` | `circuits/ghz4.json` | `svg/ghz4.svg` | `results/ghz4-probs.json` | GHZ probabilities split between 0000 and 1111. |
| qft4 | built-in | `YAO_BIN=target/debug/yao target/debug/yao example qft --nqubits 4 --json --output docs/src/examples/generated/circuits/qft4.json` | `circuits/qft4.json` | `svg/qft4.svg` | `results/qft4-probs.json` | Uniform 16-state distribution from the zero basis state. |
| phase-estimation-z | script | `YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh` | `circuits/phase-estimation-z.json` | `svg/phase-estimation-z.svg` | `results/phase-estimation-z-probs.json` | Phase bit 11 has probability 1. |
| hadamard-test-z | script | `YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh` | `circuits/hadamard-test-z.json` | `svg/hadamard-test-z.svg` | `results/hadamard-test-z-probs.json` | Minimal Z Hadamard-test circuit; intentionally matches the phase-estimation Z demo. |
| swap-test | script | `YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh` | `circuits/swap-test.json` | `svg/swap-test.svg` | `results/swap-test-probs.json` | Swap-test circuit emits an eight-state distribution. |
| bernstein-vazirani-1011 | script | `YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011` | `circuits/bernstein-vazirani-1011.json` | `svg/bernstein-vazirani-1011.svg` | `results/bernstein-vazirani-1011-probs.json` | Secret index 1011 has probability 1. |
| grover-marked-5 | script | `YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5` | `circuits/grover-marked-5.json` | `svg/grover-marked-5.svg` | `results/grover-marked-5-probs.json` | Marked index 5 has amplified probability above 0.9. |
| qaoa-maxcut-line4-depth2 | script | `YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2` | `circuits/qaoa-maxcut-line4-depth2.json` | `svg/qaoa-maxcut-line4-depth2.svg` | `results/qaoa-maxcut-line4-depth2-expect.json` | Reports expectation for Z(0)Z(1). |
| qcbm-static-depth2 | script | `YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2` | `circuits/qcbm-static-depth2.json` | `svg/qcbm-static-depth2.svg` | `results/qcbm-static-depth2-probs.json` | Static QCBM emits a 64-state distribution. |
MANIFEST
