#!/usr/bin/env bash
set -euo pipefail

# Build the QFT-n circuit acting on the uniform-superposition input
# |+>^n = H^n |0>^n. Because QFT|+>^n = |0>^n (identity on n-th roots of
# unity), the measured distribution should be a delta at index 0. Used
# by docs/src/examples/qft.md as a phase-structure verification — unlike
# the canonical |0>^n input, where QFT and H^n produce the same
# (uniform) output, this check requires every controlled-phase gate to
# carry its correct rotation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

n="${1:-4}"
require_positive_int n "$n"
n_value=$((10#$n))

tmpdir="$(make_example_tmpdir qft-plus)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/qft-from-plus.json"

start_circuit "$n_value"

# Prepare |+>^n with an initial H layer.
for ((q = 0; q < n_value; q++)); do
  append_gate H "[$q]"
done

# QFT_n ladder (same layout as `yao example qft --nqubits n`, no final SWAPs).
for ((k = 0; k < n_value; k++)); do
  append_gate H "[$k]"
  for ((j = k + 1; j < n_value; j++)); do
    exp=$((j - k))
    angle=$(awk -v e="$exp" 'BEGIN { printf "%.16f", 3.141592653589793 / (2 ^ e) }')
    append_gate Phase "[$k]" "$angle" "[$j]"
  done
done

finish_circuit

finish_example_probs "qft${n_value}-from-plus"
