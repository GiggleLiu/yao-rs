#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

n=3
marked="${1:-5}"
require_non_negative_int marked "$marked"
marked_value=$((10#$marked))
if ((marked_value >= (1 << n))); then
  die "marked state out of range for $n qubits"
fi

tmpdir="$(make_example_tmpdir grover)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/grover-marked-state.json"

marked_bit() {
  local q="$1"
  printf '%s' "$(((marked_value >> (n - 1 - q)) & 1))"
}

oracle() {
  for ((q = 0; q < n; q++)); do
    if [ "$(marked_bit "$q")" = "0" ]; then
      append_gate X "[$q]"
    fi
  done
  append_gate Z "[2]" "" "[0, 1]"
  for ((q = 0; q < n; q++)); do
    if [ "$(marked_bit "$q")" = "0" ]; then
      append_gate X "[$q]"
    fi
  done
}

diffusion() {
  for ((q = 0; q < n; q++)); do
    append_gate H "[$q]"
  done
  for ((q = 0; q < n; q++)); do
    append_gate X "[$q]"
  done
  append_gate Z "[2]" "" "[0, 1]"
  for ((q = 0; q < n; q++)); do
    append_gate X "[$q]"
  done
  for ((q = 0; q < n; q++)); do
    append_gate H "[$q]"
  done
}

start_circuit "$n"
for ((q = 0; q < n; q++)); do
  append_gate H "[$q]"
done
for ((i = 0; i < 2; i++)); do
  oracle
  diffusion
done
finish_circuit

finish_example_probs "grover-marked-${marked_value}"
