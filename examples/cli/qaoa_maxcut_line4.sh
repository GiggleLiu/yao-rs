#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

depth="${1:-2}"
require_positive_int depth "$depth"
depth_value=$((10#$depth))

gamma="${GAMMA:-0.2}"
beta_twice="${BETA_TWICE:-0.6}"

tmpdir="$(make_example_tmpdir qaoa)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/qaoa-maxcut-line4.json"

start_circuit 4
for q in 0 1 2 3; do
  append_gate H "[$q]"
done
for ((layer = 0; layer < depth_value; layer++)); do
  for edge in "0 1" "1 2" "2 3"; do
    read -r control target <<< "$edge"
    append_gate X "[$target]" "" "[$control]"
    append_gate Rz "[$target]" "$gamma"
    append_gate X "[$target]" "" "[$control]"
  done
  for q in 0 1 2 3; do
    append_gate Rx "[$q]" "$beta_twice"
  done
done
finish_circuit

"$YAO_BIN" run "$circuit" --op "Z(0)Z(1)"
