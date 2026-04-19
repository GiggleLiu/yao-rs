#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

n=6
depth="${1:-2}"
require_positive_int depth "$depth"
depth_value=$((10#$depth))

tmpdir="$(make_example_tmpdir qcbm)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/qcbm-static.json"

start_circuit "$n"
for ((layer = 0; layer <= depth_value; layer++)); do
  if ((layer > 0)); then
    for ((q = 0; q < n; q++)); do
      target=$(((q + 1) % n))
      append_gate X "[$target]" "" "[$q]"
    done
  fi

  for ((q = 0; q < n; q++)); do
    if ((layer == 0)); then
      append_gate Rx "[$q]" "0.0"
      append_gate Rz "[$q]" "0.0"
    elif ((layer == depth_value)); then
      append_gate Rz "[$q]" "0.0"
      append_gate Rx "[$q]" "0.0"
    else
      append_gate Rz "[$q]" "0.0"
      append_gate Rx "[$q]" "0.0"
      append_gate Rz "[$q]" "0.0"
    fi
  done
done
finish_circuit

simulate_probs
