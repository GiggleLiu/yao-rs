#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

n=6
depth="${1:-2}"
require_positive_int depth "$depth"
depth_value=$((10#$depth))

# Deterministic fixed parameter schedule. Cycles a small pool of non-zero
# values so the output distribution is non-trivial (not the |0>^n spike
# produced when all rotations are the identity) but still reproducible.
# Used by docs/src/examples/qcbm.md as a CLI vs. numpy reference cross-check.
thetas=(0.11 0.23 0.37 0.51 0.67 0.79 0.89)
pool="${#thetas[@]}"
param_idx=0

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
      append_gate Rx "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
      append_gate Rz "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
    elif ((layer == depth_value)); then
      append_gate Rz "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
      append_gate Rx "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
    else
      append_gate Rz "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
      append_gate Rx "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
      append_gate Rz "[$q]" "${thetas[$((param_idx % pool))]}"
      param_idx=$((param_idx + 1))
    fi
  done
done
finish_circuit

finish_example_probs "qcbm-static-depth${depth_value}"
