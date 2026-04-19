#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

tmpdir="$(make_example_tmpdir phase-estimation)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/phase-estimation-z.json"

start_circuit 2
append_gate X "[1]"
append_gate H "[0]"
append_gate Z "[1]" "" "[0]"
append_gate H "[0]"
finish_circuit

simulate_probs
