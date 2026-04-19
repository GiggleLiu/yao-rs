#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

tmpdir="$(make_example_tmpdir swap-test)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/swap-test.json"

start_circuit 3
append_gate X "[2]"
append_gate H "[0]"
append_gate SWAP "[1, 2]" "" "[0]"
append_gate H "[0]"
finish_circuit

simulate_probs
