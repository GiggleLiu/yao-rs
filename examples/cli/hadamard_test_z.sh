#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

tmpdir="$(make_example_tmpdir hadamard-test)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/hadamard-test-z.json"

start_circuit 2
append_gate X "[1]"
append_gate H "[0]"
append_gate Z "[1]" "" "[0]"
append_gate H "[0]"
finish_circuit

finish_example_probs "hadamard-test-z"
