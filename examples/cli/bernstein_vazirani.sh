#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

secret="${1:-1011}"
if [[ ! "$secret" =~ ^[01]+$ ]]; then
  die "secret must contain only 0 and 1"
fi

tmpdir="$(make_example_tmpdir bernstein-vazirani)"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/bernstein-vazirani.json"
n=${#secret}

start_circuit "$n"
for ((q = 0; q < n; q++)); do
  append_gate H "[$q]"
done
for ((q = 0; q < n; q++)); do
  if [ "${secret:q:1}" = "1" ]; then
    append_gate Z "[$q]"
  fi
done
for ((q = 0; q < n; q++)); do
  append_gate H "[$q]"
done
finish_circuit

finish_example_probs "bernstein-vazirani-${secret}"
