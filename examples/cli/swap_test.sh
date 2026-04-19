#!/usr/bin/env bash
set -euo pipefail

YAO_BIN="${YAO_BIN:-yao}"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/yao-swap-test.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/swap-test.json"
first=1

append_gate() {
  local gate="$1"
  local targets="$2"
  local params="${3:-}"
  local controls="${4:-}"
  if [ "$first" -eq 0 ]; then
    printf ',\n' >> "$circuit"
  fi
  first=0
  printf '    { "type": "gate", "gate": "%s", "targets": %s' "$gate" "$targets" >> "$circuit"
  if [ -n "$params" ]; then
    printf ', "params": [%s]' "$params" >> "$circuit"
  fi
  if [ -n "$controls" ]; then
    printf ', "controls": %s' "$controls" >> "$circuit"
  fi
  printf ' }' >> "$circuit"
}

printf '{\n  "num_qubits": 3,\n  "elements": [\n' > "$circuit"
append_gate X "[2]"
append_gate H "[0]"
append_gate SWAP "[1, 2]" "" "[0]"
append_gate H "[0]"
printf '\n  ]\n}\n' >> "$circuit"

"$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs -
