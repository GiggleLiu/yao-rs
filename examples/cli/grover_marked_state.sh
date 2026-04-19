#!/usr/bin/env bash
set -euo pipefail

YAO_BIN="${YAO_BIN:-yao}"
n=3
marked="${1:-5}"
if ((marked < 0 || marked >= (1 << n))); then
  printf 'marked state out of range for %s qubits\n' "$n" >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/yao-grover.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/grover-marked-state.json"
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

marked_bit() {
  local q="$1"
  printf '%s' "$(((marked >> (n - 1 - q)) & 1))"
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

printf '{\n  "num_qubits": %s,\n  "elements": [\n' "$n" > "$circuit"
for ((q = 0; q < n; q++)); do
  append_gate H "[$q]"
done
for ((i = 0; i < 2; i++)); do
  oracle
  diffusion
done
printf '\n  ]\n}\n' >> "$circuit"

"$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs -
