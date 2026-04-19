#!/usr/bin/env bash
set -euo pipefail

YAO_BIN="${YAO_BIN:-yao}"
depth="${1:-2}"
gamma="${GAMMA:-0.2}"
beta_twice="${BETA_TWICE:-0.6}"
if ((depth < 1)); then
  printf 'depth must be >= 1\n' >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/yao-qaoa.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/qaoa-maxcut-line4.json"
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

printf '{\n  "num_qubits": 4,\n  "elements": [\n' > "$circuit"
for q in 0 1 2 3; do
  append_gate H "[$q]"
done
for ((layer = 0; layer < depth; layer++)); do
  for edge in "0 1" "1 2" "2 3"; do
    set -- $edge
    append_gate X "[$2]" "" "[$1]"
    append_gate Rz "[$2]" "$gamma"
    append_gate X "[$2]" "" "[$1]"
  done
  for q in 0 1 2 3; do
    append_gate Rx "[$q]" "$beta_twice"
  done
done
printf '\n  ]\n}\n' >> "$circuit"

"$YAO_BIN" run "$circuit" --op "Z(0)Z(1)"
