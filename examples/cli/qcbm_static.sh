#!/usr/bin/env bash
set -euo pipefail

YAO_BIN="${YAO_BIN:-yao}"
n=6
depth="${1:-2}"
if ((depth < 1)); then
  printf 'depth must be >= 1\n' >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/yao-qcbm.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/qcbm-static.json"
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

printf '{\n  "num_qubits": %s,\n  "elements": [\n' "$n" > "$circuit"
for ((layer = 0; layer <= depth; layer++)); do
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
    elif ((layer == depth)); then
      append_gate Rz "[$q]" "0.0"
      append_gate Rx "[$q]" "0.0"
    else
      append_gate Rz "[$q]" "0.0"
      append_gate Rx "[$q]" "0.0"
      append_gate Rz "[$q]" "0.0"
    fi
  done
done
printf '\n  ]\n}\n' >> "$circuit"

"$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs -
