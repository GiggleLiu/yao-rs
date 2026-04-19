#!/usr/bin/env bash
set -euo pipefail

YAO_BIN="${YAO_BIN:-yao}"
secret="${1:-1011}"
if [[ ! "$secret" =~ ^[01]+$ ]]; then
  printf 'secret must contain only 0 and 1\n' >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/yao-bernstein-vazirani.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
circuit="$tmpdir/bernstein-vazirani.json"
n=${#secret}
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
printf '\n  ]\n}\n' >> "$circuit"

"$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs -
