# Shared helpers for the bash-based CLI examples.

: "${YAO_BIN:=yao}"

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

make_example_tmpdir() {
  local name="$1"
  mktemp -d "${TMPDIR:-/tmp}/yao-${name}.XXXXXX"
}

require_non_negative_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  require_non_negative_int "$name" "$value"
  ((10#$value >= 1)) || die "$name must be >= 1"
}

start_circuit() {
  local num_qubits="$1"
  first=1
  printf '{\n  "num_qubits": %s,\n  "elements": [\n' "$num_qubits" > "$circuit"
}

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

finish_circuit() {
  printf '\n  ]\n}\n' >> "$circuit"
}

simulate_probs() {
  "$YAO_BIN" simulate "$circuit" | "$YAO_BIN" probs -
}
