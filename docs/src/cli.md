# CLI commands

Use `yao <command> --help` for command-specific options. For setup and a complete
first run, see [installation](installation.md) and [first circuit](getting-started.md).
Input arguments accept a file path or `-` for stdin.

## Output modes

Structured results are human-readable in a terminal, JSON when piped.
`simulate` and `run` without post-processing emit binary state files; QASM
exports use raw QASM with `--output` and a JSON wrapper when piped; SVG diagrams
use SVG files. Use `--json` to force JSON for structured results in a terminal.

```bash
yao inspect bell.json              # human-readable
yao inspect bell.json --json       # force JSON
yao inspect bell.json | jq .       # auto-JSON when piped
```

Global flags available on all commands:

| Flag | Description |
|------|-------------|
| `--json` | Force JSON output |
| `-q`, `--quiet` | Suppress informational messages on stderr |
| `-o`, `--output <file>` | Write output to file |

## `yao inspect`

Display circuit information: qubit count, gate count, gate list.

```bash
yao inspect circuit.json
yao inspect circuit.json --json
cat circuit.json | yao inspect -
```

## `yao simulate`

Simulate a circuit and output the resulting quantum state. Circuits containing
noise channels automatically use a density matrix. Noiseless circuits use a
state vector unless a density-matrix input file is provided. All state-processing
commands accept both representations. Density matrices require 4^n complex
entries instead of 2^n; use [seeded trajectories](trajectories.md) for streaming
observable estimates or the tensor-network path for noise.

```bash
yao simulate circuit.json --output state.bin
yao simulate circuit.json --input initial.bin --output final.bin
yao simulate circuit.json | yao measure - --shots 100
```

Without `--output`, writes binary state data to a pipe; writing binary data directly to a terminal is rejected.

| Option | Description |
|--------|-------------|
| `--input <file>` | Input state file (defaults to \|0...0>) |
| `--output <file>` | Save state to file |

## `yao measure`

Sample measurement outcomes from a state.

```bash
yao measure state.bin --shots 1024
yao measure state.bin --shots 100 --locs 0,1
yao simulate circuit.json | yao measure - --shots 1024
```

| Option | Description |
|--------|-------------|
| `--shots <N>` | Number of measurement shots (default: 1024) |
| `--seed <u64>` | Reproduce measurement samples with the same binary version |
| `--locs <i,j,...>` | Qubit indices for partial measurement (comma-separated) |

## `yao probs`

Compute the probability distribution from a state.

```bash
yao probs state.bin
yao probs state.bin --locs 0,1
yao simulate circuit.json | yao probs -
```

| Option | Description |
|--------|-------------|
| `--locs <i,j,...>` | Qubit indices for marginal probabilities (comma-separated) |

## `yao expect`

Compute the expectation value of an operator on a state.

```bash
yao expect state.bin --op "Z(0)"
yao expect state.bin --op "0.5*Z(0)Z(1) + X(0)"
yao simulate circuit.json | yao expect - --op "Z(0)"
```

| Option | Description |
|--------|-------------|
| `--op <expr>` | Operator expression (see [operator syntax](conventions.md#operator-syntax)) |

## `yao run`

All-in-one command: simulate a circuit and optionally post-process, without intermediate files.

```bash
yao run circuit.json --shots 1024
yao run circuit.json --op "Z(0)Z(1)"
yao run circuit.json --shots 100 --locs 0,1
yao run circuit.json --output state.bin
```

| Option | Description |
|--------|-------------|
| `--input <file>` | Input state file (defaults to \|0...0>) |
| `--shots <N>` | Simulate then measure (mutually exclusive with `--op`) |
| `--seed <u64>` | Reproduce samples; requires `--shots` or `--trajectories` |
| `--trajectories <N>` | Estimate `--op` with independent noisy trajectories; conflicts with `--shots` |
| `--threads <N>` | Trajectory workers; requires `--trajectories`, and `parallel` for more than one |
| `--op <expr>` | Simulate then compute expectation (mutually exclusive with `--shots`) |
| `--locs <i,j,...>` | Qubit indices for partial measurement (used with `--shots`) |
| `--output <file>` | Save the result; binary state when neither `--shots` nor `--op` is set |

Without `--shots`, `--op`, or `--output`, the state is written to a pipe in
binary format. Writing binary state data directly to a terminal is refused.
For stochastic noise estimates, see [noisy trajectories](trajectories.md).

## `yao toeinsum`

Export a circuit as a tensor network in einsum format.

```bash
yao toeinsum circuit.json
yao toeinsum circuit.json --output tn.json
yao toeinsum circuit.json --mode dm
yao toeinsum circuit.json --mode overlap
yao toeinsum circuit.json --mode state
yao toeinsum circuit.json --op "Z(0)Z(1)"
```

| Option | Description |
|--------|-------------|
| `--mode <pure\|dm\|overlap\|state>` | Export mode: `pure` (default), `dm` (density matrix), `overlap` (scalar ⟨0\|U\|0⟩), or `state` (state vector with \|0⟩ boundary tensors) |
| `--op <expr>` | Operator expression (including sums) for expectation value TN (overrides `--mode`) |
| `--output <file>` | Save tensor network JSON to file |

See [tensor network JSON](conventions.md#tensor-network-json) for the output schema.

## `yao optimize`

Optimize contraction order for a tensor network. Requires either the `omeinsum` or `tenferro` feature.

```bash
yao optimize tn.json
yao optimize tn.json --method treesa --ntrials 20
yao toeinsum circuit.json --mode overlap | yao optimize -
```

| Option | Description |
|--------|-------------|
| `--method <greedy\|treesa>` | Optimization method (default: `greedy`) |
| `--alpha <f64>` | [greedy] Output-vs-input size balance weight (default: 0.0) |
| `--temperature <f64>` | [greedy] Temperature for stochastic selection; 0 = deterministic (default: 0.0) |
| `--ntrials <N>` | [treesa] Number of independent SA trials (default: 10) |
| `--niters <N>` | [treesa] Iterations per temperature level (default: 50) |
| `--betas <start:step:stop>` | [treesa] Inverse temperature schedule (default: "0.01:0.05:15.0") |
| `--sc-target <f64>` | [treesa] Space complexity target threshold (default: 20.0) |
| `--tc-weight <f64>` | [treesa] Time complexity weight (default: 1.0) |
| `--sc-weight <f64>` | [treesa] Space complexity weight (default: 1.0) |
| `--rw-weight <f64>` | [treesa] Read-write complexity weight (default: 0.0) |

| Memory option | Description |
|---|---|
| `--slice <labels>` | Fix comma-separated tensor labels, including signed bra labels |
| `--memory-budget <bytes>` | Estimate a memory budget and choose slices when labels are not supplied |
| `--workspace-bytes <bytes>` | Additional backend workspace reserve (default: 0) |
| `--max-slices <N>` | Maximum slice assignments (default: 1,000,000) |

Budgets are storage estimates, not process memory caps. See
[observables and memory controls](tensor-memory.md) for sliced plans.

Adds a `contraction_order` field to the TN JSON, ready for `yao contract`.

## `yao contract`

Contract a pre-optimized tensor network. Requires either the `omeinsum` or `tenferro` feature. Input must have a `contraction_order` field (produced by `yao optimize`).

```bash
yao toeinsum circuit.json | yao optimize - | yao contract -
yao toeinsum circuit.json --mode overlap | yao optimize - | yao contract -
yao toeinsum circuit.json --op "Z(0)Z(1)" | yao optimize - | yao contract -
```

With the optional [`tenferro` CLI feature](installation.md#optional-features),
select the CPU provider with:

```bash
yao contract tn.json --backend tenferro --threads 4
```

`--backend` accepts providers enabled at build time. The default is `omeinsum`
when available, otherwise `tenferro`. `--threads` accepts a positive integer and
requires tenferro; its default is 1. Both providers execute the serialized
contraction order. Tenferro reports invalid shapes/plans as errors and supports
complex128 tensors, qudit exports and exact noisy density-matrix networks.
The omeinsum provider accepts binary trees, a single unary root, and empty
networks; other node arities require tenferro.
This option selects tensor contraction; direct `simulate`/`run` use the existing
register kernels.

## `yao fromqasm`

Convert an OpenQASM 2.0 file to circuit JSON. Requires the `qasm` feature.

```bash
yao fromqasm circuit.qasm
yao fromqasm circuit.qasm --output circuit.json
yao fromqasm circuit.qasm | yao run - --shots 1024
```

## `yao toqasm`

Export a circuit as OpenQASM 2.0. Requires the `qasm` feature.

```bash
yao toqasm circuit.json
yao example bell | yao toqasm -
```

## `yao fetch`

Download benchmark circuits from online repositories.

```bash
yao fetch qasmbench list                  # List all circuits
yao fetch qasmbench list --scale small    # List only small circuits
yao fetch qasmbench grover               # Download by name (auto-detect scale)
yao fetch qasmbench qft_n4 -o qft.qasm   # Save to file
yao fetch qasmbench medium/shor_n5        # Explicit scale/name path
```

| Option | Description |
|--------|-------------|
| `--scale <small\|medium\|large>` | Filter by scale (used with `list`) |

Pipeline example:

```bash
yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100
```

## `yao example`

Print example circuit JSON to stdout.

```bash
yao example bell
yao example bell > bell.json
yao example qft --nqubits 6
```

Available examples: `bell`, `ghz`, `qft`.

| Option | Description |
|--------|-------------|
| `--nqubits <N>` | Number of qubits (default: 2 for bell, 3 for ghz, 4 for qft) |

See the [examples](examples/catalog.md) for complete algorithm walkthroughs.

## `yao visualize`

Render a circuit diagram as SVG.

```bash
yao visualize circuit.json --output circuit.svg
```

The `--output` flag is required. Only SVG output is supported.

## `yao completions`

Generate shell completion scripts.

```bash
eval "$(yao completions)"        # auto-detect shell
yao completions bash >> ~/.bashrc
yao completions zsh > _yao
```


## Data formats

See [formats and conventions](conventions.md) for circuit JSON, operator
expressions, binary states, and tensor network JSON.
