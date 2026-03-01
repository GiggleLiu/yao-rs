# Noise Model Support for yao-rs

**Date**: 2026-03-01
**Issue**: [GiggleLiu/tn-mcp-rs#8](https://github.com/GiggleLiu/tn-mcp-rs/issues/8)
**Branch**: `jg/noisy`

## Overview

Add noise channel support to yao-rs with full Yao.jl parity. Noise channels are represented as circuit elements and converted to superoperator tensors in a density-matrix-mode tensor network export. No density matrix state-vector simulation — noise only works through the einsum/TN path.

## Approach

- **Superoperator-based**: All noise channels are converted to superoperator matrices via `sum_i kron(conj(K_i), K_i)`, then inserted as `D^(4k)`-index tensors in the density matrix einsum.
- **Circuit integration**: Noise channels are explicit `CircuitElement::Channel` variants, matching Yao.jl's block-based approach.
- **Separate DM functions**: New `circuit_to_einsum_dm()` and `circuit_to_expectation_dm()` functions; existing API unchanged.
- **Label scheme**: `isize` labels with negation for dual (bra) indices, matching Yao.jl exactly.

## Component 1: `NoiseChannel` Enum (`src/noise.rs`)

```rust
pub enum NoiseChannel {
    BitFlip { p: f64 },
    PhaseFlip { p: f64 },
    Depolarizing { n: usize, p: f64 },
    PauliChannel { px: f64, py: f64, pz: f64 },
    Reset { p0: f64, p1: f64 },
    AmplitudeDamping { gamma: f64, excited_population: f64 },
    PhaseDamping { gamma: f64 },
    PhaseAmplitudeDamping { amplitude: f64, phase: f64, excited_population: f64 },
    ThermalRelaxation { t1: f64, t2: f64, time: f64, excited_population: f64 },
    Coherent { matrix: Array2<Complex64> },
    Custom { kraus_ops: Vec<Array2<Complex64>> },
}
```

### Methods

- `kraus_operators(&self) -> Vec<Array2<Complex64>>` — Construct Kraus matrices
- `superop(&self) -> Array2<Complex64>` — Build superoperator matrix
- `num_qubits(&self) -> usize` — Number of qubits the channel acts on
- Constructors with validation matching Yao.jl constraints

## Component 2: Circuit Integration (`src/circuit.rs`)

```rust
pub struct PositionedChannel {
    pub channel: NoiseChannel,
    pub locs: Vec<usize>,
}

pub enum CircuitElement {
    Gate(PositionedGate),
    Annotation(PositionedAnnotation),
    Channel(PositionedChannel),  // NEW
}
```

- `apply.rs` skips `Channel` elements (noise only in TN path)
- Validation: channel locs valid, qubit count matches channel

## Component 3: Density Matrix Einsum (`src/einsum.rs`)

### New type

```rust
pub struct TensorNetworkDM {
    pub code: EinCode<isize>,
    pub tensors: Vec<ArrayD<Complex64>>,
    pub size_dict: HashMap<isize, usize>,
}
```

### New functions

- `circuit_to_einsum_dm(circuit) -> TensorNetworkDM`
- `circuit_to_expectation_dm(circuit, operator) -> TensorNetworkDM`

### Density matrix mode algorithm

For each gate in the circuit:
1. **Unitary gate**: Add tensor with labels `[out, in]` AND conjugate tensor with labels `[-out, -in]` (ref: `push_normal_tensor!`)
2. **Diagonal gate**: Reuse labels for both ket and bra copies
3. **Noise channel**: Convert to superoperator, reshape to `D^(4k)` tensor, labels = `[out, -out, in, -in]` (ref: `add_channel!`)

Output indices: `[slots, -slots]` for full density matrix, or traced for expectation values.

## Component 4: Testing

- Julia-generated fixtures in `tests/data/noise.json`
- Test file: `tests/unit_tests/noise.rs`
- Test categories:
  1. Kraus operator matrices match Julia
  2. Superoperator = sum_i kron(conj(K_i), K_i)
  3. Noisy einsum matches Julia `yao2einsum(; mode=DensityMatrixMode())`
  4. Pure circuit DM mode matches squared amplitudes from VectorMode

## Julia Reference Map

| Rust | Julia File | Julia Function/Type |
|---|---|---|
| `NoiseChannel::BitFlip` | `lib/YaoBlocks/src/channel/errortypes.jl:31` | `BitFlipError` |
| `NoiseChannel::PhaseFlip` | `lib/YaoBlocks/src/channel/errortypes.jl:49` | `PhaseFlipError` |
| `NoiseChannel::Depolarizing` | `lib/YaoBlocks/src/channel/errortypes.jl:73` | `DepolarizingError` |
| `NoiseChannel::PauliChannel` | `lib/YaoBlocks/src/channel/errortypes.jl:99` | `PauliError` |
| `NoiseChannel::Reset` | `lib/YaoBlocks/src/channel/errortypes.jl:154` | `ResetError` |
| `NoiseChannel::AmplitudeDamping` | `lib/YaoBlocks/src/channel/errortypes.jl:351` | `AmplitudeDampingError` |
| `NoiseChannel::PhaseDamping` | `lib/YaoBlocks/src/channel/errortypes.jl:319` | `PhaseDampingError` |
| `NoiseChannel::PhaseAmplitudeDamping` | `lib/YaoBlocks/src/channel/errortypes.jl:256` | `PhaseAmplitudeDampingError` |
| `NoiseChannel::ThermalRelaxation` | `lib/YaoBlocks/src/channel/errortypes.jl:205` | `ThermalRelaxationError` |
| `NoiseChannel::Coherent` | `lib/YaoBlocks/src/channel/errortypes.jl:12` | `CoherentError` |
| `NoiseChannel::Custom` | `lib/YaoBlocks/src/channel/kraus.jl:25` | `KrausChannel` |
| `kraus_operators()` | `lib/YaoBlocks/src/channel/errortypes.jl:168-296` | `KrausChannel(err::PhaseAmplitudeDampingError)` etc. |
| `superop()` | `lib/YaoBlocks/src/channel/kraus.jl:73-78` | `SuperOp(::Type{T}, x::KrausChannel{D})` |
| `MixedUnitaryChannel` (BitFlip, PhaseFlip, Pauli) | `lib/YaoBlocks/src/channel/errortypes.jl:38,56,120` | `MixedUnitaryChannel(err)` |
| `DepolarizingChannel` superop (optimized) | `lib/YaoBlocks/src/channel/mixed_unitary_channel.jl:154-165` | `SuperOp(::Type{T}, c::DepolarizingChannel)` |
| `PositionedChannel` / circuit dispatch | `lib/YaoToEinsum/src/circuitmap.jl:107-117` | `eat_gate!(eb, b::PutBlock)` noisy branch |
| Channel → superop in TN | `lib/YaoToEinsum/src/circuitmap.jl:221-226` | `eat_gate!(eb, b::KrausChannel)` etc. |
| `add_channel!` (superop tensor insertion) | `lib/YaoToEinsum/src/circuitmap.jl:229-239` | `add_channel!(eb, SuperOp, locs)` |
| `push_normal_tensor!` (gate + conjugate) | `lib/YaoToEinsum/src/circuitmap.jl:37-44` | `push_normal_tensor!(eb, tensor, labels)` |
| `circuit_to_einsum_dm()` | `lib/YaoToEinsum/src/circuitmap.jl:353-381` | `yao2einsum(; mode=DensityMatrixMode())` |
| `EinBuilder` (label management) | `lib/YaoToEinsum/src/circuitmap.jl:16-26` | `EinBuilder{MODE, T, D}` |
| `update_slots!` | `lib/YaoToEinsum/src/circuitmap.jl:28-33` | `update_slots!(eb, indices, labels)` |
| `newlabel!` | `lib/YaoToEinsum/src/circuitmap.jl:70` | `newlabel!(eb)` |
| `add_gate_matrix!` (diagonal/general) | `lib/YaoToEinsum/src/circuitmap.jl:76-104` | `add_gate_matrix!(eb, m, locs)` |
| `add_controlled_matrix!` | `lib/YaoToEinsum/src/circuitmap.jl:153-183` | `add_controlled_matrix!(eb, k, m, locs, ctrl_locs, ctrl_vals)` |
| `eat_states!` (boundary tensors) | `lib/YaoToEinsum/src/circuitmap.jl:273-292` | `eat_states!(eb, states; conjugate)` |
| `eat_observable!` (trace with observable) | `lib/YaoToEinsum/src/circuitmap.jl:252-268` | `eat_observable!(eb, b)` |
| `trace!` (contract dual indices) | `lib/YaoToEinsum/src/circuitmap.jl:48-55` | `trace!(eb, group1, group2)` |
| `build_einsum` | `lib/YaoToEinsum/src/circuitmap.jl:390-392` | `build_einsum(eb, openindices)` |

### Kraus Operator Formulas (from `errortypes.jl`)

**BitFlip(p)**: `K0 = sqrt(1-p)*I, K1 = sqrt(p)*X` (line 38)

**PhaseFlip(p)**: `K0 = sqrt(1-p)*I, K1 = sqrt(p)*Z` (line 56)

**PauliChannel(px,py,pz)**: `K0 = sqrt(1-px-py-pz)*I, K1 = sqrt(px)*X, K2 = sqrt(py)*Y, K3 = sqrt(pz)*Z` (line 120)

**Depolarizing(n,p)**: Single-qubit = `PauliChannel(p/4, p/4, p/4)`. Multi-qubit: all n-qubit Pauli products with weight `p/4^n`, identity has extra weight `1-p`. Optimized superop at `mixed_unitary_channel.jl:154-165`.

**Reset(p0,p1)**: `K0 = sqrt(1-p0-p1)*I`, plus `sqrt(p0)*P0, sqrt(p0)*Pd` if p0>0, plus `sqrt(p1)*P1, sqrt(p1)*Pu` if p1>0. Where P0=|0><0|, P1=|1><1|, Pd=|0><1|, Pu=|1><0|. (lines 168-181)

**PhaseAmplitudeDamping(a,b,p1)** (lines 271-296):
- A0 = sqrt(1-p1) * [[1,0],[0,sqrt(1-a-b)]]
- A1 = sqrt(1-p1) * [[0,sqrt(a)],[0,0]] (if a>0)
- A2 = sqrt(1-p1) * [[0,0],[0,sqrt(b)]] (if b>0)
- B0 = sqrt(p1) * [[sqrt(1-a-b),0],[0,1]] (if p1>0)
- B1 = sqrt(p1) * [[0,0],[sqrt(a),0]] (if a>0 and p1>0)
- B2 = sqrt(p1) * [[sqrt(b),0],[0,0]] (if b>0 and p1>0)

**AmplitudeDamping(a,p1)**: = PhaseAmplitudeDamping(a, 0, p1) (line 362)

**PhaseDamping(b)**: = PhaseAmplitudeDamping(0, b, 0) (line 327)

**ThermalRelaxation(T1,T2,t,p1)** (lines 223-229):
- Tphi = (T1*T2)/(2*T1-T2)
- a = 1 - exp(-t/T1)
- b = 1 - exp(-t/Tphi)
- → PhaseAmplitudeDamping(a, b, p1)

**Coherent(U)**: Single Kraus op = U (deterministic unitary error)

### Superoperator Construction

From `kraus.jl:73-78`:
```
superop = sum_i kron(conj(K_i), K_i)
```

From `mixed_unitary_channel.jl:154-165` (optimized depolarizing):
```
S = (1-p)*I_{N^2} + (p/N) * sum_{diag} |ii><jj|
```

### Tensor Network Insertion (from `circuitmap.jl:229-239`)

For a k-qubit channel with superop S:
1. Reshape S to D^(4k) tensor
2. Labels: `[out_1..out_k, -out_1..-out_k, in_1..in_k, -in_1..-in_k]`
3. Update slots to new output labels
