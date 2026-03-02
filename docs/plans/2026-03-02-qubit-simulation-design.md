# Qubit Vector Simulation: Design & Test Plan

## Principle

**Copy the Julia implementation as closely as possible.** The Rust code should mirror Julia's structure, naming, and algorithms. Julia references are in `~/.julia/dev/`.

## Goal

Replace the current mixed-radix-based instruct functions with bit-manipulation-based qubit (d=2) fast paths. The generic qudit path stays for tensor network simulation only. Add measurement completion (reset, remove).

## Problem

Current `instruct.rs` uses `linear_to_indices()` / `mixed_radix_index()` everywhere:
- `instruct_diagonal`: O(n) Vec alloc + division per amplitude — should be zero-alloc stride loop
- `instruct_single`: 2 Vec allocs per iteration (`indices` + `insert_index`) — should be zero-alloc
- `udrows`: allocates `old_amps` Vec on every call — should use stack vars for d=2
- `instruct_controlled`: iterates ALL basis states, checks controls each time — should skip non-matching
- `marginal_probs` / `collapse_to`: same `linear_to_indices` problem

## Architecture

### 4 instruct functions (all with optional controls)

```
apply_inplace(circuit, state)
  └── for each gate:
      ├── 1-target, diagonal    → instruct_1q_diag(state, [d0,d1], loc, ctrls?)
      ├── 1-target, non-diag    → instruct_1q(state, [[a,b],[c,d]], loc, ctrls?)
      ├── 2-target, diagonal    → instruct_2q_diag(state, [d0..d3], loc0, loc1, ctrls?)
      └── 2-target, non-diag    → instruct_2q(state, gate_4x4, loc0, loc1, ctrls?)
```

### Bit utilities (new module: `bitutils.rs`)

Copy from Julia `~/.julia/dev/BitBasis/src/bit_operations.jl`:

```rust
// Julia: BitBasis/src/bit_operations.jl:90-107
// bmask(::Type{T}, positions::Int...) = reduce(+, indicator(T, b) for b in itr)
fn bmask(locs: &[usize]) -> usize

// Julia: BitBasis/src/bit_operations.jl:220-231
// flip(index::T, mask::T) = index ⊻ mask
fn flip(b: usize, mask: usize) -> usize

// Julia: BitBasis/src/bit_operations.jl:140-159
// anyone(index::T, mask::T) = (index & mask) != zero(T)
fn anyone(b: usize, mask: usize) -> bool

// Julia: BitBasis/src/bit_operations.jl:162-181
// allone(index::T, mask::T) = (index & mask) == mask
fn allone(b: usize, mask: usize) -> bool

// Julia: BitBasis/src/bit_operations.jl:184-197
// ismatch(index::T, mask::T, target::T) = (index & mask) == target
fn ismatch(b: usize, mask: usize, target: usize) -> bool

// Julia: BitBasis/src/bit_operations.jl:40-45
// indicator(::Type{T}, k::Int) = one(T) << (k-1)
fn indicator(k: usize) -> usize  // note: 0-indexed in Rust, so 1 << k
```

### IterControl (new struct in `bitutils.rs`)

Copy from Julia `~/.julia/dev/BitBasis/src/iterate_control.jl`:

```rust
// Julia: BitBasis/src/iterate_control.jl:16-25
// struct IterControl{S}
//     n::Int          # size of iteration space (2^free_bits)
//     base::Int       # base offset from control bits set to 1
//     masks::NTuple{S,Int}
//     factors::NTuple{S,Int}
// end
struct IterControl {
    n: usize,
    base: usize,
    masks: Vec<usize>,
    factors: Vec<usize>,
}

// Julia: BitBasis/src/iterate_control.jl:53-59
// function itercontrol(nbits, positions, bit_configs)
//     base = bmask(Int, positions[i] for (i, u) in enumerate(bit_configs) if u != 0)
//     masks, factors = group_shift!(nbits, positions)
//     return IterControl(1 << (nbits - length(positions)), base, masks, factors)
// end
fn itercontrol(nbits: usize, positions: &[usize], bit_configs: &[usize]) -> IterControl

// Julia: BitBasis/src/iterate_control.jl:108-130
// function group_shift!(nbits, positions)
//     sort!(positions)
//     masks = Int[]; factors = Int[]
//     k_prv = 0; i = 0
//     for k in positions
//         if k != k_prv + 1
//             push!(factors, 1<<(k_prv-i))
//             gap = k - k_prv-1
//             push!(masks, bmask(i+1:i+gap))
//             i += gap
//         end
//         k_prv = k
//     end
//     if i != nbits
//         push!(factors, 1<<(k_prv-i))
//         push!(masks, bmask(i+1:nbits))
//     end
// end
fn group_shift(nbits: usize, positions: &mut Vec<usize>) -> (Vec<usize>, Vec<usize>)

// Julia: BitBasis/src/iterate_control.jl:85-92
// getindex(it, k) = sum((k-1) & mask * factor for (mask,factor) in zip(masks,factors)) + base
impl Iterator for IterControl { ... }
```

**NOTE:** Julia uses 1-indexed bit positions. Rust uses 0-indexed. Adjust accordingly:
- Julia `indicator(T, k) = 1 << (k-1)` → Rust `1 << k`
- Julia `bmask(1:3)` = bits 1,2,3 → Rust `bmask(&[0,1,2])` = bits 0,1,2

### 4 instruct functions

**1. `instruct_1q(state, gate, loc, ctrl_locs?, ctrl_bits?)`**

Copy from Julia `~/.julia/dev/Yao/lib/YaoArrayRegister/src/instruct.jl:153-166`:

```julia
# Julia reference:
# function single_qubit_instruct!(state, U1, loc)
#     a, c, b, d = U1
#     instruct_kernel(state, loc, 1 << (loc - 1), 1 << loc, a, b, c, d)
# end
# function instruct_kernel(state, loc, step1, step2, a, b, c, d)
#     @threads for j = 0:step2:size(state, 1)-step1
#         for i = j+1:j+step1
#             u1rows!(state, i, i + step1, a, b, c, d)
#         end
#     end
# end
```

Rust (0-indexed):
```rust
fn instruct_1q(state: &mut [Complex64], a: C64, b: C64, c: C64, d: C64, loc: usize) {
    let step1 = 1 << loc;
    let step2 = 1 << (loc + 1);
    let total = state.len();
    let mut j = 0;
    while j < total {
        for i in j..(j + step1) {
            u1rows(state, i, i + step1, a, b, c, d);
        }
        j += step2;
    }
}
```

With controls — copy from Julia `instruct.jl:90-121`:
```julia
# Julia reference (generic controlled):
# locked_bits = MVector(control_locs..., locs...)
# locked_vals = MVector(control_bits..., (0 for k = 1:M)...)
# locs_raw_it = (b + 1 for b in itercontrol(N, setdiff(1:N, locs), zeros(Int, N-M)))
# locs_raw = SVector(locs_raw_it...)
# ic = itercontrol(N, locked_bits, locked_vals)
# for j = 1:length(ic)
#     i = ic[j]
#     unrows!(state, locs_raw .+ i, U)
# end
```

**2. `instruct_1q_diag(state, [d0, d1], loc, ctrl_locs?, ctrl_bits?)`**

Copy from Julia `instruct.jl:187-198`:

```julia
# Julia reference:
# function single_qubit_instruct!(state, U1::SDDiagonal, loc)
#     a, d = U1.diag
#     step = 1 << (loc - 1)
#     step_2 = 1 << loc
#     for j = 0:step_2:size(state, 1)-step
#         for i = j+1:j+step
#             mulrow!(state, i, a)
#             mulrow!(state, i + step, d)
#         end
#     end
# end
```

**3. `instruct_2q(state, gate_4x4, loc0, loc1, ctrl_locs?, ctrl_bits?)`**

Copy from Julia `instruct.jl:90-121` (generic matrix path with itercontrol):

```julia
# Julia reference (generic 2-qubit):
# Uses sort_unitary to order locs, then:
# locked_bits = MVector(control_locs..., locs...)
# locked_vals = MVector(control_bits..., 0, 0)
# locs_raw = compute raw indices for the 4 basis states
# ic = itercontrol(N, locked_bits, locked_vals)
# for j in ic: unrows!(state, locs_raw .+ j, U)
```

**4. `instruct_2q_diag(state, [d0..d3], loc0, loc1, ctrl_locs?, ctrl_bits?)`**

Same iteration as #3 but `mulrow` instead of `unrows!`.

### Measurement additions

Copy from Julia `~/.julia/dev/Yao/lib/YaoArrayRegister/src/measure.jl`:

- `measure_reset(state, locs, reset_val, rng)` — measure, then reset measured qubits to `reset_val`
- `measure_remove(state, locs, rng)` — measure, remove measured qubits, return (result, smaller_state)

Fix `marginal_probs` and `collapse_to` to use bit ops when all dims == 2.

### Dispatch in `apply_inplace`

```rust
fn apply_inplace(circuit: &Circuit, state: &mut State) {
    let all_qubit = circuit.dims.iter().all(|&d| d == 2);
    for element in &circuit.elements {
        let pg = match element { Gate(pg) => pg, _ => continue };
        if all_qubit {
            // dispatch to 4 qubit instruct functions
        } else {
            // existing generic path (for tensor network or mixed-dim)
        }
    }
}
```

## Julia Reference Files

All implementations must closely follow these Julia sources:

| Rust Module | Julia Reference File | Key Functions |
|---|---|---|
| `bitutils.rs` | `~/.julia/dev/BitBasis/src/bit_operations.jl` | `bmask`, `flip`, `anyone`, `allone`, `ismatch`, `indicator` |
| `bitutils.rs` | `~/.julia/dev/BitBasis/src/iterate_control.jl` | `IterControl`, `itercontrol`, `group_shift!`, `controldo` |
| `instruct.rs` | `~/.julia/dev/Yao/lib/YaoArrayRegister/src/instruct.jl` | `single_qubit_instruct!`, `instruct_kernel`, `instruct!(controlled)` |
| `instruct.rs` | `~/.julia/dev/Yao/lib/YaoArrayRegister/src/utils.jl` | `u1rows!`, `mulrow!`, `swaprows!`, `unrows!` |
| `measure.rs` | `~/.julia/dev/Yao/lib/YaoArrayRegister/src/measure.jl` | `measure!`, `ResetTo`, `RemoveMeasured` |

### Key differences from Julia (0-indexed vs 1-indexed)

- Julia bit positions are 1-indexed: `indicator(T, k) = 1 << (k-1)`
- Rust bit positions are 0-indexed: `indicator(k) = 1 << k`
- Julia state indices are 1-indexed: `state[i]` where `i` starts at 1
- Rust state indices are 0-indexed: `state[i]` where `i` starts at 0
- Julia `instruct_kernel` loop: `for j = 0:step2:size(state,1)-step1` then `i = j+1:j+step1`
- Rust equivalent: `for j in (0..total).step_by(step2)` then `i in j..(j+step1)`

## Test Plan

### 1. Generic unitary instruction (from Julia `test/instruct.jl:7-50`)

```julia
# Julia test reference:
# U1 = randn(ComplexF64, 2, 2)
# ST = randn(ComplexF64, 1 << 4)
# I2 = IMatrix(2)
# M = kron(I2, U1, I2, I2) * ST
# @test instruct!(Val(2), copy(ST), U1, (3,)) ≈ M
#
# U2 = rand(ComplexF64, 4, 4)
# M = kron(I2, U2, I2) * ST
# @test instruct!(Val(2), copy(ST), U2, (2, 3)) ≈ M
#
# @test instruct!(Val(2), copy(ST), kron(U1, U1), (3, 1)) ≈
#       instruct!(Val(2), instruct!(Val(2), copy(ST), U1, (3,)), U1, (1,))
```

```rust
#[test] fn test_1q_unitary_instruction()
// Random 2x2 U on 4-qubit state at loc=2
// Compare: instruct result == kron(I, U, I, I) * state_vec

#[test] fn test_2q_unitary_instruction()
// Random 4x4 U on 4-qubit state at locs=(1,2)
// Compare: instruct result == kron(I, U, I) * state_vec

#[test] fn test_separable_2q_instruction()
// instruct(kron(U1,U1), (2,0)) == instruct(U1, 2) then instruct(U1, 0)

#[test] fn test_identity_noop()
// Identity gate leaves state unchanged
```

### 2. Controlled unitary (from Julia `test/instruct.jl:68-86`)

```julia
# Julia test reference:
# ST = randn(ComplexF64, 1 << 5)
# U1 = randn(ComplexF64, 2, 2)
# @test instruct!(Val(2), copy(ST), U1, (3,), (1,), (1,)) ≈
#       general_controlled_gates(5, [Const.P1], [1], [U1], [3]) * ST
# @test instruct!(Val(2), copy(ST), U1, (3,), (1,), (0,)) ≈
#       general_controlled_gates(5, [Const.P0], [1], [U1], [3]) * ST
# U2 = kron(U1, U1)
# @test instruct!(Val(2), copy(ST), U2, (3, 4), (5, 1), (1, 0)) ≈
#       general_controlled_gates(5, [Const.P1, Const.P0], [5, 1], [U2], [3]) * ST
```

```rust
#[test] fn test_controlled_1q()
#[test] fn test_controlled_active_low()
#[test] fn test_multi_control_2q()
```

### 3. Named Pauli gates (from Julia `test/instruct.jl:89-102`)

```julia
# Julia test reference:
# for (G, M) in zip((:X, :Y, :Z), (Const.X, Const.Y, Const.Z))
#     @test linop2dense(s -> instruct!(Val(2), s, Val(G), (1,)), 1) == M
#     @test linop2dense(s -> instruct!(Val(2), s, Val(G), (1, 2, 3)), 3) == kron(M, M, M)
# end
# for (G, M) in zip((:X, :Y, :Z), (Const.X, Const.Y, Const.Z))
#     @test linop2dense(s -> instruct!(Val(2), s, Val(G), (4,), (2, 1), (0, 1)), 4) ≈
#           general_controlled_gates(4, [Const.P0, Const.P1], [2, 1], [M], [4])
# end
```

```rust
#[test] fn test_pauli_x_instruction()
#[test] fn test_pauli_y_instruction()
#[test] fn test_pauli_z_instruction()
#[test] fn test_controlled_paulis()
```

### 4. Diagonal gates (from Julia `test/instruct.jl:104-115`)

```julia
# Julia test reference:
# ST = randn(ComplexF64, 1 << 4)
# Dv = Diagonal(randn(ComplexF64, 2))
# @test instruct!(Val(2), copy(ST), Dv, (3,)) ≈
#       kron(Const.I2, Dv, Const.I2, Const.I2) * ST
```

```rust
#[test] fn test_diagonal_z_s_t()
#[test] fn test_diagonal_rz()
#[test] fn test_diagonal_phase()
```

### 5. SWAP and parametric gates (from Julia `test/instruct.jl:117-152`)

```julia
# Julia test reference:
# ST = randn(ComplexF64, 1 << 2)
# @test instruct!(Val(2), copy(ST), Val(:SWAP), (1, 2)) ≈ Const.SWAP * ST
# θ = π / 3
# @test instruct!(Val(2), copy(ST), Val(:PSWAP), (1, 2), θ) ≈
#       (cos(θ / 2) * IMatrix(4) - im * sin(θ / 2) * Const.SWAP) * ST
# for R in [:Rx, :Ry, :Rz]
#     @test instruct!(Val(2), copy(ST), Val(R), (4,), θ) ≈
#           instruct!(Val(2), copy(ST), Matrix(parametric_mat(T, Val(R), θ)), (4,))
# end
```

```rust
#[test] fn test_swap_instruction()
#[test] fn test_rotation_gates()
#[test] fn test_controlled_rotation()
```

### 6. Multi-threading regression (from Julia `test/instruct.jl:181-202`)

```julia
# Julia test reference:
# g = [0.921061-0.389418im ...] (4x4 matrix)
# n = 16
# reg1 = rand_state(n)
# for i = 1:50
#     x1 = rand(1:n); x2 = rand(1:n-1)
#     x2 = x2 >= x1 ? x2 + 1 : x2
#     instruct!(reg1, g, (x1, x2))
# end
# @test isapprox(norm(statevec(reg1)), 1.0; atol = 1e-5)
```

```rust
#[test] fn test_parallel_regression_16q()
```

### 7. Measurement (from Julia `test/measure.jl`)

```julia
# Julia test reference:
# reg = rand_state(4)
# res = measure!(YaoAPI.ResetTo(0), reg, (4,))
# @test isnormalized(reg)
# result = measure(reg; nshots = 10)
# @test all(result .< 8)
#
# reg = rand_state(6) |> focus!(1, 4, 3)
# res = measure!(YaoAPI.RemoveMeasured(), reg)
# @test nqubits(reg) == 3
# @test select(reg0, res) |> normalize! ≈ reg
```

```rust
#[test] fn test_measure_reset()
#[test] fn test_measure_remove()
#[test] fn test_measure_nshots()
#[test] fn test_measure_deterministic()
```

### 8. Consistency with existing tests

```rust
#[test] fn test_qubit_path_matches_generic_path()
// For all existing apply() ground truth cases:
// qubit fast path result == generic path result
```

## Test Dataset

Generated by `scripts/generate_instruct_test_data.jl` (run with `julia --project=scripts`).
Output: `tests/data/instruct.json` — 53 test cases with input/output state vectors.

| Category | Count | Tests |
|---|---|---|
| 1q generic unitary | 5 | random U at each loc on 4 qubits + identity |
| 2q generic unitary | 5 | random 4x4 U at various loc pairs + separability |
| Controlled 1q | 2 | active-high + active-low |
| Controlled 2q | 2 | single-ctrl + multi-ctrl |
| Pauli X/Y/Z | 15 | basis states, 3-site, controlled with active-low |
| Diagonal (Z/S/T) | 6 | uncontrolled + controlled |
| SWAP | 2 | 2-qubit + 5-qubit |
| Rotations (Rx/Ry/Rz) | 6 | uncontrolled + controlled |
| PSWAP/CPHASE | 4 | uncontrolled + controlled |
| H | 1 | 5-qubit |
| Regression (random) | 1 | 20 random 2q gates on 10 qubits |
| Measurement | 1 | Bell state probabilities |
| Position sweep 1q | 4 | all 4 positions on 4 qubits |
| Position sweep 2q | 4 | 4 position pairs on 5 qubits |

Rust tests load `tests/data/instruct.json` and compare results against Julia ground truth.

## Non-goals

- Qudit (d>2) vector simulation — use tensor network path
- Batched registers — future work
- Gate fusion / circuit compilation — future work
- Specialized named gate kernels (X-swap, Y-swap) — can optimize later
