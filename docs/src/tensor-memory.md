# Multiple observables and memory controls

`circuit_to_expectation` and `circuit_to_expectation_dm` accept complete
`OperatorPolynomial` values, including zero, identity and complex coefficients.
They share the circuit tensors across terms. A summed term-selection index
connects the local operator tensors; each coefficient appears exactly once.
This avoids constructing a dense operator on the full register. Slicing that
index evaluates terms sequentially while retaining the same circuit tensors.

The pure variant rejects channels; the density variant evaluates `Tr(O rho)`
with noise included. Identity factors support arbitrary site dimensions;
nonidentity operators require qubit sites. Invalid sites, duplicate sites in a
word and nonfinite coefficients are rejected. The CLI selects density mode for
noisy expectations automatically, or explicitly with `--mode dm`.

```bash
yao toeinsum circuit.json --op "0.3 * Z(0) + 0.7 * Y(1)" | yao optimize - | yao contract -
cargo run --example sliced_expectation --features tenferro
```

## Fixed slices and reusable execution

`SlicedPlan::new(code, sizes, tree, labels, budget)` validates a fixed tree and
slice labels without allocating tensor values. Internal-index slices sum their
contributions. Output-index slices write to the corresponding output positions;
they are not summed into a scalar. Repeated labels within an input are fixed
consistently, preserving traces and diagonals. Signed density-matrix labels work
the same way as other labels.

```rust
use yao_rs::slicing::{SliceBudget, SlicedPlan};
use yao_rs::tenferro::CpuContractor;
use yao_rs::{Circuit, Gate, put, circuit_to_einsum_with_boundary};
use yao_rs::contraction_plan::optimize_code;

let circuit = Circuit::qubits(2, vec![put(vec![0], Gate::Ry(0.4))]).unwrap();
let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
let tree = optimize_code(&tn.code, &tn.size_dict, &omeco::GreedyMethod::default()).unwrap();
let plan = SlicedPlan::new(&tn.code, &tn.size_dict, &tree,
    &[tn.code.iy[0]], SliceBudget::default()).unwrap();
let cpu = CpuContractor::new(1).unwrap();
let prepared = cpu.prepare_sliced(&plan).unwrap();
let state = cpu.execute_sliced(&prepared, &tn.tensors).unwrap();
assert_eq!(state.shape(), &[2, 2]);
```

The prepared tenferro program uses one fixed slice shape and can be reused with
new tensor values. `contractor::contract_sliced` provides the omeinsum equivalent;
its executor preparation occurs within each slice. Both preserve the supplied
tree. `execute_with` supports another executor with the documented shape and
memory contract.

Execution holds one active slice and one full output. It reuses slice buffers,
retaining no list of intermediate slice results. Reduction follows first-label
appearance order, with the final slice label varying fastest. This order is
independent of hash-map iteration. Slice-level parallelism is not enabled;
tenferro's CPU thread setting controls work within each contraction.

## Budgets and automatic selection

All `SliceBudget` quantities are **bytes**, using 16 bytes per complex128 value.
`max_bytes` limits an estimate of total tensor storage. `workspace_bytes` adds a
caller-selected reserve for provider scratch; its default zero does not mean
the backend needs no workspace. `max_slices` caps total assignments, including
output slices; the default is 1,000,000. Zero limits, invalid labels/shapes,
integer overflow and budgets below the required storage return errors.

`MemoryEstimate` reports original inputs, full output, largest intermediate,
omeco's depth-first live tensor peak, conservative worker buffers, workspace
reserve and total separately. Worker buffers count slice copies, backend input
adaptation, a temporary input-layout copy, every node output and result adaptation. This conservative tensor
model is **not a hard RSS limit**: runtime metadata, compiled graphs, allocator
retention and unreported provider workspace can make process memory higher.
The input tensors and full output must fit even if every internal index is sliced.

`SlicedPlan::auto` reuses omeco's `slice_code` and `TreeSASlicer`, reducing the
intermediate target until the complete estimate fits. It explicitly permits
replanning; it does not preserve the original tree. The current upstream slicer
requires a binary tree for nontrivial networks. Heuristics can fail to find a
feasible plan, or produce too many assignments; failure is reported instead of
silently exceeding a limit. It is not a proof that no better plan exists.

```bash
# Fixed labels from the exported network, including signed DM labels:
yao optimize tn.json --slice=3,-3 --memory-budget 67108864 --max-slices 4096
# No fixed slices: select slices and refine the tree with omeco TreeSA:
yao optimize tn.json --memory-budget 67108864 --workspace-bytes 1048576
yao contract planned.json --backend tenferro --threads 4
```

Re-optimizing a serialized sliced plan preserves its slice labels and limits
unless explicit replacement options are supplied. Unsliced legacy plans still
use `yao-tn-v1`. Plans with memory/slicing semantics use `yao-tn-v2` and include
`slice_plan` (labels, budget and estimate). Readers validate the estimate against
the expression/tree before execution. The previous CLI rejects the new format;
it cannot silently discard slices and execute an unsliced plan.

The CPU benchmark suite compares identical fixed paths separately from automatic
planning, with raw selected trees, execution times and isolated process peak RSS.
Run it with `benchmarks/run_backend.py --suite tensor-memory`; see
[benchmark methodology](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/README.md).


The [M4 CPU report](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/results/mac-tensor-memory-cpu-2026-09-07/report.md)
includes ordinary chains where slicing does not reduce RSS, a deliberately
large-intermediate path where it does, and a greedy-order comparison that avoids
the large intermediate altogether. Choose a good contraction order first; slice
when the remaining tensor-memory tradeoff benefits the workload.
