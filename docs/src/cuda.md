# CUDA simulation

The optional `cuda` feature uses published tenferro 0.4.0 for complex128 unitary
circuits and tensor contraction. It includes `tenferro-ad`. Select CUDA explicitly
through `CudaSimulator`. States, real physical parameters, intermediates and
gradients can remain on one device across calls.

## Requirements and first run

Use Rust 1.96 or newer and an NVIDIA GPU with working CUDA libraries. Tenferro
0.4.0's published CUDA dependencies select CUDA 12.8 APIs; contraction also
requires cuTENSOR. Hardware qualification uses an A800 80 GB, CUDA 12.8.90,
cuBLAS 12.8.5.5, NVRTC 12.8.93 and cuTENSOR 2.6.0.4. On the tested system,
driver 535.230.02 is paired with NVIDIA's CUDA 12.8 forward-compatibility package
570.211.01. This is one qualified configuration, not a claim about every driver
or GPU. See NVIDIA's [compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

Make CUDA shared libraries visible to the loader. Set `TENFERRO_CUTENSOR_PATH`
to the cuTENSOR shared library if needed. Unset `TENFERRO_EAGER_WHOLE_PROGRAM`:
that upstream prototype contains host-only paths and this adapter rejects it
for contraction.

```bash
CUDA_VISIBLE_DEVICES=0 cargo run --release --locked --features cuda --example cuda_loss
```

The example uploads a state, target and parameters, evaluates a smooth loss,
and updates parameters on the GPU. Each iteration starts a fresh tracked leaf
from device values so previous graphs can be released. Final results are
explicitly downloaded for display.

The regular `--all-features` suite does not initialize CUDA. Hardware tests are
ignored by default. Run them explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 cargo test --release --locked --features cuda cuda::tests -- --ignored --test-threads=1
# Debug CUDA call stacks exceed Rust's default 2 MiB test-thread stack here:
CUDA_VISIBLE_DEVICES=0 RUST_MIN_STACK=8388608 cargo test --locked --features cuda cuda::tests -- --ignored --test-threads=1
```

## Circuits and differentiation

Construct a validated `DifferentiableCircuit` and call `gpu.prepare(&circuit)`.
Upload physical parameters as an f64 tensor of shape `[num_parameters]` and the
input state as a complex128 tensor of shape `[2^n]`. Tenferro calls this complex
dtype `DType::C64` (two f64 components). State order matches `ArrayReg`: site zero
is the most significant bit.

`prepared.apply(&parameters, &state)` returns a device `EagerTensor`. It supports
existing unitary gates, custom unitary matrices, ordered targets, active-low and
multiple controls, shared/scaled/product bindings, and empty parameter vectors.
Small constants are uploaded during preparation; execution computes angles
from device parameters.

Controls become batch axes of the local gate tensor, keeping state evolution
in one contraction per gate. A gate with `c` controls and `k` targets uses a
batch of local matrices with `2^c * 4^k` complex entries; this storage is separate
from the state and AD intermediates. It avoids a full `4^(c+k)` controlled matrix.

Track inputs with `gpu.upload(&host_tensor, true)`. Compose real scalar losses
using eager operations, then use `gpu.runtime().grad`, `vjp` or `jvp`. The
[real Hermitian pairing](./differentiation.md) also applies here. For squared
state error, sum `delta.conj()?.mul(&delta)?.cast(DType::F64)?`. The equivalent
`abs(delta)^2` reaches an unsupported complex-sign derivative in upstream CUDA.

This is composed first-order AD: it retains intermediates and can use memory
proportional to depth times state size. The CPU reversible custom operation's
bounded state-memory behavior does not apply. That CPU operation cannot serve
as an implicit GPU fallback. `DifferentiableCircuit` rejects noise channels and
nonunitary custom matrices.

## Tensor networks and exact noise

`PreparedCudaContraction::new(&code, &sizes, tree)` validates shapes and preserves
an explicit contraction tree. `None` uses omeco greedy planning. N-ary nodes
execute left to right; output axes follow the requested order. Preparation does
not initialize CUDA or compile kernels. Upload tensors once, then repeatedly
call `gpu.contract(&plan, &inputs)` with values of the same shapes.

Upload ndarray tensors in column-major logical axis order:

```rust,ignore
let host = tenferro_tensor::Tensor::from_vec_col_major(
    array.shape().to_vec(), array.t().iter().copied().collect(),
)?;
let device = gpu.upload(&host, false)?;
```

`circuit_to_einsum_dm` networks support fixed-noise forward simulation, including
Kraus channels; qudit exports also work. Density tensors are exact up to
floating-point contraction error, with storage that can grow as `4^n`. This API
has no automatic GPU slicing or hard device-memory budget. Choose bounded
networks before allocating them.

Contractions compose with AD through tensor values, except when labels repeat
within an input (traces and diagonals). Those networks reject tracked inputs
up front: tenferro 0.4.0's diagonal pullback uses a host-only padding path.
Untracked trace/diagonal forward execution remains supported. This does not
differentiate serialized circuit noise parameters.

## Transfers, validation and timing

`upload` and `download` are explicit host/device boundaries. `prepare` uploads
small constants; an empty contraction uploads its scalar identity. No execution
silently switches to CPU. Foreign runtimes, host tensors, shape mismatches and
unsupported dtypes return errors. Uploads require finite, contiguous host values;
states need not be normalized. User GPU arithmetic inherits tenferro's numerical
semantics; execution does not download values to check finiteness at every call.

Call `gpu.synchronize()` immediately before and after resident timing regions.
These timings include host dispatch, allocation and AD bookkeeping, but exclude
user transfers. Time transfers separately, including output materialization.
Record context creation, preparation and first execution separately from warm
runs. Prepared objects preserve structure and constants; they do not promise
zero allocations or fused circuit kernels.

For larger eager graphs, request the gradients you need with `runtime.grad` or
`runtime.vjp`. Tenferro 0.4.0's stateful `backward()` traverses retained tracked
intermediates and requests separate pullbacks for them; it can do substantially
more work than requesting parameter and input-state gradients explicitly.
