# Comparison with Yao.jl

yao-rs is a focused Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl).
The table distinguishes circuit export from direct simulation and identifies
which differentiation and noise workflows are implemented.

| Capability | Yao.jl | yao-rs |
| --- | --- | --- |
| Circuit description (`put`/`control`) | Yes, including reusable composite blocks | Yes, flat validated circuits |
| Qudit support | Registers and circuits | Tensor-network export; direct state-vector simulation is qubit-only |
| State-vector simulation | In-place and batched registers | In-place qubit registers |
| GPU simulation | CuYao | Planned; no supported GPU backend yet |
| Symbolic computation | YaoSym | No |
| Automatic differentiation | Parameter/state gradients and ChainRules integration | Adjoint expectation gradients for parameterized unitary circuits |
| Tensor-network export | YaoToEinsum | Pure states, overlaps, expectations and density matrices |
| Diagonal tensor optimization | Yes | Yes |
| Contraction-order optimization | OMEinsumContractionOrders; slicing in current source | omeco planning and omeinsum contraction; slicing not yet exposed |
| Noise channels | Density-matrix channel execution | Kraus channels and density matrices; automatic noisy CLI simulation |
| Measurement / sampling | Register operations and in-circuit measurement blocks | Register operations and CLI sampling |
| Circuit visualization | YaoPlots | Built-in SVG rendering |
| Hamiltonian evolution | TimeEvolution and Krylov exponential action | Planned |

The tensor optimization entries are verified against the local Yao source
commit `31c7c1333b14b1e89123c511eff5742e7ac24edd`, including
`lib/YaoToEinsum/src/Core.jl` and circuit conversion code. Older documentation
may not describe every current-source capability. See Yao's
[automatic differentiation](https://docs.yaoquantum.org/stable/man/automatic_differentiation.html),
[register](https://docs.yaoquantum.org/stable/man/registers.html), and
[block](https://docs.yaoquantum.org/stable/man/blocks.html) documentation.

`expect_grad` already supports Rx, Ry, Rz, Phase, and both FSim parameters,
including controlled gates. General custom-loss/input-state differentiation
and differentiation through noise channels are separate future increments.

The tenferro feasibility fixture under `benchmarks/tenferro-probe` evaluates a
published tensor/autodiff backend without changing the supported library API.
Performance conclusions must name the workload, precision, CPU, thread count,
and whether conversion/planning is included. Use the reproducible benchmark
instructions in `benchmarks/README.md`; do not infer circuit performance from
standalone tensor-library benchmarks.
