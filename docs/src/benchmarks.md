# Benchmarks vs Yao.jl

Performance depends on circuit size and workload. The recorded CPU comparison
shows low overhead for small QFT circuits, similar QFT times at 24 qubits, and
faster Yao.jl execution for several layered and noisy circuits.

## CPU scaling

This **7 September 2026 baseline** compares the native yao-rs simulator with
Yao.jl 0.9.3 on an **Apple M4, 16 GiB RAM, macOS 26.6.2**, using one thread and
complex128 arithmetic. The toolchains were Rust 1.98.0 and Julia 1.12.4.
These are measurements of the recorded source snapshot; they are not rerun
on each release.

![Single-thread CPU time for Rx and QFT from 8 to 24 qubits on Apple M4. The logarithmic time axis shows the small-QFT advantage narrowing as the state grows.](static/benchmark-cpu-scaling.svg)

Lower time is better. The vertical axis is logarithmic. Each point is the
median of three independent process medians, with the same circuit and input
state in both implementations.

## Other workloads

The same M4 run includes gates, layered circuits, gradients, and exact noisy
simulation. A ratio **above 1 favors yao-rs; below 1 favors Yao.jl**.

| Workload | Qubits | yao-rs (ms) | Yao.jl (ms) | Yao.jl / yao-rs |
|---|---:|---:|---:|---:|
| Rz gate | 24 | 12.7267 | 18.9304 | 1.49× |
| FSim gate | 24 | 34.9472 | 25.3953 | 0.73× |
| 100 Ry–CX layers | 8 | 0.2366 | 0.1157 | 0.49× |
| 100 Ry–CX layers | 12 | 5.0920 | 1.9951 | 0.39× |
| Value + gradient, 100 layers | 8 | 1.1124 | 1.4790 | 1.33× |
| Value + gradient, 100 layers | 12 | 23.3087 | 23.6564 | 1.01× |
| Noisy circuit, exact density matrix | 4 | 0.0222 | 0.0321 | 1.44× |
| Noisy circuit, exact density matrix | 10 | 198.4158 | 155.4484 | 0.78× |

Each layer applies Ry to every qubit followed by a nearest-neighbor CX chain.
Gradient rows return both an expectation value and all parameter derivatives;
noise rows use depolarizing and amplitude-damping channels.

[Full M4 report and raw samples](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/mac-cpu-2026-09-07)
include all 48 circuit cases, one- and four-thread settings, run ranges, source
hashes, and pinned dependencies. The
[Linux Xeon Platinum 8378A results](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/linux-cpu-2026-09-07)
provide a second CPU comparison; that shared dual-socket host was not isolated
from other workloads, so its timings need to be read with the recorded variability.

## What is timed

- **Execution:** warmed library calls, including a fresh input-state copy.
  Compilation, circuit construction, CLI startup, and file I/O are excluded.
- **Agreement:** Julia validates every output entry against the Rust result
  before timing, with qubit ordering aligned. Gradient cases check both values
  and gradients.
- **Threads:** the native Rust kernels in this fixture are serial. The
  four-thread results change the Julia/provider budget; they do not make those
  Rust kernels parallel.
- **Tensor networks:** conversion, planning, and contraction are measured
  separately. Their costs and execution boundaries are in the full reports;
  the table above measures native simulation.

Additional measured workflows have their own reports:
[custom-loss differentiation](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/mac-circuit-ad-cpu-2026-09-07),
[Hamiltonian evolution](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/mac-krylov-cpu-2026-09-07),
[tensor expectations and slicing](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/mac-tensor-memory-cpu-2026-09-07),
and [CUDA vs CuYao](https://github.com/GiggleLiu/yao-rs/tree/2f99ba1b1def29ede407d6a3a4b2f837ba03185c/benchmarks/results/a800-cuda-2026-09-07).

## Reproduce

The [curated benchmark suite](https://github.com/GiggleLiu/yao-rs/tree/main/benchmarks/regression)
combines pinned QASMBench and MQT application circuits with feature-specific
workloads. It provides locked environments, full-output validation, and saved
results for future regression checks. The
[dataset survey](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/regression/SURVEY.md)
explains the selection.

From a source checkout with Rust, Julia, and uv installed:

```bash
make benchmark-setup
make benchmark BENCH_OUT=benchmarks/results/local-cpu
```

Use `BENCH_PROFILE=smoke` for a short setup check, or `BENCH_PROFILE=full` for
both thread budgets and the larger cases. See the
[benchmark guide](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/regression/README.md)
for comparing two saved runs with `make benchmark-check`.
