# Apple M4 curated CPU snapshot

Measured 8 September 2026 from [22364fbbbaff25f5b5df191b9475d63bf2d0d7a4](https://github.com/GiggleLiu/yao-rs/commit/22364fbbbaff25f5b5df191b9475d63bf2d0d7a4).
This is the initial saved baseline for the corrected curated CPU protocol.

- Device: Apple M4, 16 GiB RAM, macOS 26.6.2.
- Precision: complex128. Thread budget: one. Independent processes: six per implementation and track.
- Elapsed measurement run: 47.5 minutes, excluding initial build/setup.
- Rust 1.98.0, Julia 1.12.4, Yao.jl 0.9.3, Qulacs 0.6.14; Python and library locks are in [the environment](../../regression/environment).
- Complete outputs are validated before timing. Each of the six Rust/Julia/Qulacs orders occurs once in the circuit track.

Command, after `make benchmark-setup JULIA=/path/to/julia`:

```bash
CARGO_BUILD_JOBS=2 make benchmark \
  BENCH_OUT=/tmp/yao-curated-verified-m4 \
  JULIA=/path/to/julia
```

## Results and scope

The named execution gate reports 158 pass, 0 slower,
and 0 inconclusive comparisons (5% tolerance, per-comparison
95% bootstrap intervals). See [qualification.json](qualification.json) for every comparison.
The [readable report](report.md) selects each library's fastest measured execution
mode; [results.json](results.json) retains all native modes, preparation costs,
feature phases, and process medians.

State execution includes a fresh input copy. Circuit/fusion construction is
outside execution; fusion preparation is measured separately. Fixed-parameter
fusion uses two- and four-qubit blocks for yao-rs and four-qubit blocks for Qulacs.
Public applications use deterministic dense initial states. All selected cases,
normalizations, source hashes, provider details, and achieved Krylov errors are retained.

This profile covers 70 cases, including 18 public application circuits. The full
profile's larger circuits and four-thread setting are not covered here. Tensor
backend phases provide regression coverage; broader tensor competitors,
stochastic-accuracy comparisons, and CUDA require separate qualification.
Krylov uses `rtol=1e-7` with a validated global relative-error budget of `1e-6`;
achieved errors differ and are listed in the report.

## Reuse the baseline

On this same device and recorded environment, compare a future run with:

```bash
make benchmark-check \
  BENCH_BASELINE=benchmarks/results/mac-curated-cpu-2026-09-08/results.json \
  BENCH_CANDIDATE=path/to/candidate/results.json
```

Another device needs its own baseline. Older exploratory runs used a different
timing protocol and cannot be pooled with this snapshot.

[raw-samples.tar.gz](raw-samples.tar.gz) contains the original case files,
independent-process samples, logs, and metadata. To inspect the archive:

```bash
tar -tf raw-samples.tar.gz
```

File hashes are in [SHA256SUMS](SHA256SUMS). The [suite guide](../../regression/README.md)
explains setup, profiles, sample counts, and the comparison rules.
