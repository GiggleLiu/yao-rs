# CUDA and CPU comparison

Generated from raw Criterion and BenchmarkTools samples in this directory.

Platform: Linux-5.15.0-190-generic-x86_64-with-glibc2.35. Precision: complex128. Independent runs: 3.

Measurement notes: CUDA and CPU tenferro custom-loss cases request parameter and input-state gradients with two targeted pullbacks; native Rust and Yao use their joint reversible pullback. GPU resident execution includes fresh device copies for tracked inputs, allocation and host dispatch. Transfers and prepared constants are separate boundaries. The host is shared. GPU inventory is captured before measurement, CPU affinity and GPU clocks are not pinned; use run-to-run dispersion when interpreting results. Cases marked cuda_diagnostic_only have one process-cold probe and one warm diagnostic sample, with full-output checks; their expensive deep gradients are excluded from repeated GPU timings.

Times below are medians of per-process medians. Rust/Julia includes state copying; tensor phases are reported separately. A Julia/Rust ratio greater than one means native Rust was faster.

| Threads | Case | Native Rust µs | Yao µs | Julia/Rust | Max output error |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | cuda_density_10 | 1513946.183 | 539872.789 | 0.36 | 2.22e-16 |
| 1 | cuda_density_4 | 51.028 | 75.387 | 1.48 | 1.11e-16 |
| 1 | cuda_density_6 | 1082.708 | 1550.032 | 1.43 | 1.11e-16 |
| 1 | cuda_density_8 | 68642.269 | 20175.717 | 0.29 | 1.67e-16 |
| 1 | cuda_gradient_16_depth10 | 29251.172 | 19477.340 | 0.67 | 6.66e-15 |
| 1 | cuda_gradient_16_depth40 | 114849.583 | 72266.419 | 0.63 | 4.66e-15 |
| 1 | cuda_gradient_20_depth10 | 515975.056 | 349749.893 | 0.68 | 3.71e-15 |
| 1 | cuda_gradient_20_depth40 | 1933287.002 | 1347650.758 | 0.70 | 6.66e-15 |
| 1 | cuda_gradient_8_depth10 | 107.851 | 107.419 | 1.00 | 1.67e-16 |
| 1 | cuda_gradient_8_depth40 | 493.498 | 406.713 | 0.82 | 4.44e-16 |
| 1 | cuda_state_16_depth10 | 5695.965 | 5123.400 | 0.90 | 1.59e-17 |
| 1 | cuda_state_20_depth10 | 94358.648 | 81019.836 | 0.86 | 1.22e-17 |
| 1 | cuda_state_24_depth10 | 1585271.270 | 1488150.033 | 0.94 | 4.06e-17 |
| 1 | cuda_state_8_depth10 | 33.097 | 97.432 | 2.94 | 3.27e-17 |

## Tensor and extension phases

Each row uses the named API boundary. `tenferro_from_arrays` includes conversion, automatic planning, execution and output conversion; `omeinsum` includes its conversion/planning/execution. `tenferro_warm` uses an already prepared plan. Those prototype rows use independent planning policies. Where present, `supported_planning` compiles an existing omeco greedy tree; `supported_warm` runs the supported CPU adapter including ndarray input/output adaptation; `supported_from_arrays` combines those two phases. `omeinsum_fixed_tree` executes the identical tree, including its internal preparation. These supported rows exclude tree search and CPU context creation.

| Threads | Case | Phase | Median µs | Range of run medians µs |
| --- | --- | --- | ---: | ---: |
| 1 | cuda_density_10 | conversion | 108.983 | 97.632–112.016 |
| 1 | cuda_density_10 | cuda_resident | 20700.093 | 18573.240–21162.306 |
| 1 | cuda_density_10 | cuda_transfer_inclusive | 153157.147 | 148483.973–153484.009 |
| 1 | cuda_density_10 | omeinsum | 22242.527 | 22058.696–22334.791 |
| 1 | cuda_density_10 | omeinsum_fixed_tree | 22389.634 | 20645.130–22462.981 |
| 1 | cuda_density_10 | planning | 2329.058 | 2051.764–2330.296 |
| 1 | cuda_density_10 | supported_from_arrays | 57904.380 | 50466.961–58592.782 |
| 1 | cuda_density_10 | supported_planning | 1485.054 | 1473.092–1487.851 |
| 1 | cuda_density_10 | supported_warm | 45719.128 | 39692.848–46165.240 |
| 1 | cuda_density_10 | tenferro_from_arrays | 40504.340 | 39509.522–41368.630 |
| 1 | cuda_density_10 | tenferro_warm | 32406.185 | 24918.190–33546.921 |
| 1 | cuda_density_4 | conversion | 36.720 | 31.575–37.166 |
| 1 | cuda_density_4 | cuda_resident | 2802.219 | 2642.046–3322.786 |
| 1 | cuda_density_4 | cuda_transfer_inclusive | 7102.685 | 4157.041–7738.654 |
| 1 | cuda_density_4 | omeinsum | 470.841 | 470.835–471.813 |
| 1 | cuda_density_4 | omeinsum_fixed_tree | 209.116 | 182.177–218.252 |
| 1 | cuda_density_4 | planning | 612.354 | 525.734–612.999 |
| 1 | cuda_density_4 | supported_from_arrays | 5399.994 | 5370.958–5470.144 |
| 1 | cuda_density_4 | supported_planning | 267.221 | 265.484–267.529 |
| 1 | cuda_density_4 | supported_warm | 4012.696 | 4006.723–4070.962 |
| 1 | cuda_density_4 | tenferro_from_arrays | 2943.188 | 2917.056–3130.322 |
| 1 | cuda_density_4 | tenferro_warm | 222.931 | 216.207–235.447 |
| 1 | cuda_density_6 | conversion | 67.112 | 65.600–67.803 |
| 1 | cuda_density_6 | cuda_resident | 5056.025 | 4174.838–5419.595 |
| 1 | cuda_density_6 | cuda_transfer_inclusive | 11315.598 | 11248.885–11622.005 |
| 1 | cuda_density_6 | omeinsum | 1129.757 | 1126.315–1131.021 |
| 1 | cuda_density_6 | omeinsum_fixed_tree | 408.551 | 405.422–410.893 |
| 1 | cuda_density_6 | planning | 1193.173 | 1189.261–1195.504 |
| 1 | cuda_density_6 | supported_from_arrays | 9624.572 | 9623.930–9741.602 |
| 1 | cuda_density_6 | supported_planning | 699.632 | 697.513–702.560 |
| 1 | cuda_density_6 | supported_warm | 6776.045 | 6729.332–6842.132 |
| 1 | cuda_density_6 | tenferro_from_arrays | 6564.586 | 6071.126–6650.323 |
| 1 | cuda_density_6 | tenferro_warm | 461.845 | 443.758–463.100 |
| 1 | cuda_density_8 | conversion | 88.819 | 88.664–89.536 |
| 1 | cuda_density_8 | cuda_resident | 17061.483 | 15716.538–17276.354 |
| 1 | cuda_density_8 | cuda_transfer_inclusive | 31213.658 | 28902.988–33610.964 |
| 1 | cuda_density_8 | omeinsum | 1852.228 | 1838.788–1855.677 |
| 1 | cuda_density_8 | omeinsum_fixed_tree | 1057.399 | 985.828–1093.065 |
| 1 | cuda_density_8 | planning | 1716.528 | 1715.128–1718.146 |
| 1 | cuda_density_8 | supported_from_arrays | 17076.632 | 13083.697–17368.037 |
| 1 | cuda_density_8 | supported_planning | 1137.743 | 1136.507–1148.117 |
| 1 | cuda_density_8 | supported_warm | 12483.851 | 11914.299–12679.059 |
| 1 | cuda_density_8 | tenferro_from_arrays | 11721.078 | 10942.708–12312.723 |
| 1 | cuda_density_8 | tenferro_warm | 1466.475 | 1446.579–1487.791 |
| 1 | cuda_gradient_16_depth10 | cuda_resident | 820455.710 | 803895.180–842132.233 |
| 1 | cuda_gradient_16_depth10 | cuda_transfer_inclusive | 828091.739 | 790818.471–845958.599 |
| 1 | cuda_gradient_16_depth10 | tenferro_circuit_ad | 97430.260 | 96247.110–98510.088 |
| 1 | cuda_gradient_16_depth40 | tenferro_circuit_ad | 280305.186 | 244491.366–281916.900 |
| 1 | cuda_gradient_20_depth10 | cuda_resident | 899168.090 | 892693.659–929964.813 |
| 1 | cuda_gradient_20_depth10 | cuda_transfer_inclusive | 910279.267 | 897715.860–979739.602 |
| 1 | cuda_gradient_20_depth10 | tenferro_circuit_ad | 1477831.433 | 1470866.928–1499085.428 |
| 1 | cuda_gradient_20_depth40 | tenferro_circuit_ad | 4554135.190 | 4526771.722–4596738.587 |
| 1 | cuda_gradient_8_depth10 | cuda_resident | 791352.604 | 739652.630–797157.882 |
| 1 | cuda_gradient_8_depth10 | cuda_transfer_inclusive | 807767.395 | 753822.772–808986.581 |
| 1 | cuda_gradient_8_depth10 | tenferro_circuit_ad | 2502.142 | 1843.818–2512.272 |
| 1 | cuda_gradient_8_depth40 | tenferro_circuit_ad | 3280.712 | 2745.801–3300.149 |
| 1 | cuda_state_16_depth10 | cuda_resident | 72228.419 | 64610.039–73528.562 |
| 1 | cuda_state_16_depth10 | cuda_transfer_inclusive | 62008.719 | 55599.364–70613.704 |
| 1 | cuda_state_20_depth10 | cuda_resident | 55145.812 | 55047.396–58942.356 |
| 1 | cuda_state_20_depth10 | cuda_transfer_inclusive | 61136.338 | 58911.490–62263.092 |
| 1 | cuda_state_24_depth10 | cuda_resident | 76267.904 | 76180.361–76292.779 |
| 1 | cuda_state_24_depth10 | cuda_transfer_inclusive | 812824.838 | 811159.610–851906.153 |
| 1 | cuda_state_8_depth10 | cuda_resident | 70525.109 | 70049.614–71548.640 |
| 1 | cuda_state_8_depth10 | cuda_transfer_inclusive | 64550.656 | 42954.578–70896.791 |

Raw confidence intervals and samples are in `*-rust.json` and `*-gpu.json`; Julia trial samples are in `*-julia.json`. CUDA memory logs report device process snapshots and host peak RSS, with the measurement limits described below. See `metadata.json` and pinned manifests for reproducibility.

## Plots

![CPU scaling](cpu-scaling.svg)

![Circuit and contraction costs](tensor-costs.svg)

![AD memory](ad-memory.svg)

![Supported adapter, same contraction tree](supported-costs.svg)


## CUDA execution and transfers

Synchronized resident execution includes host dispatch, allocation and fresh AD leaves, with inputs already on the device. Transfer-inclusive execution additionally uploads every input and downloads complete outputs. Both reuse prepared structure and constants. Context creation and preparation are excluded. CPU native/Yao rows include state copies. CPU tenferro AD uses the reversible custom primitive; GPU AD uses ordinary tensor composition.

| Case | GPU resident ms | GPU transfer-inclusive ms | Native CPU / GPU resident | GPU max output error |
| --- | ---: | ---: | ---: | ---: |
| cuda_state_8_depth10 | 70.525 | 64.551 | 0.000469 | 8.36e-17 |
| cuda_state_16_depth10 | 72.228 | 62.009 | 0.0789 | 6.13e-18 |
| cuda_state_20_depth10 | 55.146 | 61.136 | 1.71 | 1.93e-18 |
| cuda_state_24_depth10 | 76.268 | 812.825 | 20.8 | 5.42e-19 |
| cuda_gradient_8_depth10 | 791.353 | 807.767 | 0.000136 | 4.44e-16 |
| cuda_gradient_8_depth40 | Diagnostic only | Diagnostic only | — | 1.78e-15 |
| cuda_gradient_16_depth10 | 820.456 | 828.092 | 0.0357 | 8.44e-15 |
| cuda_gradient_16_depth40 | Diagnostic only | Diagnostic only | — | 2.89e-15 |
| cuda_gradient_20_depth10 | 899.168 | 910.279 | 0.574 | 5.33e-15 |
| cuda_gradient_20_depth40 | Diagnostic only | Diagnostic only | — | 1.15e-14 |
| cuda_density_4 | 2.802 | 7.103 | 0.0182 | 2.22e-16 |
| cuda_density_6 | 5.056 | 11.316 | 0.214 | 3.33e-16 |
| cuda_density_8 | 17.061 | 31.214 | 4.02 | 3.89e-16 |
| cuda_density_10 | 20.700 | 153.157 | 73.1 | 3.33e-16 |

A CPU/GPU ratio greater than one means this GPU boundary was faster. These are medians of independent process medians; raw confidence intervals and samples are preserved. Transfer-inclusive figures are distinct from a complete setup-inclusive application run.

## Process-cold setup and memory snapshots

One process per case, with the persistent compiler/driver disk caches left in place. Context, preparation plus input uploads, and first synchronized execution are timed separately. NVIDIA process-memory snapshots include context, workspaces and the allocator pool. They are observed snapshots, not exact live tensor bytes or a continuous peak. Host peak RSS is a separate process metric and includes the native correctness reference.

| Case | Context ms | Prepare + upload ms | First resident ms | Warm diagnostic ms | Context GPU MiB | Prepared GPU MiB | First result GPU MiB | After repeats GPU MiB | Host peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda_state_8_depth10 | 284.515 | 302.045 | 375.975 | 42.760 | 414.000 | 536.000 | 536.000 | 536.000 | 782.094 |
| cuda_state_16_depth10 | 257.359 | 328.964 | 480.314 | 93.229 | 414.000 | 632.000 | 632.000 | 632.000 | 852.402 |
| cuda_state_20_depth10 | 263.193 | 390.317 | 474.262 | 43.243 | 414.000 | 568.000 | 632.000 | 632.000 | 877.516 |
| cuda_state_24_depth10 | 258.233 | 1392.691 | 520.384 | 76.068 | 414.000 | 1176.000 | 2136.000 | 2136.000 | 2543.297 |
| cuda_gradient_8_depth10 | 264.757 | 316.989 | 4618.640 | 601.871 | 414.000 | 536.000 | 856.000 | 856.000 | 973.117 |
| cuda_gradient_8_depth40 | 260.070 | 349.216 | 131526.143 | 15598.376 | 414.000 | 536.000 | 856.000 | 856.000 | 3345.770 |
| cuda_gradient_16_depth10 | 252.826 | 294.441 | 4212.555 | 491.928 | 414.000 | 632.000 | 920.000 | 920.000 | 1088.727 |
| cuda_gradient_16_depth40 | 273.416 | 380.063 | 135386.661 | 15421.097 | 414.000 | 632.000 | 920.000 | 920.000 | 3423.500 |
| cuda_gradient_20_depth10 | 261.618 | 454.435 | 5020.683 | 537.465 | 414.000 | 600.000 | 1080.000 | 1112.000 | 1103.246 |
| cuda_gradient_20_depth40 | 258.836 | 474.535 | 135174.636 | 7584.271 | 414.000 | 600.000 | 1080.000 | 1112.000 | 3439.918 |
| cuda_density_4 | 264.963 | 312.951 | 7.605 | 5.391 | 414.000 | 536.000 | 536.000 | 536.000 | 729.844 |
| cuda_density_6 | 261.258 | 312.551 | 12.746 | 10.443 | 414.000 | 536.000 | 856.000 | 856.000 | 730.781 |
| cuda_density_8 | 258.803 | 327.741 | 15.509 | 27.143 | 414.000 | 536.000 | 920.000 | 920.000 | 796.301 |
| cuda_density_10 | 259.069 | 315.941 | 66.939 | 14.472 | 414.000 | 536.000 | 984.000 | 984.000 | 802.594 |

Warm diagnostic values summarize samples within one probe process; they are not independent-run statistics. Deep cases marked diagnostic-only have one such sample and are excluded from the repeated GPU latency comparison.

![CUDA and CPU latency](cuda-latency.svg)

![Observed GPU memory versus depth](cuda-memory.svg)
