# Curated CPU comparison

Profile: regression. Independent runs: 6.

Times are medians of independent process medians. Each library uses its fastest measured execution mode. Ratios above 1 favor yao-rs. Fusion preparation is outside warmed execution.

| Case | Threads | yao-rs mode | yao-rs ms | Fastest measured competitor | Competitor ms | Competitor / yao-rs |
|---|---:|---|---:|---|---:|---:|
| circuit-ad/custom_gradient_12_depth10 | 1 | execute | 0.3238 | julia | 0.5731 | 1.77× |
| circuit-ad/custom_gradient_12_depth100 | 1 | execute | 3.1170 | julia | 5.7058 | 1.83× |
| circuit-ad/custom_gradient_8_depth10 | 1 | execute | 0.0254 | julia | 0.0448 | 1.76× |
| circuit-ad/custom_gradient_8_depth100 | 1 | execute | 0.2428 | julia | 0.4103 | 1.69× |
| circuits/cry_low_far_12 | 1 | execute | 0.0015 | qulacs | 0.0048 | 3.12× |
| circuits/cry_low_far_16 | 1 | fused4_execute | 0.0300 | julia | 0.0451 | 1.50× |
| circuits/cry_low_far_8 | 1 | fused2_execute | 0.0001 | qulacs | 0.0015 | 13.33× |
| circuits/fsim_adjacent_12 | 1 | fused2_execute | 0.0039 | julia | 0.0106 | 2.70× |
| circuits/fsim_adjacent_16 | 1 | fused2_execute | 0.0671 | julia | 0.1014 | 1.51× |
| circuits/fsim_adjacent_8 | 1 | fused2_execute | 0.0003 | qulacs_fused4 | 0.0019 | 7.12× |
| circuits/gradient100_12 | 1 | execute | 10.8483 | julia | 23.0744 | 2.13× |
| circuits/gradient100_4 | 1 | execute | 0.0662 | julia | 0.0997 | 1.51× |
| circuits/gradient100_8 | 1 | execute | 0.5634 | julia | 1.3402 | 2.38× |
| circuits/gradient10_12 | 1 | execute | 1.1002 | julia | 2.3558 | 2.14× |
| circuits/gradient10_4 | 1 | execute | 0.0068 | julia | 0.0104 | 1.53× |
| circuits/gradient10_8 | 1 | execute | 0.0565 | julia | 0.1343 | 2.38× |
| circuits/layers100_12 | 1 | execute | 1.9030 | julia | 1.9668 | 1.03× |
| circuits/layers100_4 | 1 | fused4_execute | 0.0003 | qulacs_fused4 | 0.0016 | 5.02× |
| circuits/layers100_8 | 1 | execute | 0.0938 | julia | 0.1157 | 1.23× |
| circuits/layers10_12 | 1 | execute | 0.1919 | julia | 0.1977 | 1.03× |
| circuits/layers10_4 | 1 | fused4_execute | 0.0003 | qulacs_fused4 | 0.0016 | 5.02× |
| circuits/layers10_8 | 1 | execute | 0.0094 | julia | 0.0115 | 1.22× |
| circuits/mqt-ghz-12 | 1 | execute | 0.0071 | qulacs | 0.0079 | 1.11× |
| circuits/mqt-ghz-16 | 1 | execute | 0.1604 | qulacs | 0.1676 | 1.05× |
| circuits/mqt-ghz-8 | 1 | execute | 0.0004 | julia | 0.0009 | 2.34× |
| circuits/mqt-grover-8 | 1 | execute | 0.3270 | qulacs | 0.4259 | 1.30× |
| circuits/mqt-qaoa-12 | 1 | execute | 0.2232 | qulacs | 0.3244 | 1.45× |
| circuits/mqt-qaoa-16 | 1 | execute | 5.7873 | qulacs | 8.6630 | 1.50× |
| circuits/mqt-qaoa-8 | 1 | execute | 0.0074 | qulacs | 0.0121 | 1.65× |
| circuits/mqt-qft-12 | 1 | fused2_execute | 0.1918 | qulacs | 0.4697 | 2.45× |
| circuits/mqt-qft-16 | 1 | fused2_execute | 5.2966 | julia | 13.3240 | 2.52× |
| circuits/mqt-qft-8 | 1 | fused2_execute | 0.0058 | qulacs | 0.0150 | 2.58× |
| circuits/mqt-randomcircuit-12 | 1 | execute | 0.9696 | qulacs | 1.3307 | 1.37× |
| circuits/mqt-randomcircuit-16 | 1 | execute | 31.3139 | qulacs | 48.7837 | 1.56× |
| circuits/mqt-randomcircuit-8 | 1 | execute | 0.0353 | qulacs | 0.0471 | 1.33× |
| circuits/noisy_10 | 1 | execute | 124.0126 | julia | 154.1985 | 1.24× |
| circuits/noisy_4 | 1 | execute | 0.0153 | julia | 0.0302 | 1.98× |
| circuits/noisy_6 | 1 | execute | 0.2408 | julia | 0.3715 | 1.54× |
| circuits/noisy_8 | 1 | execute | 4.8054 | julia | 7.0760 | 1.47× |
| circuits/qasmbench-adder_n10 | 1 | execute | 0.0261 | qulacs | 0.0322 | 1.23× |
| circuits/qasmbench-ising_n10 | 1 | fused2_execute | 0.0707 | qulacs_fused4 | 0.1642 | 2.32× |
| circuits/qasmbench-qaoa_n6 | 1 | fused2_execute | 0.0022 | qulacs_fused4 | 0.0061 | 2.74× |
| circuits/qasmbench-qpe_n9 | 1 | execute | 0.0127 | qulacs | 0.0203 | 1.61× |
| circuits/qasmbench-vqe_uccsd_n8 | 1 | fused2_execute | 0.6362 | qulacs | 0.6192 | 0.97× |
| circuits/qft_12 | 1 | execute | 0.0542 | qulacs | 0.1649 | 3.04× |
| circuits/qft_16 | 1 | execute | 1.3681 | julia | 2.9073 | 2.13× |
| circuits/qft_8 | 1 | execute | 0.0020 | qulacs | 0.0066 | 3.21× |
| circuits/rx_12 | 1 | fused4_execute | 0.0027 | julia | 0.0037 | 1.40× |
| circuits/rx_16 | 1 | fused4_execute | 0.0466 | julia | 0.0579 | 1.24× |
| circuits/rx_8 | 1 | fused4_execute | 0.0002 | julia | 0.0004 | 2.14× |
| circuits/rz_12 | 1 | fused2_execute | 0.0021 | julia | 0.0045 | 2.18× |
| circuits/rz_16 | 1 | fused4_execute | 0.0378 | qulacs_fused4 | 0.0522 | 1.38× |
| circuits/rz_8 | 1 | fused2_execute | 0.0001 | julia | 0.0004 | 2.95× |
| circuits/swap_far_12 | 1 | execute | 0.0014 | julia | 0.0032 | 2.23× |
| circuits/swap_far_16 | 1 | fused2_execute | 0.0291 | qulacs_fused4 | 0.0316 | 1.08× |
| circuits/swap_far_8 | 1 | fused2_execute | 0.0001 | julia | 0.0004 | 3.59× |
| circuits/tensor_state_4 | 1 | execute | 0.0002 | julia | 0.0003 | 1.97× |
| circuits/tensor_state_8 | 1 | execute | 0.0019 | julia | 0.0025 | 1.31× |
| krylov/krylov_heisenberg_12q_tol7 | 1 | execute | 10.0981 | julia | 15.7988 | 1.56× |
| krylov/krylov_heisenberg_4q_tol7 | 1 | execute | 0.0030 | julia | 0.0324 | 10.90× |
| krylov/krylov_heisenberg_8q_tol7 | 1 | execute | 0.2070 | julia | 0.7795 | 3.77× |
| krylov/krylov_ising_12q_tol7 | 1 | execute | 6.5098 | julia | 7.3002 | 1.12× |
| krylov/krylov_ising_4q_tol7 | 1 | execute | 0.0074 | julia | 0.0496 | 6.67× |
| krylov/krylov_ising_8q_tol7 | 1 | execute | 0.2609 | julia | 0.6261 | 2.40× |
| tensor-memory/expectation_4_terms5 | 1 | execute | 0.0007 | julia | 0.0027 | 3.70× |
| tensor-memory/expectation_6_terms5 | 1 | execute | 0.0019 | julia | 0.0045 | 2.37× |
| tensor-memory/expectation_dm_4_terms5 | 1 | execute | 0.0115 | julia | 0.0414 | 3.58× |
| tensor-memory/expectation_dm_6_terms5 | 1 | execute | 0.1456 | julia | 0.2313 | 1.59× |
| trajectories/noisy_expectation_4 | 1 | execute | 0.0393 | julia | 0.0975 | 2.48× |
| trajectories/noisy_expectation_6 | 1 | execute | 0.6211 | julia | 1.0046 | 1.62× |

This table is descriptive. Run the regression gate against a compatible saved run; it rejects incomplete or inconclusive comparisons.
Feature-specific phases and all raw samples are retained in results.json and the track directories.

## Achieved evolution accuracy

The solver parameter is rtol=1e-7; the validated global relative-error budget is 1e-6. Achieved errors differ, so timing at this budget is not an equal-error comparison.

| Case | yao-rs relative error (maximum) | Yao.jl relative error (maximum) |
|---|---:|---:|
| krylov/krylov_heisenberg_12q_tol7 | 4.036e-08 | 3.054e-07 |
| krylov/krylov_heisenberg_4q_tol7 | 8.438e-16 | 3.114e-16 |
| krylov/krylov_heisenberg_8q_tol7 | 2.494e-15 | 7.274e-08 |
| krylov/krylov_ising_12q_tol7 | 4.064e-08 | 1.938e-07 |
| krylov/krylov_ising_4q_tol7 | 4.327e-08 | 1.928e-09 |
| krylov/krylov_ising_8q_tol7 | 3.543e-08 | 4.046e-07 |
