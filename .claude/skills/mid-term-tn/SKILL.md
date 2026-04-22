---
name: mid-term-tn
description: Run the tensor-network circuit simulation demo (tn/demo_midterm.jl) for 中期答辩 — covers 指标 #4 (60比特10层量子线路模拟)
user_invocable: true
---

# Mid-term Demo: Tensor Network Circuit Simulation

Helps 刘金国 demonstrate 指标 #4 (60比特10层量子线路模拟) of national key R&D project 2024YFB4504004 (课题4: 量超融合).

## What it demonstrates (4 parts)

- **Part 1 — Small-circuit correctness**: 4-qubit rectangular lattice, exact vs TN statevector agreement to machine precision (<1e-16)
- **Part 2 — Scale proof**: 70-qubit × 12-layer Google Bristlecone supremacy circuit, amplitude `⟨0|U|0⟩` via TreeSA-optimized contraction (exceeds 指标 #4: 60比特10层)
- **Part 3a — Expectation value API validation**: `⟨Z₁Z₂⟩` via DensityMatrixMode vs exact (<1e-16 agreement)
- **Part 3b — Physical observable demo**: 60-qubit × 10-layer 1D kicked-Ising chain, `⟨Z₃₀⟩` as function of Rx(θ), clean + depolarizing noise (p=0.005) — reproduces IBM Eagle utility-experiment pattern
- **Part 3c — Kicked-Ising cross-validation**: 10-qubit exact statevector vs TN clean vs PauliPropagation noisy (all agree to truncation)

## Why it meets the standard

- 指标 #4 requires "演示60比特、10层的量子线路模拟" → Bristlecone 70/12 (amplitude) + kicked-Ising 60/10 (expectation value) both exceed
- Tensor network method (YaoToEinsum + TreeSA) enables simulation far beyond state vector limits
- Cross-validation on small circuits proves numerical correctness
- Kicked-Ising clean → noisy sweep demonstrates integration with 指标 #1 (噪声模型) — depolarizing channel at scale

## Run

```bash
cd /Users/liujinguo/jcode/QuantumSimulationDemos
julia --project=tn tn/demo_midterm.jl
```

## Output files (`tn/results/`)

- `midterm_circuit_small.svg` — Circuit diagram of validation circuit
- Console output with device info, TreeSA / contraction timing breakdown, complexity analysis, and the kicked-Ising θ sweep table

## Explain to reviewer

1. Show device info banner: "CPU model + BLAS thread count printed at startup — all results are CPU-only, reproducible."
2. Show Part 1 + 3a: "Exact vs TN agreement to machine precision proves the simulator is correct."
3. Show Part 2 Bristlecone 70/12: "Google quantum-supremacy circuit, 638 gates. TreeSA finds contraction order with tc=2^14.2, contracted in ~1.5s. A naive state vector would need 2^70 ≈ 10^21 complex numbers — infeasible."
4. Show Part 3b kicked-Ising sweep: "`⟨Z₃₀⟩` at θ=0 is +1 exactly (no dynamics); decays with θ as expected physically. Depolarizing noise (p=0.005) damps the signal — same pattern as IBM Eagle's quantum-utility experiment at 127 qubits."
5. Key point: "Requirement is 60 qubits / 10 layers. We demonstrate 70/12 (amplitude) and 60/10 (physical observables with noise)."

## Data sources

- Primary qflex `.txt` circuits: `/Users/liujinguo/jcode/QuantumSimulationDemos/tn/data/circuits/`
- In this repo, keep the Rust library on its native circuit JSON and convert `.txt` inputs with `scripts/qflex_txt_to_yao_json.py` before loading them.
