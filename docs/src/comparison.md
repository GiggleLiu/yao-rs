# Comparison with Yao.jl

yao-rs is a focused port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl), covering the circuit description and tensor network layers.

## Feature Comparison

| Capability | Yao.jl | yao-rs |
|------------|--------|--------|
| Circuit description (`put`/`control`) | Yes | Yes |
| Qudit support | Yes | Yes |
| State-vector simulation | Yes (in-place) | Yes (full matrix) |
| GPU simulation | Yes (CuYao) | No |
| Symbolic computation | Yes (YaoSym) | No |
| Automatic differentiation | Yes | No |
| Tensor network export | Yes (YaoToEinsum) | Yes |
| Diagonal gate optimization | No | Yes |
| Contraction order optimization | No | Yes (omeco) |
| Noise channels | Yes | No |
| Measurement / sampling | Yes | No |
| Circuit visualization | Yes (YaoPlots) | No |

## Summary

**yao-rs** focuses on circuit-to-tensor-network conversion with optimizations (diagonal gates, contraction order via omeco) that are not available in Yao.jl's YaoToEinsum.

**Yao.jl** is a full quantum computing framework with GPU simulation, symbolic computation, automatic differentiation, noise channels, and visualization.
