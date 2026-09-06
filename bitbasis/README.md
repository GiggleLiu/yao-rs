# bitbasis

Bit strings and basis-index operations for qubit simulation, ported from
[BitBasis.jl](https://github.com/QuantumBFS/BitBasis.jl). Part of
[yao-rs](https://github.com/GiggleLiu/yao-rs).

```toml
[dependencies]
bitbasis = "0.1"
```

Provides `BitStr<N>`, masks, bit permutations, and `IterControl` for iterating
basis indices with fixed control bits. Bit addresses in this crate count from
the least significant bit; yao-rs circuit locations count from the most
significant bit.

See the [API reference](https://docs.rs/bitbasis). Licensed under MIT.
