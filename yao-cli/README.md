# yao-cli

Command-line quantum circuit simulation, noisy density-matrix simulation,
measurement, SVG diagrams, and tensor-network contraction for
[yao-rs](https://github.com/GiggleLiu/yao-rs).

```sh
cargo install yao-cli --locked
yao example bell > bell.json
yao run bell.json --shots 1024
yao simulate bell.json | yao probs -
yao toeinsum bell.json --mode state | yao optimize - | yao contract -
yao visualize bell.json --output bell.svg
```

The installed binary is `yao`. Use `yao --help` for commands and
`yao <command> --help` for options. OpenQASM and native contraction are enabled
by default. The development version can be installed from a checkout with
`cargo install --path yao-cli --locked`.

For the optional tenferro CPU backend (Rust 1.96+), install the development
checkout with `cargo install --path yao-cli --features tenferro --locked`:

```sh
yao toeinsum bell.json --mode state | yao optimize - | yao contract - --backend tenferro --threads 4
```

The default backend remains omeinsum when both are built. A build with
`--no-default-features --features tenferro` uses tenferro by default.
`--threads` is a positive tenferro thread count, defaulting to 1.

Circuits with channels automatically use a density matrix. State files retain
their representation across pipelines. `--seed` on `run --shots` and `measure`
reproduces sampling within the same binary version.

See the [user guide](https://giggleliu.github.io/yao-rs/cli.html).
Licensed under MIT.
