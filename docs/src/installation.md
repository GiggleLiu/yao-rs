# Installation

Install the published CLI with a current stable Rust toolchain:

```bash
cargo install yao-cli --locked
yao --help
```

To build the development version from source:

```bash
git clone https://github.com/GiggleLiu/yao-rs.git
cd yao-rs
cargo install --path yao-cli --locked
yao --help
```

Cargo installs `yao` into its binary directory, usually `~/.cargo/bin`.
Ensure that directory is on your `PATH`.

Continue with [your first circuit](getting-started.md).

## Use as a Rust library

Add yao-rs to your project's `Cargo.toml`:

```toml
[dependencies]
yao-rs = "0.1"
```

The [circuits guide](circuits.md) shows how to build and validate a circuit.

## Optional features

The library has no default features. Enable only the integrations you need:

| Feature | Adds | Included in the CLI by default |
|---|---|---|
| `omeinsum` | Native tensor network contraction | Yes |
| `qasm` | OpenQASM 2.0 import and export | Yes |
| `parallel` | Rayon support for parallel operations | No |
| `tenferro` | CPU tensor contraction with reusable plans | No |
| `tenferro-ad` | Differentiable CPU circuit simulation | Library only |
| `cuda` | Device-resident CUDA simulation, gradients, and contraction | Library only |
| `optimizer-example` | Optimizer integration for the custom-loss example | Library only |

For example, to use OpenQASM and native contraction from Rust:

```toml
[dependencies]
yao-rs = { version = "0.1", features = ["qasm", "omeinsum"] }
```

Tenferro integrations require Rust 1.96 or newer. From a repository checkout,
add the CPU contractor to the CLI with:

```bash
cargo install --path yao-cli --features tenferro --locked
```

The [CUDA guide](cuda.md#requirements-and-first-run) lists GPU runtime requirements.
