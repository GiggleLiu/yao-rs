# Contributing

Use a current stable Rust toolchain supporting edition 2024. The release
readiness checks were also run with Rust 1.93. Python 3.11+ and Bash are used
by the CLI example tests and release tools; mdBook 0.5.2 builds the book.

```sh
make check-all
cargo test --workspace --no-default-features --locked
cargo clippy --workspace --all-targets --no-default-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked
python3 -m unittest discover -s scripts/tests
make doc
```

Keep `Cargo.lock` committed so CI and CLI builds resolve the same dependencies.
Use `cargo update` deliberately, then rerun the checks. Library consumers still
resolve compatible dependency versions through Cargo.toml.

Add regression tests for bugs and numerical changes. Unit tests live alongside
their modules or under `src/unit_tests/`; cross-module reference tests live in
`tests/suites/`, and command-line tests live in `yao-cli/tests/`. Compare
asymmetric states as well as Bell/GHZ states to catch qubit-ordering errors.
Use fixed seeds for reproducible randomized tests. See `CLAUDE.md` for architecture.

Describe the problem, resulting behavior, and checks in each PR. Keep public
API changes and serialization compatibility explicit. Report bugs with a minimal
circuit, command, expected result, actual result, and `yao --version`/`rustc --version`.
