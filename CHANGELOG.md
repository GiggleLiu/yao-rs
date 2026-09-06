# Changelog

## Unreleased

- Automatically simulate noisy CLI circuits with density matrices, including
  saved states, probabilities, measurements, and expectations. Pure state files
  retain the `yao-state-v1` format; density matrices use `yao-density-v1`.
- Add reproducible measurement with `--seed` to `run --shots` and `measure`.
- Fix density-matrix expectation qubit ordering and reuse the library expectation
  path in the CLI.
- Reject malformed state headers, invalid measurement/operator locations, and
  unsupported multi-term tensor-network expectation exports with useful errors.
- Expand differentiation coverage and document the existing adjoint API.
- Check all workspace crates and features in CI; verify packages and API docs
  before publishing, and create GitHub releases only after publishing succeeds.
- Track Cargo.lock, improve crate metadata, and correct installation and output docs.

## 0.1.0

Initial crates.io release of the library, bitbasis primitives, and `yao` CLI.
