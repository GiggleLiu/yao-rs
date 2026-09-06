# Releasing yao-rs

`bitbasis`, `yao-rs`, and `yao-cli` share a workspace version. Releases publish
them in dependency order. Preparing a PR does not publish a release.

## Before tagging

1. Merge the release changes and update the Unreleased section of `CHANGELOG.md`.
2. Choose the version: breaking pre-1.0 API changes require a minor bump;
   compatible fixes can use a patch bump. New features should be called out.
3. Start from a clean, up-to-date `main` branch. Ensure the publisher owns all
   three crate names and `CARGO_REGISTRY_TOKEN` is configured in repository secrets.
4. Run `make check-all`, `make doc`, and `cargo package --workspace --locked`.
   Cargo's workspace packaging stages unpublished sibling crates together.
5. Run `make release V=x.y.z` (or a prerelease such as `V=0.2.0-rc.1`).

The release helper validates the version, branch, clean index, and tag uniqueness;
updates workspace and inter-crate versions plus `Cargo.lock`; runs checks;
commits the version change; and atomically pushes HEAD with only the new tag.
If checks fail, changes remain local for inspection. If the push fails, inspect
local HEAD and the tag before retrying the push; do not blindly rerun the bump.

The tag workflow verifies versions, tests, API docs, and packages before publishing.
A GitHub release appears only after all crates publish successfully. Prerelease
tags are marked as prereleases. Cargo waits for registry indexing, so no fixed
sleep is needed. See the [Cargo publish reference](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).

Registry uploads cannot be rolled back. If a publish partially succeeds, inspect
crates.io and publish only the missing crates in dependency order, then create
the GitHub release once all versions are available. Do not move a published tag.

## Installation smoke test

```sh
cargo install yao-cli --version x.y.z --locked
yao --version
yao example bell > bell.json
yao run bell.json --shots 100 --seed 42
yao simulate bell.json | yao probs -
yao toeinsum bell.json --mode state | yao optimize - | yao contract -
```
