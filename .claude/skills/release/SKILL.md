---
name: release
description: Use when preparing a new yao-rs release, bumping the version, or tagging a release
---

# Release

Guide for cutting a new release of yao-rs. The workspace publishes three crates
to crates.io: `bitbasis`, `yao-rs`, and `yao-cli`.

## Step 1: Determine the version bump

Compare against the last release tag:

```bash
git tag -l 'v*' | sort -V          # latest tag (empty = first release is v0.1.0)
git log <last-tag>..HEAD --oneline  # review commits
git diff <last-tag>..HEAD --stat    # review scope
```

Apply semver for 0.x (pre-1.0):
- **Patch** (0.x.Y) — bug fixes, docs, CI only
- **Minor** (0.X.0) — new features, new public API
- **Major** — reserved for post-1.0

All three crates share one version via `[workspace.package]`, so they always
release together.

## Step 2: Verify a clean state

```bash
make check-all
```

`fmt-check`, `clippy -D warnings`, and the test suite must all pass first.
Optionally dry-run the publish to catch metadata issues early:

```bash
cargo publish --dry-run -p bitbasis
cargo publish --dry-run -p yao-rs
```

## Step 3: Release

```bash
make release V=x.y.z
```

This bumps `[workspace.package] version` plus the inter-crate dependency
versions (`bitbasis` in the root, `yao-rs` in `yao-cli`), runs
`cargo check --workspace`, commits `release: vx.y.z`, tags `vx.y.z`, and pushes.

The `.github/workflows/release.yml` workflow then:
1. creates a GitHub release with auto-generated notes, and
2. publishes `bitbasis` → `yao-rs` → `yao-cli` to crates.io (in dependency
   order, with indexing waits).

## Prerequisites (one-time)

- `CARGO_REGISTRY_TOKEN` must be set as a repository secret (Settings →
  Secrets → Actions) — it authenticates `cargo publish`.
- Each crate needs `description` + `license` (already set; `license`/
  `repository` are inherited from `[workspace.package]`).
- Crate names `bitbasis`, `yao-rs`, `yao-cli` must remain owned by the
  publisher on crates.io.

## Notes

- The first release at the current version doesn't need a bump — if
  `Cargo.toml` already reads `version = "0.1.0"` and `v0.1.0` isn't tagged yet,
  you can tag it directly: `git tag -a v0.1.0 -m "Release v0.1.0" && git push origin HEAD --tags`.
- Releases are driven entirely by the `v*.*.*` tag; pushing the tag is what
  triggers publishing.
