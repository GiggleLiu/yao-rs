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

This requires a clean `main` branch and an unused version tag. It bumps
`[workspace.package] version` plus inter-crate dependency versions and updates
`Cargo.lock`, runs `make check-all`, commits and tags the release, and atomically
pushes HEAD with only the new tag.

The `.github/workflows/release.yml` workflow then:
1. verifies the tag, tests, API docs, and workspace packages,
2. publishes `bitbasis` → `yao-rs` → `yao-cli` with Cargo's workspace publishing
   and index polling, and
3. creates a GitHub release with auto-generated notes after publishing succeeds.

See `RELEASING.md` for the full procedure and partial-publish recovery.

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
