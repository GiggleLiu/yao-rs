# CLI Example Pages Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every CLI example visible as its own mdBook page with CLI-only copy-paste walkthroughs and Python visualization scripts in `scripts/`.

**Architecture:** Keep `docs/src/examples/cli-visualization.md` as an index/overview, but move detailed walkthroughs into separate pages under `docs/src/examples/cli/`. Move the plot helper from `examples/cli/plot_results.py` to `scripts/plot_cli_results.py`, update the bash artifact generator and tests, and ensure mdBook `SUMMARY.md` lists every example page.

**Tech Stack:** mdBook Markdown, bash, Python standard library SVG generation, Rust integration tests.

---

## File Structure

- Create `scripts/plot_cli_results.py`: dependency-free JSON-to-SVG result plotter.
- Delete `examples/cli/plot_results.py`: old plotter location.
- Modify `examples/cli/generate_artifacts.sh`: call the plotter from `scripts/`.
- Modify `docs/src/SUMMARY.md`: list all per-example pages under the examples section.
- Modify `docs/src/examples/catalog.md`: link to the per-example pages.
- Modify `docs/src/examples/cli-visualization.md`: keep overview/gallery/result tables, remove long walkthrough bodies, link to pages.
- Create `docs/src/examples/cli/bell.md`.
- Create `docs/src/examples/cli/ghz4.md`.
- Create `docs/src/examples/cli/qft4.md`.
- Create `docs/src/examples/cli/phase-estimation-z.md`.
- Create `docs/src/examples/cli/hadamard-test-z.md`.
- Create `docs/src/examples/cli/swap-test.md`.
- Create `docs/src/examples/cli/bernstein-vazirani-1011.md`.
- Create `docs/src/examples/cli/grover-marked-5.md`.
- Create `docs/src/examples/cli/qaoa-maxcut-line4-depth2.md`.
- Create `docs/src/examples/cli/qcbm-static-depth2.md`.
- Modify `docs/src/examples/generated/manifest.md`: use `scripts/plot_cli_results.py`.
- Modify `yao-cli/tests/integration.rs`: assert script path, page existence, no Rust code snippets in CLI pages, summary links, and generated artifact links.

## Chunk 1: Regression Tests

### Task 1: Add Tests for Page Visibility and Script Location

**Files:**
- Modify: `yao-cli/tests/integration.rs`
- Test: `yao-cli/tests/integration.rs`

- [ ] Add a helper list for the 10 example pages and expected headings.
- [ ] Update plotter helper to call `scripts/plot_cli_results.py`.
- [ ] Assert `examples/cli/plot_results.py` is absent and `scripts/plot_cli_results.py` exists.
- [ ] Assert `docs/src/SUMMARY.md` links every per-example page.
- [ ] Assert every page exists, contains CLI/bash commands, contains the Python plot command, contains no fenced `rust` code blocks, and links generated SVG/result/plot artifacts.
- [ ] Run `cargo test -p yao-cli --test integration cli_visualization_docs_reference_commands_and_generated_artifacts --no-default-features`.
- [ ] Expected: FAIL because pages and script move are not implemented.

## Chunk 2: Script Move and Generator Wiring

### Task 2: Move Plot Script to `scripts/`

**Files:**
- Create: `scripts/plot_cli_results.py`
- Delete: `examples/cli/plot_results.py`
- Modify: `examples/cli/generate_artifacts.sh`
- Modify: `docs/src/examples/generated/manifest.md`
- Test: `yao-cli/tests/integration.rs`

- [ ] Move the existing Python plotter implementation to `scripts/plot_cli_results.py`.
- [ ] Update generator invocation to `python3 "$REPO_ROOT/scripts/plot_cli_results.py" "$out/results" "$out/plots"`.
- [ ] Update all documentation commands to `python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots`.
- [ ] Regenerate artifacts with `YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated`.
- [ ] Run focused integration tests for artifact generation and plot cleanup.
- [ ] Commit as `docs: move CLI plot helper to scripts`.

## Chunk 3: Per-Example mdBook Pages

### Task 3: Add CLI-Only Example Pages

**Files:**
- Create: `docs/src/examples/cli/*.md`
- Modify: `docs/src/SUMMARY.md`
- Modify: `docs/src/examples/catalog.md`
- Modify: `docs/src/examples/cli-visualization.md`
- Test: `yao-cli/tests/integration.rs`

- [ ] Add one page per example with this structure: overview, setup, generate/run commands, visualization command, embedded circuit SVG, embedded plot SVG, result interpretation.
- [ ] Keep every page CLI/Python-only. Do not include Rust code blocks.
- [ ] For scripted examples, use `YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash ...` when artifacts are referenced.
- [ ] Update `SUMMARY.md` so all pages appear in the sidebar.
- [ ] Rewrite `cli-visualization.md` as an index/gallery that links to pages instead of containing all walkthroughs.
- [ ] Update `catalog.md` to link each supported workflow to its page.
- [ ] Run mdBook and integration docs tests.
- [ ] Commit as `docs: split CLI examples into pages`.

## Chunk 4: Verification and PR Update

### Task 4: Final Verification

**Files:**
- Test only.

- [ ] Run `cargo fmt -- --check`.
- [ ] Run `cargo clippy -- -D warnings`.
- [ ] Run `cargo test -p yao-cli --test integration --no-default-features`.
- [ ] Run `cargo test -p yao-rs svg`.
- [ ] Run `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items`.
- [ ] Run `mdbook build docs`.
- [ ] Run `python3 scripts/plot_cli_results.py docs/src/examples/generated/results /tmp/yao-plot-check`.
- [ ] Request final code review.
- [ ] Push PR 36 after review passes.
