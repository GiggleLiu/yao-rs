# Maintaining the examples

The published catalog is `catalog.md`. Each algorithm has one walkthrough;
its circuit JSON, SVG, and result plots live in `generated/`.

From the repository root, regenerate the CLI assets with:

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
```

The generator also refreshes plots and the artifact manifest. Generated files
are downloadable assets, not separate documentation chapters. Keep legacy URL
redirects in `docs/book.toml` when consolidating pages.
