.PHONY: test test-verbose clippy fmt check doc doc-serve doc-open clean example-qft

# Run all tests
test:
	cargo test

# Run tests with verbose output
test-verbose:
	cargo test -- --nocapture

# Run clippy lints
clippy:
	cargo clippy -- -D warnings

# Format code
fmt:
	cargo fmt

# Check formatting without modifying
fmt-check:
	cargo fmt -- --check

# Full check: format, clippy, tests
check: fmt-check clippy test

# Build mdBook documentation
doc:
	mdbook build docs

# Serve mdBook documentation locally
doc-serve:
	mdbook serve docs

# Open mdBook documentation in browser
doc-open: doc
	open docs/book/index.html 2>/dev/null || xdg-open docs/book/index.html

# Build Rust API docs
rustdoc:
	cargo doc --no-deps

# Run the QFT example
example-qft:
	cargo run --example qft

# Clean build artifacts
clean:
	cargo clean
	rm -rf docs/book
