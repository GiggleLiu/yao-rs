.DEFAULT_GOAL := help
.PHONY: help build build-release check fmt fmt-check clippy test check-all clean doc doc-serve doc-open rustdoc example-qft run-plan

CARGO ?= cargo
DOC_PORT ?= 3001
DOC_HOST ?= 127.0.0.1

help:
	@printf "Rust targets:\n"
	@printf "  build         Build the project\n"
	@printf "  build-release Build release binary\n"
	@printf "  check         Run cargo check\n"
	@printf "  fmt           Format code\n"
	@printf "  fmt-check     Check formatting\n"
	@printf "  clippy        Run clippy (deny warnings)\n"
	@printf "  test          Run the test suite\n"
	@printf "  check-all     Run fmt-check, clippy, and test\n"
	@printf "  clean         Clean build artifacts\n"
	@printf "\nDocumentation:\n"
	@printf "  doc           Build mdBook documentation\n"
	@printf "  doc-serve     Serve mdBook at http://%s:%s\n" "$(DOC_HOST)" "$(DOC_PORT)"
	@printf "  doc-open      Build and open mdBook in browser\n"
	@printf "  rustdoc       Build Rust API docs\n"
	@printf "\nAutomation:\n"
	@printf "  run-plan      Execute a plan with Claude autorun\n"
	@printf "\nExamples:\n"
	@printf "  example-qft   Run the QFT example\n"

build:
	$(CARGO) build

build-release:
	$(CARGO) build --release

check:
	$(CARGO) check

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt -- --check

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --all-features

check-all: fmt-check clippy test
	@echo "All checks passed."

doc:
	mdbook build docs

doc-serve:
	mdbook serve docs -p $(DOC_PORT) -n $(DOC_HOST)

doc-open: doc
	open docs/book/index.html 2>/dev/null || xdg-open docs/book/index.html

rustdoc:
	$(CARGO) doc --no-deps --all-features --open

example-qft:
	$(CARGO) run --example qft

clean:
	$(CARGO) clean
	rm -rf docs/book

# Plan execution
# Usage: make run-plan [PLAN_FILE=...] [OUTPUT=output.log]
PLAN_FILE ?= $(shell ls -t docs/plans/*.md 2>/dev/null | head -1)
OUTPUT ?= claude-output.log

run-plan:
	@BRANCH=$$(git branch --show-current); \
	PLAN_FILE="$(PLAN_FILE)"; \
	PROMPT="Execute the plan in '$$PLAN_FILE'."; \
	NL=$$'\n'; \
	PROMPT="$${PROMPT}$${NL}$${NL}## Process$${NL}1. Read the plan file$${NL}2. Execute the plan — it specifies which skill(s) to use$${NL}3. Push: git push origin $$BRANCH$${NL}4. If a PR already exists for this branch, skip. Otherwise create one."; \
	PROMPT="$${PROMPT}$${NL}$${NL}## Rules$${NL}- Tests should be strong enough to catch regressions.$${NL}- Do not modify tests to make them pass.$${NL}- Test failure must be reported."; \
	echo "=== Prompt ===" && echo "$$PROMPT" && echo "===" ; \
	claude --dangerously-skip-permissions \
		--model opus \
		--verbose \
		--max-turns 500 \
		-p "$$PROMPT" 2>&1 | tee "$(OUTPUT)"
