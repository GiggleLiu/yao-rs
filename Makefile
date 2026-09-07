.DEFAULT_GOAL := help
.PHONY: help build build-release check fmt fmt-check clippy test check-all release clean doc doc-serve doc-open rustdoc example-qft run-plan cli bench bench-gates bench-qft bench-density bench-julia bench-compare

CARGO ?= cargo
CARGO_TARGET_DIR ?= target
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
	@printf "  release V=x.y.z  Bump version, tag, and push (CI publishes to crates.io)\n"
	@printf "  clean         Clean build artifacts\n"
	@printf "\nDocumentation:\n"
	@printf "  doc           Build mdBook documentation\n"
	@printf "  doc-serve     Serve mdBook at http://%s:%s\n" "$(DOC_HOST)" "$(DOC_PORT)"
	@printf "  doc-open      Build and open mdBook in browser\n"
	@printf "  rustdoc       Build Rust API docs\n"
	@printf "\nBenchmarks:\n"
	@printf "  bench         Run all Criterion benchmarks\n"
	@printf "  bench-gates   Run single-gate benchmarks\n"
	@printf "  bench-qft     Run QFT circuit benchmarks\n"
	@printf "  bench-density Run noisy density-matrix benchmarks\n"
	@printf "  bench-julia   Run Julia (Yao.jl) benchmarks and generate ground truth\n"
	@printf "  bench-compare Compare Rust vs Julia benchmark timings\n"
	@printf "\nAutomation:\n"
	@printf "  run-plan      Execute a plan with Claude autorun\n"
	@printf "\nCLI:\n"
	@printf "  cli           Build and install the yao CLI tool\n"
	@printf "\nExamples:\n"
	@printf "  example-qft   Run the QFT example\n"

build:
	$(CARGO) build --workspace

build-release:
	$(CARGO) build --workspace --release

check:
	$(CARGO) check --workspace

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings

test:
	$(CARGO) test --workspace --all-features --locked

check-all: fmt-check clippy test
	@echo "All checks passed."

# Validate, bump, check, commit, and atomically push only the release tag.
release:
ifndef V
	$(error Usage: make release V=x.y.z)
endif
	python3 scripts/release.py "$(V)"

doc:
	mdbook build docs
	$(CARGO) doc --workspace --no-deps --all-features --locked
	rm -rf docs/book/api
	cp -r "$(CARGO_TARGET_DIR)/doc" docs/book/api

doc-serve: doc
	@-lsof -ti:$(DOC_PORT) | xargs kill 2>/dev/null || true
	@echo "Serving at http://$(DOC_HOST):$(DOC_PORT)"
	python3 -m http.server $(DOC_PORT) -b $(DOC_HOST) -d docs/book

doc-open: doc
	open docs/book/index.html 2>/dev/null || xdg-open docs/book/index.html

rustdoc:
	$(CARGO) doc --workspace --no-deps --all-features --open

example-qft:
	$(CARGO) run --example qft

cli:
	$(CARGO) install --path yao-cli

bench: bench-gates bench-qft bench-density
	@echo "All Rust benchmarks complete. Results in target/criterion/"

bench-all: bench-julia bench
	python3 benchmarks/compare.py

bench-gates:
	$(CARGO) bench --bench gates

bench-qft:
	$(CARGO) bench --bench qft

bench-density:
	$(CARGO) bench --bench density

bench-julia:
	cd benchmarks/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. generate_ground_truth.jl

bench-compare:
	python3 benchmarks/compare.py

clean:
	$(CARGO) clean
	rm -rf docs/book

# Run a plan with Codex or Claude
# Usage: make run-plan [INSTRUCTIONS="..."] [OUTPUT=output.log] [AGENT_TYPE=<codex|claude>]
# PLAN_FILE defaults to the most recently modified file in docs/plans/
INSTRUCTIONS ?=
OUTPUT ?= run-plan-output.log
AGENT_TYPE ?= $(RUNNER)
PLAN_FILE ?= $(shell ls -t docs/plans/*.md 2>/dev/null | head -1)

run-plan:
	@. scripts/make_helpers.sh; \
	NL=$$'\n'; \
	BRANCH=$$(git branch --show-current); \
	PLAN_FILE="$(PLAN_FILE)"; \
	if [ "$(AGENT_TYPE)" = "claude" ]; then \
		PROCESS="1. Read the plan file$${NL}2. Execute the plan — it specifies which skill(s) to use$${NL}3. Push: git push origin $$BRANCH$${NL}4. If a PR already exists for this branch, skip. Otherwise create one."; \
	else \
		PROCESS="1. Read the plan file$${NL}2. If the plan references repo-local workflow docs under .claude/skills/*/SKILL.md, open and follow them directly. Treat slash-command names as aliases for those files.$${NL}3. Execute the tasks step by step. For each task, implement and test before moving on.$${NL}4. Push: git push origin $$BRANCH$${NL}5. If a PR already exists for this branch, skip. Otherwise create one."; \
	fi; \
	PROMPT="Execute the plan in '$$PLAN_FILE'."; \
	if [ "$(AGENT_TYPE)" != "claude" ]; then \
		PROMPT="$${PROMPT}$${NL}$${NL}Repo-local skills live in .claude/skills/*/SKILL.md. Treat any slash-command references in the plan as aliases for those skill files."; \
	fi; \
	if [ -n "$(INSTRUCTIONS)" ]; then \
		PROMPT="$${PROMPT}$${NL}$${NL}## Additional Instructions$${NL}$(INSTRUCTIONS)"; \
	fi; \
	PROMPT="$${PROMPT}$${NL}$${NL}## Process$${NL}$${PROCESS}$${NL}$${NL}## Rules$${NL}- Tests should be strong enough to catch regressions.$${NL}- Do not modify tests to make them pass.$${NL}- Test failure must be reported."; \
	echo "=== Prompt ===" && echo "$$PROMPT" && echo "===" ; \
	RUNNER="$(AGENT_TYPE)" run_agent "$(OUTPUT)" "$$PROMPT"
