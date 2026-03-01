---
name: review-implementation
description: Use after implementing a feature or completing a plan task to review code quality, completeness, and correctness before committing
---

# Review Implementation

Dispatch two parallel review subagents to verify code quality and correctness.

## Usage

```
/review-implementation [model|rule|generic]
```

If no argument, auto-detect from `git diff`.

## Steps

### 1. Detect What Changed

```bash
# Detect new/modified files
git diff --name-only --diff-filter=A main..HEAD
git diff --name-only main..HEAD
```

Detection rules:
- New file in `src/tools/` → tool review
- New file in `src/` → module review
- Only modified files → quality review only
- Explicit argument overrides auto-detection

### 2. Gather Context

```bash
# Get diff summary
DIFF_SUMMARY=$(git diff --stat main..HEAD)

# Check for linked issue (if PR exists)
PR_NUM=$(gh pr view --json number -q '.number' 2>/dev/null)
if [ -n "$PR_NUM" ]; then
  ISSUE_CONTEXT=$(gh pr view --json body -q '.body' | grep -oP '#\d+' | head -1)
fi

# Build status
cargo clippy --all-targets -- -D warnings 2>&1
cargo test 2>&1
```

### 3. Dispatch Parallel Review Subagents

Dispatch **two** subagents using the `Agent` tool with `subagent_type="superpowers:code-reviewer"`:

**Subagent A — Structural Reviewer** (foreground):
```
Review the following code changes for structural completeness:

## What changed
{DIFF_SUMMARY}

## Review checklist
- [ ] All public items have doc comments
- [ ] Error types use thiserror, not string errors
- [ ] MCP tools follow pattern: parameter struct + inner method + tool wrapper
- [ ] Tests cover happy path AND edge cases
- [ ] No unwrap() in library code (only in tests)
- [ ] Clippy passes with -D warnings
- [ ] JSON fixtures have no line breaks (single-line)

{ISSUE_CONTEXT}
```

**Subagent B — Quality Reviewer** (background):
```
Review the following code changes for quality:

## What changed
{DIFF_SUMMARY}

## Review criteria
- DRY: No duplicated logic
- KISS: Simplest approach that works
- Error handling: anyhow for app code, thiserror for library errors
- Test quality: Meaningful assertions, not just "doesn't panic"
- No over-engineering: No unnecessary abstractions
- Performance: No obvious inefficiencies
- Security: No command injection, path traversal, etc.
```

### 4. Collect and Present Report

After both subagents complete, merge their findings:

```markdown
## Review Report

### Build Status
- [ ] `cargo clippy -- -D warnings`: PASS/FAIL
- [ ] `cargo test`: PASS/FAIL

### Structural Review
<findings from Subagent A>

### Quality Review
<findings from Subagent B>

### Auto-fixed
<list of issues fixed automatically>

### Remaining Items
<list of items needing user decision>
```

### 5. Auto-fix Simple Issues

Fix automatically:
- Missing doc comments on public items
- Clippy warnings
- Formatting (`cargo fmt`)

Do NOT auto-fix:
- Architectural concerns
- Test coverage gaps (ask user first)
- API design changes

## Integration

- After `superpowers:executing-plans` completes a task
- Before `superpowers:finishing-a-development-branch`
- Standalone: `/review-implementation`
