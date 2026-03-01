---
name: review-implementation
description: Use after implementing a model, rule, or any code change to verify completeness and correctness before committing
---

# Review Implementation

Dispatches a quality review subagent with fresh context (no implementation history bias) to check code quality, test coverage, and correctness.

## Invocation

- `/review-implementation` -- auto-detect from git diff

## Step 1: Detect What Changed

```bash
BASE_SHA=$(git merge-base main HEAD)
HEAD_SHA=$(git rev-parse HEAD)
git diff --stat $BASE_SHA..$HEAD_SHA
git diff --name-only $BASE_SHA..$HEAD_SHA
```

### Detect Linked Issue

```bash
PR_NUM=$(gh pr view --json number -q .number 2>/dev/null)
if [ -n "$PR_NUM" ]; then
  ISSUE_NUM=$(gh pr view $PR_NUM --json body -q .body | grep -oE '#[0-9]+' | head -1 | tr -d '#')
fi
if [ -n "$ISSUE_NUM" ]; then
  ISSUE_BODY=$(gh issue view $ISSUE_NUM --json title,body -q '"# " + .title + "\n\n" + .body')
fi
```

If an issue is found, pass it as `{ISSUE_CONTEXT}` to the subagent. If not, set to "No linked issue found."

## Step 2: Dispatch Quality Reviewer

Dispatch using `Agent` tool with `subagent_type="superpowers:code-reviewer"`.

Fill this prompt template and pass it as the agent prompt:

---

### Quality Reviewer Prompt Template

> First, read `.claude/rules/rust-writing.md` to understand the project's Rust design patterns and conventions. Use these as your review baseline.
>
> You are reviewing code changes for quality in the `tn-mcp-rs` Rust codebase (a tensor network MCP server for quantum circuit simulation). You have NO context about prior implementation work -- review the code fresh.
>
> **What Changed:** {DIFF_SUMMARY}
>
> **Changed Files:** {CHANGED_FILES}
>
> **Plan Step Context:** {PLAN_STEP}
>
> **Linked Issue:** {ISSUE_CONTEXT}
>
> **Git Range:** Base: {BASE_SHA} / Head: {HEAD_SHA}
>
> Start by running:
> ```bash
> git diff --stat {BASE_SHA}..{HEAD_SHA}
> git diff {BASE_SHA}..{HEAD_SHA}
> ```
> Then read the changed files in full.
>
> **Review Criteria:**
>
> 1. **DRY** -- Duplicated logic that should be extracted?
> 2. **KISS** -- Unnecessarily complex? Over-engineered abstractions, convoluted control flow?
> 3. **HC/LC** -- Single responsibility? God functions (>50 lines doing multiple things)?
>
> **HCI (if MCP tools changed, i.e. `src/tools/`):**
>
> 4. **Error messages** -- Actionable? Bad: `"invalid parameter"`. Good: `"circuit JSON must include num_qubits field"`.
> 5. **Discoverability** -- Missing documentation on tool parameters?
> 6. **Consistency** -- Similar operations expressed similarly? Parameter names, output formats uniform?
> 7. **Least surprise** -- Output matches expectations? No silent failures?
> 8. **Feedback** -- Tool confirms what it did?
>
> **Test Quality:**
>
> 9. **Naive Test Detection** -- Flag tests that:
>    - Only check types/shapes, not values (e.g., `assert!(result.is_ok())` without checking the result)
>    - Mirror the implementation (recomputing the same formula proves nothing)
>    - Lack adversarial cases (only happy path, no malformed inputs or edge cases)
>    - Use trivial instances only (1-qubit circuits may pass with bugs; need multi-qubit tests)
>    - Assert count too low (1-2 asserts for non-trivial code is insufficient)
>
> **Output format:**
>
> ```
> ## Code Quality Review
>
> ### Design Principles
> - DRY: OK / ISSUE -- [description with file:line]
> - KISS: OK / ISSUE -- [description with file:line]
> - HC/LC: OK / ISSUE -- [description with file:line]
>
> ### HCI (if MCP tools changed)
> - Error messages: OK / ISSUE -- [description]
> - Discoverability: OK / ISSUE -- [description]
> - Consistency: OK / ISSUE -- [description]
> - Least surprise: OK / ISSUE -- [description]
> - Feedback: OK / ISSUE -- [description]
>
> ### Test Quality
> - Naive test detection: OK / ISSUE
>   - [specific tests flagged with reason and file:line]
>
> ### Issues
>
> #### Critical (Must Fix)
> [Bugs, correctness issues, data loss risks]
>
> #### Important (Should Fix)
> [Architecture problems, missing tests, poor error handling]
>
> #### Minor (Nice to Have)
> [Code style, optimization opportunities]
>
> ### Summary
> - [list of action items with severity]
> ```

---

## Step 3: Collect and Address Findings

When the subagent returns:

1. **Parse results** -- identify ISSUE items from the report
2. **Fix automatically** -- clear issues at Important+ severity
3. **Report to user** -- ambiguous issues, Minor items, anything uncertain

## Step 4: Present Consolidated Report

```
## Review: [Description of Changes]

### Build Status
- `make test`: PASS / FAIL
- `make clippy`: PASS / FAIL

### Code Quality
- DRY: OK / ...
- KISS: OK / ...
- HC/LC: OK / ...

### HCI (if MCP tools changed)
...

### Test Quality
...

### Fixes Applied
- [list of issues automatically fixed]

### Remaining Items (needs user decision)
- [list of issues that need user input]
```
