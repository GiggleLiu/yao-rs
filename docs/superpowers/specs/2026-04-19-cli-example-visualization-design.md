# CLI Example Visualization Design

## Goal

Make yao-rs CLI examples reproduce the Yao.jl and QuAlgorithmZoo.jl documentation examples through bash-driven workflows, then visualize the generated artifacts in a way that is useful for both verification and documentation.

## Source Context

The upstream references are local checkouts:

- `~/.julia/dev/Yao`
- `~/.julia/dev/QuAlgorithmZoo`

Relevant upstream patterns:

- Yao.jl quick-start and plotting docs emphasize `YaoPlots.vizcircuit`, especially for QFT and basic circuit examples.
- QuAlgorithmZoo examples emphasize algorithm outcomes: recovered secrets, amplified marked states, expectation values, trained distributions, and loss curves.
- Some upstream examples are directly representable with current yao-rs CLI primitives; others require capabilities outside the current CLI scope, such as training loops, richer oracle builders, Hamiltonian time evolution, or external chemistry workflows.

## Design Summary

Use a layered visualization model:

1. **Algorithm Trace** is the documentation spine.
   Each example records its upstream source, bash command, generated circuit JSON, generated SVG, and numeric evidence.

2. **Circuit Gallery** gives YaoPlots-like visual parity.
   Each supported circuit has an SVG thumbnail and an expandable full render.

3. **Generated Results** show the algorithm outcome.
   Probability distributions, expectation values, and later training curves are rendered from generated data as the primary result evidence.

The bash generator is the reproducibility contract: if the docs show a circuit or result, a bash command must regenerate it.

## Current Generated Dataset

The exploratory bash generation step produced artifacts under `/tmp/yao-cli-visual-data`:

- `circuits/*.json`
- `svg/*.svg`
- `results/*-probs.json`
- `results/qaoa-maxcut-line4-depth2-expect.json`

Generated examples:

| Example | Source Alignment | Generated Evidence |
| --- | --- | --- |
| Bell | yao-rs starter / Yao basics | `00` and `11` each have probability `0.5` |
| GHZ 4 | Yao GHZ pattern | `0000` and `1111` each have probability `0.5` |
| QFT 4 | Yao EasyBuild QFT | Uniform probabilities from `|0000>` |
| Phase estimation Z | Yao EasyBuild / QuAlgorithmZoo phase estimation class | phase bit result at state `3` |
| Hadamard test Z | QuAlgorithmZoo README class | probability output over two qubits |
| Swap test | QuAlgorithmZoo README class | probability output over three qubits |
| Bernstein-Vazirani `1011` | QuAlgorithmZoo example | secret state index `11` has probability `1.0` |
| Grover marked state `5` | QuAlgorithmZoo Grover class | marked state index `5` has probability about `0.9453` |
| QAOA MaxCut line4 depth2 | QuAlgorithmZoo QAOA class | `Z(0)Z(1)` expectation real part about `0.3074` |
| QCBM static depth2 | Yao / QuAlgorithmZoo QCBM class | static zero-parameter circuit currently concentrates on state `0` |

## Documentation Structure

Update or extend `docs/src/examples/catalog.md` as the main index, and add a dedicated mdBook page for a generated visualization report. This page should be designed for mdBook, not copied from the browser companion prototype.

Recommended files:

- `docs/src/examples/catalog.md`: concise index, coverage table, and links.
- `docs/src/examples/cli-visualization.md`: documentation-native visualization page with copy-paste commands, generated SVGs, and generated result summaries.
- `docs/src/examples/generated/`: checked-in SVG and small JSON or Markdown summaries if generated artifacts should be browsable in published docs.

Recommended page sections:

1. **Overview**
   State that CLI examples are reproducible bash workflows, not just prose snippets.

2. **Copy-Paste Reproduction Commands**
   Every example should show the exact commands a reader can run. Prefer short shell blocks over prose-only descriptions. The commands should include both one-shot script usage and artifact-generation usage where applicable.

   Example shape:

   ```bash
   YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
   YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh /tmp/yao-cli-examples
   ```

3. **Trace Table**
   Columns:
   - Example
   - Upstream source
   - Copy-paste command
   - Generated circuit artifact
   - Visualization artifact
   - Generated result artifact
   - Coverage status

4. **Circuit Gallery**
   Small SVG thumbnails from generated data for examples where circuit structure is central.
   QFT, phase estimation, Grover, QAOA, and QCBM should be visible first.

5. **Generated Results**
   Compact plots or tables sourced from generated result JSON, not hand-entered numbers.
   For probability distributions, show significant states first and avoid dumping all zero entries.
   For expectation values, show the command and result side by side.

6. **Deferred Coverage**
   Keep explicit notes for examples that need future capabilities, such as VQE, full QCBM training, QuGAN, HHL, Shor arithmetic oracles, and Hamiltonian/time-evolution workflows.

## mdBook Visualization Page

The browser companion pages are useful design prototypes, but they are session artifacts under `.superpowers/brainstorm`. Do not publish those files directly and do not try to recreate the companion UI exactly.

Instead, build a normal mdBook page that makes the generated data useful:

- preserve the same conceptual layers: commands, trace, circuit gallery, and generated results;
- show all reproduction commands as copy-paste shell blocks;
- use checked-in SVG outputs for circuit thumbnails;
- use Markdown tables for the reproducibility trace;
- use generated JSON summaries to populate probability and expectation tables;
- link every displayed SVG or result summary back to the bash command that regenerates it;
- keep the styling compatible with mdBook instead of depending on the visual companion frame CSS or JavaScript.

The docs page should be static and reviewable in git. It should feel like documentation with embedded visual evidence, not like a standalone dashboard app.

## Generated Data Use

Generated data should be first-class documentation input:

- SVG files become embedded circuit figures.
- Result JSON files become compact summaries in Markdown tables.
- Circuit JSON files are linked as downloadable or inspectable artifacts.
- A manifest records each example name, source, command, generated files, and key evidence.

The documentation should avoid manual transcription where possible. If a result value appears in `cli-visualization.md`, it should come from generated data or a checked-in generated summary.

## Bash Artifact Generator

Add a repo script, likely `examples/cli/generate_artifacts.sh`, that:

- accepts an output directory, defaulting to a temporary directory;
- accepts `YAO_BIN`, defaulting to `yao`;
- builds or copies each circuit JSON;
- runs `yao visualize` to produce SVG files;
- runs `yao simulate | yao probs` or `yao run --op ...` to produce result JSON;
- prints a short manifest at the end.

The generator should reuse existing example scripts where practical, but may need helper modes so scripts can write circuit JSON without immediately discarding their temporary files.

## Implementation Boundaries

In scope:

- bash generation of data;
- docs updates;
- current CLI SVG visualization;
- probability and expectation summaries;
- explicit upstream mapping to local Yao.jl and QuAlgorithmZoo examples.

Out of scope for this iteration:

- implementing training loops;
- implementing missing algorithm builders;
- adding a web application to the distributed package;
- changing core simulator behavior;
- changing SVG renderer semantics unless required by a specific generated circuit.

## Error Handling

The generator should fail fast when:

- `YAO_BIN` is unavailable;
- a generated circuit does not parse;
- SVG generation fails;
- a result JSON file cannot be produced.

The docs should label partial reproductions clearly. For example, QCBM is currently a static ansatz demonstration, not a reproduction of the full trained-distribution workflow.

## Testing

Tests should cover:

- existing CLI example scripts continue to produce expected JSON results;
- the artifact generator completes with `YAO_BIN=target/debug/yao`;
- generated result JSON has expected key evidence for Bell, GHZ, Bernstein-Vazirani, Grover, and QAOA;
- generated SVG files exist and contain valid SVG wrappers.

Follow TDD for implementation: write failing tests for the generator behavior before changing production scripts or docs.

## Review Note

The brainstorming skill normally calls for a spec-review subagent. This harness only permits subagents when the user explicitly asks for delegated agent work, so this spec has not received a subagent review.
