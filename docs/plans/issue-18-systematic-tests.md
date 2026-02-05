# Systematic Tests with Yao.jl Ground Truth

Issue: #18

## Context

yao-rs needs systematic testing against the original Yao.jl implementation to ensure correctness. Currently, 169 inline tests are scattered across src files, and there's no validation against the Julia reference implementation.

## Approach

1. Generate ground truth data from Yao.jl (gate matrices, state evolution, einsum results, measurements)
2. Reorganize all tests into `tests/` folder mirroring `src/` structure
3. Add ground truth validation tests that compare yao-rs output against Yao.jl

## Tasks

### Phase 1: Ground Truth Data Generation

1. Create `scripts/generate_test_data.jl` with Yao.jl to generate:
   - `tests/data/gates.json` - All 16 gate types with multiple parameter values (0, π/4, π/2, π, 3π/2, 2π), edge cases (near-zero angles)
   - `tests/data/apply.json` - State evolution cases: single gates on basis states, multi-qubit circuits (up to 8 qubits), controlled gates (single/multi, active-low), qudit circuits (d=3,4), all easybuild circuits
   - `tests/data/einsum.json` - Tensor network contraction results for same circuits
   - `tests/data/measure.json` - Probability distributions for known circuits

2. Run the Julia script to generate JSON files with 15 decimal precision

### Phase 2: Test Infrastructure

3. Create `tests/common/mod.rs` with shared utilities:
   - `load_test_data<T>(filename)` - deserialize JSON from `tests/data/`
   - `assert_states_close(a, b, tol, msg)` - compare complex state vectors
   - `assert_matrices_close(a, b, tol, msg)` - compare gate matrices
   - `state_from_json()` / `matrix_from_json()` - JSON to ndarray conversion

### Phase 3: Test File Reorganization

4. Rename existing test files to mirror src/:
   - `test_circuit.rs` → `circuit.rs`
   - `test_gates.rs` + `test_new_gates.rs` → `gate.rs`
   - `test_einsum.rs` → `einsum.rs`
   - `test_json.rs` → `json.rs`
   - `test_operator.rs` → `operator.rs`
   - `test_state.rs` → `state.rs`
   - `test_tensors.rs` → `tensors.rs`
   - `test_apply.rs` → `apply.rs`
   - `test_easybuild.rs` → `easybuild.rs`

5. Move inline tests from src files to tests/:
   - `src/apply.rs` (21 tests) → `tests/apply.rs`
   - `src/instruct.rs` (92 tests) → `tests/instruct.rs` (new file)
   - `src/measure.rs` (36 tests) → `tests/measure.rs` (new file)
   - `src/index.rs` (14 tests) → `tests/index.rs` (new file)
   - `src/easybuild.rs` (3 tests) → `tests/easybuild.rs`
   - `src/torch_contractor.rs` (3 tests) → `tests/torch_contractor.rs` (new file)

6. Merge and delete old test files:
   - `test_integration.rs` → split into relevant module tests
   - `test_boundary.rs` → merge into relevant module tests
   - `test_omeco.rs` → merge into `einsum.rs`

### Phase 4: Ground Truth Tests

7. Add ground truth validation tests:
   - `tests/gate.rs`: `test_gate_matrices_ground_truth()` - validate all gate matrices
   - `tests/apply.rs`: `test_apply_ground_truth()` - validate state evolution
   - `tests/einsum.rs`: `test_einsum_ground_truth()` - validate tensor network results
   - `tests/measure.rs`: `test_measure_ground_truth()` - validate probability distributions

### Phase 5: Verification

8. Remove all `#[cfg(test)]` modules from src files
9. Run `make check-all` to verify all tests pass
10. Verify test count is preserved or increased

## Acceptance Criteria

- [ ] No `#[cfg(test)]` modules remain in src/ files
- [ ] Test file names in tests/ mirror src/ structure exactly
- [ ] `tests/data/` contains ground truth JSON files covering:
  - All 16 gate types with multiple parameter values
  - Edge cases (zero angles, near-identity matrices)
  - Circuits: single-gate, multi-qubit (up to 8), controlled, multi-controlled, qudit (d=3,4)
  - All easybuild circuits (QFT, variational, phase estimation, supremacy)
- [ ] `scripts/generate_test_data.jl` can regenerate all JSON files
- [ ] Ground truth tests compare yao-rs output against Yao.jl with tolerance 1e-10
- [ ] `make check-all` passes
