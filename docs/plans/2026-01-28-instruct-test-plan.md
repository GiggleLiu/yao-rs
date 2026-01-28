# Instruct Module Test Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Comprehensive test coverage for all supported `instruct` functionality, matching YaoArrayRegister's test structure.

**Architecture:** Add tests to `src/instruct.rs` covering all gate types, control configurations, and edge cases.

**Tech Stack:** Rust, ndarray, num-complex, approx (for floating point comparison)

---

## Task 1: Pauli Gate Instruction Tests

**Files:**
- Modify: `src/instruct.rs` (add tests in `#[cfg(test)]` module)

**Tests to add:**

```rust
#[test]
fn test_instruct_pauli_x() {
    // X|0⟩ = |1⟩, X|1⟩ = |0⟩
    // Test on qubit 0, 1, 2 of 3-qubit system
}

#[test]
fn test_instruct_pauli_y() {
    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
}

#[test]
fn test_instruct_pauli_z() {
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    // Should use instruct_diagonal
}

#[test]
fn test_instruct_pauli_on_superposition() {
    // Apply X, Y, Z to |+⟩ state
}
```

**Commit:** `test(instruct): add Pauli gate instruction tests`

---

## Task 2: Hadamard and Single-Qubit Gate Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_h_creates_superposition() {
    // H|0⟩ = |+⟩, H|1⟩ = |-⟩
}

#[test]
fn test_instruct_h_on_each_qubit() {
    // 3-qubit system, H on qubit 0, 1, 2 separately
}

#[test]
fn test_instruct_s_gate() {
    // S = diag(1, i), should use diagonal path
}

#[test]
fn test_instruct_t_gate() {
    // T = diag(1, e^(iπ/4)), diagonal path
}

#[test]
fn test_instruct_identity() {
    // I gate should leave state unchanged
}
```

**Commit:** `test(instruct): add H, S, T, I gate tests`

---

## Task 3: Parametric Rotation Gate Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_rx_various_angles() {
    // Rx(0) = I, Rx(π) = -iX, Rx(π/2)
}

#[test]
fn test_instruct_ry_various_angles() {
    // Ry(0) = I, Ry(π) = -iY, Ry(π/2)
}

#[test]
fn test_instruct_rz_various_angles() {
    // Rz uses diagonal path
    // Rz(0) = I, Rz(π) = -iZ
}

#[test]
fn test_instruct_phase_various_angles() {
    // Phase(θ) = diag(1, e^(iθ))
    // Phase(π/4) = T, Phase(π/2) = S, Phase(π) = Z
}

#[test]
fn test_instruct_rotation_on_superposition() {
    // Apply rotations to |+⟩ and verify
}
```

**Commit:** `test(instruct): add parametric rotation tests`

---

## Task 4: SWAP Gate Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_swap_basic() {
    // SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩
}

#[test]
fn test_instruct_swap_symmetric() {
    // SWAP|00⟩ = |00⟩, SWAP|11⟩ = |11⟩
}

#[test]
fn test_instruct_swap_on_superposition() {
    // SWAP on entangled state
}

#[test]
fn test_instruct_swap_non_adjacent() {
    // SWAP qubits 0 and 2 in 3-qubit system
}

#[test]
fn test_instruct_swap_preserves_norm() {
    // Verify unitarity
}
```

**Commit:** `test(instruct): add SWAP gate tests`

---

## Task 5: Controlled Gate Comprehensive Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_cnot_all_basis_states() {
    // CNOT on |00⟩, |01⟩, |10⟩, |11⟩
}

#[test]
fn test_instruct_cnot_reversed() {
    // Control on qubit 1, target on qubit 0
}

#[test]
fn test_instruct_cz_gate() {
    // CZ = controlled-Z, diagonal gate
}

#[test]
fn test_instruct_cy_gate() {
    // CY = controlled-Y
}

#[test]
fn test_instruct_controlled_h() {
    // CH = controlled-Hadamard
}

#[test]
fn test_instruct_controlled_phase() {
    // Controlled-Phase(θ)
}

#[test]
fn test_instruct_controlled_on_superposition() {
    // Control in superposition creates entanglement
}
```

**Commit:** `test(instruct): add comprehensive controlled gate tests`

---

## Task 6: Multi-Control Gate Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_toffoli_all_basis() {
    // CCX on all 8 basis states of 3 qubits
}

#[test]
fn test_instruct_ccz_gate() {
    // CCZ = Toffoli with Z instead of X
}

#[test]
fn test_instruct_three_controls() {
    // CCCX on 4 qubits
}

#[test]
fn test_instruct_controlled_swap() {
    // Fredkin gate: controlled-SWAP
}

#[test]
fn test_instruct_mixed_control_values() {
    // Control on |0⟩ instead of |1⟩
    // ctrl_configs = [0] instead of [1]
}

#[test]
fn test_instruct_multi_control_mixed_values() {
    // ctrl_configs = [0, 1] - first control on |0⟩, second on |1⟩
}
```

**Commit:** `test(instruct): add multi-control gate tests`

---

## Task 7: Qudit (Higher Dimension) Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_qutrit_x_gate() {
    // X on d=3: cyclic permutation |0⟩→|1⟩→|2⟩→|0⟩
}

#[test]
fn test_instruct_qutrit_z_gate() {
    // Z on d=3: diag(1, ω, ω²) where ω = e^(2πi/3)
}

#[test]
fn test_instruct_qutrit_hadamard() {
    // Generalized Hadamard for d=3
}

#[test]
fn test_instruct_mixed_qubit_qutrit() {
    // System with dims = [2, 3, 2]
    // Apply gates to each site
}

#[test]
fn test_instruct_controlled_qutrit() {
    // Qubit controls qutrit gate
}

#[test]
fn test_instruct_qutrit_controls_qubit() {
    // Qutrit (value=2) controls qubit gate
}

#[test]
fn test_instruct_ququart() {
    // d=4 system
}
```

**Commit:** `test(instruct): add qudit (d>2) tests`

---

## Task 8: Diagonal Gate Optimization Tests

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_diagonal_z_matches_general() {
    // instruct_diagonal for Z matches instruct_single
}

#[test]
fn test_diagonal_s_matches_general() {
    // S gate via diagonal vs general path
}

#[test]
fn test_diagonal_t_matches_general() {
    // T gate via diagonal vs general path
}

#[test]
fn test_diagonal_phase_matches_general() {
    // Phase(θ) via diagonal vs general path
}

#[test]
fn test_diagonal_rz_matches_general() {
    // Rz via diagonal vs general path
}

#[test]
fn test_diagonal_custom_gate() {
    // Custom diagonal gate
}

#[test]
fn test_diagonal_qutrit() {
    // Diagonal gate on qutrit
}
```

**Commit:** `test(instruct): add diagonal optimization verification tests`

---

## Task 9: Edge Cases and Error Handling

**Files:**
- Modify: `src/instruct.rs`

**Tests to add:**

```rust
#[test]
fn test_instruct_single_qubit_system() {
    // n=1 system edge case
}

#[test]
fn test_instruct_large_system() {
    // n=10 qubits to verify scaling
}

#[test]
fn test_instruct_gate_on_last_qubit() {
    // Gate on highest index qubit
}

#[test]
fn test_instruct_non_contiguous_targets() {
    // Multi-qubit gate on qubits 0 and 2 (skipping 1)
}

#[test]
fn test_instruct_preserves_normalization() {
    // All gate types preserve |ψ|² = 1
}

#[test]
fn test_instruct_multiple_gates_sequence() {
    // Apply sequence of gates, verify final state
}

#[test]
#[should_panic]
fn test_instruct_invalid_location() {
    // Gate on non-existent qubit should panic
}

#[test]
#[should_panic]
fn test_instruct_gate_dimension_mismatch() {
    // 2x2 gate on qutrit should panic
}
```

**Commit:** `test(instruct): add edge case and error handling tests`

---

## Task 10: Parallel Implementation Verification

**Files:**
- Modify: `src/instruct.rs`

**Tests to add (feature-gated):**

```rust
#[cfg(feature = "parallel")]
mod parallel_comprehensive_tests {
    #[test]
    fn test_parallel_all_paulis() {
        // X, Y, Z via parallel path match sequential
    }

    #[test]
    fn test_parallel_rotations() {
        // Rx, Ry, Rz parallel match sequential
    }

    #[test]
    fn test_parallel_controlled() {
        // CNOT, Toffoli parallel match sequential
    }

    #[test]
    fn test_parallel_large_circuit() {
        // 16-qubit circuit with many gates
    }

    #[test]
    fn test_parallel_qutrit() {
        // Qutrit operations in parallel
    }

    #[test]
    fn test_parallel_deterministic() {
        // Same result across multiple runs
    }
}
```

**Commit:** `test(instruct): add comprehensive parallel verification tests`

---

## Summary

| Task | Tests | Coverage |
|------|-------|----------|
| 1 | 4 | Pauli gates |
| 2 | 5 | H, S, T, I |
| 3 | 5 | Rx, Ry, Rz, Phase |
| 4 | 5 | SWAP |
| 5 | 7 | Controlled gates |
| 6 | 6 | Multi-control |
| 7 | 7 | Qudits |
| 8 | 7 | Diagonal optimization |
| 9 | 8 | Edge cases |
| 10 | 6 | Parallel |
| **Total** | **60** | |

This brings test coverage in line with YaoArrayRegister's `instruct.jl` for all currently supported features.
