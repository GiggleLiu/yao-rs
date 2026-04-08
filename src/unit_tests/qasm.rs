use crate::apply::apply;
use crate::qasm::{from_qasm, to_qasm};
use crate::register::ArrayReg;
use approx::assert_abs_diff_eq;

fn probs_from_qasm(qasm: &str) -> Vec<f64> {
    let result = from_qasm(qasm).unwrap();
    let reg = ArrayReg::zero_state(result.circuit.num_sites());
    let out = apply(&result.circuit, &reg);
    crate::measure::probs(&out, None)
}

#[test]
fn test_bell_state() {
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#,
    );
    // |00⟩ + |11⟩
    assert_abs_diff_eq!(probs[0], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[2], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[3], 0.5, epsilon = 1e-10);
}

#[test]
fn test_ghz_state() {
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
"#,
    );
    // |000⟩ + |111⟩
    assert_abs_diff_eq!(probs[0], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[7], 0.5, epsilon = 1e-10);
    let interior_sum: f64 = probs[1..7].iter().sum();
    assert_abs_diff_eq!(interior_sum, 0.0, epsilon = 1e-10);
}

#[test]
fn test_x_gate() {
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
x q[0];
"#,
    );
    // |0⟩ → |1⟩
    assert_abs_diff_eq!(probs[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 1.0, epsilon = 1e-10);
}

#[test]
fn test_hadamard() {
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
"#,
    );
    // |0⟩ → |+⟩
    assert_abs_diff_eq!(probs[0], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 0.5, epsilon = 1e-10);
}

#[test]
fn test_rotation_gates() {
    // rx(pi) ≈ X (up to global phase), so |0⟩ → |1⟩
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(pi) q[0];
"#,
    );
    assert_abs_diff_eq!(probs[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 1.0, epsilon = 1e-10);
}

#[test]
fn test_ry_gate() {
    // ry(pi/2) on |0⟩ → equal superposition
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
ry(pi/2) q[0];
"#,
    );
    assert_abs_diff_eq!(probs[0], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 0.5, epsilon = 1e-10);
}

#[test]
fn test_rz_gate() {
    // rz on |0⟩ → still |0⟩ (only phase change)
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(pi/4) q[0];
"#,
    );
    assert_abs_diff_eq!(probs[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 0.0, epsilon = 1e-10);
}

#[test]
fn test_s_and_t_gates() {
    // S and T are diagonal, don't change |0⟩ probs
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
s q[0];
t q[0];
"#,
    );
    assert_abs_diff_eq!(probs[0], 1.0, epsilon = 1e-10);
}

#[test]
fn test_swap_gate() {
    // |10⟩ → |01⟩ via SWAP
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
swap q[0], q[1];
"#,
    );
    // x q[0] → state with qubit 0 flipped, then SWAP decomposes to 3 CNOTs.
    // Verify by comparing with a native yao-rs SWAP circuit.
    use crate::circuit::{Circuit, put};
    use crate::gate::Gate;
    let native = Circuit::qubits(2, vec![
        put(vec![0], Gate::X),
        put(vec![0, 1], Gate::SWAP),
    ]).unwrap();
    let native_probs = {
        let reg = ArrayReg::zero_state(2);
        let out = apply(&native, &reg);
        crate::measure::probs(&out, None)
    };
    for (p1, p2) in probs.iter().zip(native_probs.iter()) {
        assert_abs_diff_eq!(p1, p2, epsilon = 1e-10);
    }
}

#[test]
fn test_toffoli() {
    // CCX: |110⟩ → |111⟩
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
x q[1];
ccx q[0], q[1], q[2];
"#,
    );
    // |111⟩ = index 7
    assert_abs_diff_eq!(probs[7], 1.0, epsilon = 1e-10);
}

#[test]
fn test_cz_gate() {
    // CZ on |1+⟩ should give |1-⟩
    // Start: x q[0]; h q[1]; → |1⟩|+⟩
    // After cz: |1⟩|-⟩
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
h q[1];
cz q[0], q[1];
h q[1];
"#,
    );
    // After h on q[1]: |1⟩|-⟩ → |1⟩|1⟩ = |11⟩ = index 3
    assert_abs_diff_eq!(probs[3], 1.0, epsilon = 1e-10);
}

#[test]
fn test_measurement_collection() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#,
    )
    .unwrap();
    assert_eq!(result.measurements.len(), 2);
    assert_eq!(result.measurements[0], (0, 0));
    assert_eq!(result.measurements[1], (1, 1));
}

#[test]
fn test_barrier_is_noop() {
    // Barrier shouldn't affect results
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
barrier q[0], q[1];
cx q[0], q[1];
"#,
    );
    assert_abs_diff_eq!(probs[0], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[3], 0.5, epsilon = 1e-10);
}

#[test]
fn test_phase_gate() {
    // p(pi) on |+⟩ → |-⟩, then H → |1⟩
    let probs = probs_from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
u1(pi) q[0];
h q[0];
"#,
    );
    assert_abs_diff_eq!(probs[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 1.0, epsilon = 1e-10);
}

#[test]
fn test_to_qasm_bell() {
    use crate::circuit::{Circuit, control, put};
    use crate::gate::Gate;
    let elements = vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)];
    let circuit = Circuit::qubits(2, elements).unwrap();
    let qasm = to_qasm(&circuit).unwrap();
    assert!(qasm.contains("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("h q[0];"));
    assert!(qasm.contains("cx q[0],q[1];"));
}

#[test]
fn test_to_qasm_parametric() {
    use crate::circuit::{Circuit, put};
    use crate::gate::Gate;
    let elements = vec![
        put(vec![0], Gate::Rx(1.5)),
        put(vec![0], Gate::Phase(0.25)),
    ];
    let circuit = Circuit::qubits(1, elements).unwrap();
    let qasm = to_qasm(&circuit).unwrap();
    assert!(qasm.contains("rx(1.5) q[0];"));
    assert!(qasm.contains("u1(0.25) q[0];"));
}

#[test]
fn test_roundtrip_simulation() {
    // Build circuit → export to QASM → import back → simulate both → compare
    use crate::circuit::{Circuit, control, put};
    use crate::gate::Gate;
    let elements = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
        put(vec![1], Gate::S),
    ];
    let circuit = Circuit::qubits(2, elements).unwrap();

    let qasm = to_qasm(&circuit).unwrap();
    let reimported = from_qasm(&qasm).unwrap();

    let reg = ArrayReg::zero_state(2);
    let out1 = apply(&circuit, &reg);
    let out2 = apply(&reimported.circuit, &reg);

    let probs1 = crate::measure::probs(&out1, None);
    let probs2 = crate::measure::probs(&out2, None);

    for (p1, p2) in probs1.iter().zip(probs2.iter()) {
        assert_abs_diff_eq!(p1, p2, epsilon = 1e-10);
    }
}

#[test]
fn test_parse_error() {
    let result = from_qasm("this is not valid qasm");
    assert!(result.is_err());
}

#[test]
fn test_unsupported_opaque_gate() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
qreg q[1];
opaque my_gate a;
my_gate q[0];
"#,
    );
    assert!(result.is_err());
}

#[test]
fn test_empty_circuit_preserves_qubit_count() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
"#,
    )
    .unwrap();
    assert_eq!(result.circuit.num_sites(), 3);
    assert_eq!(result.circuit.elements.len(), 0);
}

#[test]
fn test_measurements_only_circuit() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#,
    )
    .unwrap();
    assert_eq!(result.circuit.num_sites(), 2);
    assert_eq!(result.circuit.elements.len(), 0);
    assert_eq!(result.measurements.len(), 2);
}

#[test]
fn test_no_include_with_primitive_gates() {
    // Without qelib1.inc, only U and CX primitives are available
    let result = from_qasm(
        r#"
OPENQASM 2.0;
qreg q[2];
U(3.14159265358979, 0, 3.14159265358979) q[0];
CX q[0], q[1];
"#,
    )
    .unwrap();
    assert_eq!(result.circuit.num_sites(), 2);
    assert!(result.circuit.elements.len() >= 2);
}

#[test]
fn test_export_uses_standard_qelib1_names() {
    use crate::circuit::{Circuit, control, put};
    use crate::gate::Gate;
    let elements = vec![
        put(vec![0], Gate::Phase(0.5)),
        control(vec![0], vec![1], Gate::Phase(0.25)),
    ];
    let circuit = Circuit::qubits(2, elements).unwrap();
    let qasm = to_qasm(&circuit).unwrap();
    // Should use u1/cu1 (standard qelib1.inc) not p/cp
    assert!(qasm.contains("u1(0.5) q[0];"));
    assert!(qasm.contains("cu1(0.25) q[0],q[1];"));
    assert!(!qasm.contains("p("));
    assert!(!qasm.contains("cp("));
}
