use crate::apply::apply;
use crate::circuit::CircuitElement;
use crate::density_matrix::DensityMatrix;
use crate::noise::NoiseChannel;
use crate::qasm::{QasmError, from_qasm, from_qasm_file, to_qasm};
use crate::register::{ArrayReg, Register};
use approx::assert_abs_diff_eq;
use std::time::{SystemTime, UNIX_EPOCH};

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
    let native =
        Circuit::qubits(2, vec![put(vec![0], Gate::X), put(vec![0, 1], Gate::SWAP)]).unwrap();
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
        put(vec![0], Gate::SqrtX),
    ];
    let circuit = Circuit::qubits(1, elements).unwrap();
    let qasm = to_qasm(&circuit).unwrap();
    assert!(qasm.contains("rx(1.5) q[0];"));
    assert!(qasm.contains("u1(0.25) q[0];"));
    assert!(qasm.contains("sx q[0];"));
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

#[test]
fn test_roundtrip_controlled_gates() {
    use crate::circuit::{Circuit, control, put};
    use crate::gate::Gate;

    let elements = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),          // cx
        control(vec![0], vec![1], Gate::Z),          // cz
        control(vec![0], vec![1], Gate::Rx(1.0)),    // crx (decomposed inline)
        control(vec![0], vec![1], Gate::Ry(0.5)),    // cry (decomposed inline)
        control(vec![0], vec![1], Gate::Rz(0.3)),    // crz
        control(vec![0], vec![1], Gate::Phase(0.7)), // cu1
        put(vec![0, 1], Gate::SWAP),                 // swap (decomposed to 3 cx)
    ];
    let circuit = Circuit::qubits(2, elements).unwrap();

    let qasm = to_qasm(&circuit).unwrap();
    let reimported = from_qasm(&qasm).unwrap();

    let reg = ArrayReg::zero_state(2);
    let probs1 = crate::measure::probs(&apply(&circuit, &reg), None);
    let probs2 = crate::measure::probs(&apply(&reimported.circuit, &reg), None);

    for (p1, p2) in probs1.iter().zip(probs2.iter()) {
        assert_abs_diff_eq!(p1, p2, epsilon = 1e-10);
    }
}

#[test]
fn test_export_uses_extended_gate_names() {
    use crate::circuit::{Circuit, control, put};
    use crate::gate::Gate;
    let circuit = Circuit::qubits(
        2,
        vec![
            put(vec![0, 1], Gate::SWAP),
            put(vec![0], Gate::SqrtX),
            control(vec![0], vec![1], Gate::Rx(1.0)),
        ],
    )
    .unwrap();
    let qasm = to_qasm(&circuit).unwrap();
    assert!(qasm.contains("swap q[0],q[1];"));
    assert!(qasm.contains("sx q[0];"));
    assert!(qasm.contains("crx(1) q[0],q[1];"));
}

#[test]
fn test_reset_imports_as_reset_channel() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
x q[0];
reset q[0];
"#,
    )
    .unwrap();

    let reset_channels: Vec<_> = result
        .circuit
        .elements
        .iter()
        .filter(|element| {
            matches!(
                element,
                CircuitElement::Channel(pc)
                    if pc.locs == vec![0]
                        && matches!(pc.channel, NoiseChannel::Reset { p0, p1 } if p0 == 1.0 && p1 == 0.0)
            )
        })
        .collect();

    assert_eq!(reset_channels.len(), 1);
}

#[test]
fn test_reset_density_matrix_returns_qubit_to_zero() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
x q[0];
reset q[0];
"#,
    )
    .unwrap();

    let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(1));
    dm.apply(&result.circuit);

    let probs = crate::measure::probs(&dm, None);
    assert_abs_diff_eq!(probs[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(probs[1], 0.0, epsilon = 1e-10);
}

#[test]
fn test_reset_renders_in_svg() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
reset q[0];
"#,
    )
    .unwrap();

    let svg = result.circuit.to_svg();
    assert!(svg.contains("Reset"));
}

#[test]
fn test_scientific_notation_in_u_gate_imports() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q0[1];
u(0.5,-4.5e-14,-pi) q0[0];
"#,
    )
    .unwrap();

    let rz_angles: Vec<f64> = result
        .circuit
        .elements
        .iter()
        .filter_map(|element| match element {
            CircuitElement::Gate(pg) => match &pg.gate {
                crate::gate::Gate::Rz(theta) => Some(*theta),
                _ => None,
            },
            _ => None,
        })
        .collect();

    assert!(
        rz_angles
            .iter()
            .any(|theta| (*theta + 4.5e-14).abs() < 1e-12)
    );
}

#[test]
fn test_scientific_notation_in_rz_imports() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q0[1];
rz(1e5) q0[0];
"#,
    )
    .unwrap();

    let rz_angle = result
        .circuit
        .elements
        .iter()
        .find_map(|element| match element {
            CircuitElement::Gate(pg) => match &pg.gate {
                crate::gate::Gate::Rz(theta) => Some(*theta),
                _ => None,
            },
            _ => None,
        })
        .unwrap();

    assert_abs_diff_eq!(rz_angle, 100000.0, epsilon = 1e-6);
}

#[test]
fn test_from_qasm_file_expands_scientific_notation_with_relative_include() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "yao-rs-qasm-scientific-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).unwrap();

    let include_path = temp_dir.join("defs.inc");
    let qasm_path = temp_dir.join("main.qasm");
    std::fs::write(&include_path, "gate big(theta) q { U(0,0,theta) q; }\n").unwrap();
    std::fs::write(
        &qasm_path,
        "OPENQASM 2.0;\ninclude \"defs.inc\";\nqreg q[1];\nbig(1e5) q[0];\n",
    )
    .unwrap();

    let result = from_qasm_file(qasm_path.to_str().unwrap()).unwrap();
    let rz_angle = result
        .circuit
        .elements
        .iter()
        .find_map(|element| match element {
            CircuitElement::Gate(pg) => match &pg.gate {
                crate::gate::Gate::Rz(theta) => Some(*theta),
                _ => None,
            },
            _ => None,
        })
        .unwrap();

    assert_abs_diff_eq!(rz_angle, 100000.0, epsilon = 1e-6);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_extra_gate_definitions_import() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ryy(0.5) q[0],q[1];
rzx(0.25) q[1],q[2];
dcx q[0],q[1];
ccz q[0],q[1],q[2];
"#,
    )
    .unwrap();

    assert_eq!(result.circuit.num_sites(), 3);
    assert!(!result.circuit.elements.is_empty());
}

#[test]
fn test_classical_conditional_returns_unsupported_error() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
measure q[1] -> c[0];
if (c==1) x q[1];
"#,
    );
    assert!(matches!(
        result,
        Err(QasmError::Unsupported(msg))
            if msg.contains("classical conditional")
    ));
}

#[test]
fn test_64_bit_classical_conditional_returns_unsupported_error() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c0[64];
if(c0==9223372036854775808) x q[0];
"#,
    );
    assert!(matches!(
        result,
        Err(QasmError::Unsupported(msg))
            if msg.contains("classical conditional")
    ));
}

#[test]
fn test_too_large_conditional_literal_returns_error_without_panicking() {
    let result = from_qasm(
        r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c0[66];
if(c0==36893488147419103232) x q[0];
"#,
    );
    assert!(result.is_err());
}
