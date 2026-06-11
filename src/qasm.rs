//! OpenQASM 2.0 import and export for quantum circuits.
//!
//! Uses the `openqasm` crate to parse QASM 2.0 source code and convert it
//! to yao-rs `Circuit` objects. Also provides export from `Circuit` to QASM 2.0.
//!
//! # Example
//! ```
//! use yao_rs::qasm::{from_qasm, to_qasm};
//!
//! let qasm = r#"
//! OPENQASM 2.0;
//! include "qelib1.inc";
//! qreg q[2];
//! h q[0];
//! cx q[0], q[1];
//! "#;
//!
//! let result = from_qasm(qasm).unwrap();
//! assert_eq!(result.circuit.num_sites(), 2);
//! // Gates are decomposed to U+CX primitives (Rz, Ry, CX)
//! assert!(result.circuit.elements.len() >= 2);
//! ```

use crate::circuit::{Circuit, CircuitElement, channel, control, put};
use crate::gate::Gate;
use crate::noise::NoiseChannel;
use openqasm::{GateWriter, GenericError, Linearize, ProgramVisitor, Value};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

// ========== Error type ==========

/// Error type for QASM operations.
#[derive(Debug)]
pub enum QasmError {
    /// QASM parsing or type-checking failed.
    Parse(String),
    /// Gate linearization failed.
    Linearize(String),
    /// The QASM program uses a construct that yao-rs does not model.
    Unsupported(String),
    /// A QASM gate has no equivalent in yao-rs.
    UnsupportedGate(String),
    /// Circuit construction failed.
    Circuit(crate::circuit::CircuitError),
}

impl std::fmt::Display for QasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QasmError::Parse(msg) => write!(f, "QASM parse error: {msg}"),
            QasmError::Linearize(msg) => write!(f, "QASM linearize error: {msg}"),
            QasmError::Unsupported(msg) => write!(f, "unsupported QASM construct: {msg}"),
            QasmError::UnsupportedGate(name) => write!(f, "unsupported QASM gate: {name}"),
            QasmError::Circuit(e) => write!(f, "circuit error: {e}"),
        }
    }
}

impl std::error::Error for QasmError {}

impl From<crate::circuit::CircuitError> for QasmError {
    fn from(e: crate::circuit::CircuitError) -> Self {
        QasmError::Circuit(e)
    }
}

// ========== Result type ==========

/// Result of parsing a QASM file.
#[derive(Debug)]
pub struct QasmResult {
    /// The quantum circuit (gates/channels only — measurements are separate).
    pub circuit: Circuit,
    /// Measurement instructions collected from the QASM source: `(qubit, classical_bit)`.
    pub measurements: Vec<(usize, usize)>,
}

// ========== Import: QASM → Circuit ==========

/// Extra gate definitions appended to the bundled qelib1.inc.
/// These cover modern Qiskit gates (swap, sx, p, cp, etc.) that
/// many real-world QASM files use but the crate's bundled qelib1.inc lacks.
const EXTRA_GATES: &str = "\n\
// Extended gates (modern Qiskit qelib1.inc)\n\
gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }\n\
gate p(lambda) q { U(0,0,lambda) q; }\n\
gate swap a,b { cx a,b; cx b,a; cx a,b; }\n\
gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }\n\
gate sx a { sdg a; h a; sdg a; }\n\
gate sxdg a { s a; h a; s a; }\n\
gate cp(lambda) a,b { u1(lambda/2) a; cx a,b; u1(-lambda/2) b; cx a,b; u1(lambda/2) b; }\n\
gate crx(theta) a,b { u1(pi/2) b; cx a,b; u3(-theta/2,0,0) b; cx a,b; u3(theta/2,-pi/2,0) b; }\n\
gate cry(theta) a,b { ry(theta/2) b; cx a,b; ry(-theta/2) b; cx a,b; }\n\
gate csx a,b { h b; cp(pi/2) a,b; h b; }\n\
gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; cx a,b; rz(theta) b; cx a,b; rx(-pi/2) a; rx(-pi/2) b; }\n\
gate rzx(theta) a,b { h b; cx a,b; rz(theta) b; cx a,b; h b; }\n\
gate dcx a,b { cx a,b; cx b,a; }\n\
gate ccz a,b,c { h c; ccx a,b,c; h c; }\n\
gate rxx(theta) a,b { u3(pi/2,theta,0) a; h b; cx a,b; u1(-theta) b; cx a,b; h b; u2(-pi,pi-theta) a; }\n\
gate rzz(theta) a,b { cx a,b; u1(theta) b; cx a,b; }\n";

/// Create a parser whose hardcoded qelib1.inc is extended with modern gates.
///
/// Uses `FilePolicy::with_file` to override the bundled qelib1.inc with an
/// extended version, avoiding brittle string replacement on user source code.
fn make_parser<'a>(cache: &'a mut openqasm::SourceCache) -> openqasm::Parser<'a> {
    // The default FilePolicy::filesystem() embeds the original qelib1.inc.
    // We parse that source, append our extras, and replace the hardcoded file.
    let mut policy = openqasm::parser::FilePolicy::filesystem();
    // Append extra gates to the existing hardcoded qelib1.inc
    if let openqasm::parser::FilePolicy::FileSystem { ref mut hardcoded } = policy
        && let Some(base) = hardcoded.get_mut(std::path::Path::new("qelib1.inc"))
    {
        base.push_str(EXTRA_GATES);
    }
    openqasm::Parser::new(cache).with_file_policy(policy)
}

/// Finish parsing, type-check, and linearize to a circuit.
fn finish_and_linearize(parser: openqasm::Parser<'_>) -> Result<QasmResult, QasmError> {
    let program = parser
        .done()
        .to_errors()
        .map_err(|e| QasmError::Parse(format!("{e}")))?;

    if has_classical_conditional(&program) {
        return Err(QasmError::Unsupported(
            "classical conditional (if): classical feed-forward is not supported".to_string(),
        ));
    }

    program
        .type_check()
        .to_errors()
        .map_err(|e| QasmError::Parse(format!("{e}")))?;

    // Count declared qubits from the AST before linearization,
    // so empty circuits preserve their qubit count.
    let declared_qubits = count_declared_qubits(&program);

    let state = Rc::new(RefCell::new(BuilderState::default()));
    let builder = CircuitBuilder {
        state: Rc::clone(&state),
    };

    // depth=100: fully expand all gate definitions to U+CX primitives.
    let mut linearizer = Linearize::new(builder, 100);
    linearizer
        .visit_program(&program)
        .map_err(|e| QasmError::Linearize(format!("{e}")))?;

    let st = state.borrow();
    // Use the larger of declared qubits vs initialized qubits
    // (handles empty circuits with no gates).
    let nqubits = st.nqubits.max(declared_qubits);
    let circuit = Circuit::qubits(nqubits, st.elements.clone())?;
    Ok(QasmResult {
        circuit,
        measurements: st.measurements.clone(),
    })
}

/// Count total declared qubits from qreg declarations in the AST.
fn count_declared_qubits(program: &openqasm::Program) -> usize {
    let mut total = 0usize;
    for decl in &program.decls {
        if let openqasm::Decl::QReg { ref reg } = **decl {
            if let Some(size) = reg.index {
                total += size as usize;
            } else {
                total += 1;
            }
        }
    }
    total
}

fn has_classical_conditional(program: &openqasm::Program) -> bool {
    program
        .decls
        .iter()
        .any(|decl| matches!(&**decl, openqasm::Decl::Stmt(stmt) if stmt_is_conditional(stmt)))
}

fn stmt_is_conditional(stmt: &openqasm::Stmt) -> bool {
    matches!(stmt, openqasm::Stmt::Conditional { .. })
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn normalize_qasm_source(mut source: String) -> String {
    source = source.replace("\r\n", "\n");
    if source.is_empty() {
        source.push(' ');
    }
    source
}

fn expand_float_exponents(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut expanded = String::with_capacity(src.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
            expanded.push_str(&src[start..i]);
            continue;
        }

        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
            expanded.push_str(&src[start..i]);
            continue;
        }

        if bytes[i].is_ascii_digit() && (i == 0 || !is_identifier_byte(bytes[i - 1])) {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }

            if i < bytes.len() && bytes[i] == b'.' {
                let frac_start = i + 1;
                let mut frac_end = frac_start;
                while frac_end < bytes.len() && bytes[frac_end].is_ascii_digit() {
                    frac_end += 1;
                }
                if frac_end > frac_start {
                    i = frac_end;
                }
            }

            let mantissa_end = i;
            if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
                let mut exponent_end = i + 1;
                if exponent_end < bytes.len() && matches!(bytes[exponent_end], b'+' | b'-') {
                    exponent_end += 1;
                }
                let exponent_digits_start = exponent_end;
                while exponent_end < bytes.len() && bytes[exponent_end].is_ascii_digit() {
                    exponent_end += 1;
                }

                let next_is_identifier =
                    exponent_end < bytes.len() && is_identifier_byte(bytes[exponent_end]);
                if exponent_end > exponent_digits_start && !next_is_identifier {
                    let literal = &src[start..exponent_end];
                    if let Ok(value) = literal.parse::<f64>() {
                        expanded.push_str(&format!("{value}"));
                        i = exponent_end;
                        continue;
                    }
                }
            }

            expanded.push_str(&src[start..mantissa_end]);
            continue;
        }

        expanded.push(bytes[i] as char);
        i += 1;
    }

    expanded
}

/// Parse an OpenQASM 2.0 source string into a circuit.
pub fn from_qasm(source: &str) -> Result<QasmResult, QasmError> {
    let mut cache = openqasm::SourceCache::new();
    let mut parser = make_parser(&mut cache);
    parser.parse_source(expand_float_exponents(source), None::<&str>);
    finish_and_linearize(parser)
}

/// Parse an OpenQASM 2.0 file into a circuit.
///
/// Include paths are resolved relative to the file's parent directory.
pub fn from_qasm_file(path: &str) -> Result<QasmResult, QasmError> {
    let mut cache = openqasm::SourceCache::new();
    let mut parser = make_parser(&mut cache);
    let source = std::fs::read_to_string(path).map_err(|e| QasmError::Parse(format!("{e}")))?;
    let source = normalize_qasm_source(source);
    parser.parse_source(expand_float_exponents(&source), Some(path));
    finish_and_linearize(parser)
}

// ========== Internal: GateWriter ==========

#[derive(Default)]
struct BuilderState {
    nqubits: usize,
    elements: Vec<CircuitElement>,
    measurements: Vec<(usize, usize)>,
}

struct CircuitBuilder {
    state: Rc<RefCell<BuilderState>>,
}

/// Convert a `Value` (rational + rational·π) to f64.
fn value_to_f64(v: &Value) -> f64 {
    let a = *v.a.numer() as f64 / *v.a.denom() as f64;
    let b = *v.b.numer() as f64 / *v.b.denom() as f64;
    a + b * PI
}

impl GateWriter for CircuitBuilder {
    type Error = QasmError;

    fn initialize(
        &mut self,
        qubits: &[openqasm::Symbol],
        _bits: &[openqasm::Symbol],
    ) -> Result<(), Self::Error> {
        self.state.borrow_mut().nqubits = qubits.len();
        Ok(())
    }

    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
        self.state
            .borrow_mut()
            .elements
            .push(control(vec![copy], vec![xor], Gate::X));
        Ok(())
    }

    fn write_u(
        &mut self,
        theta: Value,
        phi: Value,
        lambda: Value,
        reg: usize,
    ) -> Result<(), Self::Error> {
        let th = value_to_f64(&theta);
        let ph = value_to_f64(&phi);
        let la = value_to_f64(&lambda);

        // U(θ,φ,λ) = Rz(φ)·Ry(θ)·Rz(λ) (up to global phase)
        // Circuit order (left-to-right in time): Rz(λ), Ry(θ), Rz(φ)
        let mut st = self.state.borrow_mut();
        if la.abs() > 1e-15 {
            st.elements.push(put(vec![reg], Gate::Rz(la)));
        }
        if th.abs() > 1e-15 {
            st.elements.push(put(vec![reg], Gate::Ry(th)));
        }
        if ph.abs() > 1e-15 {
            st.elements.push(put(vec![reg], Gate::Rz(ph)));
        }
        Ok(())
    }

    fn write_opaque(
        &mut self,
        name: &openqasm::Symbol,
        _params: &[Value],
        _args: &[usize],
    ) -> Result<(), Self::Error> {
        // With depth=100, all qelib1.inc gates are fully expanded to U+CX.
        // Only truly opaque (body-less) gates reach here.
        Err(QasmError::UnsupportedGate(name.as_str().to_string()))
    }

    fn write_barrier(&mut self, _regs: &[usize]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
        self.state.borrow_mut().measurements.push((from, to));
        Ok(())
    }

    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
        // Reset is represented as a deterministic channel. Pure-state `apply`
        // skips channels today, so this is meaningful on density-matrix and
        // tensor-network paths while staying consistent with existing semantics.
        self.state
            .borrow_mut()
            .elements
            .push(channel(vec![reg], NoiseChannel::Reset { p0: 1.0, p1: 0.0 }));
        Ok(())
    }

    fn start_conditional(
        &mut self,
        _reg: usize,
        _count: usize,
        _val: u64,
    ) -> Result<(), Self::Error> {
        Err(QasmError::UnsupportedGate(
            "conditional execution (if)".to_string(),
        ))
    }

    fn end_conditional(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// ========== Export: Circuit → QASM ==========

/// Export a circuit as OpenQASM 2.0 source code.
///
/// Uses the extended qelib1.inc gate set (same as IBM Qiskit).
/// Gates like swap, sx, crx, cry are emitted directly.
pub fn to_qasm(circuit: &Circuit) -> Result<String, QasmError> {
    let n = circuit.num_sites();
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{n}];\n"));

    for element in &circuit.elements {
        match element {
            CircuitElement::Gate(pg) => {
                // Handle active-low controls: wrap with X gates
                let neg_ctrls: Vec<usize> = pg
                    .control_configs
                    .iter()
                    .enumerate()
                    .filter(|&(_, c)| !c)
                    .map(|(i, _)| pg.control_locs[i])
                    .collect();
                for &q in &neg_ctrls {
                    out.push_str(&format!("x q[{q}];\n"));
                }

                if pg.control_locs.is_empty() {
                    out.push_str(&uncontrolled_gate_qasm(&pg.gate, &pg.target_locs)?);
                } else {
                    out.push_str(&controlled_gate_qasm(
                        &pg.gate,
                        &pg.control_locs,
                        &pg.target_locs,
                    )?);
                }

                for &q in &neg_ctrls {
                    out.push_str(&format!("x q[{q}];\n"));
                }
            }
            CircuitElement::Annotation(pa) => {
                let crate::circuit::Annotation::Label(text) = &pa.annotation;
                out.push_str(&format!("// {text}\n"));
            }
            CircuitElement::Channel(_) => {
                return Err(QasmError::UnsupportedGate(
                    "noise channels cannot be exported to QASM".to_string(),
                ));
            }
        }
    }
    Ok(out)
}

fn fmt_qubits(locs: &[usize]) -> String {
    locs.iter()
        .map(|l| format!("q[{l}]"))
        .collect::<Vec<_>>()
        .join(",")
}

fn uncontrolled_gate_qasm(gate: &Gate, targets: &[usize]) -> Result<String, QasmError> {
    let q = fmt_qubits(targets);
    let line = match gate {
        Gate::X => format!("x {q};\n"),
        Gate::Y => format!("y {q};\n"),
        Gate::Z => format!("z {q};\n"),
        Gate::H => format!("h {q};\n"),
        Gate::S => format!("s {q};\n"),
        Gate::T => format!("t {q};\n"),
        Gate::SWAP => format!("swap {q};\n"),
        Gate::SqrtX => format!("sx {q};\n"),
        Gate::Phase(theta) => format!("u1({theta}) {q};\n"),
        Gate::Rx(theta) => format!("rx({theta}) {q};\n"),
        Gate::Ry(theta) => format!("ry({theta}) {q};\n"),
        Gate::Rz(theta) => format!("rz({theta}) {q};\n"),
        Gate::SqrtY | Gate::SqrtW | Gate::ISWAP | Gate::FSim(_, _) => {
            return Err(QasmError::UnsupportedGate(format!(
                "{gate} has no standard QASM 2.0 equivalent"
            )));
        }
        Gate::Custom { label, .. } => {
            return Err(QasmError::UnsupportedGate(format!(
                "custom gate '{label}' cannot be auto-exported to QASM"
            )));
        }
    };
    Ok(line)
}

fn controlled_gate_qasm(
    gate: &Gate,
    ctrls: &[usize],
    targets: &[usize],
) -> Result<String, QasmError> {
    let all = |locs: &[&[usize]]| -> String {
        locs.iter()
            .flat_map(|l| l.iter())
            .map(|l| format!("q[{l}]"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let line = match (gate, ctrls.len(), targets.len()) {
        (Gate::X, 1, 1) => format!("cx {};\n", all(&[ctrls, targets])),
        (Gate::X, 2, 1) => format!("ccx {};\n", all(&[ctrls, targets])),
        (Gate::Y, 1, 1) => format!("cy {};\n", all(&[ctrls, targets])),
        (Gate::Z, 1, 1) => format!("cz {};\n", all(&[ctrls, targets])),
        (Gate::H, 1, 1) => format!("ch {};\n", all(&[ctrls, targets])),
        (Gate::SWAP, 1, 2) => format!("cswap {};\n", all(&[ctrls, targets])),
        (Gate::SqrtX, 1, 1) => format!("csx {};\n", all(&[ctrls, targets])),
        (Gate::Rx(theta), 1, 1) => format!("crx({theta}) {};\n", all(&[ctrls, targets])),
        (Gate::Ry(theta), 1, 1) => format!("cry({theta}) {};\n", all(&[ctrls, targets])),
        (Gate::Rz(theta), 1, 1) => format!("crz({theta}) {};\n", all(&[ctrls, targets])),
        (Gate::Phase(theta), 1, 1) => format!("cu1({theta}) {};\n", all(&[ctrls, targets])),
        _ => {
            return Err(QasmError::UnsupportedGate(format!(
                "controlled {gate} with {} controls has no QASM 2.0 equivalent",
                ctrls.len()
            )));
        }
    };
    Ok(line)
}

#[cfg(test)]
#[path = "unit_tests/qasm.rs"]
mod tests;
