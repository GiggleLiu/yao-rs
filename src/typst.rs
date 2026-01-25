//! Typst-based PDF generation for quantum circuits.
//!
//! This module provides PDF rendering of quantum circuits by embedding
//! the Typst compiler. Requires the `typst` feature to be enabled.

use crate::circuit::Circuit;
use crate::json::circuit_to_json;
use typst::layout::PagedDocument;
use typst_as_lib::TypstEngine;

/// The embedded circuit rendering template.
const CIRCUIT_TEMPLATE: &str = include_str!("../visualization/circuit.typ");

/// Error type for PDF generation.
#[derive(Debug)]
pub enum PdfError {
    /// Typst compilation failed.
    Compilation(String),
    /// PDF export failed.
    PdfExport(String),
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdfError::Compilation(msg) => write!(f, "Typst compilation error: {}", msg),
            PdfError::PdfExport(msg) => write!(f, "PDF export error: {}", msg),
        }
    }
}

impl std::error::Error for PdfError {}

/// Escape a JSON string for embedding in a Typst string literal.
fn escape_json_for_typst(json: &str) -> String {
    let mut escaped = String::with_capacity(json.len() * 2);
    for c in json.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Generate a complete Typst document for a circuit.
fn generate_typst_source(circuit: &Circuit) -> String {
    let json = circuit_to_json(circuit);
    let escaped_json = escape_json_for_typst(&json);

    // Note: json.decode is deprecated in Typst 0.14 but json() expects a file path.
    // We use json.decode since we're embedding JSON as a string literal.
    format!(
        r#"#set page(width: auto, height: auto, margin: 5pt)

{template}

#{{
  let data = json.decode("{json}")
  render-circuit-impl(data)
}}
"#,
        template = CIRCUIT_TEMPLATE,
        json = escaped_json
    )
}

/// Compile a circuit to PDF bytes.
///
/// # Arguments
/// * `circuit` - The quantum circuit to render
///
/// # Returns
/// PDF file contents as bytes, or an error if compilation fails.
///
/// # Example
/// ```ignore
/// use yao_rs::{Circuit, put, control, Gate};
/// use yao_rs::typst::to_pdf;
///
/// let circuit = Circuit::new(vec![2, 2, 2], vec![
///     put(vec![0], Gate::H),
///     control(vec![0], vec![1], Gate::X),
/// ]).unwrap();
///
/// let pdf_bytes = to_pdf(&circuit)?;
/// std::fs::write("circuit.pdf", pdf_bytes)?;
/// ```
pub fn to_pdf(circuit: &Circuit) -> Result<Vec<u8>, PdfError> {
    let source = generate_typst_source(circuit);

    // Build the Typst engine with system fonts and package support
    let engine = TypstEngine::builder()
        .main_file(source)
        .search_fonts_with(Default::default())
        .with_package_file_resolver()
        .build();

    // Compile the document
    let compiled = engine.compile::<PagedDocument>();
    let document = compiled.output.map_err(|e| PdfError::Compilation(format!("{}", e)))?;

    // Export to PDF (use {:?} as EcoVec<SourceDiagnostic> doesn't implement Display)
    typst_pdf::pdf(&document, &Default::default())
        .map_err(|e| PdfError::PdfExport(format!("{:?}", e)))
}
