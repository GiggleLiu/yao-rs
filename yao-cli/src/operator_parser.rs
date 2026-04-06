use anyhow::{Context, anyhow, bail};
use num_complex::Complex64;
use yao_rs::{Op, OperatorPolynomial, OperatorString};

pub fn parse_operator(input: &str) -> anyhow::Result<OperatorPolynomial> {
    let input = input.trim();
    if input.is_empty() {
        bail!("Empty operator expression");
    }

    let mut coeffs = Vec::new();
    let mut opstrings = Vec::new();

    for term in split_terms(input) {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }

        let (coeff, ops) =
            parse_term(term).with_context(|| format!("Failed to parse term: '{term}'"))?;
        coeffs.push(coeff);
        opstrings.push(OperatorString::new(ops));
    }

    if coeffs.is_empty() {
        bail!("No valid terms in operator expression: '{input}'");
    }

    Ok(OperatorPolynomial::new(coeffs, opstrings))
}

fn split_terms(input: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut paren_depth = 0usize;

    for ch in input.chars() {
        match ch {
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth = paren_depth.saturating_sub(1);
                current.push(ch);
            }
            '+' | '-' if paren_depth == 0 && !current.trim().is_empty() => {
                terms.push(current);
                current = String::new();
                if ch == '-' {
                    current.push(ch);
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.trim().is_empty() {
        terms.push(current);
    }

    terms
}

fn parse_term(term: &str) -> anyhow::Result<(Complex64, Vec<(usize, Op)>)> {
    let term = term.trim();

    let (coeff, ops_str) = if let Some(idx) = term.find('*') {
        let coeff_str = term[..idx].trim().replace(' ', "");
        let ops_str = term[idx + 1..].trim();
        let coeff: f64 = coeff_str
            .parse()
            .with_context(|| format!("Invalid coefficient: '{coeff_str}'"))?;
        (Complex64::new(coeff, 0.0), ops_str)
    } else {
        let (sign, ops_str) = if let Some(rest) = term.strip_prefix('-') {
            (-1.0, rest.trim())
        } else if let Some(rest) = term.strip_prefix('+') {
            (1.0, rest.trim())
        } else {
            (1.0, term)
        };
        (Complex64::new(sign, 0.0), ops_str)
    };

    let ops = parse_op_string(ops_str)
        .with_context(|| format!("Failed to parse operator string: '{ops_str}'"))?;
    if ops.is_empty() {
        bail!("No operators found in term: '{term}'");
    }

    Ok((coeff, ops))
}

fn parse_op_string(input: &str) -> anyhow::Result<Vec<(usize, Op)>> {
    let mut ops = Vec::new();
    let mut rest = input.trim();

    while !rest.is_empty() {
        let (op, after_name) = parse_op_name(rest)?;
        let after_name = after_name.trim();
        if !after_name.starts_with('(') {
            bail!(
                "Expected '(' after operator name, got: '{}'",
                &after_name[..after_name.len().min(10)]
            );
        }

        let after_paren = &after_name[1..];
        let close = after_paren
            .find(')')
            .ok_or_else(|| anyhow!("Missing closing ')' in '{input}'"))?;

        let site_str = after_paren[..close].trim();
        let site: usize = site_str
            .parse()
            .with_context(|| format!("Invalid site index: '{site_str}'"))?;

        ops.push((site, op));
        rest = after_paren[close + 1..].trim();
    }

    Ok(ops)
}

fn parse_op_name(input: &str) -> anyhow::Result<(Op, &str)> {
    let candidates = [
        ("P0", Op::P0),
        ("P1", Op::P1),
        ("Pu", Op::Pu),
        ("Pd", Op::Pd),
        ("I", Op::I),
        ("X", Op::X),
        ("Y", Op::Y),
        ("Z", Op::Z),
    ];

    for (name, op) in candidates {
        if let Some(rest) = input.strip_prefix(name) {
            return Ok((op, rest));
        }
    }

    bail!("Unknown operator at: '{}'", &input[..input.len().min(10)]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(r: f64) -> Complex64 {
        Complex64::new(r, 0.0)
    }

    #[test]
    fn test_single_op() {
        let poly = parse_operator("Z(0)").unwrap();
        assert_eq!(poly.len(), 1);
        assert_eq!(poly.coeffs()[0], c(1.0));
        assert_eq!(poly.opstrings()[0].ops(), &[(0, Op::Z)]);
    }

    #[test]
    fn test_multi_site() {
        let poly = parse_operator("Z(0)Z(1)").unwrap();
        assert_eq!(poly.len(), 1);
        assert_eq!(poly.opstrings()[0].ops(), &[(0, Op::Z), (1, Op::Z)]);
    }

    #[test]
    fn test_sum() {
        let poly = parse_operator("Z(0) + X(1)").unwrap();
        assert_eq!(poly.len(), 2);
        assert_eq!(poly.coeffs()[0], c(1.0));
        assert_eq!(poly.coeffs()[1], c(1.0));
        assert_eq!(poly.opstrings()[0].ops(), &[(0, Op::Z)]);
        assert_eq!(poly.opstrings()[1].ops(), &[(1, Op::X)]);
    }

    #[test]
    fn test_difference() {
        let poly = parse_operator("X(0)Y(1) - Y(0)X(1)").unwrap();
        assert_eq!(poly.len(), 2);
        assert_eq!(poly.coeffs()[1], c(-1.0));
    }

    #[test]
    fn test_coeff() {
        let poly = parse_operator("0.5 * Z(0)Z(1) + 0.3 * X(0)").unwrap();
        assert_eq!(poly.len(), 2);
        assert!((poly.coeffs()[0] - c(0.5)).norm() < 1e-12);
        assert!((poly.coeffs()[1] - c(0.3)).norm() < 1e-12);
    }

    #[test]
    fn test_projectors() {
        let poly = parse_operator("P0(0) + P1(1)").unwrap();
        assert_eq!(poly.len(), 2);
        assert_eq!(poly.opstrings()[0].ops(), &[(0, Op::P0)]);
        assert_eq!(poly.opstrings()[1].ops(), &[(1, Op::P1)]);
    }

    #[test]
    fn test_raising_lowering() {
        let poly = parse_operator("Pu(0)Pd(1)").unwrap();
        assert_eq!(poly.opstrings()[0].ops(), &[(0, Op::Pu), (1, Op::Pd)]);
    }

    #[test]
    fn test_negative_leading_term() {
        let poly = parse_operator("-Z(0)").unwrap();
        assert_eq!(poly.coeffs()[0], c(-1.0));
    }

    #[test]
    fn test_error_on_invalid() {
        assert!(parse_operator("Q(0)").is_err());
        assert!(parse_operator("Z(abc)").is_err());
        assert!(parse_operator("").is_err());
    }
}
