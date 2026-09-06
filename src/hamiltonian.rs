//! Hermitian Pauli Hamiltonians and product-formula time evolution.
//!
//! `R_P(theta) = exp(-i theta P / 2)`. Evolution over a term `c P` uses
//! `theta = 2 c dt`. Builders use only existing one/two-qubit gates and never
//! allocate a full Hamiltonian matrix. These builders require at least one qubit.
//!
//! ```
//! use yao_rs::{ArrayReg, hamiltonian::{ising, Boundary, ProductFormula}};
//! let model = ising(3, -0.7, 0.4, Boundary::Open)?;
//! let evolution = model.evolve(0.8, 8, ProductFormula::Suzuki2)?;
//! assert_eq!(evolution.parameters(), &[0.8, -0.7, 0.4]);
//! let state = evolution.apply(&ArrayReg::zero_state(3))?;
//! assert_eq!(state.state.len(), 8);
//! # Ok::<(), String>(())
//! ```

use crate::parameters::{BoundCircuit, ParameterBinding};
use crate::{Circuit, CircuitElement, Gate, Op, OperatorPolynomial, OperatorString, control, put};

/// Product-formula order. Approximation error is distinct from roundoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductFormula {
    /// Forward term order each step; global error is generally O(1 / steps).
    LieTrotter,
    /// Forward half-step followed by reversed half-step; O(1 / steps²).
    Suzuki2,
}

/// One-dimensional model boundary. Periodic models require at least three sites
/// to avoid ambiguous self-bonds or doubled two-site bonds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Open,
    Periodic,
}

/// A real Pauli sum with explicit sharing of coefficient parameters.
///
/// Duplicate Pauli terms are retained in input order, which defines the product
/// formula. Within a term, duplicate sites and non-Pauli operators are rejected.
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    nqubits: usize,
    terms: Vec<(OperatorString, usize)>,
    coefficients: Vec<f64>,
}

impl PauliHamiltonian {
    /// Convert a numeric polynomial; each term coefficient is an independent
    /// physical parameter, reused across every product-formula step.
    pub fn new(nqubits: usize, polynomial: &OperatorPolynomial) -> Result<Self, String> {
        let mut coefficients = Vec::with_capacity(polynomial.len());
        let mut terms = Vec::with_capacity(polynomial.len());
        for (index, (coefficient, word)) in polynomial.iter().enumerate() {
            if coefficient.im != 0. {
                return Err("Hamiltonian coefficients must be real".into());
            }
            coefficients.push(coefficient.re);
            terms.push((word.clone(), index));
        }
        Self::with_shared_coefficients(nqubits, terms, coefficients)
    }

    /// Build a Pauli sum where each `(word, index)` refers to one coefficient.
    /// Repeated indices tie different terms to the same physical coupling.
    pub fn with_shared_coefficients(
        nqubits: usize,
        mut terms: Vec<(OperatorString, usize)>,
        coefficients: Vec<f64>,
    ) -> Result<Self, String> {
        if nqubits == 0 {
            return Err("Hamiltonian builders require at least one qubit".into());
        }
        for (word, index) in &mut terms {
            validate_word(nqubits, word)?;
            *word = OperatorString::new(word.ops().to_vec());
            if *index >= coefficients.len() {
                return Err("Coefficient parameter index out of range".into());
            }
        }
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err("Hamiltonian coefficients must be finite".into());
        }
        Ok(Self {
            nqubits,
            terms,
            coefficients,
        })
    }

    /// Register size.
    pub fn nqubits(&self) -> usize {
        self.nqubits
    }
    /// Physical coupling values, in construction order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Evaluate the existing numeric polynomial representation.
    pub fn polynomial(&self) -> OperatorPolynomial {
        OperatorPolynomial::new(
            self.terms
                .iter()
                .map(|(_, i)| self.coefficients[*i].into())
                .collect(),
            self.terms.iter().map(|(word, _)| word.clone()).collect(),
        )
    }

    /// Build `exp(-i H time)` with a fixed positive number of product steps.
    ///
    /// Physical parameters are `[time, coefficients...]`. Their sharing across
    /// terms and steps is recorded in the returned circuit's bindings. The
    /// graph is retained at zero time/couplings, so derivatives there are valid.
    /// For exact zero rotation angles, [`BoundCircuit::apply`] returns the
    /// input unchanged. Nonzero formulas incur the selected approximation error.
    pub fn evolve(
        &self,
        time: f64,
        steps: usize,
        formula: ProductFormula,
    ) -> Result<BoundCircuit, String> {
        if !time.is_finite() {
            return Err("Evolution time must be finite".into());
        }
        if steps == 0 {
            return Err("Product formula needs at least one step".into());
        }
        let mut builder = RotationBuilder::default();
        if self.terms.is_empty() {
            return builder.finish(
                self.nqubits,
                std::iter::once(time)
                    .chain(self.coefficients.iter().copied())
                    .collect(),
            );
        }
        let passes = if formula == ProductFormula::Suzuki2 {
            2
        } else {
            1
        };
        let gate_bound = self
            .terms
            .iter()
            .try_fold(0usize, |sum, (word, _)| {
                word.len().checked_mul(6)?.checked_add(2)?.checked_add(sum)
            })
            .and_then(|n| n.checked_mul(steps))
            .and_then(|n| n.checked_mul(passes))
            .and_then(|n| n.checked_mul(size_of::<CircuitElement>()));
        if gate_bound.is_none_or(|bytes| bytes > isize::MAX as usize) {
            return Err("Product formula length exceeds addressable storage".into());
        }
        let scale = if formula == ProductFormula::Suzuki2 {
            1.
        } else {
            2.
        } / steps as f64;
        for _ in 0..steps {
            for (word, index) in &self.terms {
                builder.rotation(
                    word,
                    ParameterBinding::Product {
                        left: 0,
                        right: index + 1,
                        scale,
                    },
                );
            }
            if formula == ProductFormula::Suzuki2 {
                for (word, index) in self.terms.iter().rev() {
                    builder.rotation(
                        word,
                        ParameterBinding::Product {
                            left: 0,
                            right: index + 1,
                            scale,
                        },
                    );
                }
            }
        }
        let parameters = std::iter::once(time)
            .chain(self.coefficients.iter().copied())
            .collect();
        builder.finish(self.nqubits, parameters)
    }
}

/// Build a Pauli-string rotation with physical parameter vector `[theta]`.
/// Identity strings retain their global phase using `Rz(theta) Phase(-theta)`.
/// This phase remains observable if the lowered gates are subsequently controlled.
pub fn pauli_rotation(
    nqubits: usize,
    word: &OperatorString,
    theta: f64,
) -> Result<BoundCircuit, String> {
    if nqubits == 0 {
        return Err("Pauli rotation requires at least one qubit".into());
    }
    validate_word(nqubits, word)?;
    let mut builder = RotationBuilder::default();
    builder.rotation(
        &OperatorString::new(word.ops().to_vec()),
        ParameterBinding::Scaled {
            index: 0,
            scale: 1.,
        },
    );
    builder.finish(nqubits, vec![theta])
}

/// Ising `H = J Σ Z_i Z_(i+1) + h Σ X_i`, using Pauli matrices (not spin/2).
/// Coefficient parameters are `[J, h]` with shared values across the chain.
pub fn ising(
    nqubits: usize,
    j: f64,
    h: f64,
    boundary: Boundary,
) -> Result<PauliHamiltonian, String> {
    let mut terms = bonds(nqubits, boundary)?
        .into_iter()
        .map(|(a, b)| (OperatorString::new(vec![(a, Op::Z), (b, Op::Z)]), 0))
        .collect::<Vec<_>>();
    terms.extend((0..nqubits).map(|i| (OperatorString::new(vec![(i, Op::X)]), 1)));
    PauliHamiltonian::with_shared_coefficients(nqubits, terms, vec![j, h])
}

/// Heisenberg/XYZ `H = Σ(Jx XX + Jy YY + Jz ZZ) + h Σ Z_i`.
/// Coefficient parameters are `[Jx, Jy, Jz, h]`; set equal couplings for the
/// isotropic model. Equal numeric values remain independently trainable here;
/// use [`PauliHamiltonian::with_shared_coefficients`] to tie those three as well.
pub fn heisenberg(
    nqubits: usize,
    j: [f64; 3],
    h: f64,
    boundary: Boundary,
) -> Result<PauliHamiltonian, String> {
    let mut terms = Vec::new();
    for (a, b) in bonds(nqubits, boundary)? {
        for (index, op) in [Op::X, Op::Y, Op::Z].into_iter().enumerate() {
            terms.push((OperatorString::new(vec![(a, op), (b, op)]), index));
        }
    }
    terms.extend((0..nqubits).map(|i| (OperatorString::new(vec![(i, Op::Z)]), 3)));
    PauliHamiltonian::with_shared_coefficients(nqubits, terms, vec![j[0], j[1], j[2], h])
}

fn bonds(nqubits: usize, boundary: Boundary) -> Result<Vec<(usize, usize)>, String> {
    if nqubits == 0 {
        return Err("Model requires at least one qubit".into());
    }
    let mut bonds = (0..nqubits - 1).map(|i| (i, i + 1)).collect::<Vec<_>>();
    if boundary == Boundary::Periodic {
        if nqubits < 3 {
            return Err("Periodic model requires at least three qubits".into());
        }
        bonds.push((nqubits - 1, 0));
    }
    Ok(bonds)
}

fn validate_word(nqubits: usize, word: &OperatorString) -> Result<(), String> {
    let mut previous = None;
    for &(site, op) in word.ops() {
        if site >= nqubits {
            return Err("Pauli site out of range".into());
        }
        if previous.is_some_and(|last| site <= last) {
            return Err("Pauli sites must be unique and sorted".into());
        }
        if !matches!(op, Op::I | Op::X | Op::Y | Op::Z) {
            return Err("Hamiltonian requires Pauli operators I/X/Y/Z".into());
        }
        previous = Some(site);
    }
    Ok(())
}

#[derive(Default)]
struct RotationBuilder {
    gates: Vec<CircuitElement>,
    bindings: Vec<ParameterBinding>,
    identity_slots: Vec<usize>,
}
impl RotationBuilder {
    fn rotation(&mut self, word: &OperatorString, angle: ParameterBinding) {
        let sites = word.ops();
        // Basis changes map each Pauli to Z; parity is accumulated on the
        // last site. Each gate is at most two-qubit, independent of word size.
        for &(site, op) in sites {
            if op == Op::Y {
                self.gates
                    .push(put(vec![site], Gate::Phase(-std::f64::consts::FRAC_PI_2)));
                self.bindings
                    .push(ParameterBinding::Fixed(-std::f64::consts::FRAC_PI_2));
            }
            if matches!(op, Op::X | Op::Y) {
                self.gates.push(put(vec![site], Gate::H));
            }
        }
        let target = sites.last().map_or(0, |(site, _)| *site);
        for &(site, _) in sites.iter().take(sites.len().saturating_sub(1)) {
            self.gates.push(control(vec![site], vec![target], Gate::X));
        }
        self.identity_slots.push(self.bindings.len());
        self.gates.push(put(vec![target], Gate::Rz(0.)));
        self.bindings.push(angle.clone());
        if sites.is_empty() {
            let negative = match angle {
                ParameterBinding::Fixed(x) => ParameterBinding::Fixed(-x),
                ParameterBinding::Scaled { index, scale } => ParameterBinding::Scaled {
                    index,
                    scale: -scale,
                },
                ParameterBinding::Product { left, right, scale } => ParameterBinding::Product {
                    left,
                    right,
                    scale: -scale,
                },
            };
            self.gates.push(put(vec![target], Gate::Phase(0.)));
            self.bindings.push(negative);
        }
        for &(site, _) in sites.iter().take(sites.len().saturating_sub(1)).rev() {
            self.gates.push(control(vec![site], vec![target], Gate::X));
        }
        for &(site, op) in sites.iter().rev() {
            if matches!(op, Op::X | Op::Y) {
                self.gates.push(put(vec![site], Gate::H));
            }
            if op == Op::Y {
                self.gates.push(put(vec![site], Gate::S));
            }
        }
    }
    fn finish(self, nqubits: usize, parameters: Vec<f64>) -> Result<BoundCircuit, String> {
        let circuit = Circuit::qubits(nqubits, self.gates).map_err(|e| e.to_string())?;
        Ok(BoundCircuit::new(circuit, parameters, self.bindings)?
            .with_identity_slots(self.identity_slots))
    }
}

#[cfg(test)]
#[path = "unit_tests/hamiltonian.rs"]
mod tests;
