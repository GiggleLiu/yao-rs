//! Matrix-free `exp(-i H t) x` for Hermitian complex operators on the CPU.
//!
//! Uses reorthogonalized Lanczos and adaptive time steps. Only the small real
//! tridiagonal projection is diagonalized (with the existing `faer` dependency).
//! Basis storage is O(`krylov_dim * x.len()`); no full Hamiltonian is constructed.
//! Callback operators must be linear, Hermitian, and unchanged throughout a call.
//! Checks on the explored subspace detect some violations, not a general proof
//! of Hermiticity. Inputs need not be normalized. No differentiation is provided.
//!
//! The step criterion implements the defect bound of Jawecki, Auzinger & Koch,
//! Theorem 1, <https://doi.org/10.1007/s10543-019-00771-6>. Lanczos/restart design
//! was also informed by KrylovKit v0.10.2 (`775546b`); this is an independent
//! implementation, not a translation of its phi-function integrator.
//!
//! ```
//! use yao_rs::{ArrayReg, evolution::EvolutionOptions, hamiltonian::{ising, Boundary}};
//! let h = ising(6, -0.7, 0.4, Boundary::Open)?;
//! let result = h.evolve_krylov(&ArrayReg::zero_state(6), 0.8, EvolutionOptions::default())?;
//! assert_eq!(result.info.time_reached, 0.8);
//! assert!(result.info.estimated_error <= result.info.tolerance);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use num_complex::Complex64 as C;
use serde::Serialize;

/// Accuracy and bounded work for Hermitian evolution.
#[derive(Debug, Clone, Copy)]
pub struct EvolutionOptions {
    /// Maximum basis size per step (at least 2; capped by vector length).
    pub krylov_dim: usize,
    /// Maximum total operator applications, including an unfinished last step.
    pub max_matvecs: usize,
    /// Absolute tolerance on the final vector's Euclidean norm error.
    pub atol: f64,
    /// Relative tolerance, scaled by the initial vector norm.
    pub rtol: f64,
}

impl Default for EvolutionOptions {
    fn default() -> Self {
        Self {
            krylov_dim: 30,
            max_matvecs: 10_000,
            atol: 1e-12,
            rtol: 1e-10,
        }
    }
}

/// Diagnostics for completed steps. The error estimate includes the Lanczos
/// truncation bound and discarded reorthogonalization corrections. It is not a
/// certified floating-point bound: callback, inner-product and eigensolver
/// roundoff are not fully bounded. Very tight tolerances may be unattainable.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EvolutionInfo {
    pub time_reached: f64,
    pub matvecs: usize,
    pub steps: usize,
    pub max_krylov_dim: usize,
    pub estimated_error: f64,
    pub tolerance: f64,
}

/// Evolved vector and diagnostics. Successful calls reach the requested time.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub state: Vec<C>,
    pub info: EvolutionInfo,
}

/// Why evolution failed. A work/precision limit never becomes a successful
/// full-time result; inspect [`EvolutionError::partial`] to recover progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionFailure {
    InvalidInput,
    Operator,
    Numerical,
    WorkLimit,
    PrecisionLimit,
}

pub struct EvolutionError {
    pub kind: EvolutionFailure,
    pub message: String,
    /// Last accepted state and its actual time, if execution had started.
    /// An unfinished basis does not advance this state, but counts as work.
    pub partial: Option<Box<EvolutionResult>>,
}

impl std::fmt::Debug for EvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionError")
            .field("kind", &self.kind)
            .field("message", &self.message)
            .field("partial_info", &self.partial.as_ref().map(|p| &p.info))
            .field("state_len", &self.partial.as_ref().map(|p| p.state.len()))
            .finish()
    }
}

impl std::fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for EvolutionError {}

impl EvolutionError {
    fn new(kind: EvolutionFailure, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            partial: None,
        }
    }
    fn progress(mut self, state: &[C], info: &EvolutionInfo) -> Self {
        self.partial = Some(Box::new(EvolutionResult {
            state: state.to_vec(),
            info: info.clone(),
        }));
        self
    }
}

/// Apply `exp(-i H time)` to a complex vector using a matrix-free callback.
///
/// `apply_h(x, y)` must overwrite every entry of `y` with `H*x` and return an
/// error if unavailable. Buffers have the input vector's length; their storage
/// is reused. Negative time is supported. Zero time/vector skips the callback.
/// Tolerance is `atol + rtol * norm(input)`, distributed over accepted steps.
/// Work exhaustion and numerical failures return explicit errors with progress.
pub fn exponential_action<F>(
    input: &[C],
    time: f64,
    options: EvolutionOptions,
    mut apply_h: F,
) -> Result<EvolutionResult, EvolutionError>
where
    F: FnMut(&[C], &mut [C]) -> Result<(), String>,
{
    use EvolutionFailure::*;
    if input.is_empty() || !finite(input) || !time.is_finite() {
        return Err(EvolutionError::new(
            InvalidInput,
            "Evolution needs a nonempty finite vector and finite time",
        ));
    }
    if options.krylov_dim < 2
        || options.max_matvecs == 0
        || !options.atol.is_finite()
        || options.atol < 0.
        || !options.rtol.is_finite()
        || options.rtol < 0.
        || (options.atol == 0. && options.rtol == 0.)
    {
        return Err(EvolutionError::new(
            InvalidInput,
            "Invalid Krylov dimension, work limit, or tolerances",
        ));
    }
    let initial_norm = norm(input);
    let tolerance = options.atol + options.rtol * initial_norm;
    if !initial_norm.is_finite() || !tolerance.is_finite() {
        return Err(EvolutionError::new(
            InvalidInput,
            "Vector norm or tolerance exceeds finite range",
        ));
    }
    let mut info = EvolutionInfo {
        tolerance,
        ..Default::default()
    };
    let mut state = input.to_vec();
    if time == 0. || initial_norm == 0. {
        info.time_reached = time;
        return Ok(EvolutionResult { state, info });
    }
    if tolerance == 0. {
        return Err(EvolutionError::new(
            InvalidInput,
            "Requested tolerance underflows to zero",
        ));
    }
    let k = options.krylov_dim.min(input.len());
    let bytes = input
        .len()
        .checked_mul(k.saturating_add(3))
        .and_then(|n| n.checked_mul(size_of::<C>()));
    if bytes.is_none_or(|n| n > isize::MAX as usize) {
        return Err(EvolutionError::new(
            InvalidInput,
            "Krylov workspace exceeds addressable storage",
        ));
    }
    let mut basis = vec![vec![C::new(0., 0.); input.len()]];
    let mut work = vec![C::new(0., 0.); input.len()];
    let mut output = work.clone();
    let duration = time.abs();
    let direction = time.signum();
    while info.time_reached.abs() < duration {
        let remaining = duration - info.time_reached.abs();
        let budget = tolerance - info.estimated_error;
        let state_norm = norm(&state);
        for (q, x) in basis[0].iter_mut().zip(&state) {
            *q = *x / state_norm;
        }
        let mut projection = Projection::default();
        for j in 0..k {
            if info.matvecs == options.max_matvecs {
                return Err(EvolutionError::new(
                    WorkLimit,
                    "Krylov operator-application limit reached before the requested time",
                )
                .progress(&state, &info));
            }
            work.fill(C::new(f64::NAN, f64::NAN));
            info.matvecs += 1;
            apply_h(&basis[j], &mut work)
                .map_err(|e| EvolutionError::new(Operator, e).progress(&state, &info))?;
            if !finite(&work) {
                return Err(EvolutionError::new(
                    Operator,
                    "Operator must overwrite all outputs with finite values",
                )
                .progress(&state, &info));
            }
            projection
                .expand(&basis[..=j], &mut work)
                .map_err(|e| e.progress(&state, &info))?;
            info.max_krylov_dim = info.max_krylov_dim.max(j + 1);
            // Stop as soon as the remaining interval satisfies its error budget.
            // A tiny residual ends basis expansion, but still enters the error
            // estimate; it is never silently rounded down to exact breakdown.
            if projection.error(state_norm, remaining) <= budget
                || projection.residual <= f64::EPSILON * projection.operator_scale
                || j + 1 == k
            {
                break;
            }
            if basis.len() == j + 1 {
                basis.push(work.clone());
            }
            for (q, w) in basis[j + 1].iter_mut().zip(&work) {
                *q = *w / projection.residual;
            }
        }
        if projection.operator_scale == 0. {
            // H*x is exactly zero: preserve even unnormalized input bits.
            info.time_reached = time;
            info.steps += 1;
            return Ok(EvolutionResult { state, info });
        }
        let dt = projection.step(state_norm, remaining, budget);
        let advanced = info.time_reached.abs() + dt;
        if !dt.is_finite() || dt <= 0. || advanced <= info.time_reached.abs() {
            return Err(EvolutionError::new(
                PrecisionLimit,
                "Krylov error budget cannot support a representable time step",
            )
            .progress(&state, &info));
        }
        let error = projection.error(state_norm, dt);
        if !error.is_finite() || error > budget * (dt / remaining) {
            return Err(EvolutionError::new(
                PrecisionLimit,
                "Krylov error estimate exceeds the remaining tolerance rate",
            )
            .progress(&state, &info));
        }
        projection
            .evaluate(&basis, direction * dt, state_norm, &mut output)
            .map_err(|e| e.progress(&state, &info))?;
        std::mem::swap(&mut state, &mut output);
        info.estimated_error += error;
        info.steps += 1;
        info.time_reached = if dt == remaining {
            time
        } else {
            direction * advanced
        };
    }
    Ok(EvolutionResult { state, info })
}

#[derive(Default)]
struct Projection {
    diagonal: Vec<f64>,
    subdiagonal: Vec<f64>,
    residual: f64,
    log_product: f64,
    log_factorial: f64,
    defect: f64,
    operator_scale: f64,
}

impl Projection {
    fn expand(&mut self, basis: &[Vec<C>], work: &mut [C]) -> Result<(), EvolutionError> {
        let j = basis.len() - 1;
        let scale = norm(work);
        self.operator_scale = self.operator_scale.max(scale);
        let diagonal = dot(&basis[j], work);
        if !scale.is_finite() || !diagonal.re.is_finite() || !diagonal.im.is_finite() {
            return Err(EvolutionError::new(
                EvolutionFailure::Numerical,
                "Krylov inner product exceeds finite range",
            ));
        }
        let hermitian_tolerance = 1e-11 * scale;
        if diagonal.im.abs() > hermitian_tolerance {
            return Err(EvolutionError::new(
                EvolutionFailure::Operator,
                "Operator is not Hermitian on the explored Krylov subspace",
            ));
        }
        self.diagonal.push(diagonal.re);
        subtract(work, &basis[j], C::new(diagonal.re, 0.));
        if j > 0 {
            self.subdiagonal.push(self.residual);
            subtract(work, &basis[j - 1], C::new(self.residual, 0.));
        }
        // Twice-modified Gram–Schmidt restores orthogonality. Corrections to
        // the tridiagonal relation are retained in a Frobenius defect estimate.
        let mut correction = 0.;
        for _ in 0..2 {
            for q in basis {
                let overlap = dot(q, work);
                correction += overlap.norm();
                subtract(work, q, overlap);
            }
        }
        if !correction.is_finite() || correction > hermitian_tolerance {
            return Err(EvolutionError::new(
                EvolutionFailure::Operator,
                "Operator violates the Hermitian Lanczos relation",
            ));
        }
        self.defect = self.defect.hypot(correction);
        self.residual = norm(work);
        self.log_product += self.residual.ln();
        self.log_factorial += ((j + 1) as f64).ln();
        Ok(())
    }

    fn error(&self, state_norm: f64, dt: f64) -> f64 {
        // Theorem 1: ||x|| * product(beta_1..beta_m) * |dt|^m / m!.
        // Logarithms avoid overflow of factorials/products. Add the discarded
        // relation defect using unitarity of both full and projected evolution.
        let log_truncation =
            state_norm.ln() + self.log_product + self.diagonal.len() as f64 * dt.ln()
                - self.log_factorial;
        log_truncation.exp() + state_norm * self.defect * dt
    }

    fn step(&self, state_norm: f64, remaining: f64, budget: f64) -> f64 {
        if self.error(state_norm, remaining) <= budget {
            return remaining;
        }
        let rate = 0.8 * (budget / remaining) / state_norm - self.defect;
        if rate <= 0. || self.diagonal.len() < 2 {
            return 0.;
        }
        ((rate.ln() - self.log_product + self.log_factorial) / (self.diagonal.len() - 1) as f64)
            .exp()
            .min(remaining)
    }

    fn evaluate(
        &self,
        basis: &[Vec<C>],
        time: f64,
        state_norm: f64,
        output: &mut [C],
    ) -> Result<(), EvolutionError> {
        let m = self.diagonal.len();
        let matrix = faer::Mat::from_fn(m, m, |r, c| {
            if r == c {
                self.diagonal[r]
            } else if r == c + 1 {
                self.subdiagonal[c]
            } else if c == r + 1 {
                self.subdiagonal[r]
            } else {
                0.
            }
        });
        let eigen = matrix.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
            EvolutionError::new(
                EvolutionFailure::Numerical,
                format!("Projected eigendecomposition failed: {e:?}"),
            )
        })?;
        let mut coefficients = vec![C::new(0., 0.); m];
        for k in 0..m {
            let phase = -time * eigen.S()[k];
            if !phase.is_finite() {
                return Err(EvolutionError::new(
                    EvolutionFailure::Numerical,
                    "Projected eigenvalue times time exceeds finite range",
                ));
            }
            let weight = C::from_polar(1., phase) * eigen.U()[(0, k)];
            for (j, c) in coefficients.iter_mut().enumerate() {
                *c += eigen.U()[(j, k)] * weight;
            }
        }
        output.fill(C::new(0., 0.));
        for (q, weight) in basis.iter().zip(coefficients) {
            for (out, q) in output.iter_mut().zip(q) {
                *out += weight * q;
            }
        }
        for out in output.iter_mut() {
            *out *= state_norm;
        }
        if !finite(output) {
            return Err(EvolutionError::new(
                EvolutionFailure::Numerical,
                "Evolved vector exceeds finite range",
            ));
        }
        Ok(())
    }
}

fn finite(x: &[C]) -> bool {
    x.iter().all(|z| z.re.is_finite() && z.im.is_finite())
}
fn dot(x: &[C], y: &[C]) -> C {
    if x.len() <= 64 {
        x.iter().zip(y).map(|(x, y)| x.conj() * y).sum()
    } else {
        let m = x.len() / 2;
        dot(&x[..m], &y[..m]) + dot(&x[m..], &y[m..])
    }
}
fn subtract(x: &mut [C], y: &[C], a: C) {
    for (x, y) in x.iter_mut().zip(y) {
        *x -= a * y;
    }
}
fn norm(x: &[C]) -> f64 {
    let squared = norm_squared(x);
    if squared.is_finite() && squared > 0. {
        return squared.sqrt();
    }
    let scale = x
        .iter()
        .fold(0f64, |s, z| s.max(z.re.abs()).max(z.im.abs()));
    if scale == 0. {
        return 0.;
    }
    scale
        * x.iter()
            .map(|z| (*z / scale).norm_sqr())
            .sum::<f64>()
            .sqrt()
}

// Pairwise reductions avoid a dimension-dependent normalization drift that
// would otherwise dominate the reorthogonalization defect on large states.
fn norm_squared(x: &[C]) -> f64 {
    if x.len() <= 64 {
        x.iter().map(|z| z.norm_sqr()).sum()
    } else {
        let m = x.len() / 2;
        norm_squared(&x[..m]) + norm_squared(&x[m..])
    }
}

pub(crate) fn evolve_pauli(
    h: &crate::hamiltonian::PauliHamiltonian,
    input: &crate::ArrayReg,
    time: f64,
    options: EvolutionOptions,
) -> Result<EvolutionResult, EvolutionError> {
    let n = h.nqubits();
    let dim = 1usize.checked_shl(u32::try_from(n).unwrap_or(u32::MAX));
    if input.nqubits() != n || dim != Some(input.state.len()) {
        return Err(EvolutionError::new(
            EvolutionFailure::InvalidInput,
            "Hamiltonian and input register dimensions must match",
        ));
    }
    let terms: Vec<_> = h
        .polynomial()
        .iter()
        .map(|(coefficient, word)| {
            let (mut flip, mut phase, mut ys) = (0usize, 0usize, 0u32);
            for &(site, op) in word.ops() {
                let bit = 1 << (n - 1 - site);
                if matches!(op, crate::Op::X | crate::Op::Y) {
                    flip |= bit;
                }
                if matches!(op, crate::Op::Z | crate::Op::Y) {
                    phase |= bit;
                }
                if op == crate::Op::Y {
                    ys += 1;
                }
            }
            (flip, phase, *coefficient * C::new(0., 1.).powu(ys % 4))
        })
        .collect();
    exponential_action(&input.state, time, options, |x, y| {
        y.fill(C::new(0., 0.));
        for &(flip, phase, coefficient) in &terms {
            if coefficient.im == 0. {
                accumulate_pauli(x, y, flip, phase, |value| value * coefficient.re);
            } else {
                accumulate_pauli(x, y, flip, phase, |value| coefficient * value);
            }
        }
        Ok(())
    })
}

#[inline]
fn accumulate_pauli(x: &[C], y: &mut [C], flip: usize, phase: usize, scale: impl Fn(C) -> C) {
    // X-only terms have no phase parity to compute. Keep that decision, and
    // real-vs-complex coefficient dispatch, outside the amplitude loop.
    if phase == 0 {
        for (column, &value) in x.iter().enumerate() {
            y[column ^ flip] += scale(value);
        }
    } else {
        for (column, &value) in x.iter().enumerate() {
            let value = scale(value);
            y[column ^ flip] += if (column & phase).count_ones().is_multiple_of(2) {
                value
            } else {
                -value
            };
        }
    }
}

#[cfg(test)]
#[path = "unit_tests/evolution.rs"]
mod tests;
