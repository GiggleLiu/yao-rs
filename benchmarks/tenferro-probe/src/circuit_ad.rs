//! Shared custom-loss workload; a specialized circuit primitive versus ordinary
//! tenferro tensor composition. The workload acts on the two lowest bits of a
//! full asymmetric complex state. This is not a universal circuit compiler.
use crate::Result;
use num_complex::Complex64 as C;
use std::sync::Arc;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_tensor::{DType, SliceConfig, Tensor};
use yao_rs::differentiable::DifferentiableCircuit;
use yao_rs::{ArrayReg, Circuit, CircuitElement, Gate, Op, control, put};

pub fn circuit(n: usize, depth: usize) -> Result<DifferentiableCircuit> {
    if n < 2 || depth == 0 {
        return Err("expected at least two qubits and one layer".into());
    }
    let mut gates = Vec::new();
    for k in 0..depth {
        gates.extend([
            put(vec![n - 1], Gate::Ry(0.1 + 0.001 * k as f64)),
            put(vec![n - 1], Gate::Rz(-0.2 + 0.002 * k as f64)),
            control(vec![n - 1], vec![n - 2], Gate::X),
            put(vec![n - 2], Gate::Rx(0.3 - 0.001 * k as f64)),
        ]);
    }
    Ok(DifferentiableCircuit::from_circuit(Circuit::qubits(
        n, gates,
    )?)?)
}
pub fn target(n: usize) -> Vec<C> {
    let mut v = (0..1usize << n)
        .map(|k| C::new((0.23 * k as f64).cos(), (0.17 * k as f64).sin()))
        .collect::<Vec<_>>();
    let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    for z in &mut v {
        *z /= norm;
    }
    v
}
pub fn native(c: &DifferentiableCircuit, x: &ArrayReg, t: &[C]) -> Result<Vec<C>> {
    let (value, g) = c.value_and_grad(c.template().parameters(), x, |y| {
        let delta = y
            .state
            .iter()
            .zip(t)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let loss = delta.iter().map(|x| x.norm_sqr()).sum();
        Ok((
            loss,
            ArrayReg::from_vec(x.nqubits(), delta.into_iter().map(|z| 2. * z).collect()),
        ))
    })?;
    Ok(std::iter::once(C::from(value))
        .chain(g.parameters.into_iter().map(C::from))
        .chain(g.input.state)
        .collect())
}
pub struct Evaluation {
    pub parameters: EagerTensor,
    pub input: EagerTensor,
    pub loss: EagerTensor,
}
pub fn prepare(
    c: Arc<DifferentiableCircuit>,
    x: &ArrayReg,
    t: &[C],
    ctx: Arc<EagerRuntime>,
    composed: bool,
) -> Result<Evaluation> {
    let p = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![c.num_parameters()], c.template().parameters().to_vec())?,
        ctx.clone(),
    )?;
    let input = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![x.state.len()], x.state.clone())?,
        ctx.clone(),
    )?;
    let y = if composed {
        compose(&c, &p, &input, ctx.clone())?
    } else {
        yao_rs::tenferro_ad::circuit_apply_eager(c, &p, &input)?
    };
    let target =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![t.len()], t.to_vec())?, ctx)?;
    let a = y.sub(&target)?.abs()?;
    Ok(Evaluation {
        parameters: p,
        input,
        loss: a.mul(&a)?.reduce_sum(Some(&[0]))?,
    })
}
pub fn finish(e: &Evaluation) -> Result<Vec<C>> {
    let _g = e.loss.backward()?;
    let loss = e.loss.value()?.as_slice::<f64>()?[0];
    let p = e.parameters.grad()?.ok_or("missing parameter gradient")?;
    let x = e.input.grad()?.ok_or("missing input gradient")?;
    Ok(std::iter::once(C::from(loss))
        .chain(p.as_slice::<f64>()?.iter().copied().map(C::from))
        .chain(x.as_slice::<C>()?.iter().copied())
        .collect())
}
fn compose(
    c: &DifferentiableCircuit,
    p: &EagerTensor,
    input: &EagerTensor,
    ctx: Arc<EagerRuntime>,
) -> Result<EagerTensor> {
    let constant = |data: Vec<C>| {
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4, 4], data)?, ctx.clone())
    };
    let id = constant((0..16).map(|i| C::from(f64::from(i % 5 == 0))).collect())?;
    let cx = constant(
        (0..16)
            .map(|i| {
                let col = i / 4;
                let row = i % 4;
                C::from(f64::from(row == if col & 1 == 1 { col ^ 2 } else { col }))
            })
            .collect(),
    )?;
    let mut y = input.reshape(&[4, c.state_len() / 4])?;
    let n = c.template().circuit().nbits;
    let mut slot = 0;
    for element in &c.template().circuit().elements {
        let CircuitElement::Gate(pg) = element else {
            return Err("composition fixture requires only gates".into());
        };
        let matrix = if !pg.control_locs.is_empty() {
            if pg.gate != Gate::X
                || pg.control_locs != [n - 1]
                || pg.target_locs != [n - 2]
                || pg.control_configs != [true]
            {
                return Err("unsupported composition fixture control".into());
            }
            cx.clone()
        } else {
            let op = match pg.gate {
                Gate::Rx(_) => Op::X,
                Gate::Ry(_) => Op::Y,
                Gate::Rz(_) => Op::Z,
                _ => return Err("unsupported composition gate".into()),
            };
            let bit = n - 1 - pg.target_locs[0];
            if bit > 1 {
                return Err("composition fixture supports last two qubits".into());
            }
            let pauli = yao_rs::operator::op_matrix(&op);
            let generator = constant(
                (0..16)
                    .map(|i| {
                        let row = i % 4;
                        let col = i / 4;
                        if ((row >> (1 - bit)) & 1) == ((col >> (1 - bit)) & 1) {
                            C::new(0., -1.) * pauli[[(row >> bit) & 1, (col >> bit) & 1]]
                        } else {
                            C::new(0., 0.)
                        }
                    })
                    .collect(),
            )?;
            let theta = p
                .slice(SliceConfig {
                    starts: vec![slot],
                    limits: vec![slot + 1],
                    strides: vec![1],
                })?
                .reshape(&[])?
                .scale_real(0.5)?;
            slot += 1;
            id.mul(&theta.cos()?.cast(DType::C64)?)?
                .add(&generator.mul(&theta.sin()?.cast(DType::C64)?)?)?
        };
        y = matrix.matmul(&y)?;
    }
    Ok(y.reshape(&[c.state_len()])?)
}

#[test]
fn custom_and_composed_losses_and_gradients_match_native() -> Result<()> {
    for n in [2, 4] {
        let c = Arc::new(circuit(n, 3)?);
        let mut x = ArrayReg::from_vec(
            n,
            (0..1usize << n)
                .map(|i| C::new((0.1 * i as f64).cos(), (0.2 * i as f64).sin()))
                .collect(),
        );
        let norm = x.state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        for z in &mut x.state {
            *z /= norm;
        }
        let t = target(n);
        let expected = native(&c, &x, &t)?;
        for composed in [false, true] {
            let ctx = yao_rs::tenferro_ad::eager_cpu_runtime(1)?;
            let got = finish(&prepare(c.clone(), &x, &t, ctx, composed)?)?;
            assert!(
                got.iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).norm() < 1e-10)
            );
        }
    }
    Ok(())
}
