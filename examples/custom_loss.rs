//! Fit an entangled complex target state using tenferro AD and argmin L-BFGS.
//! Run: cargo run --release --example custom_loss --features optimizer-example
use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};
use num_complex::Complex64 as C;
use std::{cell::RefCell, sync::Arc};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_tensor::Tensor;
use yao_rs::differentiable::DifferentiableCircuit;
use yao_rs::tenferro_ad::{circuit_apply_eager, eager_cpu_runtime};
use yao_rs::{ArrayReg, Circuit, Gate, control, put};

type Evaluation = (Vec<f64>, f64, Vec<f64>);
struct StateFit {
    circuit: Arc<DifferentiableCircuit>,
    ctx: Arc<EagerRuntime>,
    target: Vec<C>,
    // argmin may ask separately for value and gradient at the same parameters.
    cached: RefCell<Option<Evaluation>>,
}
impl StateFit {
    fn evaluate(&self, params: &[f64]) -> Result<(f64, Vec<f64>), Error> {
        if let Some((p, value, g)) = &*self.cached.borrow()
            && p == params
        {
            return Ok((*value, g.clone()));
        }
        let p = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![params.len()], params.to_vec())?,
            self.ctx.clone(),
        )?;
        let x = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![4], ArrayReg::zero_state(2).state)?,
            self.ctx.clone(),
        )?;
        let target = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![4], self.target.clone())?,
            self.ctx.clone(),
        )?;
        let y = circuit_apply_eager(self.circuit.clone(), &p, &x)?;
        let delta = y.sub(&target)?;
        let magnitude = delta.abs()?;
        let loss = magnitude.mul(&magnitude)?.reduce_sum(Some(&[0]))?;
        let _gradients = loss.backward()?;
        let value = loss.value()?.as_slice::<f64>()?[0];
        let gradient = p
            .grad()?
            .ok_or_else(|| Error::msg("missing parameter gradient"))?
            .as_slice::<f64>()?
            .to_vec();
        *self.cached.borrow_mut() = Some((params.to_vec(), value, gradient.clone()));
        Ok((value, gradient))
    }
}
impl CostFunction for StateFit {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<f64, Error> {
        Ok(self.evaluate(p)?.0)
    }
}
impl Gradient for StateFit {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Vec<f64>, Error> {
        Ok(self.evaluate(p)?.1)
    }
}
fn main() -> Result<(), Error> {
    let circuit = Arc::new(
        DifferentiableCircuit::from_circuit(Circuit::qubits(
            2,
            vec![
                put(vec![0], Gate::Ry(0.2)),
                control(vec![0], vec![1], Gate::X),
                put(vec![1], Gate::Rz(0.1)),
            ],
        )?)
        .map_err(Error::msg)?,
    );
    let target = circuit
        .forward(&[0.9, -0.4], &ArrayReg::zero_state(2))
        .map_err(Error::msg)?
        .state;
    let problem = StateFit {
        circuit,
        ctx: eager_cpu_runtime(1).map_err(Error::msg)?,
        target,
        cached: RefCell::new(None),
    };
    let initial = vec![0.2, 0.1];
    let before = problem.cost(&initial)?;
    let solver = LBFGS::new(MoreThuenteLineSearch::new(), 7);
    let result = Executor::new(problem, solver)
        .configure(|s| s.param(initial).max_iters(50))
        .run()?;
    let after = result.state().get_best_cost();
    println!("Squared state error: {before:.6e} -> {after:.6e}");
    println!("Fitted [Ry, Rz]: {:?}", result.state().get_best_param());
    if after > 1e-10 {
        return Err(Error::msg("optimizer did not reach the target state"));
    }
    Ok(())
}
