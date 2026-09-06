use super::*;
use crate::{Circuit, Gate, put};
use ::tenferro_ad::EagerTensor;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}
fn model() -> Arc<DifferentiableCircuit> {
    Arc::new(
        DifferentiableCircuit::from_circuit(
            Circuit::qubits(
                2,
                vec![put(vec![0], Gate::Ry(0.3)), put(vec![1], Gate::Rx(-0.2))],
            )
            .unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn eager_custom_real_loss_matches_native_vjp_and_updates_inputs() -> Result {
    let c = model();
    let ctx = eager_cpu_runtime(1)?;
    let x = ArrayReg::from_vec(
        2,
        vec![
            C::new(0.3, 0.4),
            C::new(-0.2, 0.1),
            C::new(0.7, -0.5),
            C::new(0.2, 0.8),
        ],
    );
    let target = vec![C::new(0.1, 0.2); 4];
    for params in [vec![0.3, -0.2], vec![-0.7, 0.4]] {
        let p = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![2], params.clone())?,
            ctx.clone(),
        )?;
        let input = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![4], x.state.clone())?,
            ctx.clone(),
        )?;
        let y = circuit_apply_eager(c.clone(), &p, &input)?;
        let goal = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![4], target.clone())?,
            ctx.clone(),
        )?;
        let delta = y.sub(&goal)?;
        let a = delta.abs()?;
        let loss = a.mul(&a)?.reduce_sum(Some(&[0]))?;
        let _gradients = loss.backward()?;
        let expected = c.forward(&params, &x)?;
        let seed = ArrayReg::from_vec(
            2,
            expected
                .state
                .iter()
                .zip(&target)
                .map(|(y, t)| 2. * (y - t))
                .collect(),
        );
        let g = c.vjp(&params, &x, &seed)?;
        close(
            loss.value()?.as_slice::<f64>()?[0],
            expected
                .state
                .iter()
                .zip(&target)
                .map(|(y, t)| (y - t).norm_sqr())
                .sum(),
        );
        for (a, b) in p
            .grad()?
            .unwrap()
            .as_slice::<f64>()?
            .iter()
            .zip(g.parameters)
        {
            close(*a, b);
        }
        for (a, b) in input
            .grad()?
            .unwrap()
            .as_slice::<C>()?
            .iter()
            .zip(g.input.state)
        {
            close((*a - b).norm(), 0.);
        }
    }
    Ok(())
}

#[test]
fn traced_vjp_accepts_nonreal_seed_and_rejects_higher_order() -> Result {
    use tenferro_runtime::GraphCompiler;
    let c = model();
    let p = TracedTensor::input_concrete_shape(DType::F64, &[2])?;
    let x = TracedTensor::input_concrete_shape(DType::C64, &[4])?;
    let y = circuit_apply(c.clone(), &p, &x)?;
    let seed = TracedTensor::from_vec_col_major(vec![4], vec![C::new(0.2, 0.7); 4])?;
    let ad = ad_context()?;
    let g = ad.vjp(&y, &p, &seed)?;
    assert!(
        ad.vjp(
            &g,
            &p,
            &TracedTensor::from_vec_col_major(vec![2], vec![1_f64; 2])?
        )
        .is_err()
    );
    let program = GraphCompiler::new()
        .compile_with_input_specs(&g, &[(&p, DType::F64, &[2]), (&x, DType::C64, &[4])])?;
    let params = Tensor::from_vec_col_major(vec![2], vec![0.3, -0.2])?;
    let input = Tensor::from_vec_col_major(vec![4], vec![C::new(0.5, -0.1); 4])?;
    let out = cpu_runtime(1)?.run_compiled(&program, &[&params, &input])?;
    let native = c.vjp(
        &[0.3, -0.2],
        &ArrayReg::from_vec(2, vec![C::new(0.5, -0.1); 4]),
        &ArrayReg::from_vec(2, vec![C::new(0.2, 0.7); 4]),
    )?;
    for (a, b) in out[0].as_slice::<f64>()?.iter().zip(native.parameters) {
        close(*a, b);
    }
    Ok(())
}

#[test]
fn fixed_circuit_has_empty_parameter_gradient() -> Result {
    let c = Arc::new(DifferentiableCircuit::from_circuit(
        Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap(),
    )?);
    let ctx = eager_cpu_runtime(1)?;
    let p = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![0], Vec::<f64>::new())?,
        ctx.clone(),
    )?;
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![C::new(1., 0.), C::new(0., 0.)])?,
        ctx,
    )?;
    let y = circuit_apply_eager(c, &p, &x)?;
    let a = y.abs()?;
    let _g = a.mul(&a)?.reduce_sum(Some(&[0]))?.backward()?;
    assert!(p.grad()?.unwrap().as_slice::<f64>()?.is_empty());
    close(x.grad()?.unwrap().as_slice::<C>()?[0].re, 2.);
    Ok(())
}

#[test]
fn semantic_jvp_masks_match_native_and_reject_higher_order() -> Result {
    use tenferro_runtime::GraphCompiler;
    let c = model();
    let p = TracedTensor::input_concrete_shape(DType::F64, &[2])?;
    let x = TracedTensor::input_concrete_shape(DType::C64, &[4])?;
    let y = circuit_apply(c.clone(), &p, &x)?;
    let source = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&p, DType::F64, &[2]), (&x, DType::C64, &[4])])?;
    let params = Tensor::from_vec_col_major(vec![2], vec![0.3, -0.2])?;
    let input = Tensor::from_vec_col_major(vec![4], vec![C::new(0.2, 0.7); 4])?;
    let dp = Tensor::from_vec_col_major(vec![2], vec![0.6, -0.1])?;
    let dx = Tensor::from_vec_col_major(vec![4], vec![C::new(-0.3, 0.1); 4])?;
    let ad = ad_context()?;
    let runtime = cpu_runtime(1)?;
    for mask in [[true, false], [false, true], [true, true]] {
        let transformed = ad.jvp_program(source.frozen_program(), &mask)?;
        assert!(
            ad.jvp_program(
                transformed.frozen(),
                &vec![true; transformed.frozen().program.inputs().len()]
            )
            .is_err()
        );
        let program = GraphCompiler::new().compile_frozen_program(transformed.frozen())?;
        let mut inputs = vec![&params, &input];
        if mask[0] {
            inputs.push(&dp);
        }
        if mask[1] {
            inputs.push(&dx);
        }
        let got = runtime.run_compiled(&program, &inputs)?;
        let pt = if mask[0] {
            vec![0.6, -0.1]
        } else {
            vec![0.; 2]
        };
        let xt = if mask[1] {
            vec![C::new(-0.3, 0.1); 4]
        } else {
            vec![C::new(0., 0.); 4]
        };
        let (_, expected) = c.jvp(
            &[0.3, -0.2],
            &ArrayReg::from_vec(2, vec![C::new(0.2, 0.7); 4]),
            &pt,
            &ArrayReg::from_vec(2, xt),
        )?;
        for (a, b) in got[0].as_slice::<C>()?.iter().zip(expected.state) {
            close((*a - b).norm(), 0.);
        }
    }
    Ok(())
}

#[test]
fn eager_rejects_wrong_shape_dtype_and_context() -> Result {
    let c = model();
    let ctx = eager_cpu_runtime(1)?;
    let p = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.3, -0.2])?,
        ctx.clone(),
    )?;
    for data in [
        Tensor::from_vec_col_major(vec![3], vec![C::new(1., 0.); 3])?,
        Tensor::from_vec_col_major(vec![4], vec![1_f64; 4])?,
    ] {
        let x = EagerTensor::from_tensor_in(data, ctx.clone())?;
        assert!(circuit_apply_eager(c.clone(), &p, &x).is_err());
    }
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], vec![C::new(1., 0.); 4])?,
        eager_cpu_runtime(1)?,
    )?;
    assert!(circuit_apply_eager(c, &p, &x).is_err());
    assert!(cpu_runtime(0).is_err());
    assert!(eager_cpu_runtime(0).is_err());
    Ok(())
}

#[test]
fn eager_shared_physical_gradients_at_zero_time() -> Result {
    use crate::hamiltonian::{Boundary, ProductFormula, ising};
    let c = Arc::new(DifferentiableCircuit::new(
        ising(3, -0.7, 0.4, Boundary::Open)?.evolve(0., 3, ProductFormula::Suzuki2)?,
    )?);
    let ctx = eager_cpu_runtime(1)?;
    let x = ArrayReg::from_vec(3, vec![C::new(0.3, 0.2); 8]);
    let target = vec![C::new(0.1, -0.7); 8];
    let p = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![0., -0.7, 0.4])?,
        ctx.clone(),
    )?;
    let input = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![8], x.state.clone())?,
        ctx.clone(),
    )?;
    let t = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![8], target.clone())?, ctx)?;
    let y = circuit_apply_eager(c.clone(), &p, &input)?;
    let a = y.sub(&t)?.abs()?;
    let _g = a.mul(&a)?.reduce_sum(Some(&[0]))?.backward()?;
    let (_, expected) = c.value_and_grad(&[0., -0.7, 0.4], &x, |y| {
        let d = y
            .state
            .iter()
            .zip(&target)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        Ok((
            d.iter().map(|x| x.norm_sqr()).sum(),
            ArrayReg::from_vec(3, d.into_iter().map(|z| 2. * z).collect()),
        ))
    })?;
    let got = p.grad()?.unwrap();
    for (a, b) in got.as_slice::<f64>()?.iter().zip(expected.parameters) {
        close(*a, b);
    }
    assert!(got.as_slice::<f64>()?[0].abs() > 0.1);
    close(got.as_slice::<f64>()?[1], 0.);
    close(got.as_slice::<f64>()?[2], 0.);
    Ok(())
}
