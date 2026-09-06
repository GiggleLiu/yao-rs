//! Diagnostic Rust-heap accounting. Run each process in isolation for peak RSS.
use num_complex::Complex64 as C;
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::Tensor;
use yao_rs::{Circuit, Gate, put};
use yao_tenferro_probe::{Result, convert, execute, prepare, subscripts};
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);
struct Heap;
// SAFETY: all memory operations and their unchanged layouts are delegated to System.
unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let live = LIVE.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
            TOTAL.fetch_add(l.size(), Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::SeqCst);
        unsafe { System.dealloc(p, l) };
    }
}
#[global_allocator]
static HEAP: Heap = Heap;
fn phase<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let live = LIVE.load(Ordering::SeqCst);
    let total = TOTAL.load(Ordering::SeqCst);
    PEAK.store(live, Ordering::SeqCst);
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed().as_nanos();
    let peak = PEAK.load(Ordering::SeqCst).saturating_sub(live);
    let retained = LIVE.load(Ordering::SeqCst).saturating_sub(live);
    let allocated = TOTAL.load(Ordering::SeqCst) - total;
    println!(
        "{}",
        json!({"phase":name,"allocated_bytes":allocated,"peak_additional_rust_heap_bytes":peak,"retained_additional_rust_heap_bytes":retained,"diagnostic_ns":elapsed})
    );
    out
}
fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("trajectories") {
        return trajectory_memory();
    }
    if std::env::args().nth(1).as_deref() == Some("tensor-memory") {
        return tensor_memory();
    }
    if std::env::args().nth(1).as_deref() == Some("circuit-ad") {
        return circuit_ad_memory();
    }
    if std::env::args().nth(1).as_deref() == Some("evolution") {
        return evolution_memory();
    }
    let n: usize = std::env::args().nth(1).unwrap_or("12".into()).parse()?;
    let depth: usize = std::env::args().nth(2).unwrap_or("10".into()).parse()?;
    if !(1..=20).contains(&n) || depth == 0 || depth > 1000 {
        return Err("expected 1..20 qubits and 1..1000 layers".into());
    }
    let nonlinear = std::env::args().nth(3).as_deref() == Some("nonlinear");
    let circuit = Circuit::qubits(n, (0..n).map(|q| put(vec![q], Gate::Ry(0.3))).collect())?;
    let network = phase("export", || {
        yao_rs::einsum::circuit_to_einsum_with_boundary(&circuit, &[])
    });
    let tensors = phase("conversion", || convert(&network.tensors))?;
    let code = subscripts(&network.code)?;
    let plan = phase("planning", || prepare(&tensors, &code))?;
    let mut backend = CpuBackend::with_threads(1)?;
    let _out = phase("first_execution", || execute(&plan, &tensors, &mut backend))?;
    let _out = phase("warm_execution", || execute(&plan, &tensors, &mut backend))?;
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1)?)?;
    let x = phase("ad_input", || {
        EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![1 << n], vec![C::new(0.3, 0.4); 1 << n])?,
            ctx.clone(),
        )
    })?;
    let scale = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![C::new(0.3, 0.0)])?,
        ctx.clone(),
    )?;
    let loss = phase("ad_forward_tape", || -> Result<_> {
        let mut y = x.clone();
        for _ in 0..depth {
            y = if nonlinear {
                y.sin()?.mul(&scale)?
            } else {
                y.conj()?
            };
        }
        let m = y.abs()?;
        Ok(m.mul(&m)?.reduce_sum(Some(&[0]))?)
    })?;
    let _gradient = phase("ad_backward", || ctx.grad(&loss, &x))?;
    println!(
        "{}",
        json!({"qubits":n,"depth":depth,"operation":if nonlinear { "scaled_sin" } else { "conj" },"state_bytes":(1<<n)*16,"note":"Rust allocator instrumentation excludes provider allocations; timings include instrumentation. Not RSS."})
    );
    Ok(())
}

fn circuit_ad_memory() -> Result<()> {
    use yao_tenferro_probe::circuit_ad as ad;
    let backend = std::env::args()
        .nth(2)
        .ok_or("expected native/custom/composed")?;
    if !["native", "custom", "composed"].contains(&backend.as_str()) {
        return Err("unknown AD backend".into());
    }
    let n: usize = std::env::args().nth(3).ok_or("expected qubits")?.parse()?;
    let depth: usize = std::env::args().nth(4).ok_or("expected depth")?.parse()?;
    if !(2..=16).contains(&n) || !(1..=100).contains(&depth) {
        return Err("expected 2..16 qubits and 1..100 layers".into());
    }
    let c = std::sync::Arc::new(ad::circuit(n, depth)?);
    let input = yao_rs::ArrayReg::zero_state(n);
    let target = ad::target(n);
    println!(
        "{}",
        json!({"backend":backend,"qubits":n,"depth":depth,"parameters":c.num_parameters(),"state_bytes":(1usize<<n)*16})
    );
    let result = if backend == "native" {
        phase("native_value_and_grad", || ad::native(&c, &input, &target))?
    } else {
        let ctx = yao_rs::tenferro_ad::eager_cpu_runtime(1)?;
        let evaluation = phase("circuit_ad_forward_tape", || {
            ad::prepare(c, &input, &target, ctx, backend == "composed")
        })?;
        phase("circuit_ad_backward", || ad::finish(&evaluation))?
    };
    if result
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err("nonfinite AD result".into());
    }
    println!(
        "{}",
        json!({"status":"complete","output_values":result.len(),"loss":result[0].re,"note":"Rust heap instrumentation; excludes native provider allocations. RSS includes startup, forward and backward. Diagnostic times are not benchmark timings."})
    );
    Ok(())
}

fn evolution_memory() -> Result<()> {
    use yao_rs::hamiltonian::{Boundary, ProductFormula, heisenberg, ising};
    let model = std::env::args().nth(2).ok_or("expected model")?;
    let n: usize = std::env::args().nth(3).ok_or("expected qubits")?.parse()?;
    let steps: usize = std::env::args().nth(4).ok_or("expected steps")?.parse()?;
    if !(3..=20).contains(&n) || !(1..=1000).contains(&steps) {
        return Err("expected 3..20 qubits and 1..1000 steps".into());
    }
    let h = phase("model", || match model.as_str() {
        "ising" => ising(n, -0.7, 0.4, Boundary::Open),
        "heisenberg" => heisenberg(n, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic),
        _ => Err("expected ising or heisenberg".into()),
    })?;
    let bound = phase("circuit_construction", || {
        h.evolve(0.8, steps, ProductFormula::Suzuki2)
    })?;
    let input = phase("state_input", || yao_rs::ArrayReg::zero_state(n));
    let out = phase("native_execution", || bound.apply(&input))?;
    let norm: f64 = out.state.iter().map(|x| x.norm_sqr()).sum();
    if (norm - 1.).abs() > 1e-9 {
        return Err("evolution state norm mismatch".into());
    }
    println!(
        "{}",
        json!({"model":model,"qubits":n,"steps":steps,"order":2,
        "gates":bound.circuit().elements.len(), "gate_parameters":bound.circuit().num_params(),
        "physical_parameters":bound.parameters().len(), "state_bytes":(1usize<<n)*16,"norm_squared":norm,
        "note":"Isolated model construction and native execution; Rust heap excludes provider allocations. Timings include instrumentation. RSS includes startup and allocator retention."})
    );
    Ok(())
}

fn tensor_memory() -> Result<()> {
    use yao_tenferro_probe::tensor_memory as tm;
    let backend = std::env::args().nth(2).ok_or("expected backend")?;
    let n: usize = std::env::args()
        .nth(3)
        .ok_or("expected matrix dimension")?
        .parse()?;
    let mode = std::env::args().nth(4).ok_or("expected slicing mode")?;
    let kind = std::env::args().nth(5).unwrap_or("chain".into());
    if !tm::matrix_workloads()
        .iter()
        .any(|(k, d, modes)| *k == kind && *d == n && modes.contains(&mode.as_str()))
    {
        return Err("unknown or out-of-bounds matrix memory fixture".into());
    }
    let tn = phase("matrix_inputs", || {
        if kind == "outer" {
            tm::outer_network(n)
        } else {
            tm::matrix_network(n)
        }
    });
    let plan = phase("slice_planning", || tm::matrix_plan(&tn, &mode))?;
    println!(
        "{}",
        json!({"dimension":n,"kind":kind,"mode":mode,"backend":backend,"labels":plan.slicing(),"estimate":plan.estimate(),"tree":omeco::json::NestedEinsumTree::from(plan.tree())})
    );
    let result = match backend.as_str() {
        "tenferro" => {
            let cpu = yao_rs::tenferro::CpuContractor::new(1)?;
            let prepared = phase("slice_compilation", || cpu.prepare_sliced(&plan))?;
            phase("slice_execution", || {
                cpu.execute_sliced(&prepared, &tn.tensors)
            })?
        }
        "omeinsum" => phase("slice_execution", || {
            yao_rs::contractor::contract_sliced(&plan, &tn.tensors)
        })?,
        _ => return Err("unknown matrix backend".into()),
    };
    if result
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err("nonfinite matrix contraction result".into());
    }
    println!(
        "{}",
        json!({"status":"complete","checksum":result.iter().map(|z|z.norm_sqr()).sum::<f64>()})
    );
    Ok(())
}

fn trajectory_memory() -> Result<()> {
    use yao_rs::{
        ArrayReg, DensityMatrix, Register,
        trajectories::{TrajectoryCircuit, TrajectoryOptions},
    };
    use yao_tenferro_probe::trajectories as tr;
    let backend = std::env::args()
        .nth(2)
        .ok_or("expected trajectory/density")?;
    let n: usize = std::env::args().nth(3).ok_or("expected qubits")?.parse()?;
    let count: usize = std::env::args()
        .nth(4)
        .ok_or("expected sample count")?
        .parse()?;
    let threads: usize = std::env::args().nth(5).ok_or("expected threads")?.parse()?;
    if ![4, 8, 10, 12, 16].contains(&n)
        || ![1, 4].contains(&threads)
        || (backend == "density" && (n > 10 || count != 0 || threads != 1))
        || (backend == "trajectory" && ![64, 256].contains(&count))
        || !["trajectory", "density"].contains(&backend.as_str())
    {
        return Err("out-of-bounds trajectory memory fixture".into());
    }
    let entangled = n <= 8;
    let circuit = phase("circuit", || tr::circuit(n, entangled))?;
    let op = tr::observable(n, entangled);
    let input = phase("input", || ArrayReg::zero_state(n));
    let mean = if backend == "trajectory" {
        let simulator = phase("trajectory_prepare", || TrajectoryCircuit::new(circuit))?;
        let stats = phase("trajectory_execute", || {
            simulator.expectation(
                &input,
                &op,
                TrajectoryOptions {
                    trajectories: count,
                    seed: 19,
                    threads,
                },
            )
        })?;
        println!("{}", json!({"statistics":stats}));
        stats.mean
    } else {
        let mut density = phase("density_input", || DensityMatrix::from_reg(&input));
        phase("density_execute", || {
            density.apply(&circuit);
            yao_rs::expect_dm(&density, &op)
        })
    };
    if !mean.re.is_finite() || !mean.im.is_finite() {
        return Err("nonfinite memory result".into());
    }
    println!(
        "{}",
        json!({"status":"complete","backend":backend,"qubits":n,"trajectories":count,"threads":threads,"state_bytes":(1usize<<n)*16,"mean":mean,
        "note":"Rust heap instrumented; diagnostic times are not benchmark times. RSS includes process startup and allocator retention."})
    );
    Ok(())
}
