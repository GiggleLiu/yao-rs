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
