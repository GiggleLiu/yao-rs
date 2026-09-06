//! One process per case: cold/setup stages and allocator-inclusive GPU snapshots.
use serde_json::json;
use std::{hint::black_box, time::Instant};
use yao_rs::cuda::CudaSimulator;
use yao_tenferro_probe::{
    Result, cases,
    cuda::{PreparedCase, check, reference},
};

fn snapshot(phase: &str) -> Result<()> {
    let memory = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !memory.status.success() {
        return Err("nvidia-smi memory query failed".into());
    }
    let pid = std::process::id().to_string();
    let mib = String::from_utf8(memory.stdout)?
        .lines()
        .find_map(|line| {
            let (p, m) = line.split_once(',')?;
            (p.trim() == pid)
                .then(|| m.trim().parse::<u64>().ok())
                .flatten()
        })
        .ok_or("GPU process memory not reported")?;
    println!(
        "{}",
        json!({"phase":phase,"device_process_bytes":mib * 1024 * 1024})
    );
    Ok(())
}
fn main() -> Result<()> {
    let id = std::env::args().nth(1).ok_or("usage: cuda_probe CASE_ID")?;
    let case = cases()?
        .into_iter()
        .find(|c| c.id == id)
        .ok_or("unknown case")?;
    let start = Instant::now();
    let gpu = CudaSimulator::new(0)?;
    gpu.synchronize()?;
    let context_ns = start.elapsed().as_nanos();
    snapshot("context")?;
    let start = Instant::now();
    let prepared = PreparedCase::new(&case, &gpu)?;
    gpu.synchronize()?;
    let preparation_upload_ns = start.elapsed().as_nanos();
    snapshot("prepared")?;
    let start = Instant::now();
    let output = prepared.resident(&gpu)?;
    gpu.synchronize()?;
    let first_resident_ns = start.elapsed().as_nanos();
    snapshot("first_result")?;
    let expected = reference(&case)?;
    let error = check(&prepared.download(&gpu, &output)?, &expected)?;
    drop(output);
    let mut samples = Vec::new();
    for _ in 0..if case.cuda_diagnostic_only { 1 } else { 3 } {
        gpu.synchronize()?;
        let start = Instant::now();
        let result = prepared.resident(&gpu)?;
        gpu.synchronize()?;
        samples.push(start.elapsed().as_nanos());
        black_box(result);
    }
    snapshot("after_repeats")?;
    let transfer_error = check(&prepared.end_to_end(&gpu)?, &expected)?;
    println!(
        "{}",
        json!({"status":"complete","id":case.id,"device":gpu.device_name(),"precision":"complex128",
        "context_ns":context_ns,"preparation_upload_ns":preparation_upload_ns,"first_resident_ns":first_resident_ns,
        "resident_samples_ns":samples,"max_error":error,"transfer_max_error":transfer_error,"input_bytes":prepared.input_bytes()})
    );
    Ok(())
}
