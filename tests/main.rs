mod common;

#[path = "suites/ad.rs"]
mod ad;
#[path = "suites/apply.rs"]
mod apply;
#[path = "suites/benchmark_validation.rs"]
mod benchmark_validation;
#[path = "suites/einsum.rs"]
mod einsum;
#[path = "suites/einsum_dm.rs"]
mod einsum_dm;
#[path = "suites/gate.rs"]
mod gate;
#[path = "suites/integration.rs"]
mod integration;
#[path = "suites/measure.rs"]
mod measure;
#[cfg(feature = "torch")]
#[path = "suites/torch_contractor.rs"]
mod torch_contractor;
