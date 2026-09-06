//! Emit Hamiltonians, tolerances, and small equivalent product-formula cases.
use yao_rs::{Circuit, circuit_to_json, hamiltonian::ProductFormula};
use yao_tenferro_probe::{Result, krylov};
fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: krylov_cases OUTPUT_JSON")?;
    let mut cases = Vec::new();
    for name in ["ising", "heisenberg"] {
        for n in [4, 8, 12, 16] {
            let h = krylov::model(name, n)?;
            let time = if n == 4 { 0.8 } else { 2.1 };
            let circuit: serde_json::Value =
                serde_json::from_str(&circuit_to_json(&Circuit::qubits(n, vec![])?))?;
            for power in [4, 7, 10] {
                let spec = krylov::Specification {
                    model: name.into(),
                    hamiltonian: h.polynomial(),
                    time,
                    rtol: 10f64.powi(-power),
                    krylov_dim: 20,
                };
                cases.push(serde_json::json!({"id":format!("krylov_{name}_{n}q_tol{power}"),
                    "mode":"krylov", "tensor":false,"initial":"zero","circuit":circuit,"krylov":spec}));
            }
            if n == 4 {
                let spec = krylov::Specification {
                    model: name.into(),
                    hamiltonian: h.polynomial(),
                    time,
                    rtol: 1e-10,
                    krylov_dim: 20,
                };
                cases.push(serde_json::json!({"id":format!("krylov_{name}_{n}q_asymmetric"),
                    "mode":"krylov", "tensor":false,"initial":"deterministic","circuit":circuit,"krylov":spec}));
                for steps in [2, 8, 32] {
                    let bound = h.evolve(time, steps, ProductFormula::Suzuki2)?;
                    let circuit: serde_json::Value =
                        serde_json::from_str(&circuit_to_json(bound.circuit()))?;
                    cases.push(serde_json::json!({"id":format!("krylov_product_{name}_{n}q_steps{steps}"),
                        "mode":"state","tensor":true,"initial":"zero","circuit":circuit,
                        "evolution":{"model":name,"order":2,"steps":steps,"time":time,"hamiltonian":h.polynomial()}}));
                }
            }
        }
    }
    if std::env::args().nth(2).as_deref() == Some("--smoke") {
        cases.retain(|c| c["id"].as_str().is_some_and(|id| id.ends_with("_4q_tol7")));
    }
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
