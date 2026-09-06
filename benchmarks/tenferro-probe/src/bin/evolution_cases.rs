//! Emit shared product-formula circuits plus numeric Hamiltonians for Yao's
//! independent dense exponential accuracy oracle. No timing occurs here.
use yao_rs::hamiltonian::{Boundary, ProductFormula, heisenberg, ising};
use yao_tenferro_probe::Result;

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: evolution_cases OUTPUT_JSON")?;
    let mut cases = Vec::new();
    let n = 3;
    for (name, h) in [
        ("ising", ising(n, -0.7, 0.4, Boundary::Open)?),
        (
            "heisenberg",
            heisenberg(n, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic)?,
        ),
    ] {
        for (order, formula) in [
            (1, ProductFormula::LieTrotter),
            (2, ProductFormula::Suzuki2),
        ] {
            for steps in [1, 2, 4, 8, 16] {
                let time = 0.8;
                let bound = h.evolve(time, steps, formula)?;
                let circuit: serde_json::Value =
                    serde_json::from_str(&yao_rs::circuit_to_json(bound.circuit()))?;
                // All accuracy comparisons use the same nontrivial initial
                // state. Zero-state TN timing is a separately named workload.
                cases.push(serde_json::json!({
                    "id":format!("evolution_{name}_order{order}_steps{steps}"),
                    "mode":"state", "tensor":false, "initial":"deterministic",
                    "circuit":circuit, "evolution":{
                        "model":name, "order":order, "steps":steps, "time":time,
                        "hamiltonian":h.polynomial(), "gates":bound.circuit().elements.len(),
                        "physical_parameters":bound.parameters(), "gate_parameters":bound.circuit().num_params(),
                    }
                }));
            }
        }
        let bound = h.evolve(0.8, 2, ProductFormula::Suzuki2)?;
        let circuit: serde_json::Value =
            serde_json::from_str(&yao_rs::circuit_to_json(bound.circuit()))?;
        cases.push(serde_json::json!({"id":format!("evolution_tensor_{name}"),
            "mode":"state", "tensor":true, "initial":"zero", "circuit":circuit}));
    }
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
