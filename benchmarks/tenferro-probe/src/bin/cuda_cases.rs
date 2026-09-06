//! GPU fixtures shared with native Rust and the existing pinned Yao runner.
use serde_json::json;
use yao_rs::{Circuit, Gate, NoiseChannel, channel, control, put};
use yao_tenferro_probe::{Result, circuit_ad};

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: cuda_cases OUTPUT [--smoke]")?;
    let smoke = std::env::args().any(|x| x == "--smoke");
    let mut cases = Vec::new();
    let mut add = |id, mode, initial, circuit: &Circuit, tensor| -> Result<()> {
        cases.push(json!({"id":id,"mode":mode,"initial":initial,"tensor":tensor,
            "circuit":serde_json::from_str::<serde_json::Value>(&yao_rs::circuit_to_json(circuit))?}));
        Ok(())
    };
    for n in if smoke { vec![4] } else { vec![8, 16, 20, 24] } {
        let depth = if smoke { 2 } else { 10 };
        let mut gates = Vec::new();
        for k in 0..depth {
            let site = k % n;
            gates.extend([
                put(vec![site], Gate::Ry(0.2 + k as f64 * 0.01)),
                put(vec![(site + 1) % n], Gate::Rz(-0.4)),
                control(vec![site], vec![(site + n / 2) % n], Gate::X),
                put(vec![(site + n - 1) % n, site], Gate::FSim(0.17, -0.31)),
            ]);
        }
        add(
            format!("cuda_state_{n}_depth{depth}"),
            "state",
            "deterministic",
            &Circuit::qubits(n, gates)?,
            false,
        )?;
    }
    for n in if smoke { vec![4] } else { vec![8, 16, 20] } {
        for depth in if smoke { vec![2] } else { vec![10, 40] } {
            let c = circuit_ad::circuit(n, depth)?;
            add(
                format!("cuda_gradient_{n}_depth{depth}"),
                "custom_gradient",
                "deterministic",
                c.template().circuit(),
                false,
            )?;
        }
    }
    for n in if smoke { vec![2] } else { vec![4, 6, 8] } {
        let mut gates = Vec::new();
        for site in 0..n {
            gates.push(put(vec![site], Gate::Ry(0.3 + site as f64 * 0.02)));
            if site > 0 {
                gates.push(control(vec![site - 1], vec![site], Gate::X));
            }
            gates.push(channel(
                vec![site],
                NoiseChannel::AmplitudeDamping {
                    gamma: 0.13,
                    excited_population: 0.2,
                },
            ));
            gates.push(channel(
                vec![site],
                NoiseChannel::Depolarizing { n: 1, p: 0.07 },
            ));
        }
        add(
            format!("cuda_density_{n}"),
            "density",
            "zero",
            &Circuit::qubits(n, gates)?,
            true,
        )?;
    }
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
