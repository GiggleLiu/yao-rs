use omeco::{contraction_complexity, optimize_code, GreedyMethod};
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate};
use yao_rs::einsum::circuit_to_einsum;
use yao_rs::gate::Gate;

// Helper function to wrap a PositionedGate in CircuitElement::Gate
fn gate(g: PositionedGate) -> CircuitElement {
    CircuitElement::Gate(g)
}

#[test]
fn test_optimize_bell_circuit() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    let complexity = contraction_complexity(&nested, &tn.size_dict, &tn.code.ixs);
    assert!(complexity.tc > 0.0);
}

#[test]
fn test_optimize_ghz_circuit() {
    // 5-qubit GHZ preparation: H on 0, then CNOT chain
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    assert!(nested.is_binary());
}

#[test]
fn test_optimize_larger_circuit() {
    // 10-qubit circuit with H layer, CNOT chain, Rz layer
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
}

#[test]
fn test_single_gate_circuit_produces_valid_optimization() {
    // A single H gate on 1 qubit: produces 1 tensor, optimizer should return a leaf
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    // Single tensor produces a leaf in the contraction tree
    assert!(nested.is_leaf());
}

#[test]
fn test_single_cnot_gate_produces_valid_optimization() {
    // A single CNOT gate on 2 qubits: produces 1 tensor, optimizer should return a leaf
    let circuit = Circuit::new(
        vec![2, 2],
        vec![gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());

    let nested = optimized.unwrap();
    assert!(nested.is_leaf());
}

#[test]
fn test_eincode_is_always_valid_single_gate() {
    // Single H gate
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_bell_circuit() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_ghz_circuit() {
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_eincode_is_always_valid_larger_circuit() {
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());
}

#[test]
fn test_optimize_complexity_ghz_circuit() {
    // Verify contraction complexity is computable for the GHZ circuit
    let mut gates = vec![gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))];
    for i in 0..4 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }

    let circuit = Circuit::new(vec![2; 5], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();
    let complexity = contraction_complexity(&optimized, &tn.size_dict, &tn.code.ixs);

    assert!(complexity.tc > 0.0);
    assert!(complexity.sc > 0.0);
}

#[test]
fn test_optimize_complexity_larger_circuit() {
    // Verify contraction complexity for the 10-qubit circuit
    let mut gates = Vec::new();
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(Gate::H, vec![i], vec![], vec![])));
    }
    for i in 0..9 {
        gates.push(gate(PositionedGate::new(
            Gate::X,
            vec![i + 1],
            vec![i],
            vec![true],
        )));
    }
    for i in 0..10 {
        gates.push(gate(PositionedGate::new(
            Gate::Rz(0.5 * i as f64),
            vec![i],
            vec![],
            vec![],
        )));
    }

    let circuit = Circuit::new(vec![2; 10], gates).unwrap();
    let tn = circuit_to_einsum(&circuit);

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();
    let complexity = contraction_complexity(&optimized, &tn.size_dict, &tn.code.ixs);

    assert!(complexity.tc > 0.0);
    assert!(complexity.sc > 0.0);
}

#[test]
fn test_single_diagonal_gate_valid_eincode() {
    // Single diagonal gate (Z) produces a valid EinCode
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(Gate::Z, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
    assert!(optimized.unwrap().is_leaf());
}

#[test]
fn test_single_rz_gate_valid_eincode() {
    // Single Rz gate (diagonal parametric) produces a valid EinCode
    let circuit = Circuit::new(
        vec![2],
        vec![gate(PositionedGate::new(
            Gate::Rz(1.0),
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    let tn = circuit_to_einsum(&circuit);
    assert!(tn.code.is_valid());

    let optimized = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default());
    assert!(optimized.is_some());
    assert!(optimized.unwrap().is_leaf());
}
