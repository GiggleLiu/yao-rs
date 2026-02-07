#!/usr/bin/env julia
#
# Generate ground-truth test data for yao-rs using Yao.jl.
#
# Usage: julia scripts/generate_test_data.jl
#
# Generates:
#   tests/data/gates.json   - Gate matrix data for all 16 gate types
#   tests/data/apply.json   - State evolution test cases
#   tests/data/einsum.json  - Tensor network contraction results (same circuits as apply)
#   tests/data/measure.json - Probability distributions for known circuits
#
# Convention mapping (critical):
#   yao-rs qubit i (0-indexed, MSB first) = Yao.jl qubit (n - i) (1-indexed, LSB first)
#   State vector indexing is the SAME in both conventions.
#
# This script uses Yao.jl's native API (mat, apply!, probs, yao2einsum)
# to generate ground-truth data.

using Yao
using Yao.EasyBuild
using JSON
using LinearAlgebra

const OUTPUT_DIR = joinpath(@__DIR__, "..", "tests", "data")
const PRECISION = 15

# ============================================================================
# Utility functions
# ============================================================================

"""Round a float to PRECISION decimal digits.
For maximum precision, we use Float64 which gives ~15-17 significant digits.
JSON.jl uses the shortest representation that round-trips exactly."""
rnd(x::Real) = Float64(x)

"""Convert a complex matrix to separate real and imaginary 2D arrays."""
function matrix_to_re_im(m::AbstractMatrix{<:Complex})
    re = [rnd(real(m[i,j])) for i in 1:size(m,1), j in 1:size(m,2)]
    im = [rnd(imag(m[i,j])) for i in 1:size(m,1), j in 1:size(m,2)]
    # Convert to vector of vectors for JSON
    re_vv = [re[i,:] |> collect for i in 1:size(re,1)]
    im_vv = [im[i,:] |> collect for i in 1:size(im,1)]
    return re_vv, im_vv
end

"""Convert a complex vector to separate real and imaginary 1D arrays."""
function vec_to_re_im(v::AbstractVector{<:Complex})
    re = [rnd(real(x)) for x in v]
    im = [rnd(imag(x)) for x in v]
    return re, im
end

"""
Map yao-rs qubit index (0-indexed, MSB) to Yao.jl qubit index (1-indexed, LSB).
For n qubits: yao_rs_qubit i -> yao_jl_qubit (n - i)
"""
yaors_to_yaojl(i::Int, n::Int) = n - i

"""Create a Yao.jl gate block from a gate name and parameters.
Uses Yao.jl's native gate constructors (mat, EasyBuild, ConstGate)."""
function make_gate_block(name::String, params::Vector)
    if name == "X"
        return X
    elseif name == "Y"
        return Y
    elseif name == "Z"
        return Z
    elseif name == "H"
        return H
    elseif name == "S"
        return ConstGate.S
    elseif name == "T"
        return ConstGate.T
    elseif name == "SWAP"
        return SWAP
    elseif name == "Phase"
        return shift(params[1])
    elseif name == "Rx"
        return Rx(params[1])
    elseif name == "Ry"
        return Ry(params[1])
    elseif name == "Rz"
        return Rz(params[1])
    elseif name == "SqrtX"
        return SqrtX
    elseif name == "SqrtY"
        return SqrtY
    elseif name == "SqrtW"
        return SqrtW
    elseif name == "ISWAP"
        return ISWAP
    elseif name == "FSim"
        return FSimGate(params[1], params[2])
    else
        error("Unknown gate: $name")
    end
end

"""
Build a Yao.jl circuit block from a list of gate specifications.
Each gate spec is a Dict with keys: name, params, targets, controls, control_configs
All qubit indices are in yao-rs convention (0-indexed).
"""
function build_yao_circuit(num_qubits::Int, gates::Vector{Dict{String,Any}})
    blocks = []
    for g in gates
        gate_block = make_gate_block(g["name"], get(g, "params", []))
        targets_rs = g["targets"]  # 0-indexed
        controls_rs = get(g, "controls", Int[])
        control_configs_rs = get(g, "control_configs", Bool[])

        # Convert to Yao.jl indices (1-indexed)
        n = num_qubits
        targets_jl = [yaors_to_yaojl(t, n) for t in targets_rs]
        controls_jl = [yaors_to_yaojl(c, n) for c in controls_rs]

        if length(targets_jl) == 1
            target_spec = targets_jl[1] => gate_block
        else
            target_spec = (targets_jl...,) => gate_block
        end

        if isempty(controls_jl)
            push!(blocks, put(n, target_spec))
        else
            # Handle active-low controls
            if isempty(control_configs_rs) || all(control_configs_rs)
                # All active-high
                ctrl_locs = controls_jl
            else
                # Mix of active-high and active-low
                ctrl_locs = []
                for (idx, c) in enumerate(controls_jl)
                    if control_configs_rs[idx]
                        push!(ctrl_locs, c)  # active-high
                    else
                        push!(ctrl_locs, -c)  # active-low (negative in Yao.jl)
                    end
                end
            end
            if length(ctrl_locs) == 1
                push!(blocks, control(n, ctrl_locs[1], target_spec))
            else
                push!(blocks, control(n, (ctrl_locs...,), target_spec))
            end
        end
    end
    return chain(num_qubits, blocks...)
end

"""Apply a circuit to |0...0> using apply! and return the output state vector."""
function apply_circuit(num_qubits::Int, gates::Vector{Dict{String,Any}})
    circuit = build_yao_circuit(num_qubits, gates)
    reg = zero_state(num_qubits)
    apply!(reg, circuit)
    return statevec(reg)
end

"""Get probability distribution using Yao's probs function."""
function circuit_probs(num_qubits::Int, gates::Vector{Dict{String,Any}})
    circuit = build_yao_circuit(num_qubits, gates)
    reg = zero_state(num_qubits)
    apply!(reg, circuit)
    return [rnd(p) for p in probs(reg)]
end

"""Get einsum contraction result using yao2einsum."""
function circuit_einsum(num_qubits::Int, gates::Vector{Dict{String,Any}})
    circuit = build_yao_circuit(num_qubits, gates)
    # Use yao2einsum to convert circuit to tensor network, then contract
    tn = yao2einsum(circuit; initial_state=Dict([i=>0 for i in 1:num_qubits]), final_state=Dict{Int,Int}())
    result = contract(tn)
    return vec(result)
end

# ============================================================================
# 1. Generate gates.json - using mat() for all gates
# ============================================================================

function generate_gates()
    println("Generating gates.json...")

    # Parameter values to test
    angles = [0.0, pi/4, pi/2, pi, 3*pi/2, 2*pi, 1e-10, -pi/3]

    gate_entries = []

    # Constant gates (no parameters) - use mat() for all
    for (name, gate) in [
        ("X", X), ("Y", Y), ("Z", Z), ("H", H),
        ("S", ConstGate.S), ("T", ConstGate.T),
        ("SWAP", SWAP),
        ("SqrtX", SqrtX), ("SqrtY", SqrtY), ("SqrtW", SqrtW),
        ("ISWAP", ISWAP),
    ]
        m = mat(gate)
        re, im_part = matrix_to_re_im(m)
        push!(gate_entries, Dict(
            "name" => name,
            "params" => [],
            "matrix_re" => re,
            "matrix_im" => im_part,
        ))
    end

    # Single-parameter gates: Phase, Rx, Ry, Rz - use mat()
    for name in ["Phase", "Rx", "Ry", "Rz"]
        for theta in angles
            gate = make_gate_block(name, [theta])
            m = mat(gate)
            re, im_part = matrix_to_re_im(m)
            push!(gate_entries, Dict(
                "name" => name,
                "params" => [rnd(theta)],
                "matrix_re" => re,
                "matrix_im" => im_part,
            ))
        end
    end

    # Two-parameter gate: FSim - use mat(FSimGate(theta, phi))
    fsim_param_pairs = [
        (0.0, 0.0),
        (pi/4, pi/6),
        (pi/2, pi/4),
        (pi, pi),
        (pi/2, pi/6),  # Google Sycamore FSim
        (1e-10, 1e-10),
    ]
    for (theta, phi) in fsim_param_pairs
        gate = FSimGate(theta, phi)
        m = mat(gate)
        re, im_part = matrix_to_re_im(m)
        push!(gate_entries, Dict(
            "name" => "FSim",
            "params" => [rnd(theta), rnd(phi)],
            "matrix_re" => re,
            "matrix_im" => im_part,
        ))
    end

    data = Dict("gates" => gate_entries)
    open(joinpath(OUTPUT_DIR, "gates.json"), "w") do f
        JSON.print(f, data, 2)
    end
    println("  Generated $(length(gate_entries)) gate entries")
end

# ============================================================================
# 2. Generate apply.json (and shared circuit definitions for einsum.json)
# ============================================================================

"""Helper to create a gate spec dict for JSON."""
function gate_spec(name::String; params=[], targets=[0], controls=Int[], control_configs=Bool[])
    return Dict{String,Any}(
        "name" => name,
        "params" => [rnd(p) for p in params],
        "targets" => targets,
        "controls" => controls,
        "control_configs" => control_configs,
    )
end

"""Helper to create a Custom gate spec dict with matrix data for JSON.
The matrix is specified as separate real and imaginary 2D arrays."""
function custom_gate_spec(matrix::AbstractMatrix{<:Complex}; targets=[0], controls=Int[], control_configs=Bool[], label="Custom")
    re_vv, im_vv = matrix_to_re_im(matrix)
    return Dict{String,Any}(
        "name" => "Custom",
        "params" => [],
        "targets" => targets,
        "controls" => controls,
        "control_configs" => control_configs,
        "matrix_re" => re_vv,
        "matrix_im" => im_vv,
        "label" => label,
    )
end

"""
Define all test circuits. Returns a list of (label, dims, gates) tuples.
dims is a Vector{Int} of per-site dimensions (e.g., [2, 2] for 2 qubits).
These are shared between apply.json and einsum.json.
"""
function define_test_circuits()
    circuits = Tuple{String, Vector{Int}, Vector{Dict{String,Any}}}[]

    # ------------------------------------------------------------------
    # Single gates on |0>
    # ------------------------------------------------------------------
    for name in ["X", "Y", "Z", "H", "S", "T", "SqrtX", "SqrtY", "SqrtW"]
        push!(circuits, ("$name on |0>", [2], [gate_spec(name)]))
    end

    # Parameterized single gates on |0>
    for name in ["Rx", "Ry", "Rz", "Phase"]
        for theta in [pi/4, pi/2, pi]
            push!(circuits, ("$name($theta) on |0>", [2], [gate_spec(name; params=[theta])]))
        end
    end

    # Single gates on |1> (start by flipping with X)
    for name in ["X", "Y", "Z", "H"]
        push!(circuits, ("$name on |1>", [2], [gate_spec("X"), gate_spec(name)]))
    end

    # ------------------------------------------------------------------
    # Two-qubit gates on |00>
    # ------------------------------------------------------------------
    push!(circuits, ("SWAP on |00>", [2,2], [gate_spec("SWAP"; targets=[0,1])]))
    push!(circuits, ("SWAP on |10>", [2,2], [gate_spec("X"; targets=[0]), gate_spec("SWAP"; targets=[0,1])]))
    push!(circuits, ("ISWAP on |00>", [2,2], [gate_spec("ISWAP"; targets=[0,1])]))
    push!(circuits, ("ISWAP on |10>", [2,2], [gate_spec("X"; targets=[0]), gate_spec("ISWAP"; targets=[0,1])]))
    push!(circuits, ("ISWAP on |01>", [2,2], [gate_spec("X"; targets=[1]), gate_spec("ISWAP"; targets=[0,1])]))

    # FSim gates
    push!(circuits, ("FSim(pi/4,pi/6) on |00>", [2,2],
        [gate_spec("FSim"; params=[pi/4, pi/6], targets=[0,1])]))
    push!(circuits, ("FSim(pi/2,pi/6) on |10>", [2,2],
        [gate_spec("X"; targets=[0]), gate_spec("FSim"; params=[pi/2, pi/6], targets=[0,1])]))
    push!(circuits, ("FSim(pi/2,pi/6) on |01>", [2,2],
        [gate_spec("X"; targets=[1]), gate_spec("FSim"; params=[pi/2, pi/6], targets=[0,1])]))
    push!(circuits, ("FSim(pi/2,pi/6) on |11>", [2,2],
        [gate_spec("X"; targets=[0]), gate_spec("X"; targets=[1]),
         gate_spec("FSim"; params=[pi/2, pi/6], targets=[0,1])]))

    # ------------------------------------------------------------------
    # Controlled gates
    # ------------------------------------------------------------------
    # CNOT
    push!(circuits, ("CNOT(0->1) on |00>", [2,2],
        [gate_spec("X"; controls=[0], targets=[1], control_configs=[true])]))
    push!(circuits, ("CNOT(0->1) on |10>", [2,2],
        [gate_spec("X"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true])]))
    push!(circuits, ("CNOT(1->0) on |01>", [2,2],
        [gate_spec("X"; targets=[1]),
         gate_spec("X"; controls=[1], targets=[0], control_configs=[true])]))

    # Bell state
    push!(circuits, ("Bell state", [2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true])]))

    # Controlled-Z
    push!(circuits, ("CZ on |++>", [2,2],
        [gate_spec("H"; targets=[0]), gate_spec("H"; targets=[1]),
         gate_spec("Z"; controls=[0], targets=[1], control_configs=[true])]))

    # Controlled-Phase
    push!(circuits, ("CPhase(pi/4) on H|00>", [2,2],
        [gate_spec("H"; targets=[0]), gate_spec("H"; targets=[1]),
         gate_spec("Phase"; params=[pi/4], controls=[0], targets=[1], control_configs=[true])]))

    # ------------------------------------------------------------------
    # Active-low controls
    # ------------------------------------------------------------------
    push!(circuits, ("Active-low CNOT on |00>", [2,2],
        [gate_spec("X"; controls=[0], targets=[1], control_configs=[false])]))
    push!(circuits, ("Active-low CNOT on |10>", [2,2],
        [gate_spec("X"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[false])]))

    # ------------------------------------------------------------------
    # Multi-controlled gates
    # ------------------------------------------------------------------
    # Toffoli (CCX)
    push!(circuits, ("Toffoli on |110>", [2,2,2],
        [gate_spec("X"; targets=[0]), gate_spec("X"; targets=[1]),
         gate_spec("X"; controls=[0,1], targets=[2], control_configs=[true,true])]))
    push!(circuits, ("Toffoli on |100>", [2,2,2],
        [gate_spec("X"; targets=[0]),
         gate_spec("X"; controls=[0,1], targets=[2], control_configs=[true,true])]))
    push!(circuits, ("Toffoli on |010>", [2,2,2],
        [gate_spec("X"; targets=[1]),
         gate_spec("X"; controls=[0,1], targets=[2], control_configs=[true,true])]))

    # Mixed active-high/low multi-control
    push!(circuits, ("Mixed control(0-high,1-low) X on |10>", [2,2,2],
        [gate_spec("X"; targets=[0]),
         gate_spec("X"; controls=[0,1], targets=[2], control_configs=[true,false])]))

    # ------------------------------------------------------------------
    # Multi-qubit circuits (3-8 qubits)
    # ------------------------------------------------------------------

    # GHZ state on 3 qubits
    push!(circuits, ("GHZ 3 qubits", [2,2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
         gate_spec("X"; controls=[1], targets=[2], control_configs=[true])]))

    # GHZ state on 4 qubits
    push!(circuits, ("GHZ 4 qubits", [2,2,2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
         gate_spec("X"; controls=[1], targets=[2], control_configs=[true]),
         gate_spec("X"; controls=[2], targets=[3], control_configs=[true])]))

    # All-H on 4 qubits
    push!(circuits, ("H on all 4 qubits", [2,2,2,2],
        [gate_spec("H"; targets=[i]) for i in 0:3]))

    # Mixed gates on 5 qubits
    push!(circuits, ("Mixed 5-qubit circuit", [2,2,2,2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("H"; targets=[1]),
         gate_spec("X"; controls=[0], targets=[2], control_configs=[true]),
         gate_spec("X"; controls=[1], targets=[3], control_configs=[true]),
         gate_spec("Rz"; params=[pi/4], targets=[4]),
         gate_spec("X"; controls=[2], targets=[4], control_configs=[true])]))

    # 6-qubit circuit
    push!(circuits, ("6-qubit entangled circuit", fill(2, 6),
        [gate_spec("H"; targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
         gate_spec("H"; targets=[2]),
         gate_spec("X"; controls=[2], targets=[3], control_configs=[true]),
         gate_spec("H"; targets=[4]),
         gate_spec("X"; controls=[4], targets=[5], control_configs=[true]),
         gate_spec("X"; controls=[1], targets=[2], control_configs=[true])]))

    # 8-qubit circuit (chain of CNOTs after H)
    push!(circuits, ("8-qubit CNOT chain", fill(2, 8),
        vcat(
            [gate_spec("H"; targets=[0])],
            [gate_spec("X"; controls=[i], targets=[i+1], control_configs=[true]) for i in 0:6]
        )))

    # ------------------------------------------------------------------
    # QFT circuits (with bit-reversal SWAPs, i.e. textbook QFT)
    # Note: yao-rs easybuild::qft_circuit matches Yao.jl's qft_circuit
    # and does NOT include the SWAP layer. These test cases include SWAPs
    # to test the full textbook QFT as an explicit gate sequence.
    # ------------------------------------------------------------------
    for n in [3, 4, 5]
        gates = Dict{String,Any}[]
        for i in 0:(n-1)
            push!(gates, gate_spec("H"; targets=[i]))
            for j in 1:(n-1-i)
                theta = 2*pi / (1 << (j+1))
                push!(gates, gate_spec("Phase"; params=[theta],
                    controls=[i+j], targets=[i], control_configs=[true]))
            end
        end
        # SWAP for bit reversal
        for i in 0:(div(n,2)-1)
            push!(gates, gate_spec("SWAP"; targets=[i, n-1-i]))
        end
        push!(circuits, ("QFT $n qubits", fill(2, n), gates))
    end

    # ------------------------------------------------------------------
    # Variational circuits (matching yao-rs easybuild::variational_circuit)
    # ------------------------------------------------------------------
    for n in [3, 4]
        nlayer = 1  # Keep it simple
        # pair_ring topology
        pairs = [(i, (i+1) % n) for i in 0:(n-1)]

        gates = Dict{String,Any}[]

        for layer in 0:nlayer
            # CNOT entangler (not on first layer)
            if layer > 0
                for (ctrl, tgt) in pairs
                    push!(gates, gate_spec("X"; controls=[ctrl], targets=[tgt],
                        control_configs=[true]))
                end
            end

            # Rotor block
            for qubit in 0:(n-1)
                if layer == 0
                    # noleading: Rx(0), Rz(0)
                    push!(gates, gate_spec("Rx"; params=[0.0], targets=[qubit]))
                    push!(gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
                elseif layer == nlayer
                    # notrailing: Rz(0), Rx(0)
                    push!(gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
                    push!(gates, gate_spec("Rx"; params=[0.0], targets=[qubit]))
                else
                    # full: Rz(0), Rx(0), Rz(0)
                    push!(gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
                    push!(gates, gate_spec("Rx"; params=[0.0], targets=[qubit]))
                    push!(gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
                end
            end
        end

        push!(circuits, ("Variational $n qubits nlayer=$nlayer", fill(2, n), gates))
    end

    # ------------------------------------------------------------------
    # Controlled-SWAP (Fredkin gate)
    # ------------------------------------------------------------------
    push!(circuits, ("Fredkin on |110>", [2,2,2],
        [gate_spec("X"; targets=[0]), gate_spec("X"; targets=[1]),
         gate_spec("SWAP"; controls=[0], targets=[1,2], control_configs=[true])]))

    # ------------------------------------------------------------------
    # Near-zero angle edge cases
    # ------------------------------------------------------------------
    push!(circuits, ("Rx(1e-10) on |0>", [2], [gate_spec("Rx"; params=[1e-10])]))
    push!(circuits, ("Ry(1e-10) on |0>", [2], [gate_spec("Ry"; params=[1e-10])]))
    push!(circuits, ("Rz(1e-10) on |0>", [2], [gate_spec("Rz"; params=[1e-10])]))
    push!(circuits, ("Phase(1e-10) on |0>", [2], [gate_spec("Phase"; params=[1e-10])]))

    # ------------------------------------------------------------------
    # 2*pi angle edge cases (should be near-identity)
    # ------------------------------------------------------------------
    push!(circuits, ("Rx(2pi) on |0>", [2], [gate_spec("Rx"; params=[2*pi])]))
    push!(circuits, ("Ry(2pi) on |0>", [2], [gate_spec("Ry"; params=[2*pi])]))
    push!(circuits, ("Rz(2pi) on |0>", [2], [gate_spec("Rz"; params=[2*pi])]))

    # ------------------------------------------------------------------
    # Hadamard test circuit (matching yao-rs easybuild::hadamard_test_circuit)
    # ------------------------------------------------------------------
    push!(circuits, ("Hadamard test X phi=0", [2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("Rz"; params=[0.0], targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
         gate_spec("H"; targets=[0])]))

    push!(circuits, ("Hadamard test X phi=pi/4", [2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("Rz"; params=[pi/4], targets=[0]),
         gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
         gate_spec("H"; targets=[0])]))

    # ------------------------------------------------------------------
    # Swap test circuit (matching yao-rs easybuild::swap_test_circuit)
    # ------------------------------------------------------------------
    push!(circuits, ("Swap test nbit=1 nstate=2 phi=0", [2,2,2],
        [gate_spec("H"; targets=[0]),
         gate_spec("Rz"; params=[0.0], targets=[0]),
         gate_spec("SWAP"; controls=[0], targets=[1,2], control_configs=[true]),
         gate_spec("H"; targets=[0])]))

    return circuits
end

"""
Define qudit test circuits. Returns a list of (label, dims, gates, input_state, output_state) tuples.
These circuits use non-qubit dimensions and are computed manually (no Yao.jl).
The states are ComplexF64 vectors.
"""
function define_qudit_circuits()
    qudit_cases = Tuple{String, Vector{Int}, Vector{Dict{String,Any}}, Vector{ComplexF64}, Vector{ComplexF64}}[]

    # ------------------------------------------------------------------
    # Single qutrit (d=3) gate: cyclic permutation P|i> = |i+1 mod 3>
    # ------------------------------------------------------------------
    P3 = ComplexF64[0 0 1; 1 0 0; 0 1 0]
    input_3 = ComplexF64[1, 0, 0]
    output_3 = P3 * input_3  # = [0, 1, 0]
    push!(qudit_cases, (
        "Qutrit cyclic permutation on |0>",
        [3],
        [custom_gate_spec(P3; targets=[0], label="CyclicPerm3")],
        input_3,
        output_3,
    ))

    # ------------------------------------------------------------------
    # Single ququart (d=4) gate: diagonal phase gate
    # ------------------------------------------------------------------
    D4 = diagm(ComplexF64[1, im, -1, -im])
    input_4 = ComplexF64[1, 0, 0, 0]
    output_4 = D4 * input_4
    push!(qudit_cases, (
        "Ququart diagonal phase on |0>",
        [4],
        [custom_gate_spec(D4; targets=[0], label="DiagPhase4")],
        input_4,
        output_4,
    ))

    # ------------------------------------------------------------------
    # Two-qutrit circuit (d=3,3) with a custom 9x9 gate
    # ------------------------------------------------------------------
    SWAP9 = zeros(ComplexF64, 9, 9)
    for i in 0:2, j in 0:2
        src = i * 3 + j + 1  # 1-indexed: |ij>
        dst = j * 3 + i + 1  # 1-indexed: |ji>
        SWAP9[dst, src] = 1.0
    end
    input_33 = zeros(ComplexF64, 9); input_33[1] = 1.0
    state_after_p3 = zeros(ComplexF64, 9)
    state_after_p3[4] = 1.0  # |10>
    output_33 = SWAP9 * state_after_p3  # = |01>
    push!(qudit_cases, (
        "Two-qutrit P3 then SWAP on |00>",
        [3, 3],
        [custom_gate_spec(P3; targets=[0], label="CyclicPerm3"),
         custom_gate_spec(SWAP9; targets=[0, 1], label="SWAP3")],
        input_33,
        output_33,
    ))

    # ------------------------------------------------------------------
    # Mixed qubit-qutrit circuit (d=[2,3])
    # ------------------------------------------------------------------
    X2 = ComplexF64[0 1; 1 0]
    input_23 = zeros(ComplexF64, 6); input_23[1] = 1.0
    output_23 = zeros(ComplexF64, 6)
    output_23[5] = 1.0  # |11>
    push!(qudit_cases, (
        "Mixed qubit-qutrit X then P3 on |00>",
        [2, 3],
        [custom_gate_spec(X2; targets=[0], label="X"),
         custom_gate_spec(P3; targets=[1], label="CyclicPerm3")],
        input_23,
        output_23,
    ))

    return qudit_cases
end

function generate_apply()
    println("Generating apply.json...")

    circuits = define_test_circuits()
    cases = []

    for (label, dims, gates) in circuits
        num_qubits = length(dims)
        # Use apply! to apply circuit and get output state
        sv = apply_circuit(num_qubits, gates)

        # Input state is always |0...0>
        total_dim = prod(dims)
        input_re = zeros(Float64, total_dim)
        input_im = zeros(Float64, total_dim)
        input_re[1] = 1.0

        out_re, out_im = vec_to_re_im(sv)

        push!(cases, Dict(
            "label" => label,
            "num_qubits" => num_qubits,
            "dims" => dims,
            "gates" => gates,
            "input_state_re" => input_re,
            "input_state_im" => input_im,
            "output_state_re" => out_re,
            "output_state_im" => out_im,
        ))
    end

    # ------------------------------------------------------------------
    # Qudit cases (computed manually, not via Yao.jl)
    # ------------------------------------------------------------------
    qudit_circuits = define_qudit_circuits()
    for (label, dims, gates, input_state, output_state) in qudit_circuits
        in_re, in_im = vec_to_re_im(input_state)
        out_re, out_im = vec_to_re_im(output_state)

        push!(cases, Dict(
            "label" => label,
            "num_qubits" => length(dims),
            "dims" => dims,
            "gates" => gates,
            "input_state_re" => in_re,
            "input_state_im" => in_im,
            "output_state_re" => out_re,
            "output_state_im" => out_im,
        ))
    end

    data = Dict("cases" => cases)
    open(joinpath(OUTPUT_DIR, "apply.json"), "w") do f
        JSON.print(f, data, 2)
    end
    println("  Generated $(length(cases)) apply test cases ($(length(qudit_circuits)) qudit)")
end

# ============================================================================
# 3. Generate einsum.json - using yao2einsum for tensor network contraction
# ============================================================================

function generate_einsum()
    println("Generating einsum.json...")

    # Use the same circuits as apply.json for cross-verification
    circuits = define_test_circuits()
    cases = []

    for (label, dims, gates) in circuits
        num_qubits = length(dims)
        # Use yao2einsum to convert circuit to tensor network and contract
        circuit = build_yao_circuit(num_qubits, gates)
        tn = yao2einsum(circuit; initial_state=Dict([i=>0 for i in 1:num_qubits]), final_state=Dict{Int,Int}())
        result = contract(tn)
        sv = vec(result)
        out_re, out_im = vec_to_re_im(sv)

        push!(cases, Dict(
            "label" => label,
            "num_qubits" => num_qubits,
            "dims" => dims,
            "gates" => gates,
            "output_state_re" => out_re,
            "output_state_im" => out_im,
        ))
    end

    # ------------------------------------------------------------------
    # Qudit cases (computed manually, not via Yao.jl)
    # ------------------------------------------------------------------
    qudit_circuits = define_qudit_circuits()
    for (label, dims, gates, _input_state, output_state) in qudit_circuits
        out_re, out_im = vec_to_re_im(output_state)

        push!(cases, Dict(
            "label" => label,
            "num_qubits" => length(dims),
            "dims" => dims,
            "gates" => gates,
            "output_state_re" => out_re,
            "output_state_im" => out_im,
        ))
    end

    data = Dict("cases" => cases)
    open(joinpath(OUTPUT_DIR, "einsum.json"), "w") do f
        JSON.print(f, data, 2)
    end
    println("  Generated $(length(cases)) einsum test cases ($(length(qudit_circuits)) qudit)")
end

# ============================================================================
# 4. Generate measure.json - using probs() from Yao.jl
# ============================================================================

function generate_measure()
    println("Generating measure.json...")

    cases = []

    # ------------------------------------------------------------------
    # Simple probability distributions - use Yao's probs()
    # ------------------------------------------------------------------

    # H|0> = 50/50
    push!(cases, Dict(
        "label" => "H|0> probabilities",
        "num_qubits" => 1,
        "gates" => [gate_spec("H")],
        "probabilities" => circuit_probs(1, [gate_spec("H")]),
    ))

    # |0> = deterministic 0
    push!(cases, Dict(
        "label" => "|0> probabilities",
        "num_qubits" => 1,
        "gates" => Dict{String,Any}[],
        "probabilities" => circuit_probs(1, Dict{String,Any}[]),
    ))

    # X|0> = |1> = deterministic 1
    push!(cases, Dict(
        "label" => "X|0> probabilities",
        "num_qubits" => 1,
        "gates" => [gate_spec("X")],
        "probabilities" => circuit_probs(1, [gate_spec("X")]),
    ))

    # Bell state
    bell_gates = [
        gate_spec("H"; targets=[0]),
        gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
    ]
    push!(cases, Dict(
        "label" => "Bell state probabilities",
        "num_qubits" => 2,
        "gates" => bell_gates,
        "probabilities" => circuit_probs(2, bell_gates),
    ))

    # H on all 2 qubits = equal superposition
    h2_gates = [gate_spec("H"; targets=[0]), gate_spec("H"; targets=[1])]
    push!(cases, Dict(
        "label" => "H|00> probabilities",
        "num_qubits" => 2,
        "gates" => h2_gates,
        "probabilities" => circuit_probs(2, h2_gates),
    ))

    # GHZ 3 qubits
    ghz3_gates = [
        gate_spec("H"; targets=[0]),
        gate_spec("X"; controls=[0], targets=[1], control_configs=[true]),
        gate_spec("X"; controls=[1], targets=[2], control_configs=[true]),
    ]
    push!(cases, Dict(
        "label" => "GHZ 3 qubits probabilities",
        "num_qubits" => 3,
        "gates" => ghz3_gates,
        "probabilities" => circuit_probs(3, ghz3_gates),
    ))

    # QFT 3 qubits on |0>
    qft3_gates = Dict{String,Any}[]
    n = 3
    for i in 0:(n-1)
        push!(qft3_gates, gate_spec("H"; targets=[i]))
        for j in 1:(n-1-i)
            theta = 2*pi / (1 << (j+1))
            push!(qft3_gates, gate_spec("Phase"; params=[theta],
                controls=[i+j], targets=[i], control_configs=[true]))
        end
    end
    for i in 0:(div(n,2)-1)
        push!(qft3_gates, gate_spec("SWAP"; targets=[i, n-1-i]))
    end
    push!(cases, Dict(
        "label" => "QFT 3 qubits probabilities",
        "num_qubits" => 3,
        "gates" => qft3_gates,
        "probabilities" => circuit_probs(3, qft3_gates),
    ))

    # H + Rx(pi/4) on qubit 0
    hrx_gates = [gate_spec("H"), gate_spec("Rx"; params=[pi/4])]
    push!(cases, Dict(
        "label" => "H Rx(pi/4) probabilities",
        "num_qubits" => 1,
        "gates" => hrx_gates,
        "probabilities" => circuit_probs(1, hrx_gates),
    ))

    # 4-qubit mixed circuit
    mixed4_gates = [
        gate_spec("H"; targets=[0]),
        gate_spec("H"; targets=[1]),
        gate_spec("X"; controls=[0], targets=[2], control_configs=[true]),
        gate_spec("Ry"; params=[pi/3], targets=[3]),
    ]
    push!(cases, Dict(
        "label" => "Mixed 4-qubit probabilities",
        "num_qubits" => 4,
        "gates" => mixed4_gates,
        "probabilities" => circuit_probs(4, mixed4_gates),
    ))

    # Variational 3 qubits (all angles 0 = identity)
    var3_gates = Dict{String,Any}[]
    n = 3; nlayer = 1
    pairs = [(i, (i+1) % n) for i in 0:(n-1)]
    for layer in 0:nlayer
        if layer > 0
            for (ctrl, tgt) in pairs
                push!(var3_gates, gate_spec("X"; controls=[ctrl], targets=[tgt],
                    control_configs=[true]))
            end
        end
        for qubit in 0:(n-1)
            if layer == 0
                push!(var3_gates, gate_spec("Rx"; params=[0.0], targets=[qubit]))
                push!(var3_gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
            elseif layer == nlayer
                push!(var3_gates, gate_spec("Rz"; params=[0.0], targets=[qubit]))
                push!(var3_gates, gate_spec("Rx"; params=[0.0], targets=[qubit]))
            end
        end
    end
    push!(cases, Dict(
        "label" => "Variational 3 qubits probabilities",
        "num_qubits" => 3,
        "gates" => var3_gates,
        "probabilities" => circuit_probs(3, var3_gates),
    ))

    # 5-qubit H on all then measure
    h5_gates = [gate_spec("H"; targets=[i]) for i in 0:4]
    push!(cases, Dict(
        "label" => "H on all 5 qubits probabilities",
        "num_qubits" => 5,
        "gates" => h5_gates,
        "probabilities" => circuit_probs(5, h5_gates),
    ))

    data = Dict("cases" => cases)
    open(joinpath(OUTPUT_DIR, "measure.json"), "w") do f
        JSON.print(f, data, 2)
    end
    println("  Generated $(length(cases)) measure test cases")
end

# ============================================================================
# Main
# ============================================================================

function main()
    mkpath(OUTPUT_DIR)
    println("Output directory: $OUTPUT_DIR")
    println()

    generate_gates()
    generate_apply()
    generate_einsum()
    generate_measure()

    println()
    println("All test data generated successfully.")

    # Quick validation
    println()
    println("=== Quick Validation ===")

    # Check gate count
    gates_data = JSON.parsefile(joinpath(OUTPUT_DIR, "gates.json"))
    println("gates.json: $(length(gates_data["gates"])) gate entries")

    # Check apply count
    apply_data = JSON.parsefile(joinpath(OUTPUT_DIR, "apply.json"))
    println("apply.json: $(length(apply_data["cases"])) test cases")

    # Check einsum count
    einsum_data = JSON.parsefile(joinpath(OUTPUT_DIR, "einsum.json"))
    println("einsum.json: $(length(einsum_data["cases"])) test cases")

    # Check measure count
    measure_data = JSON.parsefile(joinpath(OUTPUT_DIR, "measure.json"))
    println("measure.json: $(length(measure_data["cases"])) test cases")

    # Verify apply and einsum have same number of cases
    @assert length(apply_data["cases"]) == length(einsum_data["cases"]) "apply and einsum case count mismatch"
    println("apply and einsum case counts match.")

    # Verify all cases have dims field
    for (i, c) in enumerate(apply_data["cases"])
        @assert haskey(c, "dims") "apply case $i ($(c["label"])) missing dims field"
    end
    for (i, c) in enumerate(einsum_data["cases"])
        @assert haskey(c, "dims") "einsum case $i ($(c["label"])) missing dims field"
    end
    println("All apply/einsum cases have dims field.")

    # Verify first few cases match between apply and einsum
    for i in 1:min(5, length(apply_data["cases"]))
        a = apply_data["cases"][i]
        e = einsum_data["cases"][i]
        @assert a["label"] == e["label"] "Label mismatch at case $i"
        @assert a["output_state_re"] == e["output_state_re"] "output_state_re mismatch at case $i: $(a["label"])"
        @assert a["output_state_im"] == e["output_state_im"] "output_state_im mismatch at case $i: $(a["label"])"
        @assert a["dims"] == e["dims"] "dims mismatch at case $i: $(a["label"])"
    end
    println("First 5 apply/einsum outputs match.")

    # Verify some known results
    # X|0> should give |1> = [0, 1]
    x_case = apply_data["cases"][findfirst(c -> c["label"] == "X on |0>", apply_data["cases"])]
    @assert x_case["output_state_re"] == [0.0, 1.0] "X|0> real part wrong"
    @assert x_case["output_state_im"] == [0.0, 0.0] "X|0> imag part wrong"
    @assert x_case["dims"] == [2] "X|0> dims wrong"
    println("X|0> = |1> verified.")

    # H|0> should give [1/sqrt(2), 1/sqrt(2)]
    h_case = apply_data["cases"][findfirst(c -> c["label"] == "H on |0>", apply_data["cases"])]
    expected_val = round(1/sqrt(2), digits=PRECISION)
    @assert abs(h_case["output_state_re"][1] - expected_val) < 1e-10 "H|0> wrong"
    @assert abs(h_case["output_state_re"][2] - expected_val) < 1e-10 "H|0> wrong"
    println("H|0> = |+> verified.")

    # Bell state should give [1/sqrt(2), 0, 0, 1/sqrt(2)]
    bell_case = apply_data["cases"][findfirst(c -> c["label"] == "Bell state", apply_data["cases"])]
    @assert abs(bell_case["output_state_re"][1] - expected_val) < 1e-10 "Bell state wrong"
    @assert abs(bell_case["output_state_re"][2]) < 1e-10 "Bell state wrong"
    @assert abs(bell_case["output_state_re"][3]) < 1e-10 "Bell state wrong"
    @assert abs(bell_case["output_state_re"][4] - expected_val) < 1e-10 "Bell state wrong"
    println("Bell state verified.")

    # Verify qudit cases
    qutrit_case = apply_data["cases"][findfirst(c -> c["label"] == "Qutrit cyclic permutation on |0>", apply_data["cases"])]
    @assert qutrit_case["dims"] == [3] "Qutrit dims wrong"
    @assert length(qutrit_case["output_state_re"]) == 3 "Qutrit output size wrong"
    @assert qutrit_case["output_state_re"] == [0.0, 1.0, 0.0] "Qutrit output wrong: P|0> should be |1>"
    println("Qutrit cyclic permutation verified.")

    ququart_case = apply_data["cases"][findfirst(c -> c["label"] == "Ququart diagonal phase on |0>", apply_data["cases"])]
    @assert ququart_case["dims"] == [4] "Ququart dims wrong"
    @assert length(ququart_case["output_state_re"]) == 4 "Ququart output size wrong"
    @assert ququart_case["output_state_re"] == [1.0, 0.0, 0.0, 0.0] "Ququart output wrong: D|0> should be |0>"
    println("Ququart diagonal phase verified.")

    mixed_case = apply_data["cases"][findfirst(c -> c["label"] == "Mixed qubit-qutrit X then P3 on |00>", apply_data["cases"])]
    @assert mixed_case["dims"] == [2, 3] "Mixed qubit-qutrit dims wrong"
    @assert length(mixed_case["output_state_re"]) == 6 "Mixed qubit-qutrit output size wrong"
    @assert mixed_case["output_state_re"] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] "Mixed qubit-qutrit output wrong"
    println("Mixed qubit-qutrit verified.")

    # Verify hadamard test case exists
    ht_case = apply_data["cases"][findfirst(c -> c["label"] == "Hadamard test X phi=0", apply_data["cases"])]
    @assert ht_case["dims"] == [2, 2] "Hadamard test dims wrong"
    @assert length(ht_case["output_state_re"]) == 4 "Hadamard test output size wrong"
    println("Hadamard test case verified.")

    # Verify swap test case exists
    st_case = apply_data["cases"][findfirst(c -> c["label"] == "Swap test nbit=1 nstate=2 phi=0", apply_data["cases"])]
    @assert st_case["dims"] == [2, 2, 2] "Swap test dims wrong"
    @assert length(st_case["output_state_re"]) == 8 "Swap test output size wrong"
    println("Swap test case verified.")

    println()
    println("All validations passed.")
end

main()
