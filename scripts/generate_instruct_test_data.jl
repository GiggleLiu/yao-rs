#!/usr/bin/env julia
#
# Generate ground-truth test data for yao-rs qubit instruct functions.
# Runs the actual Julia instruct! tests and stores input/output pairs as JSON.
#
# Usage: julia scripts/generate_instruct_test_data.jl
#
# Generates: tests/data/instruct.json
#
# NOTE: Julia uses 1-indexed bit positions. yao-rs uses 0-indexed.
# We store locs in 0-indexed (yao-rs convention) in the JSON.
# The Julia code converts internally.
#
# Reference: ~/.julia/dev/Yao/lib/YaoArrayRegister/test/instruct.jl
#            ~/.julia/dev/BitBasis/src/

using Random
using LinearAlgebra
using JSON
using Yao
using YaoArrayRegister: linop2dense, general_controlled_gates, parametric_mat, instruct!
using LuxurySparse, SparseArrays

Random.seed!(42)

const OUTPUT_DIR = joinpath(@__DIR__, "..", "tests", "data")

# ============================================================================
# Utility
# ============================================================================

function c2pair(z::Complex)
    return [Float64(real(z)), Float64(imag(z))]
end

function vec_to_json(v::AbstractVector{<:Complex})
    return [c2pair(x) for x in v]
end

function mat_to_json(m::AbstractMatrix{<:Complex})
    return [[c2pair(m[i,j]) for j in 1:size(m,2)] for i in 1:size(m,1)]
end

# Convert 0-indexed Rust loc to 1-indexed Julia loc for n qubits
# NOTE: Julia YaoArrayRegister uses 1-indexed positions where
# position 1 = least significant bit.
# yao-rs uses 0-indexed positions where position 0 = most significant bit.
# For n qubits: rust_loc i -> julia_loc (n - i)
rust_to_julia_loc(loc::Int, n::Int) = n - loc

function make_test_case(;
    label::String,
    nbits::Int,
    input_state::Vector{ComplexF64},
    output_state::Vector{ComplexF64},
    gate_matrix=nothing,
    locs_0indexed::Vector{Int}=Int[],
    ctrl_locs_0indexed::Vector{Int}=Int[],
    ctrl_bits::Vector{Int}=Int[],
    gate_name::String="",
    theta::Float64=0.0,
)
    d = Dict{String,Any}(
        "label" => label,
        "nbits" => nbits,
        "input_state" => vec_to_json(input_state),
        "output_state" => vec_to_json(output_state),
        "locs" => locs_0indexed,
    )
    if !isempty(ctrl_locs_0indexed)
        d["ctrl_locs"] = ctrl_locs_0indexed
        d["ctrl_bits"] = ctrl_bits
    end
    if gate_matrix !== nothing
        d["gate_matrix"] = mat_to_json(Matrix{ComplexF64}(gate_matrix))
    end
    if !isempty(gate_name)
        d["gate_name"] = gate_name
    end
    if theta != 0.0
        d["theta"] = theta
    end
    return d
end

# ============================================================================
# Test case generators — mirrors Julia test/instruct.jl
# ============================================================================

function generate_cases()
    cases = Dict{String,Any}[]

    # ------------------------------------------------------------------
    # 1. General unitary instruction (test/instruct.jl:7-50)
    # ------------------------------------------------------------------

    # Random 2x2 U on 4-qubit state at Julia loc=3 (Rust loc=1)
    U1 = randn(ComplexF64, 2, 2)
    ST = randn(ComplexF64, 1 << 4)
    n = 4
    jl_loc = 3  # Julia 1-indexed
    rs_loc = n - jl_loc  # = 1, Rust 0-indexed

    out = instruct!(Val(2), copy(ST), U1, (jl_loc,))
    push!(cases, make_test_case(
        label="1q generic U at loc=$rs_loc on 4 qubits",
        nbits=n, input_state=copy(ST), output_state=out,
        gate_matrix=U1, locs_0indexed=[rs_loc],
    ))

    # Random 4x4 U on 4-qubit state at Julia locs=(2,3) -> Rust locs=(2,1)
    U2 = rand(ComplexF64, 4, 4)
    out = instruct!(Val(2), copy(ST), U2, (2, 3))
    push!(cases, make_test_case(
        label="2q generic U at locs=[2,1] on 4 qubits",
        nbits=n, input_state=copy(ST), output_state=out,
        gate_matrix=U2, locs_0indexed=[n-2, n-3],  # [2,1]
    ))

    # Separability: kron(U1,U1) on (3,1) == U1 on 3 then U1 on 1
    out_sep = instruct!(Val(2), instruct!(Val(2), copy(ST), U1, (3,)), U1, (1,))
    out_kron = instruct!(Val(2), copy(ST), kron(U1, U1), (3, 1))
    @assert out_sep ≈ out_kron "Separability test failed"
    push!(cases, make_test_case(
        label="2q separable kron(U,U) at locs=[1,3] on 4 qubits",
        nbits=n, input_state=copy(ST), output_state=out_kron,
        gate_matrix=kron(U1, U1), locs_0indexed=[n-3, n-1],  # [1,3]
    ))

    # Identity gate is no-op
    push!(cases, make_test_case(
        label="1q identity at loc=0 on 4 qubits",
        nbits=n, input_state=copy(ST),
        output_state=instruct!(Val(2), copy(ST), Matrix{ComplexF64}(I, 2, 2), (1,)),
        gate_matrix=Matrix{ComplexF64}(I, 2, 2), locs_0indexed=[n-1],
    ))

    # ------------------------------------------------------------------
    # 2. Controlled unitary (test/instruct.jl:68-86)
    # ------------------------------------------------------------------
    ST5 = randn(ComplexF64, 1 << 5)
    U1_ctrl = randn(ComplexF64, 2, 2)
    n5 = 5

    # Single control, active-high: ctrl=1, tgt=3 (Julia) -> ctrl=4, tgt=2 (Rust)
    out = instruct!(Val(2), copy(ST5), U1_ctrl, (3,), (1,), (1,))
    push!(cases, make_test_case(
        label="controlled 1q U, ctrl=4(high) tgt=2 on 5 qubits",
        nbits=n5, input_state=copy(ST5), output_state=out,
        gate_matrix=U1_ctrl, locs_0indexed=[n5-3],
        ctrl_locs_0indexed=[n5-1], ctrl_bits=[1],
    ))

    # Single control, active-low: ctrl=1(low), tgt=3 (Julia) -> ctrl=4(low), tgt=2 (Rust)
    out = instruct!(Val(2), copy(ST5), U1_ctrl, (3,), (1,), (0,))
    push!(cases, make_test_case(
        label="controlled 1q U, ctrl=4(low) tgt=2 on 5 qubits",
        nbits=n5, input_state=copy(ST5), output_state=out,
        gate_matrix=U1_ctrl, locs_0indexed=[n5-3],
        ctrl_locs_0indexed=[n5-1], ctrl_bits=[0],
    ))

    # Control 2-target gate: ctrl=1, tgt=(3,4) (Julia) -> ctrl=4, tgt=(2,1) (Rust)
    U2_ctrl = kron(U1_ctrl, U1_ctrl)
    out = instruct!(Val(2), copy(ST5), U2_ctrl, (3, 4), (1,), (1,))
    push!(cases, make_test_case(
        label="controlled 2q U, ctrl=4(high) tgt=[2,1] on 5 qubits",
        nbits=n5, input_state=copy(ST5), output_state=out,
        gate_matrix=U2_ctrl, locs_0indexed=[n5-3, n5-4],
        ctrl_locs_0indexed=[n5-1], ctrl_bits=[1],
    ))

    # Multi-control: ctrl=(5,1), bits=(1,0), tgt=(3,4) (Julia)
    # -> ctrl=(0,4), bits=(1,0), tgt=(2,1) (Rust)
    out = instruct!(Val(2), copy(ST5), U2_ctrl, (3, 4), (5, 1), (1, 0))
    push!(cases, make_test_case(
        label="multi-ctrl 2q U, ctrl=[0,4](1,0) tgt=[2,1] on 5 qubits",
        nbits=n5, input_state=copy(ST5), output_state=out,
        gate_matrix=U2_ctrl, locs_0indexed=[n5-3, n5-4],
        ctrl_locs_0indexed=[n5-5, n5-1], ctrl_bits=[1, 0],
    ))

    # ------------------------------------------------------------------
    # 3. Pauli gate instructions (test/instruct.jl:89-102)
    # ------------------------------------------------------------------
    for (G, Gname) in [(:X, "X"), (:Y, "Y"), (:Z, "Z")]
        M = if G == :X; [0.0+0im 1; 1 0] elseif G == :Y; [0.0+0im -im; im 0] else [1.0+0im 0; 0 -1] end

        # Single qubit: linop2dense equivalent
        # Apply G to |0> and |1>
        st0 = ComplexF64[1, 0]
        st1 = ComplexF64[0, 1]
        out0 = instruct!(Val(2), copy(st0), Val(G), (1,))
        out1 = instruct!(Val(2), copy(st1), Val(G), (1,))
        push!(cases, make_test_case(
            label="$Gname on |0> (1 qubit)",
            nbits=1, input_state=st0, output_state=out0,
            gate_name=Gname, locs_0indexed=[0],
        ))
        push!(cases, make_test_case(
            label="$Gname on |1> (1 qubit)",
            nbits=1, input_state=st1, output_state=out1,
            gate_name=Gname, locs_0indexed=[0],
        ))

        # Triple Pauli: G on sites (1,2,3) of 3 qubits = kron(G,G,G)
        st3 = randn(ComplexF64, 1 << 3)
        out3 = instruct!(Val(2), copy(st3), Val(G), (1, 2, 3))
        # Apply one by one for comparison
        out3_sep = copy(st3)
        for jl_loc in [1, 2, 3]
            instruct!(Val(2), out3_sep, Val(G), (jl_loc,))
        end
        @assert out3 ≈ out3_sep "Triple $Gname mismatch"
        push!(cases, make_test_case(
            label="$Gname on all 3 sites (3 qubits)",
            nbits=3, input_state=copy(st3), output_state=out3,
            gate_name=Gname, locs_0indexed=[2, 1, 0],
        ))

        # Controlled Pauli: ctrl=(2,1), bits=(0,1), tgt=4 on 4 qubits (Julia)
        # -> ctrl=(2,3), bits=(0,1), tgt=0 on 4 qubits (Rust)
        st4 = randn(ComplexF64, 1 << 4)
        out4 = instruct!(Val(2), copy(st4), Val(G), (4,), (2, 1), (0, 1))
        push!(cases, make_test_case(
            label="controlled $Gname, ctrl=[2,3](0,1) tgt=0 on 4 qubits",
            nbits=4, input_state=copy(st4), output_state=out4,
            gate_name=Gname, locs_0indexed=[0],
            ctrl_locs_0indexed=[2, 3], ctrl_bits=[0, 1],
        ))

        # Controlled Pauli: ctrl=2(low), tgt=1 on 2 qubits (Julia)
        # -> ctrl=0(low), tgt=1 on 2 qubits (Rust)
        st2 = randn(ComplexF64, 1 << 2)
        out2 = instruct!(Val(2), copy(st2), Val(G), (1,), (2,), (0,))
        push!(cases, make_test_case(
            label="controlled $Gname, ctrl=0(low) tgt=1 on 2 qubits",
            nbits=2, input_state=copy(st2), output_state=out2,
            gate_name=Gname, locs_0indexed=[1],
            ctrl_locs_0indexed=[0], ctrl_bits=[0],
        ))
    end

    # ------------------------------------------------------------------
    # 4. Diagonal gates: Z, S, T (test/instruct.jl:104-115, 265-304)
    # ------------------------------------------------------------------
    ST4 = randn(ComplexF64, 1 << 4)
    n4 = 4

    # Random diagonal 2x2 at Julia loc=3 -> Rust loc=1
    Dv = Diagonal(randn(ComplexF64, 2))
    out = instruct!(Val(2), copy(ST4), Dv, (3,))
    push!(cases, make_test_case(
        label="1q diagonal at loc=1 on 4 qubits",
        nbits=n4, input_state=copy(ST4), output_state=out,
        gate_matrix=Matrix(Dv), locs_0indexed=[1],
    ))

    # Named diagonal gates: Z, S, T on single qubit
    for (Gname, Gval) in [("Z", :Z), ("S", :S), ("T", :T)]
        st = randn(ComplexF64, 1 << 4)
        # Julia loc=3 -> Rust loc=1
        out = instruct!(Val(2), copy(st), Val(Gval), (3,))
        push!(cases, make_test_case(
            label="$Gname at loc=1 on 4 qubits",
            nbits=4, input_state=copy(st), output_state=out,
            gate_name=Gname, locs_0indexed=[1],
        ))
    end

    # Controlled Z, S, T
    for (Gname, Gval) in [("Z", :Z), ("S", :S), ("T", :T)]
        st = randn(ComplexF64, 1 << 4)
        # Julia: ctrl=1(high), tgt=3 -> Rust: ctrl=3(high), tgt=1
        out = instruct!(Val(2), copy(st), Val(Gval), (3,), (1,), (1,))
        push!(cases, make_test_case(
            label="controlled $Gname, ctrl=3(high) tgt=1 on 4 qubits",
            nbits=4, input_state=copy(st), output_state=out,
            gate_name=Gname, locs_0indexed=[1],
            ctrl_locs_0indexed=[3], ctrl_bits=[1],
        ))
    end

    # ------------------------------------------------------------------
    # 5. SWAP instruction (test/instruct.jl:117-120)
    # ------------------------------------------------------------------
    ST_swap = randn(ComplexF64, 1 << 2)
    out_swap = instruct!(Val(2), copy(ST_swap), Val(:SWAP), (1, 2))
    push!(cases, make_test_case(
        label="SWAP at locs=[1,0] on 2 qubits",
        nbits=2, input_state=copy(ST_swap), output_state=out_swap,
        gate_name="SWAP", locs_0indexed=[1, 0],
    ))

    # SWAP on larger system
    ST_swap5 = randn(ComplexF64, 1 << 5)
    # Julia (2,4) -> Rust (3,1)
    out_swap5 = instruct!(Val(2), copy(ST_swap5), Val(:SWAP), (2, 4))
    push!(cases, make_test_case(
        label="SWAP at locs=[3,1] on 5 qubits",
        nbits=5, input_state=copy(ST_swap5), output_state=out_swap5,
        gate_name="SWAP", locs_0indexed=[3, 1],
    ))

    # ------------------------------------------------------------------
    # 6. Rotation gates: Rx, Ry, Rz (test/instruct.jl:122-152)
    # ------------------------------------------------------------------
    θ = π / 3
    for (Rname, Rval) in [("Rx", :Rx), ("Ry", :Ry), ("Rz", :Rz)]
        ST5r = randn(ComplexF64, 1 << 5)

        # Without controls: Julia loc=4 -> Rust loc=1
        out_r = instruct!(Val(2), copy(ST5r), Val(Rval), (4,), θ)
        push!(cases, make_test_case(
            label="$Rname(pi/3) at loc=1 on 5 qubits",
            nbits=5, input_state=copy(ST5r), output_state=out_r,
            gate_name=Rname, locs_0indexed=[1], theta=θ,
        ))

        # With control: Julia ctrl=1(low), loc=4, theta
        # -> Rust ctrl=4(low), loc=1
        out_rc = instruct!(Val(2), copy(ST5r), Val(Rval), (4,), (1,), (0,), θ)
        push!(cases, make_test_case(
            label="controlled $Rname(pi/3), ctrl=4(low) tgt=1 on 5 qubits",
            nbits=5, input_state=copy(ST5r), output_state=out_rc,
            gate_name=Rname, locs_0indexed=[1],
            ctrl_locs_0indexed=[4], ctrl_bits=[0], theta=θ,
        ))
    end

    # ------------------------------------------------------------------
    # 7. PSWAP and CPHASE (test/instruct.jl:122-152)
    # ------------------------------------------------------------------
    for (Rname, Rval) in [("PSWAP", :PSWAP), ("CPHASE", :CPHASE)]
        ST5p = randn(ComplexF64, 1 << 5)

        # Without controls: Julia locs=(4,2) -> Rust locs=(1,3)
        out_p = instruct!(Val(2), copy(ST5p), Val(Rval), (4, 2), θ)
        push!(cases, make_test_case(
            label="$Rname(pi/3) at locs=[1,3] on 5 qubits",
            nbits=5, input_state=copy(ST5p), output_state=out_p,
            gate_name=Rname, locs_0indexed=[1, 3], theta=θ,
        ))

        # With control: Julia ctrl=1(low), locs=(4,2)
        # -> Rust ctrl=4(low), locs=(1,3)
        out_pc = instruct!(Val(2), copy(ST5p), Val(Rval), (4, 2), (1,), (0,), θ)
        push!(cases, make_test_case(
            label="controlled $Rname(pi/3), ctrl=4(low) tgt=[1,3] on 5 qubits",
            nbits=5, input_state=copy(ST5p), output_state=out_pc,
            gate_name=Rname, locs_0indexed=[1, 3],
            ctrl_locs_0indexed=[4], ctrl_bits=[0], theta=θ,
        ))
    end

    # ------------------------------------------------------------------
    # 8. H gate via matrix (test/instruct.jl:136)
    # ------------------------------------------------------------------
    ST5h = randn(ComplexF64, 1 << 5)
    out_h = instruct!(Val(2), copy(ST5h), Val(:H), (4,))
    push!(cases, make_test_case(
        label="H at loc=1 on 5 qubits",
        nbits=5, input_state=copy(ST5h), output_state=out_h,
        gate_name="H", locs_0indexed=[1],
    ))

    # ------------------------------------------------------------------
    # 9. Multi-threading regression (test/instruct.jl:181-202)
    #    Fixed seed for reproducibility
    # ------------------------------------------------------------------
    g = ComplexF64[
        0.921061-0.389418im 0.0+0.0im 0.0+0.0im 0.0+0.0im
        0.0+0.0im 0.921061-0.0im 0.0+0.0im 0.0-0.389418im
        0.0+0.0im 0.0+0.0im 0.921061-0.389418im 0.0+0.0im
        0.0+0.0im 0.0-0.389418im 0.0+0.0im 0.921061-0.0im
    ]
    n_reg = 10  # Use 10 qubits (not 16) to keep JSON small
    Random.seed!(123)
    reg1 = rand_state(n_reg)
    input_reg = copy(statevec(reg1))

    # Generate deterministic random pairs
    Random.seed!(456)
    pairs = Tuple{Int,Int}[]
    for _ in 1:20
        x1 = rand(1:n_reg)
        x2 = rand(1:n_reg-1)
        x2 = x2 >= x1 ? x2 + 1 : x2
        push!(pairs, (x1, x2))
    end

    # Apply all gates
    for (x1, x2) in pairs
        instruct!(reg1, g, (x1, x2))
    end
    output_reg = statevec(reg1)

    # Convert Julia locs (1-indexed) to Rust locs (0-indexed)
    pairs_rust = [[n_reg - x1, n_reg - x2] for (x1, x2) in pairs]

    push!(cases, Dict{String,Any}(
        "label" => "regression: 20 random 2q gates on $n_reg qubits",
        "nbits" => n_reg,
        "input_state" => vec_to_json(input_reg),
        "output_state" => vec_to_json(output_reg),
        "gate_matrix" => mat_to_json(g),
        "gate_pairs" => pairs_rust,
    ))

    # ------------------------------------------------------------------
    # 10. Measurement: marginal probabilities
    # ------------------------------------------------------------------
    # Bell state: H on qubit 1 (Julia loc=2), CNOT ctrl=2 tgt=1 (Julia)
    bell_reg = zero_state(2)
    apply!(bell_reg, chain(2, put(2, 2=>H), control(2, 2, 1=>X)))
    bell_sv = statevec(bell_reg)

    # Full probs
    bell_probs = probs(bell_reg)
    push!(cases, Dict{String,Any}(
        "label" => "measure: Bell state full probs",
        "nbits" => 2,
        "input_state" => vec_to_json(bell_sv),
        "probabilities" => Float64.(bell_probs),
    ))

    # ------------------------------------------------------------------
    # 11. Various gate sizes on different positions
    # ------------------------------------------------------------------
    # 1q gate at each position of a 4-qubit system
    U_test = randn(ComplexF64, 2, 2)
    ST_pos = randn(ComplexF64, 1 << 4)
    for jl_loc in 1:4
        rs_loc = 4 - jl_loc
        out = instruct!(Val(2), copy(ST_pos), U_test, (jl_loc,))
        push!(cases, make_test_case(
            label="1q U at loc=$rs_loc on 4 qubits (position sweep)",
            nbits=4, input_state=copy(ST_pos), output_state=out,
            gate_matrix=U_test, locs_0indexed=[rs_loc],
        ))
    end

    # 2q gate at various positions of a 5-qubit system
    U2_test = randn(ComplexF64, 4, 4)
    ST_pos5 = randn(ComplexF64, 1 << 5)
    for (jl1, jl2) in [(1,2), (1,5), (2,4), (3,5)]
        rs1, rs2 = 5 - jl1, 5 - jl2
        out = instruct!(Val(2), copy(ST_pos5), U2_test, (jl1, jl2))
        push!(cases, make_test_case(
            label="2q U at locs=[$rs1,$rs2] on 5 qubits (position sweep)",
            nbits=5, input_state=copy(ST_pos5), output_state=out,
            gate_matrix=U2_test, locs_0indexed=[rs1, rs2],
        ))
    end

    return cases
end

# ============================================================================
# Main
# ============================================================================

function main()
    mkpath(OUTPUT_DIR)
    println("Generating instruct test data...")

    cases = generate_cases()

    data = Dict("cases" => cases)
    outfile = joinpath(OUTPUT_DIR, "instruct.json")
    open(outfile, "w") do f
        JSON.print(f, data, 1)  # indent=1 for readability
    end

    println("Generated $(length(cases)) test cases -> $outfile")

    # Quick validation
    println("\n=== Validation ===")
    for (i, c) in enumerate(cases)
        label = c["label"]
        n = c["nbits"]
        expected_len = 1 << n
        if haskey(c, "probabilities")
            @assert length(c["probabilities"]) == expected_len "Prob length mismatch in case $i: $label"
        elseif haskey(c, "gate_pairs")
            # Regression test case with gate_pairs
            inp = c["input_state"]
            out = c["output_state"]
            @assert length(inp) == expected_len "Input length mismatch in case $i: $label"
            @assert length(out) == expected_len "Output length mismatch in case $i: $label"
        else
            inp = c["input_state"]
            out = c["output_state"]
            @assert length(inp) == expected_len "Input length mismatch in case $i: $label (got $(length(inp)), expected $expected_len)"
            @assert length(out) == expected_len "Output length mismatch in case $i: $label (got $(length(out)), expected $expected_len)"
        end
    end
    println("All $(length(cases)) cases validated.")
end

main()
