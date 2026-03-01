#!/usr/bin/env julia
#
# Generate ground-truth noise channel test data for yao-rs using Yao.jl.
#
# Usage: julia tests/fixtures/generate_noise.jl
#
# Generates: tests/data/noise.json
#

using Yao
using JSON
using LinearAlgebra

const OUTPUT_DIR = joinpath(@__DIR__, "..", "data")

rnd(x::Real) = Float64(x)

"""Convert a complex matrix to {re: [[...]], im: [[...]]} dict."""
function matrix_to_dict(m::AbstractMatrix{<:Complex})
    re = [[rnd(real(m[i,j])) for j in 1:size(m,2)] for i in 1:size(m,1)]
    im = [[rnd(imag(m[i,j])) for j in 1:size(m,2)] for i in 1:size(m,1)]
    return Dict("re" => re, "im" => im)
end

"""Get Kraus operator matrices from a noise error type."""
function get_kraus_matrices(err)
    ch = quantum_channel(err)
    if ch isa MixedUnitaryChannel
        # MixedUnitaryChannel stores unscaled operators + probabilities
        ops = Vector{Matrix{ComplexF64}}()
        for (p, op) in zip(ch.probs, ch.operators)
            m = sqrt(p) * Matrix(mat(ComplexF64, op))
            push!(ops, m)
        end
        return ops
    elseif ch isa KrausChannel
        # KrausChannel operators are already scaled
        return [Matrix(mat(ComplexF64, op)) for op in subblocks(ch)]
    elseif ch isa DepolarizingChannel
        # DepolarizingChannel is special - construct Kraus manually
        # Single-qubit: equivalent to PauliError(p/4, p/4, p/4)
        p = ch.p
        n = ch.n
        if n == 1
            return get_kraus_matrices(PauliError(p/4, p/4, p/4))
        else
            error("Multi-qubit depolarizing fixture generation not implemented")
        end
    else
        error("Unknown channel type: $(typeof(ch))")
    end
end

"""Compute superoperator S = sum_i kron(conj(K_i), K_i)."""
function compute_superop(kraus_ops)
    d = size(kraus_ops[1], 1)
    S = zeros(ComplexF64, d^2, d^2)
    for K in kraus_ops
        S .+= kron(conj(K), K)
    end
    return S
end

"""Apply noise channel to a density matrix rho."""
function apply_channel(kraus_ops, rho)
    result = zeros(ComplexF64, size(rho))
    for K in kraus_ops
        result .+= K * rho * K'
    end
    return result
end

function main()
    mkpath(OUTPUT_DIR)
    data = Dict()

    # ================================================================
    # 1. Kraus operators for each noise type
    # ================================================================
    kraus_cases = [
        ("bit_flip_0.1", BitFlipError(0.1)),
        ("phase_flip_0.2", PhaseFlipError(0.2)),
        ("pauli_0.1_0.2_0.05", PauliError(0.1, 0.2, 0.05)),
        ("amplitude_damping_0.3_0.0", AmplitudeDampingError(0.3, 0.0)),
        ("amplitude_damping_0.3_0.4", AmplitudeDampingError(0.3, 0.4)),
        ("phase_damping_0.2", PhaseDampingError(0.2)),
        ("phase_amp_damping_0.3_0.2_0.0", PhaseAmplitudeDampingError(0.3, 0.2, 0.0)),
        ("phase_amp_damping_0.3_0.2_0.4", PhaseAmplitudeDampingError(0.3, 0.2, 0.4)),
        ("thermal_relaxation_100_80_10_0.0", ThermalRelaxationError(100.0, 80.0, 10.0, 0.0)),
        ("depolarizing_1_0.1", PauliError(0.025, 0.025, 0.025)),
    ]

    kraus_data = Dict()
    for (name, err) in kraus_cases
        println("Generating Kraus for $name...")
        kraus_ops = get_kraus_matrices(err)
        superop = compute_superop(kraus_ops)

        # Verify completeness
        d = size(kraus_ops[1], 1)
        completeness = sum(K' * K for K in kraus_ops)
        cerr = maximum(abs.(completeness - Matrix{ComplexF64}(I, d, d)))
        @assert cerr < 1e-10 "Completeness failed for $name: err=$cerr"

        kraus_data[name] = Dict(
            "num_kraus" => length(kraus_ops),
            "kraus" => [matrix_to_dict(K) for K in kraus_ops],
            "superop" => matrix_to_dict(superop),
        )
    end
    data["kraus"] = kraus_data

    # ================================================================
    # 2. DM evolution test cases
    # ================================================================
    dm_cases = Dict()
    I2 = Matrix{ComplexF64}(I, 2, 2)

    # Helper states
    rho_plus = ComplexF64[0.5 0.5; 0.5 0.5]   # |+><+| = H|0><0|H
    rho_zero = ComplexF64[1.0 0.0; 0.0 0.0]   # |0><0|

    # Case 1: H then PhaseFlip(0.1)
    println("Generating DM: H + PhaseFlip...")
    kraus_pf = get_kraus_matrices(PhaseFlipError(0.1))
    rho_pf = apply_channel(kraus_pf, rho_plus)
    dm_cases["h_phaseflip_0.1"] = Dict("dims" => [2], "rho" => matrix_to_dict(rho_pf))

    # Case 2: BitFlip(0.1) on |0>
    println("Generating DM: BitFlip on |0>...")
    kraus_bf = get_kraus_matrices(BitFlipError(0.1))
    rho_bf_zero = apply_channel(kraus_bf, rho_zero)
    dm_cases["bitflip_0.1_on_zero"] = Dict("dims" => [2], "rho" => matrix_to_dict(rho_bf_zero))

    # Case 3: H then Depolarizing(1, 0.1)
    println("Generating DM: H + Depolarizing...")
    kraus_dep = get_kraus_matrices(PauliError(0.025, 0.025, 0.025))
    rho_dep = apply_channel(kraus_dep, rho_plus)
    dm_cases["h_depolarizing_0.1"] = Dict("dims" => [2], "rho" => matrix_to_dict(rho_dep))

    # Case 4: Bell state + Depolarizing on q0
    println("Generating DM: Bell + Depolarizing on q0...")
    bell = zeros(ComplexF64, 4)
    bell[1] = 1/sqrt(2)
    bell[4] = 1/sqrt(2)
    rho_bell = bell * bell'
    kraus_bell = [kron(K, I2) for K in kraus_dep]
    rho_bell_dep = apply_channel(kraus_bell, rho_bell)
    dm_cases["bell_depolarizing_q0_0.1"] = Dict("dims" => [2, 2], "rho" => matrix_to_dict(rho_bell_dep))

    # Case 5: H then AmplitudeDamping(0.3)
    println("Generating DM: H + AmplitudeDamping...")
    kraus_ad = get_kraus_matrices(AmplitudeDampingError(0.3, 0.0))
    rho_ad = apply_channel(kraus_ad, rho_plus)
    dm_cases["h_amplitude_damping_0.3"] = Dict("dims" => [2], "rho" => matrix_to_dict(rho_ad))

    data["dm_cases"] = dm_cases

    # ================================================================
    # 3. Expectation value test cases
    # ================================================================
    expectation_cases = Dict()
    Z_mat = ComplexF64[1.0 0.0; 0.0 -1.0]
    X_mat = ComplexF64[0.0 1.0; 1.0 0.0]

    expectation_cases["z_h_pure"] = Dict(
        "value_re" => rnd(real(tr(Z_mat * rho_plus))),
        "value_im" => rnd(imag(tr(Z_mat * rho_plus))),
    )
    expectation_cases["x_h_depolarizing_0.1"] = Dict(
        "value_re" => rnd(real(tr(X_mat * rho_dep))),
        "value_im" => rnd(imag(tr(X_mat * rho_dep))),
    )
    expectation_cases["z_h_phaseflip_0.1"] = Dict(
        "value_re" => rnd(real(tr(Z_mat * rho_pf))),
        "value_im" => rnd(imag(tr(Z_mat * rho_pf))),
    )

    data["expectations"] = expectation_cases

    # Write JSON
    open(joinpath(OUTPUT_DIR, "noise.json"), "w") do f
        JSON.print(f, data, 4)
    end
    println()
    println("Generated tests/data/noise.json successfully.")
    println("All Kraus operators satisfy completeness relation.")
end

main()
