#!/usr/bin/env julia

using BenchmarkTools
using JSON
using LinearAlgebra
using Yao
using Yao.EasyBuild

const OUTPUT_DIR = joinpath(@__DIR__, "..", "data")
const STATE_SAMPLES = 100
const STATE_EVALS = 10
const DENSITY_SAMPLES = 50
const DENSITY_EVALS = 5

# Only store full state vectors for n ≤ MAX_STATE_QUBITS (larger states are too big for JSON).
# Timings are stored for all qubit counts.
const MAX_STATE_QUBITS = 5

# yao-rs uses 0-indexed qubits in MSB-first order, while Yao uses 1-indexed
# qubits in LSB-first order. This mapping keeps the generated data aligned with
# the Rust validation tests and existing Julia fixtures in this repository.
yaors_to_yaojl(qubit::Int, nqubits::Int) = nqubits - qubit

function deterministic_state(nqubits::Int)
    dim = 1 << nqubits
    state = ComplexF64[cos(0.1 * k) + sin(0.2 * k) * im for k in 0:(dim - 1)]
    state ./ norm(state)
end

function state_to_interleaved(state::AbstractVector{<:Complex})
    result = Float64[]
    sizehint!(result, 2 * length(state))
    for amp in state
        push!(result, real(amp))
        push!(result, imag(amp))
    end
    result
end

function matrix_to_interleaved(matrix::AbstractMatrix{<:Complex})
    result = Float64[]
    sizehint!(result, 2 * length(matrix))
    for row in axes(matrix, 1), col in axes(matrix, 2)
        value = matrix[row, col]
        push!(result, real(value))
        push!(result, imag(value))
    end
    result
end

function single_gate_1q_data()
    builders = [
        ("X", nq -> put(nq, yaors_to_yaojl(2, nq) => X)),
        ("H", nq -> put(nq, yaors_to_yaojl(2, nq) => H)),
        ("T", nq -> put(nq, yaors_to_yaojl(2, nq) => ConstGate.T)),
        ("Rx_0.5", nq -> put(nq, yaors_to_yaojl(2, nq) => Rx(0.5))),
        ("Rz_0.5", nq -> put(nq, yaors_to_yaojl(2, nq) => Rz(0.5))),
    ]

    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()

    for (name, build_gate) in builders
        gate_data = Dict{String, Any}()
        gate_timing = Dict{String, Any}()
        for nq in 4:25
            println("  1q gate $name, $nq qubits")
            reg = ArrayReg(deterministic_state(nq))
            gate = build_gate(nq)
            result = copy(reg)
            apply!(result, gate)
            if nq <= MAX_STATE_QUBITS
                gate_data[string(nq)] = state_to_interleaved(statevec(result))
            end

            bench = @benchmark apply!(copy($reg), $gate) samples=STATE_SAMPLES evals=STATE_EVALS
            gate_timing[string(nq)] = minimum(bench).time
        end
        data[name] = gate_data
        timing_data[name] = gate_timing
    end

    data, timing_data
end

function single_gate_2q_data()
    builders = [
        (
            "CNOT",
            nq -> control(
                nq,
                yaors_to_yaojl(2, nq),
                yaors_to_yaojl(3, nq) => X,
            ),
        ),
        (
            "CRx_0.5",
            nq -> control(
                nq,
                yaors_to_yaojl(2, nq),
                yaors_to_yaojl(3, nq) => Rx(0.5),
            ),
        ),
    ]

    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()

    for (name, build_gate) in builders
        gate_data = Dict{String, Any}()
        gate_timing = Dict{String, Any}()
        for nq in 4:25
            println("  2q gate $name, $nq qubits")
            reg = ArrayReg(deterministic_state(nq))
            gate = build_gate(nq)
            result = copy(reg)
            apply!(result, gate)
            if nq <= MAX_STATE_QUBITS
                gate_data[string(nq)] = state_to_interleaved(statevec(result))
            end

            bench = @benchmark apply!(copy($reg), $gate) samples=STATE_SAMPLES evals=STATE_EVALS
            gate_timing[string(nq)] = minimum(bench).time
        end
        data[name] = gate_data
        timing_data[name] = gate_timing
    end

    data, timing_data
end

function single_gate_multi_data()
    data = Dict("Toffoli" => Dict{String, Any}())
    timing_data = Dict("Toffoli" => Dict{String, Any}())

    for nq in 4:25
        println("  Toffoli, $nq qubits")
        reg = ArrayReg(deterministic_state(nq))
        gate = control(
            nq,
            (yaors_to_yaojl(2, nq), yaors_to_yaojl(3, nq)),
            yaors_to_yaojl(1, nq) => X,
        )
        result = copy(reg)
        apply!(result, gate)
        if nq <= MAX_STATE_QUBITS
            data["Toffoli"][string(nq)] = state_to_interleaved(statevec(result))
        end

        bench = @benchmark apply!(copy($reg), $gate) samples=STATE_SAMPLES evals=STATE_EVALS
        timing_data["Toffoli"][string(nq)] = minimum(bench).time
    end

    data, timing_data
end

function qft_data()
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()

    for nq in 4:25
        println("  QFT, $nq qubits")
        reg = product_state(nq, 1)
        circuit = qft_circuit(nq)
        result = copy(reg)
        apply!(result, circuit)
        if nq <= MAX_STATE_QUBITS
            data[string(nq)] = state_to_interleaved(statevec(result))
        end

        bench = @benchmark apply!(copy($reg), $circuit) samples=STATE_SAMPLES evals=STATE_EVALS
        timing_data[string(nq)] = minimum(bench).time
    end

    data, timing_data
end

function build_noisy_circuit(nq::Int)
    blocks = AbstractBlock[]

    for q in 1:nq
        push!(blocks, put(nq, q => H))
    end
    for q in 1:(nq - 1)
        push!(blocks, control(nq, q, q + 1 => X))
    end
    for q in 1:nq
        push!(blocks, put(nq, q => quantum_channel(DepolarizingError(1, 0.01))))
    end
    for q in 1:nq
        push!(blocks, put(nq, q => Rz(0.3)))
    end
    for q in 1:nq
        push!(blocks, put(nq, q => quantum_channel(AmplitudeDampingError(0.05, 0.0))))
    end

    chain(nq, blocks...)
end

function build_ising_hamiltonian(nq::Int)
    zz_terms = [put(nq, q => Z) * put(nq, q + 1 => Z) for q in 1:(nq - 1)]
    x_terms = [put(nq, q => X) for q in 1:nq]
    sum(zz_terms) + 0.5 * sum(x_terms)
end

function noisy_dm_data()
    data = Dict{String, Any}()
    timing_data = Dict{String, Any}()

    for nq in 4:10
        println("  Noisy DM, $nq qubits")
        circuit = build_noisy_circuit(nq)
        dm = density_matrix(zero_state(nq))
        dm_result = copy(dm)
        apply!(dm_result, circuit)

        entry = Dict{String, Any}()
        entry["trace"] = real(tr(dm_result.state))
        entry["purity"] = sum(abs2, dm_result.state)

        reduced = partial_tr(dm_result, [1])
        # Compute von Neumann entropy manually from real eigenvalues
        # (Yao's von_neumann_entropy can fail on complex matrices)
        eigs = real.(eigvals(reduced.state))
        eigs = max.(eigs, 0.0)  # clamp small negatives
        entry["entropy"] = -sum(p -> p > 1e-15 ? p * log(p) : 0.0, eigs)

        h_ising = build_ising_hamiltonian(nq)
        exp_val = expect(h_ising, dm_result)
        entry["expect_ising"] = Dict("re" => real(exp_val), "im" => imag(exp_val))

        if nq <= 6
            entry["density_matrix"] = matrix_to_interleaved(dm_result.state)
            entry["reduced_dm"] = matrix_to_interleaved(reduced.state)
        end

        data[string(nq)] = entry

        bench = @benchmark begin
            local_dm = copy($dm)
            apply!(local_dm, $circuit)
        end samples=DENSITY_SAMPLES evals=DENSITY_EVALS
        timing_data[string(nq)] = minimum(bench).time
    end

    data, timing_data
end

function write_json(filename::AbstractString, data)
    open(joinpath(OUTPUT_DIR, filename), "w") do io
        JSON.print(io, data, 2)
    end
end

function main()
    mkpath(OUTPUT_DIR)
    timings = Dict{String, Any}()

    println("=== Task 1: Single Gates ===")
    data_1q, timing_1q = single_gate_1q_data()
    data_2q, timing_2q = single_gate_2q_data()
    data_multi, timing_multi = single_gate_multi_data()
    write_json("single_gate_1q.json", data_1q)
    write_json("single_gate_2q.json", data_2q)
    write_json("single_gate_multi.json", data_multi)
    timings["single_gate_1q"] = timing_1q
    timings["single_gate_2q"] = timing_2q
    timings["single_gate_multi"] = timing_multi

    println("\n=== Task 2: QFT ===")
    data_qft, timing_qft = qft_data()
    write_json("qft.json", data_qft)
    timings["qft"] = timing_qft

    println("\n=== Task 3: Noisy DM ===")
    data_noisy, timing_noisy = noisy_dm_data()
    write_json("noisy_circuit.json", data_noisy)
    timings["noisy_dm"] = timing_noisy

    write_json("timings.json", timings)
    println("\nDone! Generated files in $OUTPUT_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
