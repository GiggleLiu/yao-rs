# Full-output validation and warmed timing of the shared Rust/Julia circuits.
using Yao, BenchmarkTools, JSON, LinearAlgebra, Statistics

function build_circuit(spec)
    n = spec["num_qubits"]
    blocks = AbstractBlock[]
    for el in spec["elements"]
        if el["type"] == "channel"
            noise = if el["channel"] == "Depolarizing"
                quantum_channel(DepolarizingError(el["n"], el["p"]))
            elseif el["channel"] == "BitFlip"
                quantum_channel(BitFlipError(el["p"]))
            elseif el["channel"] == "AmplitudeDamping"
                quantum_channel(AmplitudeDampingError(el["gamma"], el["excited_population"]))
            else
                error("unsupported channel $(el["channel"])")
            end
            push!(blocks, put(n, Tuple(n .- el["locs"]) => noise))
            continue
        end
        name = el["gate"]
        params = get(el, "params", Float64[])
        gate = if name == "Rx"
            Rx(params[1])
        elseif name == "Ry"
            Ry(params[1])
        elseif name == "Rz"
            Rz(params[1])
        elseif name == "Phase"
            shift(params[1])
        elseif name == "FSim"
            theta, phi = params
            matblock(ComplexF64[1 0 0 0; 0 cos(theta) -im*sin(theta) 0; 0 -im*sin(theta) cos(theta) 0; 0 0 0 exp(-im*phi)])
        elseif name == "H"
            H
        elseif name == "X"
            X
        elseif name == "S"
            ConstGate.S
        elseif name == "SWAP"
            SWAP
        else
            error("unsupported gate $name")
        end
        targets = Tuple(n .- el["targets"])
        controls = get(el, "controls", [])
        configs = get(el, "control_configs", fill(true, length(controls)))
        # Mapping sites, rather than permuting states, makes complete flat vectors agree.
        if isempty(controls)
            push!(blocks, put(n, targets => gate))
        else
            active = Tuple((n-q)*(config ? 1 : -1) for (q,config) in zip(controls,configs))
            push!(blocks, control(n, active, targets => gate))
        end
    end
    chain(n, blocks...)
end

# The same reversible algorithm used by Yao's expect adjoint, returning the value too.
# Based on YaoBlocks/src/autodiff/specializes.jl (Apache-2.0); the inner product adds the value.
function value_gradient(initial,circuit,op)
    out=apply!(copy(initial),circuit)
    cotangent=apply!(2copy(out),op)
    value=real(dot(statevec(out),statevec(cotangent)))/2
    _,gradient=Yao.AD.apply_back((out,cotangent),circuit)
    value,gradient
end

function evaluate(mode, initial, circuit, op)
    if mode == "gradient"
        value,grad=value_gradient(initial,circuit,op)
        return ComplexF64[value;grad]
    elseif mode in ("expectation", "expectation_dm")
        return ComplexF64[complex_expectation(mode,initial,circuit,op)]
    elseif mode == "custom_gradient"
        return custom_gradient(initial,circuit,op)
    end
    output=apply!(copy(initial),circuit)
    mode == "density" ? vec(permutedims(output.state)) : vec(statevec(output))
end

# Preserve complex polynomial values. Yao expect intentionally projects to real.
# DM formula adapts YaoBlocks blocktools.jl expect without safe_real (Apache-2.0).
function complex_expectation(mode,initial,circuit,op)
    out=apply!(copy(initial),circuit)
    mode == "expectation_dm" ? sum(transpose(out.state).*mat(op)) : sandwich(out,op,out)
end

function custom_target(n)
    values=ComplexF64[cos(0.23*k)+im*sin(0.17*k) for k in 0:((1<<n)-1)]
    values/norm(values)
end

# General real-pairing pullback: d||psi-target||² seeds 2(psi-target).
# Yao apply_back returns the input cotangent and real gate parameters directly.
function custom_gradient(initial,circuit,target)
    out=apply!(copy(initial),circuit)
    delta=vec(statevec(out))-target
    value=sum(abs2,delta)
    (_,input_bar),params=Yao.AD.apply_back((out,ArrayReg(2delta)),circuit)
    ComplexF64[value;params;vec(statevec(input_bar))]
end

# Dense oracle built from Yao Pauli blocks, independent of the emitted circuit.
# Only used for the bounded 3-qubit accuracy fixtures, outside timing regions.
function exact_evolution(spec, n, initial, time)
    h = zeros(ComplexF64, 1<<n, 1<<n)
    paulis = Dict("I"=>I2, "X"=>X, "Y"=>Y, "Z"=>Z)
    for (coefficient, word) in zip(spec["coeffs"], spec["opstrings"])
        block = chain(n, (put(n,n-site=>paulis[op]) for (site,op) in word["ops"])...)
        h .+= complex(coefficient...) .* mat(block)
    end
    exp(-im*time*h) * vec(statevec(initial))
end

function pauli_hamiltonian(spec, n)
    paulis = Dict("I"=>I2, "X"=>X, "Y"=>Y, "Z"=>Z)
    sum(real(complex(coefficient...))*chain(n,
        (put(n,n-site=>paulis[op]) for (site,op) in word["ops"])...)
        for (coefficient,word) in zip(spec["coeffs"],spec["opstrings"]))
end

# Adapted from Yao TimeEvolution (Apache-2.0; see benchmarks/NOTICE.md).
# Same callback and solver options, exposing diagnostics
# outside timing. The real Pauli coefficients establish Hermiticity by construction.
function qualified_krylov(initial, h, time, tol)
    a = Yao.YaoBlocks.BlockMap(ComplexF64, h)
    value, info = Yao.YaoBlocks.exponentiate(a, -im*time, vec(statevec(initial));
        tol=tol, krylovdim=min(1000,size(a,1)), ishermitian=true, eager=true)
    Bool(info.converged) || error("Yao Krylov did not converge: $(info.normres)")
    value, info
end

function main()
    cases_path, reference_dir, output_path=ARGS
    threads=parse(Int,get(ENV,"YAO_BENCH_THREADS","1"))
    BLAS.set_num_threads(threads)
    records=[]
    for case in JSON.parsefile(cases_path)
        spec=case["circuit"]; n=spec["num_qubits"]; mode=case["mode"]
        construction=@benchmark build_circuit($spec) samples=10 evals=1 seconds=0.5
        circuit=build_circuit(spec)
        initial=if case["initial"] == "deterministic"
            values=ComplexF64[cos(0.1*k)+im*sin(0.2*k) for k in 0:((1<<n)-1)]
            ArrayReg(values/norm(values))
        elseif mode in ("density", "expectation_dm")
            density_matrix(zero_state(n))
        else
            zero_state(n)
        end
        if mode == "krylov"
            evolution = case["krylov"]
            h = pauli_hamiltonian(evolution["hamiltonian"], n)
            circuit = time_evolve(h, evolution["time"]; tol=evolution["rtol"], check_hermicity=false)
            construction = @benchmark pauli_hamiltonian($(evolution["hamiltonian"]),$n) samples=10 evals=1 seconds=0.5
        end
        op=if haskey(case,"operator")
            polynomial=case["operator"]
            # Explicit matrices preserve Rust's Pu/Pd convention.
            paulis=Dict("I"=>I2,"X"=>X,"Y"=>Y,"Z"=>Z,
                "P0"=>matblock(ComplexF64[1 0;0 0]), "P1"=>matblock(ComplexF64[0 0;0 1]),
                "Pu"=>matblock(ComplexF64[0 1;0 0]), "Pd"=>matblock(ComplexF64[0 0;1 0]))
            sum(complex(coefficient...)*chain(n,(put(n,n-site=>paulis[name]) for (site,name) in word["ops"])...) for (coefficient,word) in zip(polynomial["coeffs"],polynomial["opstrings"]))
        else
            mode == "custom_gradient" ? custom_target(n) : put(n,n=>Z)
        end
        # Full state/matrix/gradient comparison outside measured closures.
        got=evaluate(mode,initial,circuit,op)
        if mode == "gradient"
            _, public_grad=adjoint(expect)(op,initial=>circuit)
            isapprox(real.(got[2:end]),public_grad;atol=1e-10,rtol=1e-10) || error("public adjoint disagreement")
        end
        bytes=read(joinpath(reference_dir,case["id"]*".bin"))
        expected=reinterpret(ComplexF64,bytes)
        length(got)==length(expected) || error("output length mismatch")
        err=maximum(abs,got-expected)
        agreement = mode == "krylov" ? 10 * case["krylov"]["rtol"] : 1e-10
        isapprox(got,expected;atol=agreement,rtol=agreement) || error("$(case["id"]): error=$err")
        # Rust expect_grad includes one forward value calculation and one backward sweep.
        trial=if mode == "gradient"
            @benchmark value_gradient($initial,$circuit,$op) samples=10 evals=1 seconds=0.5
        elseif mode in ("expectation", "expectation_dm")
            @benchmark complex_expectation($mode,$initial,$circuit,$op) samples=10 evals=1 seconds=0.5
        elseif mode == "custom_gradient"
            @benchmark custom_gradient($initial,$circuit,$op) samples=10 evals=1 seconds=0.5
        else
            @benchmark apply!(copy($initial),$circuit) samples=10 evals=1 seconds=0.5
        end
        record=Dict{String,Any}("id"=>case["id"],"median_ns"=>median(trial).time,
            "samples_ns"=>trial.times,"allocations"=>trial.allocs,"allocated_bytes"=>trial.memory,
            "construction_median_ns"=>median(construction).time,"max_error"=>err)
        if haskey(case, "evolution")
            evolution=case["evolution"]
            exact=exact_evolution(evolution["hamiltonian"],n,initial,evolution["time"])
            record["approximation_error"] = norm(expected-exact)/norm(exact)
            record["yao_approximation_error"] = norm(got-exact)/norm(exact)
        end
        if mode == "krylov"
            evolution = case["krylov"]
            checked, info = qualified_krylov(initial,h,evolution["time"],evolution["rtol"])
            norm(checked-got) < 1e-12 || error("Yao public TimeEvolution disagrees with qualified solver")
            tight, tight_info = qualified_krylov(initial,h,evolution["time"],1e-13)
            oracle = n <= 8 ? exact_evolution(evolution["hamiltonian"],n,initial,evolution["time"]) : tight
            rust_tight = reinterpret(ComplexF64,read(joinpath(reference_dir,case["id"]*".tight.bin")))
            norm(rust_tight-oracle)/norm(oracle) < 1e-11 || error("tight independent evolution references disagree")
            norm(tight-oracle)/norm(oracle) < 1e-11 || error("tight Yao disagrees with dense oracle")
            record["approximation_error"] = norm(expected-oracle)/norm(oracle)
            record["yao_approximation_error"] = norm(got-oracle)/norm(oracle)
            max(record["approximation_error"],record["yao_approximation_error"]) <= 10 * evolution["rtol"] || error("requested Krylov accuracy not achieved")
            record["oracle"] = n <= 8 ? "dense exponential" : "KrylovKit tol=1e-13"
            record["tight_reference_error"] = norm(rust_tight-oracle)/norm(oracle)
            record["yao_krylov"] = Dict("numops"=>info.numops,"numiter"=>info.numiter,"normres"=>info.normres,"converged"=>info.converged,
                "oracle_numops"=>tight_info.numops,"oracle_normres"=>tight_info.normres)
        end
        push!(records,record)
        println(case["id"]," error=",err," median_ns=",median(trial).time)
        open(output_path,"w") do io
            JSON.print(io,Dict("julia"=>string(VERSION),"yao"=>string(pkgversion(Yao)),
                "threads"=>threads,"blas"=>string(BLAS.get_config()),"records"=>records),2)
        end
        GC.gc()
    end
end
main()
