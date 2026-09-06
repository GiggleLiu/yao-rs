# Run with: julia --project=~/.julia/dev/Yao scripts/generate_ad_reference.jl
# Julia sites count from the LSB (1-based); Rust sites count from the MSB (0-based).
using Yao

circuit = chain(2,
    put(2, 2 => H),
    put(2, 1 => H),
    put(2, 2 => Rx(0.31)),
    put(2, 1 => Ry(-0.47)),
    control(2, 2, 1 => Rz(0.22)),
    control(2, -1, 2 => shift(-0.63)),
)
observable = put(2, 2 => Z) + 0.5 * put(2, 1 => X)
initial = zero_state(2)
value = real(expect(observable, initial => circuit))
_, gradient = adjoint(expect)(observable, initial => circuit)
println("{\"value\":", value, ",\"parameters\":[", join(parameters(circuit), ","),
    "],\"gradient\":[", join(gradient, ","), "]}")
