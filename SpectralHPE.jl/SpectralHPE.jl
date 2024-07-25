# # HPE solver using spherical harmonics for horizontal discretization

include("setup.jl")
include("run.jl")

benchmark(choices, params, sph, [cpu, simd, MultiThread(cpu, nthreads), MultiThread(simd, nthreads)])

@time simulation(merge(choices, params), model, diags, state0; ndays=5)

