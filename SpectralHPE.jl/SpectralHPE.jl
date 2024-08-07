# # HPE solver using spherical harmonics for horizontal discretization

include("setup.jl")

include("run.jl")

# benchmark(choices, params, sph, [cpu, simd, MultiThread(cpu, nthreads), MultiThread(simd, nthreads)])

@time tape = simulation(merge(choices, params), model, diags, state0; ndays = 10);

include("movie.jl")

T850 = var_ref(diags->diags.temperature)
Omega850 = var_ref(diags->diags.Omega)
W850 = var_ref(diags->diags.Phi_dot)
V850 = var_ref(diags->-diags.uv.ucolat)

@time save(tape; T850, W850, Omega850, V850)

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
