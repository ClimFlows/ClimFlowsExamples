# # HPE solver using spherical harmonics for horizontal discretization

include("setup.jl");
include("run.jl");

@time tape = simulation(merge(choices, params), model, diags, state0);

include("movie.jl")

transp(x::Matrix) = transpose(x)
transp(x) = permutedims(x, (2,1,3))
getvar(sym::Symbol) = diags->transp(getproperty(diags, sym))
getvars(syms...) = NamedTuple{syms}(map(getvar, syms))
getref(sym::Symbol) = var_ref(diags->getproperty(diags, sym))
getrefs(refs) = NamedTuple{propertynames(refs)}(map(getref, Tuple(refs)))

ulat = diags->transp(-diags.uv.ucolat)
dulat = diags->transp(-diags.duv.ucolat)
V850 = var_ref(diags->-diags.uv.ucolat)

vars = getvars(:gradPhi, :geopotential, :pressure, :dpressure, :surface_pressure, :Phi_dot, :Omega)
vars850 = getrefs((T850=:temperature, Omega850=:Omega, W850=:Phi_dot))

@info diags

@time save(tape; ulat, dulat, V850, vars..., vars850...)
exit()

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")

include("scaling.jl")
models = [duplicate_model(choices, params, model, nt) for nt in 1:nthreads];
scaling_tendencies(models, state0)
scaling_RK4(models, state0)
# benchmark(choices, params, sph, [cpu, simd, mgr])
