# # HPE, mimetic FD on SCVT (Dubos et al., 2015)
# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh

# ## Preamble
using Revise
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

@time_imports using oneAPI, KernelAbstractions, Adapt, ManagedLoops, LoopManagers

include("setup.jl")
include("../../SpectralHPE.jl/NCARL30.jl")
include("params.jl")
include("create_model.jl")
include("run.jl")

gpu = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), oneArray)
model_gpu = model |> gpu
state0_gpu = state0 |> gpu
tape = [state0]
simulation(choices, params, model_gpu, diags, to_lonlat, state0_gpu) do state_gpu
    push!(tape, state_gpu |> PlainCPU())
end;

# tape = simulation(choices, params, model, diags, to_lonlat, state0);

include("save.jl")
save(tape, "$(choices.filename).nc") do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end
