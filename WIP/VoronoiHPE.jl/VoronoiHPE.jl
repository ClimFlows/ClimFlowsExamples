# # HPE, mimetic FD on SCVT (Dubos et al., 2015)
# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

@time_imports using oneAPI, KernelAbstractions, Adapt, ManagedLoops, LoopManagers

include("setup.jl")
include("../../SpectralHPE.jl/NCARL30.jl")
include("params.jl")

# stop as early as possible if output file is already present
ncfile = Base.Filesystem.abspath("$(choices.filename).nc")
# @assert !Base.Filesystem.ispath(ncfile) "Output file $ncfile exists, please delete/move it and re-run."

include("create_model.jl")
include("run.jl")

tape = [state0]
@time simulation(choices, params, gpu, model, diags, to_lonlat, state0) do state_gpu
#    push!(tape, state_gpu |> cpu)
end;

include("save.jl")
@time save(tape, ncfile) do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end
