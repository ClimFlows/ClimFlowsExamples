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

# stop as early as possible if output file is already present
ncfile = Base.Filesystem.abspath("$(choices.filename).nc")
@assert !Base.Filesystem.ispath(ncfile) "Output file $ncfile exists, please delete/move it and re-run."

include("create_model.jl")
include("run.jl")

tape = [state0]
if choices.try_gpu && oneAPI.functional()
    oneAPI.versioninfo()
    cpu, gpu = choices.cpu, LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), oneArray)
    simulation(choices, params, model |> gpu, diags, to_lonlat, state0 |> gpu) do state_gpu
        push!(tape, state_gpu |> cpu)
    end;
else
    simulation(choices, params, model, diags, to_lonlat, state0) do state
        push!(tape, state)
    end
end

include("save.jl")
@time save(tape, ncfile) do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end
