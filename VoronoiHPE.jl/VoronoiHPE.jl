# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

include("setup.jl")
include("NCARL30.jl")
include("config.jl")

using CFHydrostatics.Voronoi: Dynamics, Dynamics.hydrostatic_balance_HV!
Dynamics.configure(mgr, ::typeof(hydrostatic_balance_HV!)) = ManagedLoops.configure(mgr, GPUConfig(8,1))

using CFTimeSchemes
CFTimeSchemes.Update.manage(x, mgr::LoopManager) = mgr[x]
function CFTimeSchemes.update!(new, model, old, args...)
    mgr = ManagedLoops.configure(model.mgr, GPUConfig(8,8))
    CFTimeSchemes.Update.update!(new, mgr, old, args...) 
end

# ## stop early if output file is already present
ncfile = Base.Filesystem.abspath("$(choices.filename).nc")
@assert !Base.Filesystem.ispath(ncfile) "Output file $ncfile exists, please delete/move it and re-run."

include("create_model.jl")
include("run.jl")

tape = [state0]
@time simulation(choices, params, gpu, model, diags, to_lonlat, state0) do state_gpu
    # `cpu` adapts GPU arrays to the host but is the identity on CPU arrays.
    # Take a snapshot so that tape entries do not alias the live state.
    push!(tape, deepcopy(state_gpu |> cpu))
end;

include("save.jl")
@time save(tape, ncfile) do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end
