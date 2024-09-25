# # HPE, mimetic FD on SCVT (Dubos et al., 2015)
# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

include("setup.jl")
include("../../SpectralHPE.jl/NCARL30.jl")
include("params.jl")
include("create_model.jl")
include("run.jl")

@info diags
tape = simulation(choices, params, model, diags, to_lonlat, state0);

include("save.jl")
save(tape, "$(choices.filename).nc") do state
    session = open(diags; model, to_lonlat, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end
