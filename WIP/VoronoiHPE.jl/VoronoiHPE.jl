# # HPE, mimetic FD on SCVT
# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh

# ## Preamble
using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

include("setup.jl")

include("params.jl")
params = map(choices.precision, params)

sphere =
    VoronoiSphere(DYNAMICO_reader(ncread, choices.meshname); prec = choices.precision)
@info sphere

model, diags, state0 = setup(sphere, choices, params);
interp = let
    lons, lats = collect(choices.lons), collect(choices.lats)
    ClimFlowsPlots.SphericalInterpolations.lonlat_interp(sphere, lons, lats)
end

tape = [state0]

include("save.jl")

save(tape, choices.filename) do state
    session = open(diags; model, interp, state)
    return ((sym, getproperty(session, sym)) for sym in choices.outputs)
end

exit()

include("run.jl")
solver! = solver(choices, params, model, state0)

@info "Macro time step = $(solver!.dt) s"
@info "Interval = $(3600*params.hours_per_period) s"
