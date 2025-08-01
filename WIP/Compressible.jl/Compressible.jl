# Fully compressible solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__);
push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)
using Revise

@time_imports using CFCompressible

includet("setup.jl");
include("config.jl");
includet("run.jl");

CFTimeSchemes.tendencies!(slow, fast, scratch, model::CFCompressible.FCE, state, t, dt) = 
    CFCompressible.tendencies!(slow, fast, scratch, model, state, t, dt )

function CFTimeSchemes.model_dstate(::CFCompressible.FCE, state, _)
    rsim(x::AbstractArray) = similar(x)
    rsim(x::NamedTuple) = map(rsim, x)
    return rsim(state)
end

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd

# mgr = cpu

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

params = map(Float64, params)
@info "Model setup..." choices params
params = (Uplanet = params.radius * params.Omega, params...)

# initial condition
loop_HPE, case = setup(choices, params, sph, mgr, HPE)
(; diags, model) = loop_HPE

#======================================================================#

let choices = merge(choices, (TimeScheme=CFTimeSchemes.KinnmarkGray{2,5}, ndays=1)),
    params = merge(params, (; courant=4.0))
    loop_HPE, case = setup(choices, params, sph, mgr, HPE)
    (; diags, model) = loop_HPE
    state_HPE =  CFHydrostatics.initial_HPE(case, model)
    state0 = deepcopy(state_HPE)
    @time tape = simulation(merge(choices, params), loop_HPE, state0);
end;

loop_HPE, case = setup(choices, params, sph, mgr, HPE)
(; diags, model) = loop_HPE
state_HPE =  CFHydrostatics.initial_HPE(case, model)
@profview tape = simulation(merge(choices, params), loop_HPE, deepcopy(state_HPE));

ps = open(diags; model, state=state_HPE).surface_pressure
model_FCE = CFCompressible.FCE(model, params.gravity, ps, params.rhob)
state_FCE = CFCompressible.NH_state.diagnose(model_FCE, diags, state_HPE)
scheme_FCE = choices.TimeScheme(model_FCE)

diags_FCE = CFCompressible.diagnostics(model_FCE)
let session = open(diags_FCE ; model=model_FCE, state=state_FCE)
    @info "check" extrema(session.temperature)
    @info "check" extrema(session.conservative_variable)
    @info "check" extrema(session.pressure)
end

loop_FCE = TimeLoopInfo(sph, model_FCE, scheme_FCE, loop_HPE.remap_period, loop_HPE.dissipation, diags_FCE)
@time tape = simulation(merge(choices, params), loop_FCE, state_FCE);

#=
include("movie.jl")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)
exit()

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#
