# Fully compressible solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__);
push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)
using Revise

using CFCompressible

includet("setup.jl");
include("config.jl");
includet("run.jl");

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd

mgr = cpu

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

compressible = CFCompressible.FCE(model, params.gravity)
state_FCE = CFCompressible.NH_state.diagnose(compressible, diags, state_HPE)
# loop_FCE, case = setup(choices, params, sph, mgr, CFCompressible.FCE)

# @time tape = simulation(merge(choices, params), loop_HPE, state0);

#=
include("movie.jl")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)
exit()

@time movie(model, diags, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#
