# Fully compressible solver using spherical harmonics for horizontal discretization
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

@time_imports using NetCDF: ncread, ncwrite, nccreate, ncclose
@time_imports using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
@time_imports using ClimFlowsPlots: SphericalInterpolations as Interp

include("setup.jl");
include("config.jl");
include("run.jl");

rmap(fun, x) = fun(x)
rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd

# mgr = cpu

@info "Model setup..." choices params
choices, params = experiment(choices, params)
params = rmap(Float64, params)
params_testcase = (Uplanet = params.radius * params.Omega, params.testcase...)
params = (testcase=params_testcase, params...)
newton = CFCompressible.NewtonSolve(choices.newton...)

reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere
interp = let
    nlat = choices.nlat
    dlat = 180/nlat
    lons, lats = dlat*(1:2nlat), dlat*(-nlat/2:nlat/2)
    Interp.lonlat_interp(vsphere, lons, lats)
end


loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
diags_VHPE, model_VHPE = loop_VHPE.diags, loop_VHPE.model
state_VHPE =  CFHydrostatics.initial_HPE(case, model_VHPE)

model_VFCE = CFCompressible.FCE(model_VHPE, params.gravity, params.rhob, newton)
state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, diags_VHPE, state_VHPE)
diags_VFCE = CFCompressible.diagnostics(model_VFCE)

# to_deg(rad) = (180/pi)*rad
# interp = Interp.lonlat_interp(vsphere, to_deg(sph.lon[1,:]), to_deg(sph.lat[:,1]))

scheme_VFCE = choices.TimeScheme(model_VFCE)
loop_VFCE = TimeLoopInfo(vsphere, model_VFCE, scheme_VFCE, choices.remap_period, loop_VHPE.dissipation, diags_VFCE, choices.quicklook)
tape = simulation(merge(choices, params, (; ndays=1.0, interval=3600)), loop_VFCE, state_VFCE; interp);

#=
include("movie.jl")
@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#

