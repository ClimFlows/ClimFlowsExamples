# Fully compressible solver using mimetic finite differences on a spherical Voronoi mesh
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

mgr = cpu

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
let 
    slow, fast, tmp = tendencies!(void, void, void, model_VFCE, state_VFCE, 0., 0.)
    @time tendencies!(slow, fast, tmp, model_VFCE, state_VFCE, 0., 100.)
#    @time advance!(future, scheme_VFCE, state_VFCE, 0.0, 100., tmp)
end; nothing
loop_VFCE = TimeLoopInfo(vsphere, model_VFCE, scheme_VFCE, choices.remap_period, loop_VHPE.dissipation, diags_VFCE, choices.quicklook)
@profview tape = simulation(merge(choices, params, (; ndays=1/2, interval=180)), loop_VFCE, state_VFCE; interp);

#=
include("movie.jl")
@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#

using Interpolations, NetCDF

function get_topo(file)
    topographies = Dict(
        "etopo40.nc" => ("ETOPO40X", "ETOPO40Y", "ROSE"),
        "etopo20.nc" => ("ETOPO20X1_1081", "ETOPO20Y", "ROSE"),
        "etopo5.nc" => ("X", "Y", "bath"),
    )
    lon, lat, h = topographies[file]
    to_rad = pi/180
    file = joinpath(@__DIR__, file)
    lon, lat, h = map(n->NetCDF.open(file, n), (lon, lat, h)) 
    lon, lat, h = to_rad*lon[:], to_rad*lat[:], 9.81*h[:,:]
    return lon.-mean(lon), lat.-mean(lat), @. max(0, h)
end

function to_voronoi(lon, lat, data)
    nlat = size(data, 2)
    itp = linear_interpolation(axes(data), data ; extrapolation_bc=(Periodic(), Line()))    
    to_index, minlon = nlat/pi, minimum(lon)
    lon = @. 1+mod(to_index*(lon-minlon), 2nlat)
    lat = @. 1+to_index*(lat+pi/2)
    return itp.(lon, lat)
end

lon, lat, orog = get_topo("etopo5.nc")
@time orog_i = to_voronoi(vsphere.lon_i, vsphere.lat_i, orog);
heatmap(interp(orog_i))
