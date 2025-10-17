# Fully compressible solver using mimetic finite differences on a spherical Voronoi mesh
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

@time_imports using NetCDF: ncread, ncwrite, nccreate, ncclose
@time_imports using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
@time_imports using ClimFlowsPlots: SphericalInterpolations as Interp

include("DCMIP2012_orog.jl")
include("setup.jl");
include("config.jl");
include("run.jl");
include("smooth.jl")

rmap(fun, x) = fun(x)
rmap(fun, x::Vector) = fun.(x)
rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)
rmap(fun, x::Interpolations.Extrapolation) = x

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd

mgr = cpu
# mgr = simd

reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
vsphere = VoronoiSphere(reader; prec=choices.precision)
@showtime vsphere_tree = Interp.spherical_tree(vsphere);
@info vsphere

interp = let
    nlat = choices.nlat
    dlat = 180/nlat
    lons, lats = dlat*(1:2nlat), dlat*(-nlat/2:nlat/2)
    Interp.lonlat_interp(lons, lats, vsphere, vsphere_tree)
end

topo_raw = let 
    _, _, topo = get_topo(choices.etopo)
    lon = range(-pi, pi, size(topo, 1))
    lat = range(-pi/2, pi/2, size(topo, 2))
    Phis = linear_interpolation((lon, lat), topo; extrapolation_bc=(Periodic(), Line()))
    Phis.(vsphere.lon_i, vsphere.lat_i)
end;

Δs = [pi/90, pi/180, pi/360, pi/720, 0.0]
topos = [smoothed(vsphere, topo_raw, Δ) for Δ in Δs];

for (Δ, topo) in zip(Δs, topos)
    @info "smoothing" Δ extrema(topo) maximum(normgrad(vsphere, topo))
    display(heatmap(interp(topo)))
end

function stable_time((choices, params), Δ, topo ; ndays=params.ndays, interval=params.interval)
    params = rmap(Float64, params)
    X = params.X # rescaling factor

    @info "Model setup..." choices params Δ
    Phis = Interp.linear_interpolator(topo/X, vsphere, vsphere_tree)
    testcase = (; Phis, Uplanet = params.radius * params.Omega, params.testcase...)
    params = (; params..., testcase)

    loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
    diags_VHPE, model_VHPE = loop_VHPE.diags, loop_VHPE.model
    state_VHPE =  CFHydrostatics.initial_HPE(case, model_VHPE)

    @time tape = simulation(merge(choices, params, (; ndays, interval)), loop_VHPE, state_VHPE; interp);
    stable_time_HPE = (length(tape)-1)*interval

    display(heatmap(interp(topo)))
    display(heatmap(interp(model_VHPE.Phis)))
    @info "Simulation length" Δ stable_time_HPE

    newton = CFCompressible.NewtonSolve(choices.newton...)
    model_VFCE = CFCompressible.FCE(model_VHPE, params.gravity, params.rhob, newton)
    state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, diags_VHPE, state_VHPE)
    diags_VFCE = CFCompressible.diagnostics(model_VFCE)
    scheme_VFCE = choices.TimeScheme(model_VFCE)
    loop_VFCE = TimeLoopInfo(vsphere, model_VFCE, scheme_VFCE, choices.remap_period, loop_VHPE.dissipation, diags_VFCE, choices.quicklook)

    g = model_VFCE.planet.gravity
    slope = maximum(normgrad(vsphere, topo))/g/model_VFCE.planet.radius

    @showtime tape = simulation(merge(choices, params, (; ndays, interval)), loop_VFCE, state_VFCE; interp);
    stable_time_FCE = (length(tape)-1)*interval

    display(heatmap(interp(model_VFCE.Phis)))
    @info "Simulation length" g slope Δ stable_time_FCE

    X, g, slope, Δ, stable_time_HPE, stable_time_FCE # result of do ... end block
end

stable_times = map(zip(Δs, topos)) do (Δ, topo)
    stable_time(experiment(choices, params), Δ, topo ; interval=3600, ndays=1/24)
end
@info "Model stability" stable_times
