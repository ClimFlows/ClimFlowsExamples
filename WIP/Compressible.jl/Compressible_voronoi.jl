# Fully compressible solver using mimetic finite differences on a spherical Voronoi mesh
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
using Serialization

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
nthreads = Threads.nthreads()
vlen(F) = div(32, sizeof(F))
cpu, simd = PlainCPU(), VectorizedCPU(vlen(choices.precision))

# mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd
# mgr = cpu
mgr = simd

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

# Δs = [pi/90, pi/180, pi/360, pi/720, 0.0]
# Δs = [pi/90, pi/720, 0.0]
Δs = [0.0]
topos = [smoothed(vsphere, topo_raw, Δ) for Δ in Δs];

for (Δ, topo) in zip(Δs, topos)
    @info "smoothing" Δ extrema(topo) maximum(normgrad(vsphere, topo))
    display(heatmap(interp(topo)))
end

function stable_time((choices, params), Δ, topo ; ndays=params.ndays, interval=params.interval)
    plotmap(interp(topo), "Unscaled, smoothed surface geopotential")

    params = rmap(choices.precision, params)
    (; Xfactor, Zfactor) = params # rescaling factors

    @info "Model setup..." choices params Δ
    Phis = Interp.linear_interpolator((Zfactor/Xfactor)*topo, vsphere, vsphere_tree)
    testcase = (; Phis, radius=params.radius, Uplanet = params.radius * params.Omega, params.testcase...)
    params = (; params..., testcase)

    loop_VHPE, case = setup(choices, params, vsphere, cpu, HPE)
    diags_VHPE, model_VHPE = loop_VHPE.diags, loop_VHPE.model
    state_VHPE =  CFHydrostatics.initial_HPE(case, model_VHPE)

    newton = CFCompressible.NewtonSolve(choices.newton...)
    loop_VHPE_simd, _ = setup(choices, params, vsphere, simd, HPE)
    model_VFCE = CFCompressible.FCE(loop_VHPE_simd.model, params.gravity, params.rhob, newton)

    state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, diags_VHPE, state_VHPE)
    diags_VFCE = CFCompressible.diagnostics(model_VFCE)
    scheme_VFCE = choices.TimeScheme(model_VFCE)
    loop_VFCE = TimeLoopInfo(vsphere, model_VFCE, scheme_VFCE, choices.remap_period, loop_VHPE.dissipation, diags_VFCE, choices.quicklook)

    (; gravity, radius) = model_VFCE.planet
    zsurf_i = model_VFCE.Phis/gravity
    slope = maximum(normgrad(vsphere, zsurf_i))/radius
    zsurf = interp(zsurf_i)
    plotmap(zsurf, "Surface height used by FCE model - max slope $slope")
    lineplot(zsurf[1:div(size(zsurf,1),3), div(size(zsurf,2),2)])

    @showtime tape_FCE = simulation(merge(choices, params, (; ndays, interval)), loop_VFCE, state_VFCE; interp);
    stable_time_FCE = (length(tape_FCE)-1)*interval
    @info "FCE simulation length" radius maximum(zsurf) gravity slope Δ stable_time_FCE

    plotmap(interp(model_VHPE.Phis), "Surface geopotential used by HPE model")
    tape_HPE = simulation(merge(choices, params, (; ndays, interval)), loop_VHPE, state_VHPE; interp);
    stable_time_HPE = (length(tape_HPE)-1)*interval
    @info "HPE simulation length" Δ stable_time_HPE

    return (; Xfactor, Zfactor, gravity, slope, Δ, stable_time_HPE, stable_time_FCE, params, scheme_VFCE, tape_FCE)
end

stable_times = map(zip(Δs, topos)) do (Δ, topo)
    stable_time(experiment(choices, params), choices.precision(Δ), topo)
end
@info "Model stability" stable_times

serialize(joinpath(@__DIR__, "output_compressible_voronoi.jld"), stable_times[1])
