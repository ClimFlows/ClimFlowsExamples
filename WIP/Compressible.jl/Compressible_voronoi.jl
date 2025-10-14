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

reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere
interp = let
    nlat = choices.nlat
    dlat = 180/nlat
    lons, lats = dlat*(1:2nlat), dlat*(-nlat/2:nlat/2)
    Interp.lonlat_interp(vsphere, lons, lats)
end

@info "Model setup..." choices params

choices, params = experiment(choices, params)
params = rmap(Float64, params)
params_testcase = (Uplanet = params.radius * params.Omega, params.testcase...)
params = (testcase=params_testcase, params...)
newton = CFCompressible.NewtonSolve(choices.newton...)

loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
diags_VHPE, model_VHPE = loop_VHPE.diags, loop_VHPE.model
state_VHPE =  CFHydrostatics.initial_HPE(case, model_VHPE)
# @profview tape = simulation(merge(choices, params, (; ndays=1/2, interval=180)), loop_VHPE, state_VHPE; interp);

let 
    n=3
    Phis = model_VHPE.Phis   
    Ai = vsphere.Ai
    Ai = Ai/mean(Ai)
    @info "no smoothing" extrema(Phis) maximum(normgrad(vsphere, Phis))
    display(heatmap(interp(Phis)))

    for Δ in [pi/720, pi/360, pi/180, pi/90, pi/60]
        h = Helmholtz(Δ^2/n, vsphere, similar(vsphere.le_de))
        Phis_smooth = Phis
        for _ in 1:n
            (Phis_smooth, stats) = cg(h, Ai.*Phis_smooth ; verbose=0)
        end
        @info "smoothing" Δ extrema(Phis_smooth) maximum(normgrad(vsphere, Phis_smooth))
        display(heatmap(interp(Phis_smooth)))
    end
end



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
# @profview tape = simulation(merge(choices, params, (; ndays=1/2, interval=180)), loop_VFCE, state_VFCE; interp);

#=
include("movie.jl")
@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#

#= (x, stats) = cg(A, b::AbstractVector{FC};
                M=I, ldiv::Bool=false, radius::T=zero(T),
                linesearch::Bool=false, atol::T=√eps(T),
                rtol::T=√eps(T), itmax::Int=0,
                timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                callback=workspace->false, iostream::IO=kstdout)
=#
