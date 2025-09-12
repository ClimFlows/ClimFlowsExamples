# Fully compressible solver using spherical harmonics for horizontal discretization
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

@time_imports using CFCompressible
@time_imports using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!
@time_imports using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile

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

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

# initial condition

loop_HPE, case = setup(choices, params, sph, mgr, HPE)
diags_HPE, model_HPE = loop_HPE.diags, loop_HPE.model
state_HPE =  CFHydrostatics.initial_HPE(case, model_HPE)

# run_Kinnmark_Gray(params, choices, sph, mgr)
# @profview tape = simulation(merge(choices, params), loop_HPE, deepcopy(state_HPE));

#======================================================================#

model_FCE = CFCompressible.FCE(model_HPE, params.gravity, params.rhob, newton)
state_FCE = CFCompressible.NH_state.diagnose(model_FCE, diags_HPE, state_HPE)
diags_FCE = CFCompressible.diagnostics(model_FCE)

diags_HPE.specific_volume = (model, pressure, temperature) -> model.gas(:p, :T).specific_volume.(pressure, temperature)
diags_HPE.ulat = uv -> -uv.ucolat
diags_HPE.ulon = uv -> uv.ulon
diags_HPE.slow_fast = (model, state) -> CFTimeSchemes.tendencies!(void, void, void, model, state, 0.0, 0.0)
diags_HPE.slow = slow_fast -> slow_fast[1]
diags_HPE.fast = slow_fast -> slow_fast[2]
diags_HPE.slow_mass_air = (model, slow) -> synthesis_scalar!(void, slow.mass_air_spec, model.domain.layer)

if false
    session_HPE = open(diags_HPE ; model=model_HPE, state=state_HPE)
    session_FCE = open(diags_FCE ; model=model_FCE, state=state_FCE)

    diff(tag) = getproperty(session_HPE, tag)-getproperty(session_FCE, tag)

    function showdiff(tag) 
        var_HPE = getproperty(session_HPE, tag)[:,:,10]
        var_FCE = getproperty(session_FCE, tag)[:,:,10]
        println()
        @info "showing $tag" extrema(var_HPE) extrema(var_FCE) var_HPE ≈ var_FCE
        show(heatmap(var_FCE - var_HPE))
        show(scatterplot(var_FCE[:], var_HPE[:]))
    end

#    @info "check" session_HPE.conservative_variable ≈ session_FCE.conservative_variable
#    @info "check" session_HPE.temperature ≈ session_FCE.temperature
#    showdiff(:conservative_variable)
#    showdiff(:ulon)
#    showdiff(:specific_volume)
#    showdiff(:temperature)
#    showdiff(:Phi_dot)

    showdiff(:slow_mass_air)
end

#======================================================================#

# scheme_FCE = choices.TimeScheme(model_FCE)
# loop_FCE = TimeLoopInfo(sph, model_FCE, scheme_FCE, loop_HPE.remap_period, loop_HPE.dissipation, diags_FCE, choices.quicklook)

#= let 
    tau = 100.0
    slow, fast, scratch = CFTimeSchemes.tendencies!(void, void, void, model_FCE, state_FCE, 0., tau);
    slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau);
    @timev slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau);
    @profview for _ in 1:10
        slow, fast, scratch = CFTimeSchemes.tendencies!(slow, fast, scratch, model_FCE, state_FCE, 0., tau)
    end
end; =#

# tape = simulation(merge(choices, params, (; ndays=1.0, interval=3600)), loop_FCE, state_FCE);
# simulation(merge(choices, params), loop_HPE, state_HPE);

#=
include("movie.jl")
@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")

@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#

using CFDomains: VoronoiSphere
using NetCDF: ncread, ncwrite, nccreate, ncclose
using SHTnsSpheres: synthesis_scalar!

synth(x) = synth(x, eltype(x))
synth(x, ::Type) = x
synth(x, ::Type{<:Complex}) = synthesis_scalar!(void, x, sph)

reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere

loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
diags_VHPE, model_VHPE = loop_VHPE.diags, loop_VHPE.model
state_VHPE =  CFHydrostatics.initial_HPE(case, model_VHPE)

model_VFCE = CFCompressible.FCE(model_VHPE, params.gravity, params.rhob, newton)
state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, diags_VHPE, state_VHPE)
diags_VFCE = CFCompressible.diagnostics(model_VFCE)

tau = 10.0
@time slow_VFCE, fast_VFCE, tmp_VFCE = CFCompressible.tendencies!(void, void, void, model_VFCE, state_VFCE, 0., tau);
@time slow_FCE, fast_FCE, tmp_FCE = CFCompressible.tendencies!(void, void, void, model_FCE, state_FCE, 0., tau);
# CFCompressible.tendencies!(slow, fast, tmp_VFCE, model_VFCE, state_VFCE, 0., 100.);

using ClimFlowsPlots: SphericalInterpolations as Interp
using SHTnsSpheres: synthesis_scalar!
to_deg(rad) = (180/pi)*rad
interp = Interp.lonlat_interp(vsphere, to_deg(sph.lon[1,:]), to_deg(sph.lat[:,1]))

Ldiff(x,y) = round(Linf(x-y)/max(Linf(x),Linf(y)); sigdigits=2)

function vplot(voronoi; lev=10)
    vslice(x::Matrix) = x
    vslice(x::Array{<:Any, 3}) = x[lev, :, :]
    voronoi = vslice(interp(voronoi))
    display(heatmap(voronoi))
end

function diffplot(title, voronoi, sp; lev=10)
#    vslice(x::Matrix) = x[100:300, 1:200]
#    spslice(x::Matrix) = x[100:300, 1:200]
#    vslice(x::Array{<:Any, 3}) = x[1, 100:300, 1:200]
#    spslice(x::Array{<:Any, 3}) = x[100:300, 1:200, 1]
    vslice(x::Matrix) = x
    spslice(x::Matrix) = x
    vslice(x::Array{<:Any, 3}) = x[lev, :, :]
    spslice(x::Array{<:Any, 3}) = x[:, :, lev]


    voronoi = vslice(interp(voronoi))
    sp = spslice(synth(sp))
    display(heatmap(voronoi))
    display(heatmap(sp))
    err = @. (abs(voronoi-sp)+1e-40)
    display(heatmap(err))
    @info "$title : $(Ldiff(voronoi,sp))" Linf(voronoi) Linf(sp) Linf(voronoi-sp) argmax(abs.(voronoi-sp))
end

# fine
diffplot("Wl", tmp_VFCE.Wl, tmp_FCE.common.Wl)
diffplot("Phis", model_VFCE.Phis, model_FCE.Phis)
diffplot("sk", tmp_VFCE.common.sk', tmp_FCE.common.sk)
diffplot("tridiag.A", tmp_VFCE.tridiag.A[:,1,:]', tmp_FCE.tridiag.A)
diffplot("tridiag.B", tmp_VFCE.tridiag.B[:,1,:]', tmp_FCE.tridiag.B)
diffplot("tridiag.R", tmp_VFCE.tridiag.R[:,1,:]', tmp_FCE.tridiag.R)

let lev=10
    display(heatmap(tmp_FCE.Wl_new[:,:,lev]))
    display(heatmap(tmp_FCE.Wl_new[:,:,lev]))
    display(heatmap(tmp_FCE.Wl_new[:,:,lev]-tmp_FCE.common.Wl[:,:,lev]))
    display(heatmap(tau*tmp_FCE.fast_spat.dWl[:,:,lev]))
end

# OK if tau>0
diffplot("new_Phil", tmp_VFCE.new_Phil', tmp_FCE.Phil_new)
diffplot("dPhil", tmp_VFCE.fast_dPhil, tmp_FCE.fast_spat.dPhil)
diffplot("new_Wl", tmp_VFCE.new_Wl', tmp_FCE.Wl_new)
diffplot("Bernoulli", tmp_VFCE.tmp_slow.B, tmp_FCE.slow_mass.fluxes.B)
diffplot("dmass_air", slow_VFCE.mass_air, slow_FCE.mass_air_spec)
diffplot("dmass_consvar", slow_VFCE.mass_consvar, slow_FCE.mass_consvar_spec, lev=5)
diffplot("dW_slow", slow_VFCE.W, slow_FCE.W_spec, lev=5)

# noisy
diffplot("dWl", tmp_VFCE.fast_dWl, tmp_FCE.fast_spat.dWl)
diffplot("new_Wl - old_Wl", tmp_VFCE.new_Wl'-tmp_VFCE.Wl, tmp_FCE.Wl_new-tmp_FCE.common.Wl)
