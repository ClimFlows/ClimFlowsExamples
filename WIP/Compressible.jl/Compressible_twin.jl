# Fully compressible solver using spherical harmonics for horizontal discretization
using InteractiveUtils

using Pkg; Pkg.activate(@__DIR__);
# push!(LOAD_PATH, Base.Filesystem.joinpath(@__DIR__, "packages")); unique!(LOAD_PATH)

@time_imports using CFCompressible
@time_imports using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!
@time_imports using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
@time_imports using NetCDF: ncread, ncwrite, nccreate, ncclose
@time_imports using ClimFlowsPlots: SphericalInterpolations as Interp

include("setup.jl");
include("config.jl");
include("run.jl");

rmap(fun, x) = fun(x)
rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)

synth(x) = synth(x, eltype(x))
synth(x, ::Type) = x
synth(x, ::Type{<:Complex}) = synthesis_scalar!(void, x, sph)

to_deg(rad) = (180/pi)*rad

Ldiff(x,y) = round(Linf(x-y)/max(Linf(x),Linf(y)); sigdigits=2)

function vplot(voronoi; lev=10)
    vslice(x::Matrix) = x
    vslice(x::Array{<:Any, 3}) = x[lev, :, :]
    voronoi = vslice(interp(voronoi))
    display(heatmap(voronoi))
end

function diffplot(title, voronoi, sp; lev=10)
    vslice(x::Matrix) = x
    spslice(x::Matrix) = x
    vslice(x::Array{<:Any, 3}) = x[lev, :, :]
    spslice(x::Array{<:Any, 3}) = x[:, :, lev]

    voronoi = vslice(interp(voronoi))
    sp = spslice(synth(sp))
    err = @. (abs(voronoi-sp)+1e-40)
    @info "$title : $(Ldiff(voronoi,sp))" Linf(voronoi) Linf(sp) Linf(voronoi-sp) argmax(abs.(voronoi-sp))
    display(heatmap(voronoi))
    display(heatmap(sp))
    display(scatterplot(sp[:], voronoi[:]))
    display(heatmap(err))
end

struct CompressibleTwin{V,S,I}
    voronoi::V
    spectral::S
    interp::I
end

#============================  main program =========================#

threadinfo()
nthreads = 1 # Threads.nthreads()
cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = (nthreads>1) ? MultiThread(simd, nthreads) : simd
# mgr = cpu

choices, params = experiment(choices, params)
params = rmap(Float64, params)
params_testcase = (Uplanet = params.radius * params.Omega, params.testcase...)
params = (testcase=params_testcase, params...)    

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
@time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

model, state = let # twin model and initial condition
    @info "Initializing Voronoi mesh..."
    reader = DYNAMICO_reader(ncread, DYNAMICO_meshfile(choices.meshname))
    vsphere = VoronoiSphere(reader; prec=choices.precision)
    @info vsphere
    interp = Interp.lonlat_interp(vsphere, to_deg(sph.lon[1,:]), to_deg(sph.lat[:,1]))
    newton = CFCompressible.NewtonSolve(choices.newton...)
    loop_HPE, case = setup(choices, params, sph, mgr, HPE)
    loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
    model_FCE = CFCompressible.FCE(loop_HPE.model, params.gravity, params.rhob, newton)
    model_VFCE = CFCompressible.FCE(loop_VHPE.model, params.gravity, params.rhob, newton)

    state_HPE =  CFHydrostatics.initial_HPE(case, loop_HPE.model)
    state_VHPE =  CFHydrostatics.initial_HPE(case, loop_VHPE.model)
    state_FCE = CFCompressible.NH_state.diagnose(model_FCE, loop_HPE.diags, state_HPE)
    state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, loop_VHPE.diags, state_VHPE)

    CompressibleTwin(model_VFCE, model_FCE, interp), (voronoi=state_VFCE, spectral=state_FCE)
end

function CFTimeSchemes.tendencies!(slow, fast, tmp, model::CompressibleTwin, state, t, tau)
    vslow, vfast, vtmp = CFCompressible.tendencies!(slow.voronoi, fast.voronoi, tmp.voronoi, model.voronoi, state.voronoi, t, tau)
    sslow, sfast, stmp = CFCompressible.tendencies!(slow.spectral, fast.spectral, tmp.spectral, model.spectral, state.spectral, t, tau)

    stamp(str) = "$str (t=$t, τ=$tau)"
    # checks
    if rand()<0.05 # 1 in 20
        # diffplot(stamp("Wl"), vtmp.Wl, stmp.common.Wl)
        # diffplot(stamp("Phis"), model.voronoi.Phis, model.spectral.Phis)
        # diffplot(stamp("sk"), vtmp.common.sk', stmp.common.sk)
        # diffplot(stamp("tridiag.A"), vtmp.tridiag.A[:,1,:]', stmp.tridiag.A)
        # diffplot(stamp("tridiag.B"), vtmp.tridiag.B[:,1,:]', stmp.tridiag.B)
        # diffplot(stamp("tridiag.R"), vtmp.tridiag.R[:,1,:]', stmp.tridiag.R)
        # diffplot(stamp("new_Phil"), vtmp.new_Phil', stmp.Phil_new)
        # diffplot(stamp("dPhil"), vtmp.fast_dPhil, stmp.fast_spat.dPhil)
        # diffplot(stamp("new_Wl"), vtmp.new_Wl', stmp.Wl_new)
        # diffplot(stamp("Bernoulli"), vtmp.tmp_slow.B, stmp.slow_mass.fluxes.B)
        diffplot(stamp("dmass_air"), vslow.mass_air, sslow.mass_air_spec)
        # diffplot(stamp("dmass_consvar"), vslow.mass_consvar, sslow.mass_consvar_spec, lev=5)
        # diffplot(stamp("dW_slow"), vslow.W, sslow.W_spec, lev=5)
        # diffplot(stamp("dPhi_slow"), vslow.Phi, sslow.Phi_spec, lev=5)

        # noisy
        # diffplot(stamp("dWl"), vtmp.fast_dWl, stmp.fast_spat.dWl)
    end
    return (voronoi=vslow, spectral=sslow), (voronoi=vfast, spectral=sfast), (voronoi=vtmp, spectral=stmp)
end

interp = model.interp # global var used by diffplot

@time slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 0.0);
@time slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 100.0);

scheme = choices.TimeScheme(model)
solver = CFTimeSchemes.IVPSolver(scheme, 360.0)
@time future, t = CFTimeSchemes.advance!(void, solver, state, 0.0, 10*24*10)

# CFCompressible.tendencies!(slow, fast, tmp_VFCE, model_VFCE, state_VFCE, 0., 100.);

# loop = TimeLoopInfo(sph, model, scheme, choices.remap_period, nothing, nothing, choices.quicklook)
# tape = simulation(merge(choices, params, (; ndays=1.0, interval=3600)), loop, state)
# simulation(merge(choices, params), loop_HPE, state_HPE);

#=
include("movie.jl")
@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
s
@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_FCE, diags_FCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#
