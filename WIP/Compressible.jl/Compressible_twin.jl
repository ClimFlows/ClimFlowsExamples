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
    maxval = max(Linf(voronoi), Linf(sp))
    err = @. (abs(voronoi-sp)+1e-40)/maxval
    @info "$title : $(Ldiff(voronoi,sp))" Linf(voronoi) Linf(sp) Linf(voronoi-sp) argmax(abs.(voronoi-sp))
    display(heatmap(voronoi ; title="$title (voronoi)"))
    display(heatmap(sp ; title="$title (spectral)"))
    display(scatterplot(sp[:], voronoi[:]))
    display(heatmap(err, title="$title (relative difference)"))
end

struct TwinModels{V,S,I}
    voronoi::V
    spectral::S
    interp::I
end

struct FourModels{VH, VC,SH, SC, I}
    VHPE::VH
    VFCE::VC
    SHPE::SH
    SFCE::SC
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
    wfactor = params.wfactor
    loop_SHPE, case = setup(choices, params, sph, mgr, HPE)
    loop_VHPE, case = setup(choices, params, vsphere, mgr, HPE)
    model_SFCE = CFCompressible.FCE(loop_SHPE.model, params.gravity, params.rhob, newton)
    model_VFCE = CFCompressible.FCE(loop_VHPE.model, params.gravity, params.rhob, newton)

    state_SHPE =  CFHydrostatics.initial_HPE(case, loop_SHPE.model)
    state_VHPE =  CFHydrostatics.initial_HPE(case, loop_VHPE.model)
    state_SFCE = CFCompressible.NH_state.diagnose(model_SFCE, loop_SHPE.diags, state_SHPE ; wfactor)
    state_VFCE = CFCompressible.NH_state.diagnose(model_VFCE, loop_VHPE.diags, state_VHPE ; wfactor)

#    TwinModels(model_VFCE, model_SFCE, interp), (voronoi=state_VFCE, spectral=state_SFCE)
#    TwinModels(loop_VHPE.model, loop_SHPE.model, interp), (voronoi=state_VHPE, spectral=state_SHPE)
    FourModels(loop_VHPE.model, model_VFCE, loop_SHPE.model, model_SFCE, interp), (VHPE=state_VHPE, VFCE=state_VFCE, SHPE=state_SHPE, SFCE=state_SFCE)
end;
interp = model.interp # global var used by diffplot

function CFTimeSchemes.tendencies!(slow, fast, tmp, model::TwinModels, state, t, tau)
    stamp(str) = "$str (t=$t, τ=$tau)"
    vslow, vfast, vtmp = CFTimeSchemes.tendencies!(slow.voronoi, fast.voronoi, tmp.voronoi, model.voronoi, state.voronoi, t, tau)
    sslow, sfast, stmp = CFTimeSchemes.tendencies!(slow.spectral, fast.spectral, tmp.spectral, model.spectral, state.spectral, t, tau)

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
        diffplot(stamp("Bernoulli"), vtmp.fast.fast_B, stmp.locals_duv_fast.B_spec)
        # diffplot(stamp("KE"), vtmp.slow.KE, stmp.locals_slow.B_spec)
        diffplot(stamp("dmass_air"), vslow.mass_air, sslow.mass_air_spec)
        diffplot(stamp("dmass_consvar"), vslow.mass_consvar, sslow.mass_consvar_spec)
        # diffplot(stamp("dmass_consvar"), vslow.mass_consvar, sslow.mass_consvar_spec, lev=5)
        # diffplot(stamp("dW_slow"), vslow.W, sslow.W_spec, lev=5)
        # diffplot(stamp("dPhi_slow"), vslow.Phi, sslow.Phi_spec, lev=5)

        # noisy
        # diffplot(stamp("dWl"), vtmp.fast_dWl, stmp.fast_spat.dWl)
    end
    return (voronoi=vslow, spectral=sslow), (voronoi=vfast, spectral=sfast), (voronoi=vtmp, spectral=stmp)
end

function CFTimeSchemes.tendencies!(slow, fast, tmp, model::FourModels, state, t, tau)
    stamp(str) = "$str (t=$t, τ=$tau)"

    VHslow, VHfast, VHtmp = CFTimeSchemes.tendencies!(slow.VHPE, fast.VHPE, tmp.VHPE, model.VHPE, state.VHPE, t, tau)
    VCslow, VCfast, VCtmp = CFTimeSchemes.tendencies!(slow.VFCE, fast.VFCE, tmp.VFCE, model.VFCE, state.VFCE, t, tau)
    SHslow, SHfast, SHtmp = CFTimeSchemes.tendencies!(slow.SHPE, fast.SHPE, tmp.SHPE, model.SHPE, state.SHPE, t, tau)
    SCslow, SCfast, SCtmp = CFTimeSchemes.tendencies!(slow.SFCE, fast.SFCE, tmp.SFCE, model.SFCE, state.SFCE, t, tau)

    g = model.VFCE.planet.gravity
    # checks
#    if rand()<0.05 # 1 in 20
#        diffplot(stamp("Bernoulli (HPE)"), VHtmp.fast.fast_B, SHtmp.locals_duv_fast.B_spec)
#        diffplot(stamp("Bernoulli (FCE)"), VCtmp.fast_VH.dHdm, SCtmp.fast_uv.dHdm_spec)
#        diffplot(stamp("Δ(Bernoulli)"), VCtmp.fast_VH.dHdm-VHtmp.fast.fast_B, SCtmp.fast_uv.dHdm_spec-SHtmp.locals_duv_fast.B_spec)
         diffplot(stamp("KE (HPE)"), VHtmp.slow.KE, SHtmp.locals_slow.B_spec)
         diffplot(stamp("slow B (FCE)"), VCtmp.tmp_slow.B, SCtmp.slow_curl_form.B_spec)
         diffplot(stamp("Δ(slow B)"), VCtmp.tmp_slow.B-VHtmp.slow.KE, SCtmp.slow_curl_form.B_spec-SHtmp.locals_slow.B_spec)
#        diffplot(stamp("∂ₜmass_air (HPE)"), VHslow.mass_air, SHslow.mass_air_spec)
#        diffplot(stamp("∂ₜmass_air (FCE)"), VCslow.mass_air, SCslow.mass_air_spec)
#        diffplot(stamp("Δ(∂ₜmass_air)"), g*VCslow.mass_air-VHslow.mass_air, g*SCslow.mass_air_spec-SHslow.mass_air_spec)
        # diffplot(stamp("dmass_consvar"), VHslow.mass_consvar, SHslow.mass_consvar_spec)
        # diffplot(stamp("dmass_consvar"), vslow.mass_consvar, sslow.mass_consvar_spec, lev=5)
        # diffplot(stamp("dW_slow"), vslow.W, sslow.W_spec, lev=5)
        # diffplot(stamp("dPhi_slow"), vslow.Phi, sslow.Phi_spec, lev=5)

#    end
    return (; VHPE=VHslow, VFCE=VCslow, SHPE=SHslow, SFCE=SCslow, ),
            (; VHPE=VHfast, VFCE=VCfast, SHPE=SHfast, SFCE=SCfast, ),
            (; VHPE=VHtmp, VFCE=VCtmp, SHPE=SHtmp, SFCE=SCtmp, )
end
@time slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 100.);


#=
@time slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 0.);
@time slow, fast, tmp = CFTimeSchemes.tendencies!(slow, fast, tmp, model, state, 0., 0.);
@time slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 100.);
@time slow, fast, tmp = CFTimeSchemes.tendencies!(slow, fast, tmp, model, state, 0., 100.);

let dt=360., N=2
    slow, fast, tmp = CFTimeSchemes.tendencies!(void, void, void, model, state, 0., 0.);
    dstate, _ = CFTimeSchemes.tendencies!(void, void, model.voronoi, state.voronoi, 0.); # explicit API
    vplot(dstate.mass_air - slow.voronoi.mass_air)
    @info extrema(slow.voronoi.ucov + fast.voronoi.ucov - dstate.ucov)

    scheme = CFTimeSchemes.KinnmarkGray{2,5}(model.voronoi)
    solver = CFTimeSchemes.IVPSolver(scheme, dt , state.voronoi, 0.)
    @time future, t = CFTimeSchemes.advance!(void, solver, state.voronoi, 0.0, 1)
    @time future, t = CFTimeSchemes.advance!(future, solver, future, 0.0, N-1)

    scheme = choices.TimeScheme(model.voronoi)
    solver = CFTimeSchemes.IVPSolver(scheme, dt)
    @time future, t = CFTimeSchemes.advance!(void, solver, state.voronoi, 0.0, N)
end

@profview CFTimeSchemes.tendencies!(slow.voronoi, fast.voronoi, tmp.voronoi, model.voronoi, state.voronoi, 0., 100.);
=#

scheme = choices.TimeScheme(model)
solver = CFTimeSchemes.IVPSolver(scheme, 360.0)
# @time future, t = CFTimeSchemes.advance!(void, solver, state, 0.0, 10*24)
@time future, t = CFTimeSchemes.advance!(void, solver, state, 0.0, 10*24*10)

# CFCompressible.tendencies!(slow, fast, tmp_VFCE, model_VFCE, state_VFCE, 0., 100.);

# loop = TimeLoopInfo(sph, model, scheme, choices.remap_period, nothing, nothing, choices.quicklook)
# tape = simulation(merge(choices, params, (; ndays=1.0, interval=3600)), loop, state)
# simulation(merge(choices, params), loop_SHPE, state_SHPE);

#=
include("movie.jl")
@time movie(model_SFCE, diags_SFCE, tape, T850; filename = "T850.mp4")
s
@info diags

@time save(tape; dmass, ps, T850, W850, Omega850, V850, W, Omega, dulat, pressure, geopotential)

@time movie(model_SFCE, diags_SFCE, tape, T850; filename = "T850.mp4")
@time movie(model, diags, tape, Omega850; filename = "Omega850.mp4")
@time movie(model, diags, tape, W850; filename = "W850.mp4")
=#
