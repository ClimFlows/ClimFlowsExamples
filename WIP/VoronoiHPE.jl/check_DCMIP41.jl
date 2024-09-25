# # HPE, mimetic FD on SCVT (Dubos et al., 2015)
# # Hydrostatic primitive equations, mimetic finite differences on a spherical Voronoi mesh

# ## Preamble
using Pkg;
Pkg.activate(@__DIR__);

# from DCMIP41.jl/run.jl

const start_time = time()
toc(str) = "At t=$(time()-start_time)s : $str"

plain(args...) = GFBackends.PlainCPU()
omp(args...) = GFBackends.MainThread(plain())
threads(args...) = GFBackends.MultiThread(plain())
SIMD(args...) = GFBackends.VectorizedCPU(args...)
ompSIMD(args...) = GFBackends.MainThread(SIMD(args...))
autotune() = GFBackends.tune()
tSIMD(args...) = GFBackends.MultiThread(SIMD(args...))
fake(args...) = GFBackends.FakeGPU(tSIMD(args...))

include("DCMIP41.jl/preamble_voronoi.jl")

import GFTimeSchemes as TSchemes
import GFThermodynamics as Thermo
import GFHydrostatics as GFHPE
import GFTestCases
import GFPlanets
import GFModels
import GFLoops
import GFRegistries
import GFExperiments
import GFBackends

# ## Plots & movies

include("DCMIP41.jl/voronoi.jl")

# ## Setup simulation

function init_ps(lon, lat, testcase)
    ps, _ = GFTestCases.initial_surface(lon, lat, testcase)
    return ps
end

function setup_LHPE(
    sphere;
    benchmark = false,
    Float = Float64,
    periods = 24,
    hours_per_period = 1,
    Scheme = RK25,
    courant = 4.0,
    nz = 32,
    nstep = 1,
    testcase = GFTestCases.Jablonowski06,
    backend_init = GFBackends.PlainCPU,
    backend = GFBackends.PlainCPU,
    kwargs...,
)
    ## parameters : override defaults with user-defined and convert numbers to Float
    params = let
        conv(x::NamedTuple) = map(conv, x)
        conv(x::Union{Int,Symbol,Type}) = x
        conv(x::Union{Rational,Float32,Float64}) = Float(x)
        map(conv, merge(GFTestCases.default_params(testcase), kwargs))
    end

    ## setup model on host
    planet, domain, gas, vcoord = let
        radius = params.R0
        planet = GFPlanets.ShallowTradPlanet(radius, params.Uplanet / radius)
        domain = GFDomains.Shell(sphere, nz)
        vcoord = GFHPE.SigmaCoordinate(nz, params.ptop)
        Gas, gas_params... = params.gas
        gas = Gas(gas_params)
        planet, domain, gas, vcoord
    end

    ## initial condition & standard diagnostics (on host)
    state, diags, dt = let
        backend_init = backend_init()
        testcase = testcase(params)
        model =
            GFHPE.LagrangianHPE(backend_init, domain, vcoord, planet, params.ptop, gas)
        state = GFModels.initialize(model, init_ps, GFTestCases.initial_flow, testcase)
        scratch = GFModels.allocate_scratch(model)
        diags = GFModels.diagnostics(
            model;
            domain = domain,
            planet = planet,
            state = state,
            scratch = scratch,
        )

        ## time step is limited by horizontal propagation of acoustic waves
        (temp, p) = diags[:temperature, :pressure]
        maxtemp = maximum(temp)
        c_sound = Thermo.sound_speed(gas, (p = gas.p0, T = maxtemp))
        @info c_sound
        dx = params.R0 * GFDomains.laplace_dx(sphere)
        dt = Float(courant) * dx / c_sound

        @info """

        Maximum temperature = $(typeof(maxtemp)) $maxtemp
        Sound speed = $(typeof(c_sound)) $c_sound
        Effective resolution dx = $(typeof(dx)) $(round(dx/Float(1e3))) km
        Theoretical time step =  $(typeof(dt)) $(round(dt)) s"""
        state, diags, dt
    end

    ## setup model on device
    model, state = let
        backend = backend()
        state_on_device = GFLoops.to_device(state, backend)
        diags.state_on_device = state_on_device
        diags.state = state_on_device -> GFLoops.to_host(state_on_device, backend)
        GFRegistries.reset!(diags)
        temp, p = diags[:temperature, :pressure]
        @info "on-device state" maximum(temp) minimum(p) maximum(p)

        domain_device = GFLoops.to_device(domain, backend)
        model_device =
            GFHPE.LagrangianHPE(backend, domain_device, vcoord, planet, params.ptop, gas)
        @info "model_device" typeof(model_device.phisurf) typeof(
            similar(model_device.domain.layer.Ai),
        )
        copy!(model_device.phisurf, model.phisurf)
        model_device, state_on_device
    end

    if benchmark
        println("====== Starting benchmarks ======")
        benchmarks(model, diags)
        prof(model, Scheme, dt, state)
        println("=================================")
        println()
    end

    return diags, model, state
end

GFoptions = (
    Float = Float64,
    Scheme = TSchemes.RK25,
    courant = 3.4,
    nstep = 3,
    nz = 30, # 40,
    periods = 12 * 12,
    hours_per_period = 2,
    ptop = 1e4,
    R0 = 6.4e6,
    gas = (
        Gas = Thermo.IdealPerfectGas,
        p0 = 1e5,
        T0 = 273.0,
        kappa = 2 // 7,
        Cp = 287.0 * 7 // 2,
        consvar = :temperature,
    ),
)

meshname = "uni.2deg.mesh.nc"

GFsphere = read_mesh(meshname; Float = GFoptions.Float)
GFdiags, GFmodel, GFstate = setup_LHPE(
    GFsphere;
    GFoptions...,
#    backend = ompSIMD,
    nstep = 3,
#    nz = 79,
    hours_per_period = 1,
    periods = 24 * 10,
)
@info "Available diagnostics" GFdiags

convert_to_CF((mass, ucov)) = (; mass_air=mass[:,:,1], mass_consvar=mass[:,:,2], ucov )
GFstate0 = convert_to_CF(GFstate)

Linf(a,b) = maximum(abs, a-b), argmax(abs.(a-b))
Linf(a) = maximum(abs, a)

discrepancy(sym::Symbol, a, b) = discrepancy(getproperty(a, sym), getproperty(b, sym))

function discrepancy(a, b)
    m, i = Linf(a,b)
    m/Linf(a), i
end

compare(a, b)=
for sym in propertynames(a)
    @info "$sym" discrepancy(sym, a, b)
end

function GFtendencies(GFstate)
    state = (mass=cat(GFstate.mass_air, GFstate.mass_consvar; dims=3), GFstate.ucov)
    GFscratch = NamedTuple{:phi, :B, :U, :qv, :qe}(GFModels.allocate_scratch(GFmodel))
    GFdstate = GFModels.allocate_state(GFmodel)
    GFModels.tendencies!(GFdstate, state, GFscratch, GFmodel, GFmodel.backend) |> convert_to_CF, GFscratch
end

tendencies(state) = CFHydrostatics.HPE_tendencies!(void, void, model, vsphere, state, nothing);

pert(a::NamedTuple)=map(pert, a)
pert(a::Array{T}) where T = a + (T(reltol)*maximum(abs, a))*randn(eltype(a), size(a))
reltol = 2e-17

# now back to ClimFlows

include("setup.jl")
include("../../SpectralHPE.jl/NCARL30.jl")
include("params.jl")
include("create_model.jl")

dstate, scratch = tendencies(GFstate0);
GFdstate, GFscratch = GFtendencies(GFstate0);

compare(GFsphere, vsphere)
compare(GFstate0, state0)
compare(GFdstate, dstate)

@info "fcov" discrepancy(model.fcov, GFmodel.fcov)
@info "qe" discrepancy(scratch.PV.PV_e, GFscratch.qe)
@info "B" discrepancy(scratch.Bernoulli.B, GFscratch.B[:,:,1])
@info "consvar" discrepancy(scratch.mass_budget.consvar, GFscratch.B[:,:,2])
@info "exner" discrepancy(scratch.Bernoulli.exner, GFscratch.B[:,:,3])
@info "flux_air" discrepancy(scratch.mass_budget.flux_air, GFscratch.U)

state1 = map(pert, GFstate0);
dstate, scratch = tendencies(GFstate0);
dstate1, scratch1 = tendencies(state1);
compare(dstate, dstate1)
