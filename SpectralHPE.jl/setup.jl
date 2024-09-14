using Pkg;
Pkg.activate(@__DIR__);
Pkg.status();
using InteractiveUtils

@time_imports begin
    using ThreadPinning
    pinthreads(:cores)
    using SIMDMathFunctions
    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd

    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, initial_flow, initial_surface

    using UnicodePlots: heatmap
end

# let CFTimeSchemes use our multi-thread manager when updating the model state
@inline CFTimeSchemes.update!(new, model::HPE, old, args...) = CFTimeSchemes.Update.update!(new, model.mgr, old, args...)
@inline CFTimeSchemes.Update.manage(a::Array{<:Complex}, mgr::LoopManager) = no_simd(mgr)[a]

include("NCARL30.jl")

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere,Dyn,Scheme,Filter,Diags}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    remap_period::Int
    dissipation::Filter
    diags::Diags
end

#============== model setup =============#

function setup(choices, params, sph, mgr)
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    hd_n, hd_nu = params.hyperdiff_n, params.hyperdiff_nu
    # stuff independent from initial condition
    gas = params.Fluid(params)
    # vcoord = SigmaCoordinate(params.nz, params.ptop)
    vcoord = NCARL30(params.nz, params.ptop)

    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    scheme = choices.TimeScheme(model)
    dissip = (
        zeta = HyperDiffusion(model.domain, hd_n, hd_nu, :vector_curl),
        theta = HyperDiffusion(model.domain, hd_n, hd_nu, :scalar),
    )
    diags = diagnostics(model)
    info = TimeLoopInfo(sph, model, scheme, choices.remap_period, dissip, diags)

    # initial condition
    state = let
        init(lon, lat) = initial_surface(lon, lat, case)
        init(lon, lat, p) = initial_flow(lon, lat, p, case)
        CFHydrostatics.initial_HPE(init, model)
    end

    return info, state
end

#=======================  main program =====================#

choices = (
    Fluid = IdealPerfectGas,
    TimeScheme = KinnmarkGray{2,5},
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
    hyperdiff_n = 2,
    remap_period = 5,
    nlat = 96,
    ndays = 5,
)
params = (
    ptop = 225.52395239472398,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    hyperdiff_nu = 0.002,
    courant = 4.0,
    interval = 6 * 3600, # 6-hour intervals
)

threadinfo()

nthreads = Threads.nthreads()

cpu, simd = PlainCPU(), VectorizedCPU(8)
mgr = MultiThread(simd, nthreads)

@info "Initializing spherical harmonics..."
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

@info "Model setup..." choices params

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)

info, state0 = setup(choices, params, sph, mgr)
(; diags, model) = info
