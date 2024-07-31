using Pkg;
Pkg.activate(@__DIR__);
Pkg.status();
using InteractiveUtils

using ThreadPinning
pinthreads(:cores)

@time_imports begin
    using SIMDMathFunctions
    using LoopManagers: PlainCPU, VectorizedCPU, MultiThread, tune

    using CFTimeSchemes: RungeKutta4, IVPSolver, advance!
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, initial_flow, initial_surface

    using UnicodePlots: heatmap
end

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere,Dyn,Scheme,Filter,Diags}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    dissipation::Filter
    diags::Diags
end

#============== model setup =============#

function setup(choices, params, sph; mgr = VectorizedCPU())
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    hd_n, hd_nu = params.hyperdiff_n, params.hyperdiff_nu
    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    scheme = RungeKutta4(model)
    dissip = (
        zeta = HyperDiffusion(model.domain, hd_n, hd_nu, :vector_curl),
        theta = HyperDiffusion(model.domain, hd_n, hd_nu, :scalar),
    )
    diags = diagnostics(model)
    info = TimeLoopInfo(sph, model, scheme, dissip, diags)

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
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
    hyperdiff_n = 8,
    nlat = 96,
    ndays = 30,
)
params = (
    ptop = 100,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    hyperdiff_nu = 0.1,
    courant = 1.8,
    interval = 6 * 3600, # 6-hour intervals
)

@info "Initializing spherical harmonics..."
nthreads = max(1, Threads.nthreads() - 1)
(hasproperty(Main, :sph) && sph.nlat == choices.nlat) ||
    @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

@info "Model setup..."

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)

cpu, simd = PlainCPU(), VectorizedCPU(8)

info, state0 = setup(choices, params, sph; mgr = simd)
(; diags, model) = info
