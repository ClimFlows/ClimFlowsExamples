using InteractiveUtils

@time_imports begin
    using ThreadPinning
#    pinthreads(:cores)
    pinthreads([0, 2])
    
    using SIMDMathFunctions
    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
    using MutatingOrNot: void, Void

    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: RungeKutta4, KinnmarkGray, BackwardEuler, Midpoint, TRBDF2
    using CFTimeSchemes: CFTimeSchemes, IVPSolver
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06

    using UnicodePlots: heatmap
#    using Unitful: m as meter, s as second, J as Joule, K as Kelvin, kg, Pa
    using LinearAlgebra
end

# let CFTimeSchemes use our multi-thread manager when updating the model state
@inline CFTimeSchemes.update!(new, model::HPE, old, args...) = CFTimeSchemes.Update.update!(new, model.mgr, old, args...)
@inline CFTimeSchemes.Update.manage(a::Array{<:Complex}, mgr::LoopManager) = no_simd(mgr)[a]

include("../../SpectralHPE.jl/NCARL30.jl")

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

function setup(choices, params, sph, mgr, Equation)
    case = choices.TestCase(Float64)
    params = merge(choices, case.params, params)
    hd_n, hd_nu = params.hyperdiff_n, params.hyperdiff_nu
    # stuff independent from initial condition
    gas = params.Fluid(params)
    # vcoord = SigmaCoordinate(params.nz, params.ptop)
#    vcoord = NCARL30(params.nz, params.ptop)
    vcoord = SigmaCoordinate(params.nz, params.ptop)

    model = Equation(params, mgr, sph, vcoord, (lon, lat)->case(lon, lat)[2], gas)
    scheme = choices.TimeScheme(model)
    dissip = (
        zeta = HyperDiffusion(model.domain, hd_n, hd_nu, :vector_curl),
        theta = HyperDiffusion(model.domain, hd_n, hd_nu, :scalar),
    )
    diags = diagnostics(model)
    info = TimeLoopInfo(sph, model, scheme, choices.remap_period, dissip, diags)

    return info, case
end
