using InteractiveUtils

@time_imports begin
    using ThreadPinning
#    pinthreads(:cores)
    pinthreads([0, 2])
    
    using SIMDMathFunctions
    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
    using MutatingOrNot: void, Void

    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: RungeKutta4, KinnmarkGray, BackwardEuler, Midpoint, TRBDF2, ARK_TRBDF2
    using CFTimeSchemes: CFTimeSchemes, IVPSolver
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, DCMIP

    using UnicodePlots: heatmap, scatterplot
#    using Unitful: m as meter, s as second, J as Joule, K as Kelvin, kg, Pa
    using LinearAlgebra
end

# fill some CFTimeSchemes entry points

CFTimeSchemes.tendencies!(slow, fast, scratch, model::CFCompressible.FCE, state, t, dt) = 
    CFCompressible.tendencies!(slow, fast, scratch, model, state, t, dt )

function CFTimeSchemes.model_dstate(::CFCompressible.FCE, state, _)
    rsim(x::AbstractArray) = similar(x)
    rsim(x::NamedTuple) = map(rsim, x)
    return rsim(state)
end
#   use our multi-thread manager when updating the model state
@inline CFTimeSchemes.update!(new, model::HPE, old, args...) = CFTimeSchemes.Update.update!(new, model.mgr, old, args...)
@inline CFTimeSchemes.Update.manage(a::Array{<:Complex}, mgr::LoopManager) = no_simd(mgr)[a]

# small functions that help manage nested named tuples
rmap(fun, x) = fun(x)
rmap(fun, x::Union{Tuple, NamedTuple}) = map(y->rmap(fun,y), x)

override(a, b) = b
override(a::NamedTuple ; b...) = override(a, NamedTuple(b))
function override(a::NamedTuple, b::NamedTuple)
    bb = (; a... , b...) # override a with b ; a field come first, then b fields not present in a
    aa = (; bb... , a...) # extend a with new b ; extra fields ordered as in b
    # now aa and bb have the same fields in the same order
    map(override, aa, bb)
end
 
# include("../../SpectralHPE.jl/NCARL30.jl")

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere,Dyn,Scheme,Filter,Diags}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    remap_period::Int
    dissipation::Filter
    diags::Diags
    quicklook::Function # we can afford runtime dispatch when calling quicklook
end

#============== model setup =============#

function setup(choices, params, sph, mgr, Equation)
    case = choices.TestCase(Float64; params.testcase...)
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
    info = TimeLoopInfo(sph, model, scheme, choices.remap_period, dissip, diags, choices.quicklook)

    return info, case
end

