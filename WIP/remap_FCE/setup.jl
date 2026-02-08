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
    using CFDomains: SigmaCoordinate, HyperDiffusion, HVLayout, laplace_dx, void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!
    using ClimFluids: IdealPerfectGas
    using CFPlanets: ShallowTradPlanet
    using CFTransport
    using CFHydrostatics: CFHydrostatics, HPE
    using CFCompressible: CFCompressible, FCE
    using ClimFlowsTestCases: Jablonowski06, DCMIP

    using UnicodePlots: heatmap, scatterplot, lineplot
    using Statistics: mean
    using Base.Filesystem: joinpath
    using Serialization: serialize
end

# fill some CFTimeSchemes entry points
CFTimeSchemes.tendencies!(slow, fast, scratch, model::CFCompressible.FCE, state, t, dt) = 
    CFCompressible.tendencies!(slow, fast, scratch, model, state, t, dt )

#   use our multi-thread manager when updating the model state
@inline CFTimeSchemes.update!(new, model::HPE, old, args...) = CFTimeSchemes.Update.update!(new, model.mgr, old, args...)
@inline CFTimeSchemes.Update.manage(a::Array{<:Complex}, mgr::LoopManager) = no_simd(mgr)[a]
@inline CFTimeSchemes.Update.manage(x, mgr::LoopManager) = mgr[x]

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
 
#============== model setup =============#

function setup(choices, params, sph, mgr)
    case = choices.TestCase(choices.precision; params.testcase...)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    vcoord = params.vcoord(params.nz, params.ptop)
    model_HPE = HPE(params, mgr, sph, vcoord, (lon, lat)->case(lon, lat)[2], gas)
    newton = CFCompressible.NewtonSolve(choices.newton...)
    model_FCE = CFCompressible.FCE(model_HPE, params.gravity, params.rhob, newton)
    
    diags_HPE = CFHydrostatics.diagnostics(model_HPE)
    diags_FCE = CFCompressible.diagnostics(model_FCE)
    state_HPE = CFHydrostatics.initial_HPE(case, model_HPE)
    state_FCE = CFCompressible.NH_state.diagnose(model_FCE, diags_HPE, state_HPE)
    return TwinModels(model_HPE, model_FCE), (HPE=state_HPE, FCE=state_FCE), (HPE=diags_HPE, FCE=diags_FCE)
end
