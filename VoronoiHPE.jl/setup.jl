@time_imports begin
    using ThreadPinning
    pinthreads(:cores)

    # heavy dependencies
    using ClimFlowsData: DYNAMICO_reader, VoronoiMeshes
    using ClimFlowsPlots
    using NetCDF: ncread, ncwrite, nccreate, ncclose
    using UnicodePlots: heatmap, scatterplot

    # lightweight dependencies
    using SIMDMathFunctions
    using Strided: Strided, @strided
    using BenchmarkTools: @btime, @benchmark

    import ManagedLoops, LoopManagers
    using LoopManagers: LoopManager, GPUConfig, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
    import CFDomains
    using CFDomains: SigmaCoordinate, HyperDiffusion, void
    import VoronoiSpheres
    using VoronoiSpheres: VoronoiSphere
    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
    using CFPlanets: ShallowTradPlanet
    using ClimFluids: IdealPerfectGas
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: DCMIP, Isothermal, Jablonowski06, describe, initial
end

# multi-thread transposition
import CFDomains: transpose!, Void
function transpose!(x, ::MultiThread, y)
    @strided permutedims!(x, y, (2,1))
    return x # otherwise returns a StridedView
end
transpose!(::Void, ::MultiThread, y) = permutedims(y, (2,1)) # for non-ambiguity
transpose!(::Void, ::LoopManagers.KernelAbstractions_GPU, y) = CUDA.CUBLAS.geam('T', 'T', 1, y, 0, y)
transpose!(x, ::LoopManagers.KernelAbstractions_GPU, y) = CUDA.CUBLAS.geam!('T', 'T', 1, y, 0, y, x)

## model setup

struct Recurser{Fun}
    fun::Fun
end
(rec::Recurser)(x::Number) = rec.fun(x)
(rec::Recurser)(x::Union{<:Tuple, <:NamedTuple}) = map(rec, x)
rmap(fun, x) = Recurser(fun)(x)

function setup(sphere, choices, params)
    case = choices.TestCase(choices.precision; params.testcase...)
    params = merge(choices, case.params, params)
    ## physical parameters needed to run the model
    @info case

    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = choices.coordinate(params.nz, params.ptop)

    model = HPE(params, choices.cpu, sphere, vcoord, (lon, lat)->initial(case, lon, lat)[2], gas)

    ## initial condition & standard diagnostics
    state = CFHydrostatics.initial_HPE((args...)->initial(case, args...), model)

    return model, diagnostics(model), state
end
