@time_imports begin
    using ThreadPinning
    pinthreads(:cores)

    # heavy dependencies
    using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
    using ClimFlowsPlots
    using NetCDF: ncread, ncwrite, nccreate, ncclose
    using UnicodePlots: heatmap, scatterplot

    # lightweight dependencies
    using SIMDMathFunctions
    using SHTnsSpheres: SHTnsSphere
    using Strided: Strided, @strided
    using BenchmarkTools: @btime, @benchmark

    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
    using CFDomains: SigmaCoordinate, HyperDiffusion, VoronoiSphere, void
    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
    using CFPlanets: ShallowTradPlanet
    using ClimFluids: IdealPerfectGas
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, describe, initial_flow, initial_surface
end

# multi-thread transposition
import CFHydrostatics.Voronoi.Dynamics: transpose!, Void
function transpose!(x, ::MultiThread, y)
    @strided permutedims!(x, y, (2,1))
    return x # otherwise returns a StridedView
end
transpose!(::Void, ::MultiThread, y) = permutedims(y, (2,1)) # for non-ambiguity

## model setup

struct Recurser{Fun}
    fun::Fun
end
(rec::Recurser)(x::Number) = rec.fun(x)
(rec::Recurser)(x::Union{<:Tuple, <:NamedTuple}) = map(rec, x)
rmap(fun, x) = Recurser(fun)(x)

function setup(sphere, choices, params)
    case = testcase(choices.TestCase, choices.precision; params.testcase...)
    params = merge(choices, case.params, params)
    ## physical parameters needed to run the model
    @info case

    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = choices.coordinate(params.nz, params.ptop)

    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, choices.cpu, sphere, vcoord, surface_geopotential, gas)

    ## initial condition & standard diagnostics
    state = let
        init(lon, lat) = initial_surface(lon, lat, case)
        init(lon, lat, p) = initial_flow(lon, lat, p, case)
        CFHydrostatics.initial_HPE(init, model)
    end

    return model, diagnostics(model), state
end
