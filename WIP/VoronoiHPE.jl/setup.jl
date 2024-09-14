@time_imports begin
    using ThreadPinning
    pinthreads(:cores)
    # lightweight dependencies
    using SIMDMathFunctions
    using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, tune, no_simd
    using CFDomains: SigmaCoordinate, HyperDiffusion, VoronoiSphere, void
    using CFTimeSchemes: scratch_space, tendencies!, advance!
    using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
    using CFPlanets: ShallowTradPlanet
    #    import CFShallowWaters
    using ClimFluids: IdealPerfectGas
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, describe, initial_flow, initial_surface

    # heavy dependencies
    using ClimFlowsData: DYNAMICO_reader
    using ClimFlowsPlots
    using NetCDF: ncread, ncwrite, nccreate, ncclose
    using UnicodePlots: heatmap

    # graphical dependencies
    # using CairoMakie
end

## model setup

function setup(sphere, choices, params)
    case = testcase(choices.TestCase, choices.precision)
    params = merge(choices, case.params, params)
    ## physical parameters needed to run the model
    @info testcase

    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    # vcoord = NCARL30(params.nz, params.ptop)

    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, choices.mgr, sphere, vcoord, surface_geopotential, gas)

    ## initial condition & standard diagnostics
    state = let
        init(lon, lat) = initial_surface(lon, lat, case)
        init(lon, lat, p) = initial_flow(lon, lat, p, case)
        CFHydrostatics.initial_HPE(init, model)
    end

    diags = diagnostics(model)
    @info diags

    return model, diags, state
end
