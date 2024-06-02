# # HPE solver using spherical harmonics for horizontal discretization

using Pkg;
Pkg.activate(@__DIR__);
using InteractiveUtils

@time_imports begin
    # lightweight dependencies
    using MutatingOrNot: void, Void
    using LoopManagers: VectorizedCPU, MultiThread
    using SIMDMathFunctions

    using ClimFlowsTestCases:
        Jablonowski06, testcase, describe, initial_flow, initial_surface
    using CFTimeSchemes: CFTimeSchemes, advance!
    using ClimFluids: ClimFluids, AbstractFluid, SimpleFluid, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet, coriolis

    using CFDomains: CFDomains, shell, Shell
    using CFHydrostatics: CFHydrostatics, SigmaCoordinate, HPE, diagnostics

    import ClimFlowsPlots.SpectralSphere as Plots

    using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!
    # heavy dependencies
    using GeoMakie, CairoMakie, ColorSchemes
end

macro skip(args...) ; return nothing ; end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

# initial condition from test case

# model setup

function setup(choices, params, sph; hd_n = 8, hd_nu = 1e-2)
    mgr = MultiThread(VectorizedCPU())
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    state = let
        init(lon, lat) = initial_surface(lon, lat, case)
        init(lon, lat, p) = initial_flow(lon, lat, p, case)
        CFHydrostatics.initial_HPE(init, model)
    end

    # time step based on maximum sound speed
    diags = diagnostics(model)
    cmax = let session = open(diags; model, state), uv = session.uv
        maximum(session.sound_speed + @. sqrt(uv.ucolat^2 + uv.ulon^2))
    end
    dt = params.radius * params.courant / cmax / sqrt(sph.lmax * sph.lmax + 1)
    dt = divisor(dt, params.interval)
    @info "Time step" cmax dt

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating = false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, state, diags, scheme, solver
end

divisor(dt, T) = T / ceil(Int, T / dt)
upscale(x) = x

# main program

choices = (
    Fluid = IdealPerfectGas,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
    nlat = 48
)
params = (
    ptop = 100,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    courant = 2,
    interval = 6 * 3600, # 6-hour intervals
)

@info "Initializing spherical harmonics..."
# @time sph = SHTnsSphere(128)
hasproperty(Main, :sph) || @time sph = SHTnsSphere(choices.nlat)

@info "Model setup..."


params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)
@time model, state0, diags, scheme, solver = setup(choices, params, sph)

solver! = solver(true) # mutating, non-allocating
# solver = solver(false) # non-mutating, allocating

@info "Time needed to simulate 1 day ..."
let ndays = 1
    interval = params.interval
    N = Int(ndays * 24 * 3600 / interval)
    nstep = Int(params.interval / solver!.dt)
    state = deepcopy(state0)
    advance!(state, solver!, state, 0.0, 1)
    @time advance!(state, solver!, state, 0.0, N * nstep)
end

@info "Starting simulation."

diag(state) = transpose(open(diags; model, state).uv.ucolat[:, :, 1])
diag_obs = Makie.Observable(diag(state0))

lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)

@time let ndays = 6
    interval = params.interval
    N = Int(ndays * 24 * 3600 / interval)

    # separate thread running the simulation
    channel = Channel(spawn = true) do ch
        state = deepcopy(state0)
        nstep = Int(params.interval / solver!.dt)
        for iter = 1:N
            for istep = 1:nstep
                advance!(state, solver!, state, 0.0, 1)
                #                state, _ = advance!(void, solver, state, 0.0, 1)
                #                (; gh_spec, uv_spec) = state
                #                (; toroidal) = uv_spec
                #                toroidal = model.filter(toroidal, toroidal, model.sph)
                #                uv_spec = vector_spec(uv_spec.spheroidal, toroidal)
                #                state = RSW_state(gh_spec, uv_spec)
            end
            put!(ch, diag(state))
        end
        @info "Worker: finished"
    end
    # main thread making the movie
    record(fig, "$(@__DIR__)/ulat.mp4", 1:N) do i
        @info "t=$(div(interval*i,3600))h"
        diag_obs[] = take!(channel)
    end
end
