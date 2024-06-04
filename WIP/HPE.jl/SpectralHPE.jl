# # HPE solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__)
using InteractiveUtils

@time_imports begin
    using SIMDMathFunctions
    using LoopManagers: VectorizedCPU, MultiThread

    using CFTimeSchemes: CFTimeSchemes, advance!
    using CFDomains: SigmaCoordinate
    using SHTnsSpheres: SHTnsSphere

    using ClimFluids: ClimFluids, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, describe, initial_flow, initial_surface

    import ClimFlowsPlots.SpectralSphere as Plots
    using GeoMakie, CairoMakie, ColorSchemes
end

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
    solver(mutating = false) = CFTimeSchemes.IVPSolver(scheme, dt; u0=state0, mutating)
    return model, state, diags, scheme, solver
end

divisor(dt, T) = T / ceil(Int, T / dt)

function vertical_remap(model, state)
    reshp(x) = reshape(x, size(mass,1), size(mass,2), size(mass,3))
    sph = model.domain.layer
    mass = SHTnsSpheres.synthesis_scalar!(void, state.mass_spec, sph)
    uv = SHTnsSpheres.synthesis_vector!(void, state.uv_spec, sph)
    now = @views (mass=mass[:,:,:,1], massq=mass[:,:,:,2], ux=uv.ucolat, uy=uv.ulon)
    remapped = CFHydrostatics.vertical_remap!(nothing, model, void, void, now)
    mass[:,:,:,1] .= reshp(remapped.mass)
    mass[:,:,:,2] .= reshp(remapped.massq)
    mass_spec = SHTnsSpheres.analysis_scalar!(void, mass, sph)
    uv_spec = SHTnsSpheres.analysis_vector!(void, (ucolat=reshp(remapped.ux), ulon=reshp(remapped.uy)), sph)
    return (; mass_spec, uv_spec)
end

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
hasproperty(Main, :sph) || @time sph = SHTnsSphere(choices.nlat)

@info "Model setup..."

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)
@time model, state0, diags, scheme, solver = setup(choices, params, sph)

solver! = solver(true) # mutating, non-allocating
# solver = solver(false) # non-mutating, allocating

@info "Time needed to simulate 1 day ..."
new_state, t = let ndays = 1
    interval = params.interval
    N = Int(ndays * 24 * 3600 / interval)
    nstep = Int(params.interval / solver!.dt)
    state = deepcopy(state0)
    advance!(state, solver!, state, 0.0, 1)
    @time advance!(state, solver!, state, 0.0, N * nstep)
end

@info "Preparing plots..."

diag(state) = transpose(open(diags; model, state).uv.ucolat[:, :, 1])
diag_obs = Makie.Observable(diag(state0))

lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)

@info "Starting simulation."

@time let ndays = 10
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
            state = vertical_remap(model, state)
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
