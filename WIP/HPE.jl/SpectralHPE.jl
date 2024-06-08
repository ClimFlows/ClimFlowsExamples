# # HPE solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__); Pkg.status()
using InteractiveUtils

using ThreadPinning
pinthreads(:cores)

@time_imports begin
    using MutatingOrNot: void
    using SIMDMathFunctions
    using LoopManagers: VectorizedCPU, MultiThread

    using CFTimeSchemes: CFTimeSchemes, advance!
    using CFDomains: SigmaCoordinate
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, synthesis_scalar!

    using ClimFluids: ClimFluids, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet
    using CFHydrostatics: CFHydrostatics, HPE, diagnostics
    using ClimFlowsTestCases: Jablonowski06, testcase, describe, initial_flow, initial_surface

    import ClimFlowsPlots.SpectralSphere as Plots
    using GeoMakie, CairoMakie, ColorSchemes

    using UnicodePlots: heatmap
end

Dynamics = Base.get_extension(CFHydrostatics, :SHTnsSpheres_Ext)
# model setup

function setup(choices, params, sph; hd_n = 8, hd_nu = 1e-2)
    mgr = MultiThread(VectorizedCPU(), nthreads)
#    mgr = VectorizedCPU()
#    mgr = MultiThread()
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

function vertical_remap(model, state, scratch=void)
    sph = model.domain.layer
    mass_spat = SHTnsSpheres.synthesis_scalar!(scratch.mass_spat, state.mass_spec, sph)
    uv_spat = SHTnsSpheres.synthesis_vector!(scratch.uv_spat, state.uv_spec, sph)
    now = @views (mass=mass_spat[:,:,:,1], massq=mass_spat[:,:,:,2], ux=uv_spat.ucolat, uy=uv_spat.ulon)
    remapped = CFHydrostatics.vertical_remap!(model.mgr, model, scratch.remapped, scratch, now)

    reshp(x) = reshape(x, size(mass_spat,1), size(mass_spat,2), size(mass_spat,3))
    mass_spat[:,:,:,1] .= reshp(remapped.mass)
    mass_spat[:,:,:,2] .= reshp(remapped.massq)
    mass_spec = SHTnsSpheres.analysis_scalar!(state.mass_spec, mass_spat, sph)
    uv_spec = SHTnsSpheres.analysis_vector!(state.uv_spec, (ucolat=reshp(remapped.ux), ulon=reshp(remapped.uy)), sph)
    return (; mass_spec, uv_spec)
end

function scratch_remap(diags, model, state)
    flatten(x) = reshape(x, size(x,1)*size(x,2), size(x,3))
    mass_spat = open(diags ; model, state).mass
    uv_spat = open(diags ; model, state).uv
    ux = flatten(similar(uv_spat.ulon))
    uy, mass, new_mass, massq, slope, q = (similar(ux) for _ in 1:6)
    flux = similar(mass, (size(mass,1), size(mass,2)+1))
    fluxq = similar(flux)
    return (; mass_spat, uv_spat, flux, fluxq, new_mass, slope, q, remapped=(; mass, massq, ux, uy))
end

function benchmark(model, state0, solver!, params, scratch=void ; ndays=1)
    state = deepcopy(state0)
    interval = params.interval
    N = Int(ndays * 24 * 3600 / interval)
    nstep = Int(params.interval / solver!.dt)
    for iter = 1:N*nstep
        advance!(state, solver!, state, 0.0, 1)
        state = vertical_remap(model, state, scratch)
    end
end

# main program

choices = (
    Fluid = IdealPerfectGas,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
    nlat = 64
)
params = (
    ptop = 100,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    courant = 1.8,
    interval = 6 * 3600, # 6-hour intervals
)

@info "Initializing spherical harmonics..."
nthreads = max(1,Threads.nthreads()-1)
hasproperty(Main, :sph) || @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

@info "Model setup..."

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)
@time model, state0, diags, scheme, solver = setup(choices, params, sph)
solver! = solver(true) # mutating, non-allocating
# solver = solver(false) # non-mutating, allocating

@info "Time needed to simulate 1 day ..."
scratch = scratch_remap(diags, model, state0);
benchmark(model, state0, solver!, params, scratch)
@time benchmark(model, state0, solver!, params, scratch)
# @profview benchmark(model, state0, solver!, params, scratch)

@info "Preparing plots..."

diag(state) = transpose(open(diags; model, state).temperature[:, :, 3])
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
                state = vertical_remap(model, state)
            end
            put!(ch, diag(state))
        end
        @info "Worker: finished"
    end

    # main thread making the movie
    record(fig, "$(@__DIR__)/T850.mp4", 1:N) do i
        @info "t=$(div(interval*i,3600))h"
        diag_obs[] = take!(channel)
        if mod(params.interval*i, 86400) == 0
            @info "day $(i/4)"
            display(heatmap(transpose(diag_obs[])))
        end
    end
end
