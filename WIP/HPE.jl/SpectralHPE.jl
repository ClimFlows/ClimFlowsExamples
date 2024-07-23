# # HPE solver using spherical harmonics for horizontal discretization

using Pkg; Pkg.activate(@__DIR__); Pkg.status()
using InteractiveUtils

using ThreadPinning
pinthreads(:cores)

@time_imports begin
    using MutatingOrNot: void
    using SIMDMathFunctions
    using LoopManagers: PlainCPU, VectorizedCPU, MultiThread, tune
    using CookBooks:CookBook

    using CFTimeSchemes: CFTimeSchemes, advance!
    using CFDomains: SigmaCoordinate, HyperDiffusion
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

#=====  custom time loop: combines HPE solver, time integration scheme, vertical remap and hyperdiffusion ====#

# everything that does not depend on initial condition
struct TimeLoopInfo{Sphere, Dyn, Scheme, Filter}
    sphere::Sphere
    model::Dyn
    scheme::Scheme
    dissipation::Filter
    diags::CookBook
end

# some info (essentially the time step) depends on the initial condition
struct TimeLoop{Dyn, Solver, Filter}
    model::Dyn
    solver::Solver
    dissipation::Filter
    diags::CookBook
    mutating::Bool
end

function TimeLoop(info::TimeLoopInfo, u0, time_step, mutating)
    (; model, scheme, dissipation, diags) = info
    solver = CFTimeSchemes.IVPSolver(scheme, time_step; u0, mutating)
    return TimeLoop(model, solver, dissipation, diags, mutating)
end

# the extra parameter `dt` is for benchmarking purposes only
function run(timeloop::TimeLoop, N, interval, state, scratch ; dt=timeloop.solver.dt)
    (; solver, model, mutating) = timeloop
    @assert mutating # FIXME
    t, nstep = zero(dt), Int(interval / dt)
    for _ = 1:N*nstep
        advance!(state, solver, state, t, 1)
        state = vertical_remap(model, state, scratch)
    end
end

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

#============== model setup =============#

function new_setup(choices, params, sph; hd_n = 8, hd_nu = 1e-2, mgr = VectorizedCPU())
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    scheme = CFTimeSchemes.RungeKutta4(model)
    dissip = HyperDiffusion(model.domain, hd_n, hd_nu, :vector_curl)
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

function max_time_step(info::TimeLoopInfo, courant, interval, state)
    (; sphere, model, diags) = info
    # time step based on maximum sound speed and courant number `courant`, which divides `interval`
    session = open(diags; model, state)
    uv = session.uv
    cmax = maximum(session.sound_speed + @. sqrt(uv.ucolat^2 + uv.ulon^2))
    dt = model.planet.radius * courant / cmax / sqrt(sphere.lmax * sphere.lmax + 1)
    dt = divisor(dt, interval)
end

divisor(dt, T) = T / ceil(Int, T / dt)

#============== benchmark ==================#

function new_benchmark(timeloop, state, dt, interval, scratch=void ; ndays=1)
    N = max(1, Int(ndays * 24 * 3600 / interval))
    run(timeloop, N, interval, state, scratch ; dt)
end

function benchmark(choices, params, sph, mgrs)
    for mgr in mgrs
        # NB : spherical harmonics are multithread in all cases !
        @info "===== Time needed to simulate 1 day with $mgr ====="
        let (info, state0) = new_setup(choices, params, sph; mgr)
            scratch = scratch_remap(info.diags, info.model, state0)
            (; courant, interval) = params
            dt = max_time_step(info, courant, interval, state0)
            timeloop = TimeLoop(info, state0, 0.0, true) # zero time step for benchmarking
            new_benchmark(timeloop, state0, dt, interval ; ndays=0) # compile
            @time new_benchmark(timeloop, state0, dt, interval, scratch)
    #        @profview new_benchmark(timeloop, state0, dt, interval, scratch)
        end
    end
end

#=======================  main program =====================#

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
(hasproperty(Main, :sph) && sph.nlat==choices.nlat) || @time sph = SHTnsSphere(choices.nlat, nthreads)
@info sph

@info "Model setup..."

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)

cpu, simd = PlainCPU(), VectorizedCPU(8)

# benchmark(choices, params, sph, [cpu, simd, MultiThread(cpu, nthreads), MultiThread(simd, nthreads)])

info, state0 = new_setup(choices, params, sph; mgr=simd)
(; diags, model) = info

@info "Preparing plots..."

diag(state) = transpose(open(diags; model, state).temperature[:, :, 3])
diag_obs = Makie.Observable(diag(state0))

lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
fig = Plots.orthographic(lons .- 90, lats, diag_obs; colormap = :berlin)

@info "Starting simulation."
@time let ndays = 10
    scratch = scratch_remap(diags, model, state0);
    (; courant, interval) = params
    dt = max_time_step(info, courant, interval, state0)
    timeloop = TimeLoop(info, state0, dt, true) # mutating, non-allocating
    N = Int(ndays * 24 * 3600 / interval)

    # separate thread running the simulation
    channel = Channel(spawn = true) do ch
        state = deepcopy(state0)
        for iter = 1:N
            run(timeloop, 1, interval, state, scratch)
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
