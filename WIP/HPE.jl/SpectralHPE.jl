# # HPE solver using spherical harmonics for horizontal discretization

using Pkg;
Pkg.activate(@__DIR__);
using InteractiveUtils

@time_imports begin
    # lightweight dependencies
    using MutatingOrNot: void, Void
    using SIMDMathFunctions

    using ClimFlowsTestCases:
        Jablonowski06, testcase, describe, initial_flow, initial_surface
    using CFTimeSchemes: CFTimeSchemes, advance!
    using ClimFluids: ClimFluids, AbstractFluid, SimpleFluid, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet, coriolis

    using CFDomains: CFDomains, shell, Shell

    import ClimFlowsPlots.SpectralSphere as Plots

    using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!
    # heavy dependencies
    using GeoMakie, CairoMakie, ColorSchemes
end

macro skip(args...)
    return nothing
end

include("LagrangianHPE.jl")

struct HPE{Coord, Domain<:Shell, Fluid<:AbstractFluid}
    vcoord::Coord
    planet::ShallowTradPlanet{Float64}
    domain::Domain
    gas::Fluid
    fcov::Matrix{Float64} # covariant Coriolis factor = f(lat)*radius^2
    Phis::Matrix{Float64} # surface geopotential
    #    hd::HyperDiffusion
end

function HPE(params, sph::SHTnsSphere, vcoord, geopotential, gas)
    (; radius, Omega), (; lon, lat) = params, sph
    planet = ShallowTradPlanet(radius, Omega)
    f(lon, lat) = coriolis(planet, lon, lat)
    Phis = geopotential.(lon, lat)
    return HPE(vcoord, planet, CFDomains.shell(params.nz, sph), gas, f.(lon, lat), Phis)
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

# initial condition

initial_HPE(model::HPE, case) = initial_HPE(model, model.domain.layout, case)
initial_HPE(model::HPE, ::CFDomains.HVLayout, case) =
    initial_HPE_HV(model, CFDomains.nlayer(model.domain), model.domain.layer, case)

function initial_HPE_HV(model, nz, sph::SHTnsSphere, case)
    mass, ulon, ulat = initial_HPE_HV(model, nz, sph.lon, sph.lat, model.gas, case)
    mass_spec = analysis_scalar!(void, mass, sph)
    uv_spec = analysis_vector!(void, vector_spat(-ulat, ulon), sph)
    HPE_state(mass_spec, uv_spec)
end

function initial_HPE_HV(model, nz, lon, lat, gas::SimpleFluid, case)
    # mass[i, j, k, 1] = dry air mass
    # mass[i, j, k, 2] = mass-weighted conservative variable
    radius, vcoord = model.planet.radius, model.vcoord
    consvar = gas(:p, :v).conservative_variable
    alloc(dims...) = similar(lon, size(lon)..., dims...)
    mass, ulon, ulat = alloc(nz, 2), alloc(nz), alloc(nz)

    for i in axes(mass, 1), j in axes(mass, 2), k = 1:nz
        let lon = lon[i, j], lat = lat[i, j]
            ps, _ = initial_surface(lon, lat, case)
            p = pressure_level(2k - 1, ps, vcoord) # full level k
            _, uu, vv = initial_flow(lon, lat, p, case)
            ulon[i, j, k], ulat[i, j, k] = radius * uu, radius * vv
            p_lower = pressure_level(2k - 2, ps, vcoord) # lower interface
            p_upper = pressure_level(2k, ps, vcoord) # upper interface
            mg = p_lower - p_upper
            Phi_lower, _, _ = initial_flow(lon, lat, p_lower, case)
            Phi_upper, _, _ = initial_flow(lon, lat, p_upper, case)
            v = (Phi_upper - Phi_lower) / mg # dPhi = -v . dp
            mass[i, j, k, 1] = radius^2 * mg
            mass[i, j, k, 2] = (radius^2 * mg) * consvar(p, v)
        end
    end
    return mass, ulon, ulat
end

includet("spectral_modules.jl")
CFTimeSchemes.tendencies!(dstate, model::HPE, state, scratch, t) =
    Dynamics.tendencies!(dstate, model, state, scratch, t)
CFTimeSchemes.scratch_space(model::HPE, state) = Dynamics.scratch_space(model, state)
function CFTimeSchemes.model_dstate(::HPE, state)
    sim(x) = similar(x)
    sim(x::NamedTuple) = map(sim, x)
    sim(state)
end

# model setup

function setup(choices, params, sph; hd_n = 8, hd_nu = 1e-2)
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, sph, vcoord, surface_geopotential, gas)
    state = initial_HPE(model, case)

    # time step based on maximum sound speed
    diags = Diagnostics.diagnostics()
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

@info "Initializing STHns..."
# sph = SHTnsSphere(128)
@time sph = SHTnsSphere(32)

@info "Model setup..."

choices = (
    Fluid = IdealPerfectGas,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 30,
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
