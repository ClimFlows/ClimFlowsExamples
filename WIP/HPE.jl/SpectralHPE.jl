# # HPE solver using spherical harmoics for horizontal discretization

@time_imports begin
    # lightweight dependencies
    using MutatingOrNot: void, Void
    using SIMDMathFunctions

    using ClimFlowsTestCases:
        Jablonowski06, testcase, describe, initial_flow, initial_surface
    using CFTimeSchemes: CFTimeSchemes, advance!
    using CFDomains: CFDomains, HyperDiffusion
    using ClimFluids: ClimFluids, AbstractFluid, SimpleFluid, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet, coriolis

    using CFDomains: VoronoiSphere
    import ClimFlowsPlots.SpectralSphere as Plots

    using SHTnsSpheres:
        SHTnsSpheres,
        SHTnsSphere,
        void,
        erase,
        analysis_scalar!,
        synthesis_scalar!,
        analysis_vector!,
        synthesis_vector!,
        divergence!,
        curl!,
        sample_vector!,
        sample_scalar!

    # heavy dependencies
    using GeoMakie, CairoMakie, ColorSchemes

    # development
    using BenchmarkTools: @benchmark
end

include("LagrangianHPE.jl")

struct HPE{Coord,Domain<:Shell,Fluid<:AbstractFluid}
    vcoord::Coord
    planet::ShallowTradPlanet{Float64}
    domain::Domain
    gas::Fluid
    fcov::Matrix{Float64} # covariant Coriolis factor = f(lat)*radius^2
    Phis::Matrix{Float64} # surface geopotential
    #    hd::HyperDiffusion
end

function HPE(params, case, sph, gas)
    (; radius, Omega, ptop, nz), (; lon, lat) = params, sph
    vcoord = SigmaCoordinate(nz, ptop)
    planet = ShallowTradPlanet(radius, Omega)
    f(lon, lat) = coriolis(planet, lon, lat)
    geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
#    fcov = (radius * radius) * f.(lon, lat)
    Phis = geopotential.(lon, lat)
    return HPE(vcoord, planet, shell(sph, params.nz), gas, f.(lon, lat), Phis)
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

# initial condition

initial_HPE(model::HPE, case) = initial_HPE(model, model.domain, case)
initial_HPE(model::HPE, domain::Shell{nz, HVLayout}, case) where nz =
    initial_HPE_HV(model, nz, domain.layer, case)

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

    for i in axes(mass,1), j in axes(mass, 2), k in 1:nz
        let lon = lon[i,j], lat=lat[i,j]
            ps, _ = initial_surface(lon, lat, case)
            p = pressure_level(2k-1, ps, vcoord) # full level k
            _, uu, vv = initial_flow(lon, lat, p, case)
            ulon[i,j,k], ulat[i,j,k] = radius*uu, radius*vv
            p_lower = pressure_level(2k-2, ps, vcoord) # lower interface
            p_upper = pressure_level(2k, ps, vcoord) # upper interface
            mg = p_lower - p_upper
            Phi_lower, _, _ = initial_flow(lon, lat, p_lower, case)
            Phi_upper, _, _ = initial_flow(lon, lat, p_upper, case)
            v = (Phi_upper - Phi_lower)/mg # dPhi = -v . dp
            mass[i,j,k,1] = radius^2*mg
            mass[i,j,k,2] = (radius^2*mg) * consvar(p,v)
        end
    end
    return mass, ulon, ulat
end

includet("spectral_modules.jl")
using .Dynamics
using .Diagnostics

# model setup

function setup(
    choices,
    params,
    sph;
    hd_n = 8,
    hd_nu = 1e-2,
)
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    model = HPE(params, case, sph, gas)
    state = initial_HPE(model, case)

    # time step based on maximum sound speed
    diags = Diagnostics.diagnostics()
    cmax = let session = open(diags; model, state), uv=session.uv
        maximum(session.sound_speed + @. sqrt(uv.ucolat^2+uv.ulon^2))
    end
    dt = params.radius * params.courant / cmax / sqrt(sph.lmax * sph.lmax + 1)
    dt = divisor(dt, params.interval)
    @info "Time step" cmax dt

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating=false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, state, diags, scheme, solver
end

divisor(dt, T) = T / ceil(Int, T / dt)
upscale(x) = x

# main program

@info "Initializing..."

#sph = SHTnsSpheres.SHTnsSphere(128)
sph = SHTnsSpheres.SHTnsSphere(64)
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
    interval = 3600 # 1-hour intervals
)

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)
model, state0, diags, scheme, solver = setup(choices, params, sph)

# temp(state) = transpose(open(diags; model, state).temperature[:,:,1])
diag(state) = transpose(open(diags; model, state).uv.ucolat[:,:,1])
diag_obs = Makie.Observable(diag(state0))

lons = Plots.bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = Plots.bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
fig = Plots.orthographic(lons, lats, diag_obs; colormap = :berlin)

@info "Starting simulation."

CFTimeSchemes.tendencies!(dstate, model::HPE, state, scratch, t) = Dynamics.tendencies!(dstate, model, state, scratch, t)

# solver! = solver(true) # mutating, non-allocating
solver = solver(false) # non-mutating, allocating

@profview let N=120 # number of intervals to simulate
    # separate thread running the simulation
    channel = Channel(spawn=true) do ch
        state = deepcopy(state0)
        nstep = Int(params.interval/solver.dt)
        for iter in 1:N
            for istep in 1:nstep
                state, _ = advance!(void, solver, state, 0.0, 1)
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
    record(fig, "$(@__DIR__)/ulat.mp4", 1:N) do hour
        @info hour
        diag_obs[] = take!(channel)
    end
end
