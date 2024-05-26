# # HPE solver using spherical harmoics for horizontal discretization

@time_imports begin
    # lightweight dependencies
    using CookBooks
    using MutatingOrNot: void, Void
    using ClimFlowsTestCases:
        Jablonowski06, testcase, describe, initial_flow, initial_surface
    using CFTimeSchemes: CFTimeSchemes, advance!
    using CFDomains: CFDomains, HyperDiffusion
    using ClimFluids: ClimFluids, AbstractFluid, SimpleFluid, IdealPerfectGas
    using CFPlanets: CFPlanets, ShallowTradPlanet, coriolis

    using CFDomains: VoronoiSphere

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
    fcov = (radius * radius) * f.(lon, lat)
    Phis = geopotential.(lon, lat)
    return HPE(vcoord, planet, shell(sph, params.nz), gas, fcov, Phis)
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

# setup

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

function initial_HPE_point(model, k, lon, lat, gas::SimpleFluid, case)
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

function setup(
    choices,
    params,
    sph;
    courant = 2.0,
    interval = 3600.0,
    hd_n = 8,
    hd_nu = 1e-2,
)
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    model = HPE(params, case, sph, gas)
    state = initial_HPE(model, case)
    #= initial flow
    f(n) = map((lon, lat) -> initial_flow(lon, lat, case)[n], sph.lon, sph.lat)
    gh, ulon, ulat = f(1), f(2), f(3)
    state0 = initial_state(model, gh, ulon, ulat)

    # time step based on maximum angular velocity of gravity waves
    umax = sqrt(maximum(@. ulon^2 + ulat^2))
    cmax = (umax + sqrt(case.params.gH0)) / radius
    dt = courant / cmax / sqrt(sph.lmax * sph.lmax + 1)
    dt = divisor(dt, interval)
    @info "Time step" umax, cmax, dt

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating=false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, scheme, solver, state0 =#
end

divisor(dt, T) = T / ceil(Int, T / dt)

# main program

@info "Initializing..."

sph = SHTnsSpheres.SHTnsSphere(32)
choices = (
    Fluid = IdealPerfectGas,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = 10,
)
params = (
    ptop = 0,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
)
params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)

interval = 3600.0
model = setup(choices, params, sph; interval) #=, scheme, solver, state0 =#
book = diagnostics()

# Initial potential vorticity
pv = open(book; model, state = state0) do diags
    pv = diags.potential_vorticity
    pv = transpose(max.(0.0, pv))
end
# Create a Makie observable and make a plot from it
# When we later update pv, the plot will update too.
pv = Makie.Observable(upscale(pv))

# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
lons = bounds_lon(model.sph.lon[1, :] * (180 / pi))[1:2:end]
lats = bounds_lat(model.sph.lat[:, 1] * (180 / pi))[1:2:end]
fig = orthographic(lons, lats, pv; colormap = :berlin)

state = deepcopy(state0)
solver! = solver(true)
