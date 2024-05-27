# # HPE solver using spherical harmoics for horizontal discretization

@time_imports begin
    # lightweight dependencies
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

# dynamics

module Dynamics

using MutatingOrNot: void, Void
using SHTnsSpheres: analysis_scalar!, synthesis_scalar!, analysis_vector!, synthesis_vector!, divergence!, curl!

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)

function mass_flux!(flux, model, mass, uv)
    invrad2 = model.planet.radius^-2
    flux = vector_spat(
        (@. flux.ucolat = -invrad2 * mass * uv.ucolat),
        (@. flux.ulon = -invrad2 * mass * uv.ulon),
    )
end

function tendencies!(dstate, model, state, scratch, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; mass, uv, flux, flux_spec, B, B_spec, zeta, zeta_spec, qflux, qflux_spec) = scratch
    (; mass_spec, uv_spec) = state
    dmass_spec, duv_spec = dstate.mass_spec, dstate.uv_spec
    sph, invrad2, fcov = model.domain.layer, model.planet.radius^-2, model.fcov

    # flux-form mass budget:
    #   ∂Φ/∂t = -∇(Φu, Φv)
    # uv is the momentum 1-form = a(u,v)
    # gh is the 2-form a²Φ
    # divergence! is relative to the unit sphere
    #   => scale flux by radius^-2
    mass = synthesis_scalar!(mass, mass_spec, sph)
    uv = synthesis_vector!(uv, uv_spec, sph)
    flux = vector_spat(
        (@. flux.ucolat = -invrad2 * mass * uv.ucolat),
        (@. flux.ulon = -invrad2 * mass * uv.ulon),
    )
    flux_spec = analysis_vector!(flux_spec, flux, sph)
    dmass_spec = divergence!(dmass_spec, flux_spec, sph)

    # curl-form momentum budget:
    #   ∂u/∂t = (f+ζ)v - ∂B/∂x
    #   ∂v/∂t = -(f+ζ)u - ∂B/∂y
    #   B = (u²+v²)2 + gh
    # uv is momentum = a*(u,v)
    # curl! is relative to the unit sphere
    # fcov, zeta and gh are the 2-forms a²f, a²ζ, a²Φ
    #   => scale B and qflux by radius^-2
    zeta_spec = curl!(zeta_spec, uv_spec, sph)
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    qflux = vector_spat(
        (@. qflux.ucolat = invrad2 * (zeta + fcov) * uv.ulon),
        (@. qflux.ulon = -invrad2 * (zeta + fcov) * uv.ucolat),
    )
    B = @. B = invrad2 * (mass + (uv.ucolat^2 + uv.ulon^2) / 2) # FIXME
    B_spec = analysis_scalar!(B_spec, B, sph)
    qflux_spec = analysis_vector!(qflux_spec, qflux, sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )
end

hydrostatic_pressure!(::Void, model, mass_spat) = hydrostatic_pressure!(similar(@view mass_spat[:,:,:,1]), model, mass_spat)

function hydrostatic_pressure!(p::Array{Float64,3}, model, mass_spat::Array{Float64,4})
    @assert size(mass_spat,3) == size(p,3)
    radius, ptop, nz = model.planet.radius, model.vcoord.ptop, size(p,3)
    rm2 = radius^-2
    for i in axes(p,1), j in axes(p,2)
        p[i,j,nz] = ptop + rm2*mass_spat[i,j,nz,1]/2
    end
    for i in axes(p,1), j in axes(p,2), k in nz:-1:2
        p[i,j,k-1] = p[i,j,k] + (mass_spat[i,j,k,1]+mass_spat[i,j,k-1,1])*(rm2/2)
    end
    return p
end

end # module Dynamics

using .Dynamics

# diagnostics

module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres: analysis_scalar!, synthesis_scalar!, analysis_vector!, synthesis_vector!, divergence!, curl!

using ..Dynamics

diagnostics() = CookBook(;
    dstate, mass_spat, surface_pressure, pressure,
    conservative_variable, temperature)

mass_spat(model, state) = synthesis_scalar!(void, state.mass_spec, model.domain.layer)
pressure(model, mass_spat) = Dynamics.hydrostatic_pressure!(void, model, mass_spat)

dstate(model, state) = Dynamics.tendencies!(void, model, state, void, 0.0)

function surface_pressure(model, state)
    radius = model.planet.radius
    ps_spec = @views (radius^-2)*sum(state.mass_spec[:,:,1]; dims=2)
    ps_spat = synthesis_scalar!(void, ps_spec[:,1], model.domain.layer)
    return ps_spat .+ model.vcoord.ptop
end

function conservative_variable(mass_spat)
    mass = @view mass_spat[:,:,:,1]
    mass_consvar = @view mass_spat[:,:,:,2]
    return @. mass_consvar / mass
end

function temperature(model, conservative_variable, pressure)
    temp = model.gas(:p, :consvar).temperature
    return temp.(pressure, conservative_variable)
end

end #module Diagnostics

using .Diagnostics

# model setup

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
    state0 = initial_HPE(model, case)

    #=
    # time step based on maximum angular velocity of gravity waves
    umax = sqrt(maximum(@. ulon^2 + ulat^2))
    cmax = (umax + sqrt(case.params.gH0)) / radius
    dt = courant / cmax / sqrt(sph.lmax * sph.lmax + 1)
    dt = divisor(dt, interval)
    @info "Time step" umax, cmax, dt

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating=false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, scheme, solver, state0 =#
    return model, state0
end

divisor(dt, T) = T / ceil(Int, T / dt)
upscale(x) = x

# main program

@info "Initializing..."

sph = SHTnsSpheres.SHTnsSphere(128)
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
)
params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)

interval = 3600.0
model, state0 = setup(choices, params, sph; interval) #=, scheme, solver, state0 =#
book = Diagnostics.diagnostics()

ps = transpose(open(book; model, state = state0).surface_pressure)
# Create a Makie observable and make a plot from it
# When we later update pv, the plot will update too.
pv = Makie.Observable(upscale(ps))

# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
lons = bounds_lon(sph.lon[1, :] * (180 / pi)) #[1:2:end]
lats = bounds_lat(sph.lat[:, 1] * (180 / pi)) #[1:2:end]
fig = orthographic(lons, lats, ps; colormap = :berlin)

state = deepcopy(state0)
solver! = solver(true)
