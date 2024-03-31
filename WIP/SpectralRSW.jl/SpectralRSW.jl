# # Rotating shallow water

## Preamble

const debug = false
include("preamble.jl")

@time_imports begin
    using MutatingOrNot: void, Void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, void, erase,
        analysis_scalar!, synthesis_scalar!,
        analysis_vector!, synthesis_vector!,
        divergence!, curl!, sample_vector!, sample_scalar!

    using ClimFlowsTestCases: Williamson91, testcase, describe, initial_flow

    using CFTimeSchemes: CFTimeSchemes, advance!
    using BenchmarkTools

    using GeoMakie, CairoMakie, ColorSchemes
    using CookBooks
end

struct RSW{F}
    sph::SHTnsSphere
    fcov::Matrix{F} # 2-form: fcov = 2Ω * radius^2 * sin(latitude)
    radius::F
end

RSW(sph, Omega::Real, radius) = RSW(sph, (2 * radius^2 * Omega) * sph.z, radius)

function Base.show(io::IO, (; sph, fcov, radius)::RSW)
    Omega = maximum(fcov) / 2 * radius^-2
    print(io, "RSW($sph, Omega=$Omega, radius=$radius)")
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
RSW_state(gh_spec, uv_spec) = (; gh_spec, uv_spec)

function CFTimeSchemes.scratch_space((; sph)::RSW, (; gh_spec, uv_spec))
    spec() = similar(gh_spec)
    spat() = SHTnsSpheres.similar_spat(gh_spec, sph)
    vspec() = map(similar, uv_spec)
    vspat() = SHTnsSpheres.similar_spat(uv_spec, sph)
    B, zeta, gh = spat(), spat(), spat()
    flux, uv, qflux = vspat(), vspat(), vspat()
    B_spec, zeta_spec = spec(), spec()
    flux_spec, qflux_spec = vspec(), vspec()
    return (; B, zeta, gh, flux, uv, qflux, B_spec, zeta_spec, flux_spec, qflux_spec)
end

CFTimeSchemes.model_dstate((; sph)::RSW, (; gh_spec, uv_spec)) =
    RSW_state(similar(gh_spec), map(similar, uv_spec))

function CFTimeSchemes.tendencies!(dstate, model::RSW, state, scratch, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; gh, uv, flux, flux_spec, B, B_spec, zeta, zeta_spec, qflux, qflux_spec) = scratch
    (; uv_spec, gh_spec) = state
    dgh_spec, duv_spec = dstate.gh_spec, dstate.uv_spec
    (; sph, radius, fcov) = model
    invrad2 = radius^-2

    # flux-form mass budget:
    #   ∂Φ/∂t = -∇(Φu, Φv)
    # uv is the momentum 1-form = a(u,v)
    # gh is the 2-form a²Φ
    # divergence! is relative to the unit sphere
    #   => scale flux by radius^-2
    gh = synthesis_scalar!(gh, gh_spec, sph)
    uv = synthesis_vector!(uv, uv_spec, sph)
    flux = vector_spat(
        (@. flux.ucolat = -invrad2 * gh * uv.ucolat),
        (@. flux.ulon = -invrad2 * gh * uv.ulon),
    )
    flux_spec = analysis_vector!(flux_spec, flux, sph)
    dgh_spec = divergence!(dgh_spec, flux_spec, sph)

    # curl-form momentum budget:
    #   ∂u/∂t = (f+ζ)v - ∂B/∂x
    #   ∂v/∂t = -(f+ζ)u - ∂B/∂y
    #   B = (u²+v²)2 + gh
    # uv is momentum = a(u,v)
    # curl! is relative to the unit sphere
    # fcov, zeta and gh are the 2-forms a²f, a²ζ, a²Φ
    #   => scale B and qflux by radius^-2
    zeta_spec = curl!(zeta_spec, uv_spec, sph)
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    qflux = vector_spat(
        (@. qflux.ucolat = invrad2 * (zeta + fcov) * uv.ulon),
        (@. qflux.ulon = -invrad2 * (zeta + fcov) * uv.ucolat),
    )
    B = @. B = invrad2 * (gh + (uv.ucolat^2 + uv.ulon^2) / 2)
    B_spec = analysis_scalar!(B_spec, B, sph)
    qflux_spec = analysis_vector!(qflux_spec, qflux, sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )

    return RSW_state(dgh_spec, duv_spec)
end

function vorticity(model, state)
    (; sph) = model
    zeta = model.radius^-2 * curl!(void, state.uv_spec, sph)
    synthesis_scalar!(void, zeta, sph)
end

function velocity(model, state)
    (; sph, radius) = model
    uv_spec = vector_spec(
        (@. state.uv_spec.spheroidal/radius),
        (@. state.uv_spec.toroidal/radius))
    synthesis_vector!(void, uv_spec, sph)
end

function geopotential(model, state)
    (; radius, sph) = model
    radius^(-2)*synthesis_scalar!(void, state.gh_spec, sph)
end

function potential_vorticity(model, geopotential, vorticity)
    (; fcov, radius) = model
    @. ((radius^-2)*fcov+vorticity)/geopotential
end

diagnostics() = CookBook(; velocity, geopotential, vorticity, potential_vorticity)

# setup

function initial_state(model, gh, ulon, ulat)
    (; radius) = model
    gh_spec = analysis_scalar!(void, radius^2*gh, sph)
    uv_spec = analysis_vector!(void, vector_spat(-radius * ulat, radius * ulon), sph)
    RSW_state(gh_spec, uv_spec)
end

function setup(case, sph ; courant = 2.0, interval=3600.0)
    radius = case.params.R0

    model = RSW(sph, case.params.Omega, radius)

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
    return model, scheme, solver, state0
end

divisor(dt, T) = T / Int(ceil(T / dt))

# main program

@info toc("Initializing...")

sph = SHTnsSpheres.SHTnsSphere(512)
case = testcase(Williamson91{6}, Float64)
interval = 3600.0
model, scheme, solver, state0 = setup(case, sph ; interval)
book = diagnostics()

# Initial potential vorticity
pv = open(book; model, state=state0) do diags
    pv = diags.potential_vorticity
    pv = transpose(max.(0.0, pv))
end
# Create a Makie observable and make a plot from it
# When we later update pv, the plot will update too.
pv = Makie.Observable(upscale(pv))

# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
lons = bounds_lon(model.sph.lon[1,:]*(180/pi))[1:2:end]
lats = bounds_lat(model.sph.lat[:,1]*(180/pi))[1:2:end]
fig = orthographic(lons, lats, pv; colormap = :berlin)

state = deepcopy(state0)
solver! = solver(true)

@info toc("Starting simulation.")

@time let N=240 # number of hours to simulate
    # separate thread running the simulation
    channel = Channel(spawn=true) do ch
        nstep = Int(interval/solver!.dt)
        for iter in 1:N
            advance!(state, solver!, state, 0.0, nstep)
            pv_worker = open(book; model, state) do diags
                transpose(max.(0.0, diags.potential_vorticity))
            end
            put!(ch, pv_worker)
        end
        @info "Worker: finished"
    end
    # main thread making the movie
    record(fig, "$(@__DIR__)/PV.mp4", 1:N) do hour
        @info toc("") hour
        pv[] = upscale(take!(channel))
    end
end
