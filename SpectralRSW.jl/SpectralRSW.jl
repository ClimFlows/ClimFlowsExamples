# # Rotating shallow water

## Preamble

import Pkg
const cuda_project = joinpath(@__DIR__, "CUDA", "Project.toml")
const use_cuda = Base.active_project() == cuda_project
Pkg.activate(use_cuda ? dirname(cuda_project) : joinpath(@__DIR__, "CPU"))

const debug = false
include("preamble.jl")

@time_imports begin
    if use_cuda
        import CUDA
        CUDA.allowscalar(false)
    end

    using MutatingOrNot: void, Void
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, void, erase,
        analysis_scalar!, synthesis_scalar!,
        analysis_vector!, synthesis_vector!,
        divergence!, curl!, sample_vector!, sample_scalar!

    using ClimFlowsTestCases: Williamson91, describe, initial

    using CFTimeSchemes: CFTimeSchemes, advance!

    using GeoMakie, CairoMakie, ColorSchemes
    using CookBooks
    import ClimFlowsPlots.SpectralSphere as SpectralPlots
end

struct HyperDiffusion{F}
    n::Int
    nu::F
end

function (hd::HyperDiffusion)(storage, coefs, sph)
    (; n, nu), (; laplace, lmax) = hd, sph
    @. storage = (1-nu*(-laplace/(lmax*(lmax+1)))^n)*coefs
end

struct RSW{F, HD, S<:SHTnsSphere, A<:AbstractMatrix{F}}
    sph::S
    fcov::A # 2-form: fcov = 2Ω * radius^2 * sin(latitude)
    radius::F
    filter::HD
end

RSW{F, HD}(sph, fcov, radius, filter) where {F, HD} =
    RSW(sph, convert(Matrix{F}, fcov), convert(F, radius), convert(HD, filter))

RSW(sph, Omega::Real, radius, hd) = RSW(sph, (2 * radius^2 * Omega) * sph.z, radius, hd)

function Base.show(io::IO, (; sph, fcov, radius)::RSW)
    Omega = maximum(fcov) / 2 * radius^-2
    print(io, "RSW($sph, Omega=$Omega, radius=$radius)")
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
RSW_state(gh_spec, uv_spec) = (; gh_spec, uv_spec)

function CFTimeSchemes.tendencies!(dstate, scratch, model::RSW, state, t)
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

    scratch = (; B, zeta, gh, flux, uv, qflux, B_spec, zeta_spec, flux_spec, qflux_spec)
    return RSW_state(dgh_spec, duv_spec), scratch
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
    (; radius, sph) = model
    gh_spec = analysis_scalar!(void, radius^2*gh, sph)
    uv_spec = analysis_vector!(void, vector_spat(-radius * ulat, radius * ulon), sph)
    RSW_state(gh_spec, uv_spec)
end

function setup(case, sph ; courant = 2., interval=3600.0, hd_n=8, hd_nu=1e-2)
    F = eltype(sph.z)
    radius = F(case.params.R0)
    hd = HyperDiffusion(hd_n, F(hd_nu))
    model = RSW(sph, F(case.params.Omega), radius, hd)

    # The analytical test case is sampled on the host once.
    lon, lat = Array(sph.lon), Array(sph.lat)
    f(n) = map((lon, lat) -> F(case(lon, lat)[n]), lon, lat)
    gh, ulon, ulat = f(1), f(2), f(3)
    to_grid(a) = copyto!(similar(sph.z), a)
    state0 = initial_state(model, to_grid(gh), to_grid(ulon), to_grid(ulat))

    # time step based on maximum angular velocity of gravity waves
    umax = sqrt(maximum(@. ulon^2 + ulat^2))
    cmax = (umax + sqrt(F(case.params.gH0))) / radius
    dt = F(courant) / cmax / sqrt(F(sph.lmax * sph.lmax + 1))
    dt = divisor(dt, F(interval))
    @info "Time step" umax, cmax, dt

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver() = CFTimeSchemes.IVPSolver(scheme, dt)
    solver(state0) = CFTimeSchemes.IVPSolver(scheme, dt, state0, nothing)
    return model, scheme, solver, state0
end

divisor(dt, T) = T / ceil(Int, T / dt)

# main program

@info toc("Initializing spherical harmonics...")
sph = use_cuda ? SHTnsSphere(128; backend=:cuda, precision=Float32) : SHTnsSphere(128)

@info toc("Setting up...")
case = Williamson91{6}(eltype(sph.z))
interval = 3600.0
model, scheme, solver, state0 = setup(case, sph ; interval)
book = diagnostics()

# Initial potential vorticity
pv = open(book; model, state=state0) do diags
    pv = diags.potential_vorticity
    transpose(Array(max.(zero(eltype(pv)), pv)))
end
# Create a Makie observable and make a plot from it
# When we later update pv, the plot will update too.
pv = Makie.Observable(SpectralPlots.upscale(pv))

# see https://docs.makie.org/stable/explanations/colors/index.html for colormaps
lons = SpectralPlots.bounds_lon(Array(model.sph.lon)[1,:]*(180/pi))[1:2:end]
lats = SpectralPlots.bounds_lat(Array(model.sph.lat)[:,1]*(180/pi))[1:2:end]
fig = SpectralPlots.orthographic(lons, lats, pv; colormap = :berlin)

solver! = solver(state0)
stream = use_cuda ? CUDA.stream() : nothing

@info toc("Starting simulation.")

@time let N=240 # number of hours to simulate
    # separate thread running the simulation
    channel = Channel(spawn=true) do ch
        # SHTns uses the stream on which its plan was created.
        use_cuda && CUDA.stream!(stream)
        state = deepcopy(state0)
        nstep = round(Int, interval/solver!.dt)
        for iter in 1:N
            for istep in 1:nstep
                advance!(state, solver!, state, zero(solver!.dt), 1)
                (; gh_spec, uv_spec) = state
                (; toroidal) = uv_spec
                toroidal = model.filter(toroidal, toroidal, model.sph)
                uv_spec = vector_spec(uv_spec.spheroidal, toroidal)
                state = RSW_state(gh_spec, uv_spec)
            end
            pv_worker = open(book; model, state) do diags
                pv = diags.potential_vorticity
                transpose(Array(max.(zero(eltype(pv)), pv)))
            end
            put!(ch, pv_worker)
        end
        @info "Worker: finished"
    end
    # main thread making the movie
    record(fig, "$(@__DIR__)/PV.mp4", 1:N) do hour
        @info toc("") hour
        pv[] = SpectralPlots.upscale(take!(channel))
    end
end
