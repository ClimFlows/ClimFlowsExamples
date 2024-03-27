# # Rotating shallow water

# ## Preamble

const debug = false
include("preamble.jl")

@time_imports begin
    using MutatingOrNot: void, Void
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

    using ClimFlowsTestCases: Williamson91, testcase, describe, initial_flow

    using CFTimeSchemes: CFTimeSchemes, advance!
end

struct RSW{F}
    sph::SHTnsSphere
    fcov::Matrix{F}
    radius::F
end

function Base.show(io::IO, (; sph, fcov, radius)::RSW)
    Omega = maximum(fcov) / 2 * radius^-2
    print(io, "RSW($sph, Omega=$Omega, radius=$radius)")
end

# these "constructors" are *required* for type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
RSW_state(gh_spec, uv_spec) = (; uv_spec, gh_spec)

function CFTimeSchemes.tendencies!(dstate, model::RSW, state, scratch, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; sph, radius, fcov) = model
    #    (; flux, gh, uv, B, B_spec, zeta, zeta_spec, qflux, qflux_spec) = scratch
    (; uv_spec, gh_spec) = state
    duv_spec, dgh_spec = dstate.uv_spec, dstate.gh_spec
    invrad2 = radius^-2

    # mass budget: ∂gh/∂t = -∇(gh*ux, gh*uy)
    # uv is the momentum 1-form = radius^2*velocity
    # gh is the 2-form radius^2*gh
    # => scale flux by radius^-2
    gh = synthesis_scalar!(scratch.gh, state.gh_spec, sph)
    uv = synthesis_vector!(scratch.uv, state.uv_spec, sph)
    flux = vector_spat(
        (@. scratch.flux.ucolat = -invrad2 * gh * uv.ucolat),
        (@. scratch.flux.ulon = -invrad2 * gh * uv.ulon),
    )
    flux_spec = analysis_vector!(scratch.flux_spec, flux, sph)
    dgh_spec = divergence!(dgh_spec, flux_spec, sph)

    # momentum budget:
    # B = u^2+v^2)/2 + gh
    # ∂u/∂t = (f+ζ)v - ∂B/∂x
    # ∂v/∂t = -(f+ζ)u - ∂B/∂y
    # uv is momentum = radius^2*velocity
    # fcov, ζ and gh are 2-forms
    # => scale B and qflux by radius^-2
    zeta_spec = curl!(scratch.zeta_spec, uv_spec, sph)
    zeta = synthesis_scalar!(scratch.zeta, zeta_spec, sph)
    qflux = vector_spat(
        (@. scratch.qflux.ucolat = invrad2 * (zeta + fcov) * uv.ulon),
        (@. scratch.qflux.ulon = -invrad2 * (zeta + fcov) * uv.ucolat),
    )
    B = @. scratch.B = invrad2 * (gh + (uv.ucolat^2 + uv.ulon^2) / 2)
    B_spec = analysis_scalar!(scratch.B_spec, B, sph)
    qflux_spec = analysis_vector!(scratch.qflux_spec, qflux, sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )

    return RSW_state(dgh_spec, duv_spec)
end

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

# setup

function setup(case, sph, courant = 2.0)
    radius = case.params.R0
    f(n) = map((lon, lat) -> initial_flow(lon, lat, case)[n], sph.lon, sph.lat)
    ulon, ulat, gH = f(1), f(2), f(3)

    uv_spec = analysis_vector!(void, vector_spat(-(radius^2) * ulat, (radius^2) * ulon), sph)
    gh_spec = analysis_scalar!(void, gH, sph)
    state0 = RSW_state(gh_spec, uv_spec)

    fcov = (2 * radius^2 * case.params.Omega) * sph.z
    model = RSW(sph, fcov, radius)

    # time step based on maximum angular velocity of gravity waves
    umax = sqrt(maximum(@. ulon^2 + ulat^2))
    cmax = (umax + sqrt(case.params.gH0)) / radius
    dt = courant / cmax / sqrt(sph.lmax * sph.lmax + 1)

    scheme = CFTimeSchemes.RungeKutta4(model)
    solver(mutating=false) = CFTimeSchemes.IVPSolver(scheme, dt, state0, mutating)
    return model, scheme, solver, state0
end

ideal_time_step(dt, T) = T / Int(ceil(T / dt))

# main program

sph = SHTnsSpheres.SHTnsSphere(128)
case = testcase(Williamson91{6}, Float64)
model, scheme, solver, state0 = setup(case, sph);

if false
    using BenchmarkTools
    future = deepcopy(state0);

    advance!(future, solver(true), state0, 0.0, 100);
    @profview advance!(future, solver(true), state0, 0.0, 100);
    @btime advance!($future, solver(true), $state0, 0.0, 100);

    advance!(void, solver(), state0, 0.0, 100);
    @profview advance!(void, solver(), state0, 0.0, 100);
    @btime advance!(void, solver(), $state0, 0.0, 100);
end
