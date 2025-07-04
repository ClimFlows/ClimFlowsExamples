# Forward transport model similar to SpectralTransport.jl

# the following two lines may be commented out after being executed once
# Pkg.Registry.add(url = "https://github.com/ClimFlows/JuliaRegistry.git")
# Pkg.add(["ForwardDiff", "Zygote", "SHTnsSpheres", "CFTimeSchemes"])

using ForwardDiff, Zygote, SHTnsSpheres, CFTimeSchemes
Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}")

using CFTimeSchemes: IVPSolver, advance!

using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, void, erase,
    analysis_scalar!, synthesis_scalar!, analysis_vector!, divergence!,
    sample_vector!, sample_scalar!

function setup(nlat, courant, T)
    sph = SHTnsSpheres.SHTnsSphere(nlat)
    state = SHTnsSpheres.sample_scalar!(void, F, sph)

    uv = SHTnsSpheres.sample_vector!(void, solid_body, sph)
    model = FluxForm(sph, uv)
    scheme = CFTimeSchemes.RungeKutta4(model)
    Nstep, dt = optimal_step(courant / sph.lmax, T)

    function forward(state0)
        spec0 = analysis_scalar!(void, state0, sph)
        if true
            solver = IVPSolver(scheme, dt)
            spec1, t = advance!(void, solver, spec0, zero(solver.dt), Nstep)
        else
            spec1 = spec0
        end
        state1 = synthesis_scalar!(void, spec1, sph)
        return state1
    end

    fwd_with_adjoint(state) = Zygote.pullback(forward, state)

    @showtime forward(state)
    @showtime forward(state)

    @showtime state1, adjoint = fwd_with_adjoint(state)
    @showtime state1, adjoint = fwd_with_adjoint(state)

    dstate = randn(eltype(state1), size(state1))
    @showtime dstate0 = adjoint(dstate)
    @showtime dstate0 = adjoint(dstate)

    return (; scheme, Nstep, dt, state, forward, fwd_with_adjoint)
end


struct FluxForm{Domain, UV}
    domain::Domain
    uv::UV
end

function CFTimeSchemes.tendencies!(dstate, scratch, model::FluxForm, f_spec, t)
    sph = model.domain
    # get grid-point values of f
    f_spat = synthesis_scalar!(scratch.f_spat, f_spec, sph)
    # multiply velocity by g to obtain flux
    fluxlon   = @. scratch.fluxlon   = -f_spat*model.uv.ulon
    fluxcolat = @. scratch.fluxcolat = -f_spat*model.uv.ucolat
    # flux divergence
    flux_spat = (ucolat=fluxcolat, ulon=fluxlon)
    flux_spec = analysis_vector!(scratch.flux_spec, flux_spat, sph)
    return divergence!(dstate, flux_spec, sph), void
end

function optimal_step(dt, T)
    N = Int(ceil(T/dt))
    return N, T/N
end

function initial_condition(f, (; domain)::FluxForm)
    f_spat = SHTnsSpheres.sample_scalar!(void, f, domain)
    return SHTnsSpheres.analysis_scalar!(void, f_spat, domain)
end

solid_body(x, y, z, lon, lat) = (sqrt(1-z*z), 0.)  # zonal solid-body "wind"
F(x, y, z, lon, lat) = exp(-1000 * (1 - y)^4)        # initial "concentration"
