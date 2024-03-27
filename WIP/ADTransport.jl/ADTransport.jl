# # Inverse modelling
# Inversion of a forward transport model similar to SpectralTransport.jl
# The inverse problem searches for the initial condition given the final state.
# This problem is solved by minimizing a loss function
# whose gradients are computed via reverse AD with Zygote
# and fed into the Adam optimizer from Flux.Optimisers .

# ## Preamble

const debug=false
include("preamble.jl")

@time_imports begin
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere, void, erase,
                        synthesis_scalar!, analysis_vector!, divergence!,
                        sample_vector!, sample_scalar!

    import Flux, Zygote, ForwardDiff
    using Flux.Optimise: @withprogress, @logprogress
end

Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}") #src

@info toc("Local definitions")
includet("time.jl")

#=========== Transport scheme ===========#

struct FluxForm{Domain, UV}
    domain::Domain
    uv::UV
end

Base.show(io::IO, model::FluxForm) =
    print(io, "FluxFormTransport($(model.domain))")

Base.show(io::IO, ::Type{FluxForm{Domain, UV}}) where {Domain, UV} =
    print(io, "FluxFormTransport($Domain, UV}")

# ## Model definition

function tendencies!(dstate, model::FluxForm, f_spec, scratch, t)
    # get grid-point values of f
    f_spat = synthesis_scalar!(scratch.f_spat, f_spec, sph)
    # multiply velocity by g to obtain flux
    fluxlon   = @. scratch.fluxlon   = -f_spat*model.uv.ulon
    fluxcolat = @. scratch.fluxcolat = -f_spat*model.uv.ucolat
    # flux divergence
    flux_spat = (ucolat=fluxcolat, ulon=fluxlon)
    flux_spec = analysis_vector!(scratch.flux_spec, flux_spat, sph)
    return divergence!(dstate, flux_spec, sph)
end

# The above is enough for the AD-friendly, non-mutating form.
# Stuff below is only needed for the more efficient, mutating variant.

function scratch_space(::FluxForm, f_spec0::Vector{Complex{R}}) where R # R may be Dual
    spat = SHTnsSpheres.shtns_alloc_spat(R, sph)
    alloc() = similar(spat, R)
    spec() = similar(f_spec0)
    return (f_spat=alloc(), fluxlon=alloc(), fluxcolat=alloc(),
        flux_spec=(spheroidal=spec(), toroidal=spec()) )
end

model_dstate(::FluxForm, f_spec) = similar(f_spec)

#================ Model setup =============#

solid_body(x,y,z,lon,lat) = (sqrt(1-z*z), 0.)  # zonal solid-body "wind"

function initial_condition(f, (; domain)::FluxForm)
    f_spat = SHTnsSpheres.sample_scalar!(void, f, domain)
    return SHTnsSpheres.analysis_scalar!(void, f_spat, domain)
end

function optimal_step(dt, T)
    N = Int(ceil(T/dt))
    return N, T/N
end

function setup(sph; lmax = sph.lmax, courant = 2.0, velocity = solid_body, T=1.0)
    @info toc("Initializing FluxForm on $sph")
    uv = SHTnsSpheres.sample_vector!(void, velocity, sph)
    model = FluxForm(sph, uv)
    scheme = RungeKutta4(model)
    Nstep, dt = optimal_step(courant / sph.lmax, T)

    function forward(spec0) # non-mutating
        ## we copy spec0 because SHTns may erase its inputs
        solver = IVPSolver(scheme, dt, spec0)
        spec, t = advance!(void, solver, copy(spec0), 0.0, Nstep)
        return spec
    end

    function forward!(spec0) # mutating
        spec = copy(spec0)
        solver = IVPSolver(scheme, dt, spec, true)
        spec, t = advance!(spec0, solver, spec, 0.0, Nstep)
        return spec
    end

    F(x, y, z, lon, lat) = exp(-1000 * (1 - y)^4)        # initial "concentration"
    return model, scheme, initial_condition(F, model), forward, forward!
end

#=============== Optimization ============#

struct Loss{Fun,Target}
    fun::Fun
    target::Target
end

function (loss::Loss)(state)
    (; fun, target) = loss
    value = fun(state)
    return sum(abs2, value - target)
end

# ## Double-check that gradients are correct
# We check the Zygote gradient by computing
# a directional gradient with ForwardDiff

# complex dot product
@inline _cprod(z1, z2) = z1.re * z2.re + z1.im * z2.im
cprod(a, b) = @inline mapreduce(_cprod, +, a, b)

function check_grad(loss, state, dstate)
    f(x) = loss(@. state + x * dstate)
    fwd_grad = ForwardDiff.derivative(f, 0.0)
    @time zyg_grad = cprod(Zygote.gradient(loss, state)[1], dstate)
    @info typeof(loss) fwd_grad zyg_grad
    return nothing
end

# ## Inverse problem
# Adapted from `Flux.Optimise.train!`.
# In ML parlance, "model" is the set of coefficients to be optimized.
# Here it is the initial condition `state`.

function train!(loss, model, N, optim=Flux.Adam())
    tree = Flux.Optimisers.setup(optim, model)
    @info "initial loss" loss(model)
    @withprogress for i = 1:N
        l, g = Zygote.withgradient(loss, model)
        tree, model = Flux.Optimisers.update!(tree, model, g[1])
        Flux.@logprogress i / N loss=l
    end
    @info "final loss" loss(model)
    return model
end

#===================== main program ======================#

sph = SHTnsSpheres.SHTnsSphere(128);
model, scheme, state, forward, forward! = setup(sph);
dstate = randn(length(state)) .* state;

let state = copy(state) # do not touch state !
    target = forward(forward(state))
    @info typeof(target)
    ## @profview forward(state);

    loss1(state) = sum(abs2, state)
    loss2 = Loss(state->tendencies!(void, model, state, void, 0.), dstate)
    loss3 = Loss(forward, target)

    check_grad(loss1, state, dstate)
    check_grad(loss2, state, dstate)
    check_grad(loss3, state, dstate)

    optimal = train!(loss2, state, 100);
    ## @profview optimal = train!(loss3, state, 100);
    @time optimal = train!(loss3, state, 100);
end;

@time forward!(state);
