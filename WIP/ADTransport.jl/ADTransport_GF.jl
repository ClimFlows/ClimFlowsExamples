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
    using GFModels, GFTimeSchemes, GFDomains, GFBackends
    import Flux, Zygote, ForwardDiff
    using Flux.Optimise: @withprogress, @logprogress
    import SHTns_jll
    import GFModels, GFTimeSchemes, GFDomains, GFBackends
end

Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}") #src

@info toc("Local definitions")

#================== Generic ======================#

abstract type Transport end

struct FluxFormTransport{Domain, UV, B} <: Transport
    domain::Domain
    uv::UV
    manager::B
end

Base.show(io::IO, model::FluxFormTransport) =
    print(io, "FluxFormTransport($(model.domain)) running on $(model.manager)")

Base.show(io::IO, ::Type{FluxFormTransport{Domain, UV, B}}) where {Domain, UV, B} =
    print(io, "FluxFormTransport($Domain, UV, $B}")

# ## Model definition
# We adopt a non-mutating approach to allow backward AD with Zygote

const nb_calls=Ref(0)

function transport_flux(f_spec, (ucolat, ulon), sph, manager)
    nb_calls[] = nb_calls[]+1
    f_spat = synthesis_scalar(f_spec, sph, manager) # get grid-point values of f
    fluxlon = @. -ulon * f_spat
    fluxcolat = @. -ucolat * f_spat
    return analysis_div((ucolat=fluxcolat, ulon=fluxlon), sph, manager)
end

solid_body(x,y,z) = (sqrt(1-z*z), 0.)  # zonal solid-body "wind"

struct Loss{Scheme,Target}
    scheme::Scheme
    target::Target
end

function (loss::Loss)(state)
    (; scheme, target) = loss
    new = scheme(state)
    return sum(abs2, new - target)
end

# ## Double-check that gradients are correct
# We check the Zygote gradient by computing
# a directional gradient with ForwardDiff

@inline _cprod(z1, z2) = z1.re * z2.re + z1.im * z2.im
cprod(a, b) = @inline mapreduce(_cprod, +, a, b)

function forward_grad(loss, state, dstate)
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

function train!(loss, model, N, optim)
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

#=============== GeoFluids ==================#

function alloc_field(name, model::FluxFormTransport)
    (u,v) = model.uv
    GFDomains.allocate_field(name, model.domain, eltype(u))
end
function alloc_fields(names, model::FluxFormTransport)
    (u,v) = model.uv
    GFDomains.allocate_fields(names, model.domain, eltype(u))
end

## since velocity is prescribed, the model state consists only of the transported field = scalar
GFModels.allocate_state(model::Transport) = alloc_field(:scalar_spec, model)
GFModels.allocate_scratch(model::Transport) = alloc_fields((:scalar_spat, :vector_spat, :vector_spec), model)

GFModels.tendencies(f, model::FluxFormTransport, manager) =
    transport_flux(f, model.uv, model.domain, manager)


synthesis_scalar(spec, sph::GFDomains.SpectralSphere, manager) = GFDomains.synthesis_scalar(spec, sph, manager)
analysis_div(uv, sph::GFDomains.SpectralSphere, manager) = GFDomains.analysis_div(uv, sph, manager)

# ## Model initialization
# To run a simulation we must provide an initial condition for the model state.
# We will let `GFExperiments.timeloop` handle time integration.

function setup(Scheme=GFTimeSchemes.RK25, Model=FluxFormTransport, lmax=128, courant=2.0, velocity=solid_body)
    F(x,y,z) = exp(-1000*(1-y)^4)        # initial "concentration"
    dt = courant/lmax ;
    @info toc("Initializing SHTns for lmax=$lmax")
    sph = GFDomains.SHTns_sphere(lmax) ;
    model   = Model(sph, GFDomains.sample_vector(velocity, sph), GFBackends.PlainCPU()) ;
    return model, Scheme(model, dt), initial_condition(F, model)
end

function initial_condition(f, model::FluxFormTransport{<:GFDomains.SHTns_sphere})
    (; domain, manager) = model
    f = GFDomains.sample_scalar(f, domain)
    f = GFDomains.analysis_scalar(f, domain, manager)
    return GFDomains.truncate_scalar!(f, domain)
end

function (loss::Loss{<:GFTimeSchemes.TimeScheme})(state)
    (; scheme, target) = loss
    for i in 1:10
       state = GFTimeSchemes.advance(state, scheme, scheme.model.manager)
    end
    return sum(abs2, state - target)
end

function (loss::Loss{<:Transport})(state)
    model = loss.scheme
    dstate = GFModels.tendencies(state, model, model.manager)
    return sum(abs2, dstate)
end

# ## Main program

model, scheme, state = setup() ;

dstate = randn(length(state)) .* state;
## we copy state because SHTns may overwrite its inputs
target = GFTimeSchemes.advance(copy(state), scheme, model.manager);
target = GFTimeSchemes.advance(target, scheme, model.manager);

loss1(state) = sum(abs2, state)
loss2 = Loss(model, dstate)
loss3 = Loss(scheme, target)
loss4 = Loss(scheme, state)

forward_grad(loss1, state, dstate)
forward_grad(loss2, state, dstate)
forward_grad(loss3, state, dstate)
forward_grad(loss4, state, dstate)

nb_calls[] = 0
@time optim_state = train!(loss4, state, 100, Flux.Adam());
@info "Number of calls to model" nb_calls[]
