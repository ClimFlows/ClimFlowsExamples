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
    using SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP
    import Flux, Zygote, ForwardDiff
    using SHTnsSpheres: SHTnsSpheres, SHTnsSphere
    using OrdinaryDiffEq: solve, RK4, ODEProblem
    using RecursiveArrayTools: ArrayPartition

    using Flux.Optimise: @withprogress, @logprogress
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

function transport_flux(f_spec, (ucolat, ulon), sph, manager)
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

#================== ClimFlows + OrdinaryDiffEq ===============#

synthesis_scalar(spec, sph::SHTnsSphere, manager) = SHTnsSpheres.synthesis_scalar(spec, sph)
analysis_div(uv, sph::SHTnsSphere, manager) = SHTnsSpheres.analysis_div(uv, sph)

function initial_condition(f, model::FluxFormTransport{SHTnsSpheres.SHTnsSphere})
    (; domain, manager) = model
    f = SHTnsSpheres.sample_scalar(f, domain)
    f = SHTnsSpheres.analysis_scalar(f, domain)
end

function (loss::Loss{FluxFormTransport{<:SHTnsSpheres.SHTnsSphere}})(state)
    model = loss.scheme
    dstate = model(state)
    return sum(abs2, dstate)
end

(model::FluxFormTransport)(spec) = transport_flux(spec, model.uv, model.domain, model.manager)

function (model::FluxFormTransport)(spec, p, t)
    transport_flux(spec, (ucolat=p.x[1], ulon=p.x[2]), model.domain, model.manager)
end

function setup(sph; alg=RK4(), Model=FluxFormTransport, lmax=128, courant=2.0, velocity=solid_body) # , options...)
    F(x,y,z) = exp(-1000*(1-y)^4)        # initial "concentration"
    @info toc("Initializing SHTns for lmax=$lmax")
    (; ucolat, ulon) = SHTnsSpheres.sample_vector(velocity, sph)
    p = ArrayPartition(ucolat, ulon)
    model = Model(sph, (ucolat, ulon), LoopManagers.PlainCPU())
    dt = courant/sph.lmax
    tspan = (0, 10*dt)

    function forward(spec0)
        problem = ODEProblem(model, spec0, tspan, p)
        return last(solve(problem ; alg, dt, adaptive=false, saveat=last(tspan), options...).u)
    end
    return model, initial_condition(F, model), forward
end

sph = SHTnsSpheres.SHTnsSphere(128);
options = (; sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
model, state, forward = setup(sph)

dstate = randn(length(state)) .* state;
target = forward(forward(state));
@info typeof(target)
# @profview forward(state);

loss1(state) = sum(abs2, state)
loss2 = Loss(model, dstate)
loss3 = Loss(forward, target)

forward_grad(loss1, state, dstate)
forward_grad(loss2, state, dstate)
forward_grad(loss3, state, dstate)

loss4 = Loss(forward, copy(state))
forward_grad(loss4, state, dstate)
@profview optim_state = train!(loss4, state, 100, Flux.Adam());
