# # Inverse modelling
# Inversion of a forward transport model similar to SpectralTransport.jl
# The inverse problem searches for the initial condition given the final state.
# This problem is solved by minimizing a loss function
# whose gradients are computed via reverse AD with Zygote
# and fed into the Adam optimizer from Flux.Optimisers .

# ## Preamble

const debug = false
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

# ## Loss function
# Used for sanity checks and for inverse modelling.

## callable struct for loss function
## stores the forward model, its (constant) parameters and the target
struct Loss{Forward,Target,Params}
    forward::Forward
    target::Target
    params::Params
end
Loss(forward, target) = Loss(forward, target, nothing)

function (loss::Loss)(state)
    (; forward, target, params) = loss
    if isnothing(params)
        new = forward(state)
    else
        new = forward(state, params, 0.0) # SciML signature du = f(u,p,t)
    end
    return sum(abs2, new - target)
end

## Double-check that gradients are correct
## We check the Zygote gradient by computing
## a directional gradient with ForwardDiff

@inline _cprod(z1, z2) = z1.re * z2.re + z1.im * z2.im
cprod(a, b) = @inline mapreduce(_cprod, +, a, b)

function forward_grad(loss, state, dstate)
    state = copy(state)
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
        Flux.@logprogress i / N loss = l
    end
    @info "final loss" loss(model)
    return model
end

# ## Forward model definition
# We adopt a non-mutating approach to allow backward AD with Zygote

struct FluxFormTransport{Domain,UV,B}
    domain::Domain
    uv::UV
    manager::B
end

Base.show(io::IO, model::FluxFormTransport) =
    print(io, "FluxFormTransport($(model.domain)) running on $(model.manager)")

Base.show(io::IO, ::Type{FluxFormTransport{Domain,UV,B}}) where {Domain,UV,B} =
    print(io, "FluxFormTransport($Domain, ..., $B}")

## FluxFormTransport instances are callable, so that we can
## create a loss function from them.

## This form conforms to SciML requirements.
## Although we do not optimize model parameters, only the initial condition,
## SciML *requires* that we provide model parameters `p`, that they be Array-like
## *and* that they have a non-zero gradient (are used in the computation) !
## Hence we let `p` be an `ArrayPartition` storing the wind field.
function (model::FluxFormTransport)(spec, p::ArrayPartition, t)
    transport_flux(spec, (ucolat = p.x[1], ulon = p.x[2]), model.domain, model.manager)
end

## This form is for use as the `forward` field of a loss function.
## Wind field is not provided as parameter, we use the one stored in `model``
(model::FluxFormTransport)(spec) =
    transport_flux(spec, model.uv, model.domain, model.manager)

const nb_calls = Ref(0)

function transport_flux(f_spec, (ucolat, ulon), sph, manager)
    nb_calls[] = nb_calls[] + 1
    f_spat = synthesis_scalar(f_spec, sph, manager) # get grid-point values of f
    fluxlon = @. -ulon * f_spat
    fluxcolat = @. -ucolat * f_spat
    return analysis_div((ucolat = fluxcolat, ulon = fluxlon), sph, manager)
end

solid_body(x, y, z) = (sqrt(1 - z * z), 0.0)  # zonal solid-body "wind"

synthesis_scalar(spec, sph::SHTnsSphere, manager) = SHTnsSpheres.synthesis_scalar(spec, sph)
analysis_div(uv, sph::SHTnsSphere, manager) = SHTnsSpheres.analysis_div(uv, sph)

function initial_condition(f, model::FluxFormTransport{SHTnsSpheres.SHTnsSphere})
    (; domain, manager) = model
    f = SHTnsSpheres.sample_scalar(f, domain)
    f = SHTnsSpheres.analysis_scalar(f, domain)
end

# ## Main program

function setup(
    sph;
    alg = RK4(),
    Model = FluxFormTransport,
    lmax = 128,
    courant = 2.0,
    velocity = solid_body,
) # , options...)
    F(x, y, z) = exp(-1000 * (1 - y)^4)        # initial "concentration"
    @info toc("Initializing SHTns for lmax=$lmax")
    (; ucolat, ulon) = SHTnsSpheres.sample_vector(velocity, sph)
    p = ArrayPartition(ucolat, ulon)
    model = Model(sph, (ucolat, ulon), LoopManagers.PlainCPU())
    dt = courant / sph.lmax
    tspan = (0, 10 * dt)

    function forward(spec0)
        ## we copy spec0 because SHTns may erase its inputs
        problem = ODEProblem(model, copy(spec0), tspan, p)
        return last(
            solve(problem; alg, dt, adaptive = false, saveat = last(tspan), options...).u,
        )
    end
    return model, initial_condition(F, model), forward, p
end

sph = SHTnsSpheres.SHTnsSphere(128);
options = (; sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()))
model, state, forward, p = setup(sph)
dstate = randn(length(state)) .* state;

let state = copy(state) # do not touch state !
    target = forward(forward(state))
    @info typeof(target)
    ## @profview forward(state);

    loss1(state) = sum(abs2, state)
    loss2 = Loss(model, dstate, p)
    loss3 = Loss(forward, target)

    forward_grad(loss1, state, dstate)
    forward_grad(loss2, state, dstate)
    forward_grad(loss3, state, dstate)
end

loss4 = Loss(forward, copy(state)) # target = frozen copy of `state`
forward_grad(loss4, state, dstate)

# @profview optim_state = train!(loss4, copy(state), 100, Flux.Adam());
GC.gc()
nb_calls[] = 0
@time optim_state = train!(loss4, copy(state), 100, Flux.Adam());
@info "Number of calls to model" nb_calls[]

# obey SciML expectations of the r.h.s of an ODEProblem
struct SciMLFun{Model}
    model::model
end
(fun::SciMLFun)(u, p, t) = tendencies(fun.model, void, u, void, p, t) # du = tendencies(model, du, u, scratch, params, t)

# obey our expectations for the r.h.s. of an ODE problem
struct RHSFun{Model,Params}
    model::Model
    params::Params
end
(fun::RHSFun)(du, u, scratch, t) = fun.f(du, u, scratch, fun.params, t) # du = fun(du, u, scratch, params)

includet("time.jl")

struct ODE end

tendencies(::Void, ::ODE, u, ::Void, t) = copy(u)
tendencies(du::A, ::ODE, u::A, ::Nothing, t) where {A} = copy!(du, u)

scratch_space(::ODE, u0) = nothing
model_dstate(::ODE, u0) = similar(u0)

function myloss(x0)
    scheme = RungeKutta4(ODE())
    solver = IVPSolver(scheme, 0.01, x0, false)
    x, t = advance(void, solver, x0, 0.0, 100)
    sum(abs2, x)
end

x = randn(1000)
Zygote.gradient(myloss, x)
@btime Zygote.gradient(myloss, $x)

function loss(x0, dx, t)
    scheme = RungeKutta4(ODE())
    x = @. x0 + t * dx
    solver = IVPSolver(scheme, 0.01, x, true)
    y = advance(similar(x), solver, x, 0.0, 100)
    sum(abs2, y)
end

x0 = randn(1000)
dx = randn(1000)
ForwardDiff.derivative(t->loss(x0, dx, t), 0.0)

@btime ForwardDiff.derivative(t->loss($x0, $dx, t), 0.0)
