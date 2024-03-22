# void type to merge mutating and non-mutating implementations

struct Void end
const void = Void()
Base.show(io::IO, ::Void) = print(io, "void")

Base.getproperty(::Void, ::Symbol) = void
Base.getindex(::Void, args...) = void
Base.iterate(::Void, state = nothing) = (void, nothing)
Base.collect(::Void) = void

#============ ClimFlows time integration API ===============#

# Model
"""
    k = model_dstate(model, u)
Returns an object in which we can store tendencies ; u0 is passed to handle ForwardDiff.Dual
"""
function model_dstate end

"""
    space = scratch_space(model, u0)
    space = scratch_space(scheme, u0)
scratch space used to compute tendencies or keep sub-stages (RK); u0 is passed to handle ForwardDiff.Dual
"""
function scratch_space end

"""
    dstate = tendencies(dstate, model, u, scratch, t)
Computes tendencies (mutating)
"""
function tendencies end

"""
    dstate = tendencies!(model, u, t)
computes tendencies (non-mutating). Default implementation:
    tendencies!(model, u, t) = tendencies(void, model, u, void, t)
"""
tendencies!(model, u, t) = tendencies(void, model, u, void, t)

# Time scheme
"""
    stages = time_stages(scheme, model, u0)
returns an object storing tendencies at sub-stages (RK). u0 is passed to handle ForwardDiff.Dual
"""
function time_stages end

"""
    future, t = advance(future, scheme, present, t, dt, scratch)
    future, t = advance(void, scheme, present, t, dt, void)
Integrate in time by one time step, respectively mutating (non-allocating) and non-mutating
"""
function advance end

"""
    new = update(new, new, increment, factor)
    new = update(new, old, increment, factor)
    new = update(void, old, increment, factor)
Respectively:
    @. new = old + factor*increment
    new = @. old + factor*increment
Operates recursively on nested tuples / named tuples
"""
update(::Void, u, k, dt) = @. u + dt * k
update(::Void, u, ka, a, kb, b) = @. u + a * ka + b * kb
update(::Void, u, ka, a, kb, b, kc, c) = @. u + a * ka + b * kb + c * kc
update(::Void, u, ka, a, kb, b, kc, c, kd, d) = @. u + a * ka + b * kb + c * kc + d * kd

update(x, u, k, dt) = @. x = u + dt * k
update(x, u, ka, a, kb, b) = @. x = u + a * ka + b * kb
update(x, u, ka, a, kb, b, kc, c) = @. x = u + a * ka + b * kb + c * kc
update(x, u, ka, a, kb, b, kc, c, kd, d) = @. x = u + a * ka + b * kb + c * kc + d * kd

# IVPSolver
"""
    future, t = advance(future, solver, present, t, N)
    future, t = advance(void, solver, present, t, N)
"""
function advance(storage::Union{Void, State}, (; dt, scheme, scratch)::IVPSolver, state::State, t, N::Int) where State
    @assert N>0
    state = advance(storage, scheme, state, t, dt, void)
    for i=2:N
        state = advance(storage, scheme, state, t+(i-1)*dt, dt, scratch)
    end
    return state
end

"""
    solver = IVPSolver(model, scheme, dt, u0, mutating=false)
Default is non-mutating, which should work easier with AD.
With `mutating=true`, `forward` should not allocate and have better performance.
"""
struct IVPSolver{F,Scheme,Scratch}
    dt::F # time step
    scheme::Scheme
    scratch::Scratch # scratch space, or void

    function IVPSolver(scheme::S, dt::F, u0, mutating) where {F,S}
        scr = mutating ? scratch_space(scheme, u0) : void
        new{F,S,typeof(scr)}(dt, scheme, scr)
    end
end

struct RungeKutta4{Model}
    model::Model
end

function scratch_space((; model)::RungeKutta4, u0)
    k() = model_dstate(model, u0)
    return (scratch = scratch_space(model, u0), k0 = k(), k1 = k(), k2 = k(), k3 = k())
end

function advance(future, (; model)::RungeKutta4, u0, t0, dt, (; scratch, k0, k1, k2, k3))
    k0 = tendencies(k0, model, u0, scratch, t0)
    # u1 = u0 + dt/2*k0
    u1 = update(future, u0, k0, dt / 2)
    k1 = tendencies(k1, model, u1, scratch, t0 + dt / 2)
    # u2 = u1 - dt/2*k0 + dt/2*k1 == u0 + dt/2*k1
    u2 = update(future, u1, k0, -dt / 2, k1, dt / 2)
    k2 = tendencies(k2, model, u2, scratch, t0 + dt / 2)
    # u3 = u2 - dt/2*k1 + dt*k2 == u0 + dt*k2
    u3 = update(future, u2, k1, -dt / 2, k2, dt)
    k3 = tendencies(k3, model, u3, scratch, t0 + dt)
    # u4 = u0 + (k0+k3)*(dt/6) + (k1+k2)*(dt/3)
    future = update(future, u3, k0, dt / 6, k1, dt / 3, k2, -2dt / 3, k3, dt / 6)
    return future
end
