module CFTimeSchemes

# Void type and instance to merge mutating and non-mutating implementations
using MutatingOrNot: void, Void

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
    dstate = tendencies!(dstate, model, u, scratch, t) # Mutating
    dstate = tendencies!(void, model, u, void, t) # Non-mutatingœ
Computes tendencies. Pass `void` to arguments `dstate` and `scratch` for non-mutating variant.
"""
function tendencies! end

# Time scheme
"""
    stages = time_stages(scheme, model, u0)
returns an object storing tendencies at sub-stages (RK). u0 is passed to handle ForwardDiff.Dual
"""
function time_stages end

"""
    future, t = advance!(future, scheme, present, t, dt, scratch)
    future, t = advance!(void, scheme, present, t, dt, void)
Integrate in time by one time step, respectively mutating (non-allocating) and non-mutating
"""
function advance! end

"""
    new = update!(new, new, increment, factor)
    new = update!(new, old, increment, factor)
    new = update!(void, old, increment, factor)
Respectively equivalent to:
    @. new += factor*increment
    @. new = old + factor*increment
    new = @. old + factor*increment
Operates recursively on nested tuples / named tuples
"""
update!(x, u::S, a::F, ka::S) where {F, S} = LinUp((a,))(x,u,(ka,))::S
update!(x, u::S, a::F, ka::S, b::F, kb::S) where {F, S} = LinUp((a,b))(x,u,(ka,kb))::S
update!(x, u::S, a::F, ka::S, b::F, kb::S, c::F, kc::S) where {F, S} = LinUp((a,b,c))(x,u,(ka,kb,kc))::S
update!(x, u::S, a::F, ka::S, b::F, kb::S, c::F, kc::S, d::F, kd::S) where {F, S} = LinUp((a,b,c,d))(x,u,(ka,kb,kc,kd))::S

struct LinUp{F,N} # linear update with coefs=(a,b, ...)
    coefs::NTuple{N,F}
end

# LinUp on arrays
# Currently limited to 4-stage schemes

function (up::LinUp{F,1})(x, u::A, ks::NTuple{1,A}) where {F, A<:Array}
    ka, = ks
    a, = up.coefs
    return @. x = muladd(a, ka, u)
end
function (up::LinUp{F,2})(x, u::A, ks::NTuple{2,A}) where {F, A<:Array}
    ka, kb = ks
    a, b = up.coefs
    return @. x = muladd(b, kb, muladd(a, ka, u))
end
function (up::LinUp{F,3})(x, u::A, ks::NTuple{3,A}) where {F, A<:Array}
    ka, kb, kc = ks
    a, b, c = up.coefs
    return @. x = muladd(c, kc, muladd(b, kb, muladd(a, ka, u)))
end
function (up::LinUp{F,4})(x, u::A, ks::NTuple{4,A}) where {F, A<:Array}
    ka, kb, kc, kd = ks
    a, b, c, d = up.coefs
    return @. x = muladd(d, kd, muladd(c, kc, muladd(b, kb, muladd(a, ka, u))))
end

# LinUp on named tuples

svoid(::Void, u) = map(uu->void, u)
svoid(x, u) = x

function (up::LinUp{F,N})(x, u::NT, ka::NTuple{N,NT}) where {F, N, names, NT<:NamedTuple{names}}
    return map(up, svoid(x,u), u, transp(ka))
end

@inline function transp(ntup::NTuple{N,NT}) where {N, names, NT<:NamedTuple{names}}
    M = length(names) # compile-time constant
    getindexer(i) = coll->coll[i]
    t = ntuple(
        let nt=ntup
            i->map(getindexer(i), nt)
        end,
        Val{M}())
    return NamedTuple{names}(t)
end

# Initial-value problem solver
"""
    solver = IVPSolver(model, scheme, dt, u0, mutating=false)
    solver = IVPSolver(model, scheme, dt)
Use `solver` with `advance`.

By default, `solver` is non-mutating, which is known to work with Zygote but allocates.
If `u0` is provided *and* `mutating=true`, `advance` should not allocate and have better performance.
The type and shape of `u0` (but not the values) are used to allocate scratch spaces, see `scratch_space`.
This allows the non-mutating mode to work with ForwardDiff.
"""
struct IVPSolver{F,Scheme,Scratch}
    dt::F # time step
    scheme::Scheme
    scratch::Scratch # scratch space, or void

    function IVPSolver(scheme::S, dt::F, u0, mutating=false) where {F,S}
        scr = mutating ? scratch_space(scheme, u0) : void
        new{F,S,typeof(scr)}(dt, scheme, scr)
    end
    IVPSolver(scheme::S, dt::F) where {F,S} = new{F,S,Void}(dt, scheme, void)
end

"""
    future, t = advance!(future, solver, present, t, N)
    future, t = advance!(void, solver, present, t, N)
"""
function advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::IVPSolver, state::State, t, N::Int) where State
    @assert N>0
    @assert typeof(t)==typeof(dt)
    state = advance!(storage, scheme, state, t, dt, scratch)::State
    for i=2:N
        state = advance!(storage, scheme, state, t+(i-1)*dt, dt, scratch)::State
    end
    return state, t+N*dt
end


"""
    scheme = RungeKutta4(model)
4th-order Runge Kutta scheme for `model`. Pass `scheme` to `IVPSolver`.
"""
struct RungeKutta4{Model}
    model::Model
end

function scratch_space((; model)::RungeKutta4, u0)
    k() = model_dstate(model, u0)
    return (scratch = scratch_space(model, u0), k0 = k(), k1 = k(), k2 = k(), k3 = k())
end

function advance!(future, (; model)::RungeKutta4, u0, t0, dt, (; scratch, k0, k1, k2, k3))
    k0 = tendencies!(k0, model, u0, scratch, t0)
    # u1 = u0 + dt/2*k0
    u1 = update!(future, u0, dt / 2, k0)
    k1 = tendencies!(k1, model, u1, scratch, t0 + dt / 2)
    # u2 = u1 - dt/2*k0 + dt/2*k1 == u0 + dt/2*k1
    u2 = update!(future, u1, -dt / 2, k0, dt / 2, k1)
    k2 = tendencies!(k2, model, u2, scratch, t0 + dt / 2)
    # u3 = u2 - dt/2*k1 + dt*k2 == u0 + dt*k2
    u3 = update!(future, u2, -dt / 2, k1, dt, k2)
    k3 = tendencies!(k3, model, u3, scratch, t0 + dt)
    # u4 = u0 + (k0+k3)*(dt/6) + (k1+k2)*(dt/3)
    future = update!(future, u3, dt / 6, k0, dt / 3, k1, -2dt / 3, k2, dt / 6, k3)
    return future
end

end
