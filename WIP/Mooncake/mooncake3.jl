using Pkg; Pkg.activate(@__DIR__)
using Revise
using BenchmarkTools

import ForwardDiff as FD
using SIMD
using CFDomains.VoronoiOperators: pdv
using CFDomains.LazyExpressions: @lazy, LazyExpression

#==== FD + SIMD stuff ===#

# Base.promote_rule(::Type{F}, T::Type{SIMD.Vec{N,F}}) where {N,F} = T
# Base.convert(T::Type{SIMD.Vec{N,F}}, x::F) where {N,F} = T(x)

# is_compatible(::SVec{F}, ::F) where F = true
# is_compatible(::F, ::SVec{F}) where F = true

const SVec{F,N} = SIMD.Vec{N,F}

@inline FD.can_dual(::Type{SIMD.Vec{N, F}}) where {N, F} = FD.can_dual(F)

@inline FD._mul_partial(partial::SIMD.Vec, x::SIMD.Vec) = partial * x
@inline FD._mul_partial(partial::SVec{F}, x::F) where F = partial * x
@inline FD._mul_partial(partial::F, x::SVec{F}) where F = partial * x

@inline Base.:*(x::SIMD.Vec, partials::FD.Partials) = partials*x

@inline function Base.:*(partials::FD.Partials, x::SIMD.Vec)
    return FD.Partials(FD.scale_tuple(partials.values, x))
end

@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv::S, partial::FD.Partials{M,S}) where {T,F,N,M, S<:SVec{F,N}}
    return FD.Dual{T}(val, deriv*partial)   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::S, partial1::FD.Partials{M,S}, deriv2::F, partial2::FD.Partials{M,F}) where {T,F,N,M, S<:SVec{F,N}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::F, partial1::FD.Partials{M,F}, deriv2::S, partial2::FD.Partials{M,S}) where {T,F,N,M, S<:SVec{F,N}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end
@inline function FD.dual_definition_retval(::Val{T}, val::S, deriv1::S, partial1::FD.Partials{M,S}, deriv2::S, partial2::FD.Partials{M,S}) where {T,F,N,M, S<:SVec{F,N}}
    return FD.Dual{T}(val, FD._mul_partials(partial1, partial2, deriv1, deriv2))   
end

#==== test partial derivative in hot loop ====#

using ManagedLoops:@with, @vec
using LoopManagers
using SIMDMathFunctions

Base.sincos(x::SIMD.Vec) = sin(x), cos(x)
Base.rtoldefault(::Type{SVec{F,N}}) where {F,N} = Base.rtoldefault(F)

same_eltype(::Tuple{Vararg{AbstractArray{T}}}) where T = T

function loop_FD(mgr, fun::Fun, fa, a) where Fun
    @with mgr let irange = eachindex(fa)
        @vec for i in irange
            fa[i] = pdv(fun, a[i])
        end
    end
end

function loop(mgr, fun::Fun, ff, a) where Fun
    @with mgr let irange = eachindex(ff)
        @vec for i in irange
            ff[i] = fun(c[i])
        end
    end
end

function loop_FD2(mgr, fun2::Fun, fa, fb, a, b) where Fun
    @with mgr let irange = eachindex(fa,fb)
        @vec for i in irange
            @inbounds fa[i], fb[i] = pdv(fun2, a[i], b[i])
        end
    end
end

function loop_FD3(mgr, fun3::Fun, fa, fb, fc, a, b, c) where Fun
    @with mgr let irange = eachindex(fa,fb,fc)
        @vec for i in irange
            @inbounds fa[i], fb[i], fc[i] = pdv(fun3, a[i], b[i], c[i])
        end
    end
end

function collect!(mgr, output, lazy)
    @with mgr let irange = eachindex(output)
        @vec for i in irange 
           @inbounds output[i] = lazy[i]
        end
    end
end

function f(mgr, cc, a, g) 
    @lazy c(a ; g) = a+g/2
    @inline collect!(mgr, cc, c)
end

fun2(x,y) = cos(x)*sin(y)
fun3(x,y,z) = cos(x)*sin(y)*exp(z)

mgr = LoopManagers.VectorizedCPU(8)
F, N = Float32, 1024
a = randn(F, N*N);
b = randn(F, N*N);
c = randn(F, N*N);
g = F(9.81)
fa = similar(a);
fb = similar(b);
fc = similar(c);

loop_FD(mgr, sin, fa, a)
@assert fa ≈ cos.(a)
loop_FD2(mgr, *, fa, fb, a, b)
@assert fa ≈ b
@assert fb ≈ a
loop_FD3(mgr, *, fa, fb, fc, a, b, c)
@assert fa ≈ b.*c
@assert fb ≈ a.*c
@assert fc ≈ a.*b

@benchmark loop_FD($mgr, sin, $fa, $a)
@benchmark loop($mgr, cos, $fa, $a)

f(mgr, c, a, g)
f(mgr, c, a, b)

@code_native debuginfo=:none f(mgr, c, a, g)
@code_native debuginfo=:none f(mgr, c, a, b)
# @benchmark collect!(mgr, cc, c)
@benchmark f($mgr, $c, $a, $g)
@benchmark f($mgr, $c, $a, $b)

#==========================================================#

#=

macro lazy(expr)
    esc(expand_lazy(expr))
end

function expand_lazy(expr)
    # get function name, body, inputs and params
    def = splitdef(expr)
    name = def[:name]
    inputs = def[:args]
    params = def[:kwargs]
    # anonymous function taking inputs and params as regular args
    args = [inputs...; params...]
    fun = combinedef(Dict(:body=>def[:body], :args=>args, :kwargs=>[], :whereparams=>[]))
    # construct lazy expression
    params = Expr(:tuple, params...)
    inputs = Expr(:tuple, inputs...)
    :( $name = lazy_expr(($fun), ($inputs), ($params)) )
end

const Arrays{N} = Tuple{Vararg{AbstractArray{<:Any, N}}} # a tuple of arrays of rank N

function lazy_expr(fun::Fun, inputs::Arrays{N}, params) where {Fun, N}
    LazyExpression{Fun, typeof(inputs), typeof(params)}(fun, inputs, params) 
end

struct LazyExpression{Fun, Inputs, Params}
    fun::Fun
    inputs::Inputs
    params::Params
end

@prop function Base.getindex(lazy::LazyExpression{T}, i) where T
    @boundscheck foreach(x->checkbounds(x,i), lazy.inputs)
    inputs = map(y-> (@inbounds y[i]), lazy.inputs)
    params = get(lazy.params, i)
    @inline lazy.fun(inputs..., params...)
end

@inline get(x::Tuple, i) = map(y->get(y, i), x)
@inline get(x::AbstractVector, i) = @inbounds x[i]
@inline get(x::Number, i) = x

=#

#==== partial derivative of N-variable functions, N ∈ {1,2,3} ====#

# When constructing the Dual numbers xx,yy,
# dual parts must have the same type as the primal part.

#=
function pdv(fun1::T, x) where T
    xx = FD.Dual{T}(x, one(x))
    ff = fun1(xx)
    return ff.partials.values[1]
end

function pdv(fun2::T, x, y) where T
    xx = FD.Dual{T}(x, one(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y))
    ff = fun2(xx, yy)
    return ff.partials.values
end

function pdv(fun3::T, x, y, z) where T
    xx = FD.Dual{T}(x, one(x), zero(x), zero(x))
    yy = FD.Dual{T}(y, zero(y), one(y), zero(y))
    zz = FD.Dual{T}(z, zero(z), zero(z), one(z))
    ff = fun3(xx, yy, zz)
    return ff.partials.values
end
=#
