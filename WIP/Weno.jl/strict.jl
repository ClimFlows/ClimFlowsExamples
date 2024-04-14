"""
    struct StrictFloat{F<:AbstractFloat} <: AbstractFloat
    const Strict32 = StrictFloat{Float32}
    const Strict64 = StrictFloat{Float64}

    x = Strict32(1.5) <: AbstractFloat
    x = Strict64(1.5) <: AbstractFloat
`Float32/64` types that do not mix well with `Float64/32` and with each other.
Use these types to detect the unwanted occurence of mixed precision.
This is useful to get best performance when computing with `Float32`. This is for debugging only and comes with a
performance penalty. Coverage of basic function and operators may be incomplete.
To cover additional functions, see `strict_float_unary_math` and `strict_float_binary_math`
"""
struct StrictFloat{F<:AbstractFloat} <: AbstractFloat
    value::F
end

Base.show(io::IO, mime::MIME"text/plain", x::StrictFloat) = show(io, mime, x.value)

@inline StrictFloat{F}(x::StrictFloat{F}) where F<:AbstractFloat = x
@inline StrictFloat{F}(x::Rational) where F<:AbstractFloat = StrictFloat{F}(F(x))
@inline Base.Float32(x::StrictFloat) = Float32(x.value)
@inline Base.Float64(x::StrictFloat) = Float64(x.value)

" See `StrictFloat` "
const Strict32 = StrictFloat{Float32}
" See `StrictFloat` "
const Strict64 = StrictFloat{Float64}

@inline Base.one(::StrictFloat{F}) where F = StrictFloat(one(F))
@inline Base.zero(::StrictFloat{F}) where F = StrictFloat(zero(F))
@inline Base.promote_rule(::Type{StrictFloat{F}}, ::Type{F}) where F = StrictFloat{F}
@inline Base.promote_rule(::Type{F}, ::Type{StrictFloat{F}}) where F = StrictFloat{F}

for typ in (:Int, :Int32, :Irrational, :Rational)
    @eval begin
        @inline Base.promote_rule(::Type{StrictFloat{F}}, ::Type{$typ}) where F<:AbstractFloat = StrictFloat{F}
        @inline Base.promote_rule(::Type{$typ}, ::Type{StrictFloat{F}}) where F<:AbstractFloat = StrictFloat{F}
    end
end

@inline Base.randn(rng, ::Type{StrictFloat{F}}) where F = StrictFloat{F}(Base.randn(rng, F))

"""
    strict_float_unary_math((:fun1, :fun2))

Specializes one-argument functions `fun1` and `fun2` for `StrictFloat` argument.

    strict_float_binary_math((:op1, :op2))

Specializes two-argument functions `op1` and `op2` for `StrictFloat{F}` arguments with same `F`.

The implementations look like :

    @inline Base.fun1(x::StrictFloat) = StrictFloat(Base.fun1(x.value))
    @inline Base.op1(x::StrictFloat{F}, y::StrictFloat{F}) where F = StrictFloat(Base.op1(x.value, y.value))

Use these functions if you hit an error when using a mathematical function not covered.
You may also specialize the offending function manually as above.
"""
strict_float_unary_math(funs) = for fun in funs
    @eval @inline Base.$fun(x::StrictFloat) = StrictFloat(Base.$fun(x.value))
end

"See `strict_float_unary_math`"
strict_float_binary_math(ops) = for op in ops
    @eval @inline Base.$op(x::StrictFloat{F}, y::StrictFloat{F}) where F = StrictFloat(Base.$op(x.value, y.value))
end

strict_float_unary_math((:inv, :-, :cos, :sin, :exp, :log, :asin, :acos, :sqrt, :ceil, :round))
strict_float_binary_math((:/, :*, :-, :+, :^, :rem))

# Integer / boolean functions
for fun in (:Int32, :Int64)
    @eval @inline Base.$fun(x::StrictFloat) = Base.$fun(x.value)
end
# Integer / boolean binary operators
for op in (:<, :>, :<=, :>=)
    @eval @inline Base.$op(x::StrictFloat{F}, y::StrictFloat{F}) where F = Base.$op(x.value, y.value)
end

# Other
Base.round(x::S, mode::RoundingMode) where S<:StrictFloat = S(round(x.value, mode))
Base.:*(x::UInt8, y::S) where S<:StrictFloat = S(x*y.value)
Base.:*(x::Irrational, y::S) where S<:StrictFloat = S(x*y.value)
Base.eps(::Type{StrictFloat{F}}) where F = eps(F)
