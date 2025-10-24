using Pkg; Pkg.activate(@__DIR__)

import DifferentiationInterface as DI
import Mooncake
using Mooncake: zero_fcodual, CoDual, NoFData, rrule!!, primal

using ClimFlowsData: DYNAMICO_meshfile, DYNAMICO_reader
using NetCDF: ncread
using CFDomains: VoronoiSphere, Stencils
using Random

using BenchmarkTools
using InteractiveUtils

#=============================================================================#

using DifferentiationInterface
import Mooncake

f(x) = sum(sin, x)
backend = AutoMooncake(; config=nothing)
x = ones(1_000)
prep = prepare_gradient(f, backend, x)
@info "grad" gradient(f, prep, backend, x) cos.(x)

f_custom(x) = sum(sin, x)
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(f_custom), Vector{<:AbstractFloat}}

# codual = reverse codual = primal + adjoint
# fcodual = forward codual = primal + tangent

function Mooncake.rrule!!(::CoDual{typeof(f_custom), NoFData}, rx::CoDual{Vector{Float64}, Vector{Float64}})
    x = primal(rx)
    result = f(x)
    cx = cos.(x)
    function f_pb!!(dy) 
        # q is an Array => incremented in-place during reverse pass
        @. rx.dx += dy*cx
        return Mooncake.NoRData(), Mooncake.NoRData()
    end
    return zero_fcodual(result), f_pb!!
end

prep = prepare_gradient(f_custom, backend, x);
@info "grad" gradient(f_custom, prep, backend, x) ≈ gradient(f, backend, x)

@benchmark gradient(f_custom, prep, backend, x)

exit()

#=============================================================================#

function sum_grad2(f, sph)
    result = zero(eltype(f))
    @inbounds for edge in eachindex(f)
        grad = Stencils.gradient(sph, edge)
        result += grad(f)^2
    end
    return result
end

# `choices` is for discrete parameters, while `params` is for continuous parameters (floats)
# Values in `params` will be converted to `choices.precision`.

choices = (
    precision = Float32,
    meshname = DYNAMICO_meshfile("uni.1deg.mesh.nc"),
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)
grad = Stencils.gradient(vsphere)

q = randn(choices.precision, length(vsphere.lon_e))
backend = DI.AutoMooncake(; config=nothing)
prep = DI.prepare_gradient(sum_grad2, backend, q, DI.Constant(grad))
    
display(@benchmark sum_grad2($q, $grad))
display(@benchmark DI.gradient(sum_grad2, $prep, $backend, $q, DI.Constant($grad)))

# define custom rule ; keep the original function to test adjoint
sum_grad2_custom(f, sph) = sum_grad2(f, sph)

function Mooncake.rrule!!(::CoDual{typeof(sum_grad2_custom)}, q::CoDual{V,V}, sph::CoDual) where {V<:Vector}
    @info "custom rule" typeof(sph) typeof(q)
    result = sum_grad2(q.x, sph.x)
    return zero_fcodual(result), sum_grad2_pb!!
end

function sum_grad2_pb!!(args...)
    @info typeof(args)
end

prep = DI.prepare_gradient(sum_grad2_custom, backend, q, grad);
@time DI.gradient(sum_grad2_custom, prep, backend, q, grad);

Mooncake.rrule!!(zero_fcodual(sum_grad2_custom), zero_fcodual(q), zero_fcodual(grad))

rng = Random.default_rng()
mode = Mooncake.ReverseMode
Mooncake.TestUtils.test_rule(rng, sum_grad2_custom, q, grad ; mode)


