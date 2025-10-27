using Pkg; Pkg.activate(@__DIR__)
using Revise
using InteractiveUtils
using BenchmarkTools
using Cthulhu

@time_imports begin
    import DifferentiationInterface as DI
    using DifferentiationInterface: Constant as Const
    import Mooncake

    using FixedSizeArrays

    using ClimFlowsData: DYNAMICO_meshfile, DYNAMICO_reader
    using NetCDF: ncread
    using CFDomains: VoronoiSphere, Stencils
end

includet("stubs.jl")
includet("fixed_size.jl")
includet("rrules.jl")

# include("check_fixed_size.jl")

#=============================================================================#

includet("stencils.jl")

function sum_grad2(f, grad, sph, app!)
    app!(grad, Gradient(sph), f)
    return sum(x->x^2, grad)
end

# `choices` is for discrete parameters, while `params` is for continuous parameters (floats)
# Values in `params` will be converted to `choices.precision`.

choices = (
    precision = Float32,
    meshname = DYNAMICO_meshfile("uni.2deg.mesh.nc"),
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)

q = randn(choices.precision, length(vsphere.lon_i))
gradq = similar(q, length(vsphere.lon_e)) # gradient is computed on edges

@info "forward" sum_grad2(q, gradq, vsphere, apply!_internal)
@code_warntype sum_grad2(q, gradq, vsphere, apply!_internal)
# @descend sum_grad2(q, vsphere, apply!_internal)

backend = DI.AutoMooncake(; config=nothing)
prep = DI.prepare_gradient(sum_grad2, backend, q, Const(gradq), Const(vsphere), Const(apply!));
grad = DI.gradient(sum_grad2, prep, backend, q, Const(gradq), Const(vsphere), Const(apply!));
prep_internal = DI.prepare_gradient(sum_grad2, backend, q, Const(gradq), Const(vsphere), Const(apply!_internal));
grad_internal = DI.gradient(sum_grad2, prep_internal, backend, q, Const(gradq), Const(vsphere), Const(apply!_internal));
@assert grad ≈ grad_internal

display(@benchmark DI.gradient(sum_grad2, prep_internal, backend, q, Const(gradq), Const(vsphere), Const(apply!_internal)))
display(@benchmark DI.gradient(sum_grad2, prep, backend, q, Const(gradq), Const(vsphere), Const(apply!)))
