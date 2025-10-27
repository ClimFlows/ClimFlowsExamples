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

function sum_op2(f, tmp, op, app!)
    app!(tmp, op, f)
    return sum(x->x^2, tmp)
end

function test_op(q, tmp, op)
    @info "forward" sum_op2(q, tmp, op, apply!_internal)
    @code_warntype sum_op2(q, tmp, op, apply!_internal)
    
    backend = DI.AutoMooncake(; config=nothing)
    # prep_internal = DI.prepare_gradient(sum_op2, backend, q, Const(tmp), Const(op), Const(apply!_internal));
    # grad_internal = DI.gradient(sum_op2, prep_internal, backend, q, Const(tmp), Const(op), Const(apply!_internal));
    prep = DI.prepare_gradient(sum_op2, backend, q, Const(tmp), Const(op), Const(apply!));
    grad = DI.gradient(sum_op2, prep, backend, q, Const(tmp), Const(op), Const(apply!));
    # @assert grad ≈ grad_internal    

    # run_internal() = DI.gradient(sum_op2, prep_internal, backend, q, Const(tmp), Const(op), Const(apply!_internal))
    run() = DI.gradient(sum_op2, prep, backend, q, Const(tmp), Const(op), Const(apply!))
    for _ in 1:10
        @time run()
        # @time run_internal()
    end
    display(@benchmark $run())
    # display(@benchmark $run_internal())
end

# `choices` is for discrete parameters, while `params` is for continuous parameters (floats)
# Values in `params` will be converted to `choices.precision`.

choices = (
    precision = Float32,
    meshname = DYNAMICO_meshfile("uni.1deg.mesh.nc"),
)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)

#=============== TRiSK ================#

q = randn(choices.precision, length(vsphere.lon_e))
tmp = similar(q)
test_op(q, tmp, TRiSK(vsphere))

#=============== Gradient ================#

q = randn(choices.precision, length(vsphere.lon_i))
tmp = similar(q, length(vsphere.lon_e)) # gradient is computed on edges
grad_op = Gradient(vsphere)
test_op(q, tmp, grad_op)
